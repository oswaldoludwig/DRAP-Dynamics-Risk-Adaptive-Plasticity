"""Suite-normalized, frame-level risk-weighted tensor importance."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def per_sample_action_loss(logits, labels):
    """Return one causal action-token CE value per frame."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError(f"Unexpected logits/labels shapes: {logits.shape}, {labels.shape}")
    length = labels.shape[1]
    if logits.shape[1] < length:
        raise ValueError("Multimodal logits sequence is shorter than labels")
    aligned_logits = logits[:, -length:, :]
    shift_logits = aligned_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.shape[-1]
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab), shift_labels.view(-1),
        ignore_index=-100, reduction="none"
    ).view_as(shift_labels)
    valid = shift_labels.ne(-100)
    return (token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)


def consolidation_penalty(named_parameters, importance, reference, coefficient):
    terms = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or name not in importance:
            continue
        ref = reference[name]
        if ref.device != parameter.device:
            ref = ref.to(parameter.device)
        # Sum preserves a per-parameter protection force for large VLA tensors.
        terms.append(float(importance[name]) *
                     (parameter.float() - ref.float()).square().sum())
    if not terms:
        device = next((p.device for _, p in named_parameters), "cpu")
        return torch.zeros((), device=device)
    return 0.5 * float(coefficient) * torch.stack(terms).sum()


def load_state(checkpoint):
    path = Path(checkpoint) / "risk_tensor_importance.pt"
    if not path.is_file():
        return {}, {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["importance"], payload["reference"]


def move_reference_to_device(reference, named_parameters):
    named_dict = dict(named_parameters)
    return {name: value.to(named_dict[name].device, dtype=named_dict[name].dtype)
            for name, value in reference.items() if name in named_dict}


def estimate(model, dataloader, max_batches=None):
    """Estimate risk-weighted importance.

    With importance batch size 1, this is one risk-weighted squared gradient per
    demonstration frame, eliminating cross-frame gradient cancellation.
    """
    model.eval()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    sums = {name: 0.0 for name, _ in named}
    batches = 0
    examples = 0
    mean_risk = 0.0

    for batch in dataloader:
        if max_batches is not None and batches >= max_batches:
            break
        risk = batch.pop("dynamics_risk").to("cuda", dtype=torch.float32)
        batch = {k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        sample_loss = per_sample_action_loss(output.logits, batch["labels"])
        if batches == 0:
            reconstructed = float(sample_loss.mean().detach())
            native = float(output.loss.detach())
            error = abs(reconstructed - native) / max(abs(native), 1e-8)
            print(f"Importance-loss alignment: native={native:.6f}, "
                  f"reconstructed={reconstructed:.6f}, relative_error={error:.3e}, "
                  f"logits_shape={tuple(output.logits.shape)}, "
                  f"labels_shape={tuple(batch['labels'].shape)}", flush=True)
            if error > 0.05:
                raise RuntimeError("Per-frame loss differs from native loss by >5%")
        # Risk is already normalized once over the entire suite. No batch renorm.
        weighted_loss = torch.mean(risk * sample_loss)
        weighted_loss.backward()
        for name, parameter in named:
            if parameter.grad is not None:
                sums[name] += float(parameter.grad.detach().float().square().mean())
        batches += 1
        examples += int(risk.numel())
        mean_risk += float(risk.mean())

    if batches == 0:
        raise RuntimeError("No batches used for importance estimation")
    importance = {name: value / batches for name, value in sums.items()}
    positive = torch.tensor([v for v in importance.values() if v > 0], dtype=torch.float64)
    median = max(float(torch.median(positive)) if positive.numel() else 1.0, 1e-30)
    importance = {name: value / median for name, value in importance.items()}
    reference = {name: p.detach().to("cpu", dtype=torch.bfloat16) for name, p in named}
    scores = torch.tensor(list(importance.values()), dtype=torch.float32)
    report = {
        "batches": batches, "examples": examples,
        "importance_batch_size_observed": examples / batches,
        "importance_estimator": "per-frame" if examples == batches else "minibatch",
        "mean_batch_risk": mean_risk / batches,
        "risk_normalization": "offline suite median=1; no minibatch renormalization",
        "raw_positive_median": median, "n_tensors": len(importance),
        "score_p50": float(torch.quantile(scores, .50)) if scores.numel() else 0.0,
        "score_p95": float(torch.quantile(scores, .95)) if scores.numel() else 0.0,
        "score_max": max(importance.values(), default=0.0),
    }
    model.zero_grad(set_to_none=True)
    return importance, reference, report


def merge(old, new, decay=1.0):
    return {name: float(decay) * old.get(name, 0.0) + new.get(name, 0.0)
            for name in set(old) | set(new)}


def save(out_dir, importance, reference, report):
    out_dir = Path(out_dir)
    torch.save({"importance": importance, "reference": reference},
               out_dir / "risk_tensor_importance.pt")
    (out_dir / "risk_tensor_importance_report.json").write_text(json.dumps(report, indent=2))
