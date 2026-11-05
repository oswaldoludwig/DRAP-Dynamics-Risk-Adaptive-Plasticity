"""Memory-conscious online diagonal EWC state for continual OpenVLA training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

NamedParameters = List[Tuple[str, torch.nn.Parameter]]


def state_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint) / "ewc_state.pt"


def load_cpu(checkpoint: str | Path):
    path = state_path(checkpoint)
    if not path.is_file():
        return {}, {}, {"loaded": False, "source": str(path)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta", {}))
    meta.update({"loaded": True, "source": str(path)})
    return payload.get("fisher", {}), payload.get("anchor", {}), meta


def move_state_to_parameters(
    named_parameters: NamedParameters,
    fisher_cpu: Dict[str, torch.Tensor],
    anchor_cpu: Dict[str, torch.Tensor],
    fisher_device_dtype: torch.dtype = torch.float32,
):
    """Move matching state once; keep anchors BF16 to reduce GPU memory."""
    fisher, anchor = {}, {}
    names = {name for name, _ in named_parameters}
    unexpected = sorted((set(fisher_cpu) & set(anchor_cpu)) - names)
    if unexpected:
        raise KeyError(f"EWC state contains non-trainable tensors: {unexpected[:10]}")
    for name, parameter in named_parameters:
        if name not in fisher_cpu or name not in anchor_cpu:
            continue
        if fisher_cpu[name].shape != parameter.shape or anchor_cpu[name].shape != parameter.shape:
            raise ValueError(f"EWC state shape mismatch for {name}")
        fisher[name] = fisher_cpu[name].to(parameter.device, dtype=fisher_device_dtype)
        anchor[name] = anchor_cpu[name].to(parameter.device, dtype=parameter.dtype)
    return fisher, anchor


def penalty(named_parameters, fisher, anchor, coefficient):
    if coefficient < 0:
        raise ValueError("EWC coefficient must be non-negative")
    if not fisher:
        device = named_parameters[0][1].device if named_parameters else "cpu"
        return torch.zeros((), device=device)
    total = None
    # Process tensor-by-tensor so full-model temporary drift matrices are not retained.
    for name, parameter in named_parameters:
        if name not in fisher:
            continue
        drift = parameter.float() - anchor[name].float()
        term = (fisher[name].float() * drift.square()).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=named_parameters[0][1].device)
    return 0.5 * float(coefficient) * total


def _sample_positive_quantiles(tensors: Dict[str, torch.Tensor], samples_per_tensor=4096):
    """Bounded-memory diagnostics; never concatenate the full Fisher."""
    samples = []
    generator = torch.Generator(device="cpu").manual_seed(7)
    for value in tensors.values():
        flat = value.reshape(-1)
        if flat.numel() > samples_per_tensor:
            idx = torch.randint(0, flat.numel(), (samples_per_tensor,), generator=generator)
            flat = flat[idx]
        positive = flat[flat > 0].float()
        if positive.numel():
            samples.append(positive)
    if not samples:
        return 0.0, 0.0, 0.0
    sample = torch.cat(samples)
    return (
        float(torch.quantile(sample, 0.50)),
        float(torch.quantile(sample, 0.95)),
        float(sample.max()),
    )


@torch.enable_grad()
def estimate_empirical_fisher(model, dataloader, named_parameters, max_batches):
    """Frame-level diagonal empirical Fisher, accumulated on CPU in FP32."""
    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    sums = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in named_parameters
    }
    batches = examples = 0
    loss_sum = 0.0
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    for batch in dataloader:
        if batches >= max_batches:
            break
        batch = {
            key: (value.to("cuda", non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        loss = model(**batch).loss
        loss.backward()
        batch_size = int(next(v for v in batch.values() if torch.is_tensor(v)).shape[0])
        for name, parameter in named_parameters:
            if parameter.grad is None:
                continue
            # One tensor-sized transfer at a time. The temporary is released each loop.
            increment = parameter.grad.detach().float().square().cpu()
            sums[name].add_(increment)
            del increment
        model.zero_grad(set_to_none=True)
        batches += 1
        examples += batch_size
        loss_sum += float(loss.detach())
        if batches == 1 or batches % 25 == 0:
            print(json.dumps({"fisher_batch": batches, "fisher_examples": examples}), flush=True)

    if not batches:
        raise RuntimeError("No batches used for Fisher estimation")
    # Normalize in place, avoiding a second full Fisher dictionary.
    for value in sums.values():
        value.div_(float(batches))

    anchor = {
        name: parameter.detach().to("cpu", dtype=torch.bfloat16).clone()
        for name, parameter in named_parameters
    }
    p50, p95, maximum = _sample_positive_quantiles(sums)
    report = {
        "estimator": "diagonal_empirical_fisher",
        "batches": batches,
        "examples": examples,
        "observed_batch_size": examples / batches,
        "frame_level": examples == batches,
        "mean_estimation_loss": loss_sum / batches,
        "n_tensors": len(sums),
        "n_parameters": sum(value.numel() for value in sums.values()),
        "fisher_positive_p50_sampled": p50,
        "fisher_positive_p95_sampled": p95,
        "fisher_positive_max_sampled": maximum,
        "quantile_samples_per_tensor": 4096,
    }
    model.train(was_training)
    return sums, anchor, report


def merge_online_in_place(old_fisher, new_fisher, decay):
    """F_old <- decay * F_old + F_new, without allocating a third dictionary."""
    if decay < 0:
        raise ValueError("EWC decay must be non-negative")
    if not old_fisher:
        return new_fisher
    for name, new_value in new_fisher.items():
        if name in old_fisher:
            old_value = old_fisher[name]
            if old_value.shape != new_value.shape:
                raise ValueError(f"Fisher shape mismatch for {name}")
            old_value.mul_(float(decay)).add_(new_value)
        else:
            old_fisher[name] = new_value
    return old_fisher


def save(out_dir, fisher, anchor, report):
    """Serialize existing CPU tensors without making full state copies."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"fisher": fisher, "anchor": anchor, "meta": report}
    torch.save(payload, out / "ewc_state.pt")
    (out / "ewc_report.json").write_text(json.dumps(report, indent=2))
