"""
train_continual.py

Sequentially fine-tunes a VLA model across LIBERO task suites, applying one
of three adaptation strategies at each stage:

  - full        : unfreeze everything (upper bound on plasticity, upper
                   bound on forgetting risk -- the baseline everyone fears)
  - lora        : standard LoRA on attention/MLP projections (current
                   default in the OpenVLA / OpenVLA-OFT fine-tuning recipes)
  - kappatune   : this project's contribution -- per-group low-kappa
                   selective unfreezing (see kappatune_vla.py), with an
                   `--inverse-kappa` flag to run the high-kappa control.

Run one suite at a time so checkpoints and forgetting can be evaluated after
each stage (see eval_forgetting.py) -- don't just loop internally and save
only the final model, or you lose the per-stage forgetting curve, which is
the actual dependent variable of interest.

Example:
    python train_continual.py --suite libero_spatial --method kappatune \
        --budget-vision 40 --budget-language 75 --budget-action-head 20 \
        --init-checkpoint openvla/openvla-7b \
        --out-dir runs/kappatune/stage0_spatial

    python train_continual.py --suite libero_object --method kappatune \
        --budget-vision 40 --budget-language 75 --budget-action-head 20 \
        --init-checkpoint runs/kappatune/stage0_spatial \
        --out-dir runs/kappatune/stage1_object
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import SuiteSpec, load_suite_dataset
from kappatune_vla import (
    TensorGroupPatterns,
    apply_selection,
    select_high_kappa_tensors,
    select_low_kappa_tensors,
    summarize_selection,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=[
        "libero_spatial", "libero_object", "libero_goal", "libero_long", "libero_90",
    ])
    p.add_argument("--method", required=True, choices=["full", "lora", "kappatune"])
    p.add_argument("--init-checkpoint", required=True,
                    help="HF repo id or local path -- either openvla/openvla-7b "
                         "(stage 0) or a previous stage's --out-dir (stage i>0).")
    p.add_argument("--out-dir", required=True)

    # Updated default path to your local data folder
    #p.add_argument("--hf-namespace", default="/mnt/users/oswaldo_ludwig/robotics/VLA/data",
    #                help="Local datasets root folder or Hugging Face organization namespace.")

    p.add_argument("--hf-namespace", default="lerobot",
                    help="Hugging Face organization namespace.")

    # KappaTune-specific
    p.add_argument("--budget-vision", type=int, default=40)
    p.add_argument("--budget-language", type=int, default=75)
    p.add_argument("--budget-action-head", type=int, default=20)
    p.add_argument("--inverse-kappa", action="store_true",
                    help="Control condition: select HIGH-kappa tensors instead. "
                         "Always run this once per suite to confirm the mechanism, "
                         "per KappaTune paper Fig. 1.")

    # LoRA-specific
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])

    # Generic training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--num-demos", type=int, default=None,
                    help="Cap demos per suite; None = full suite.")
    return p.parse_args()


def parse_args_BKP() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True, choices=[
        "libero_spatial", "libero_object", "libero_goal", "libero_long", "libero_90",
    ])
    p.add_argument("--method", required=True, choices=["full", "lora", "kappatune"])
    p.add_argument("--init-checkpoint", required=True,
                    help="HF repo id or local path -- either openvla/openvla-7b "
                         "(stage 0) or a previous stage's --out-dir (stage i>0).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hf-namespace", default="physical-intelligence")

    # KappaTune-specific
    p.add_argument("--budget-vision", type=int, default=40)
    p.add_argument("--budget-language", type=int, default=75)
    p.add_argument("--budget-action-head", type=int, default=20)
    p.add_argument("--inverse-kappa", action="store_true",
                    help="Control condition: select HIGH-kappa tensors instead. "
                         "Always run this once per suite to confirm the mechanism, "
                         "per KappaTune paper Fig. 1.")

    # LoRA-specific
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])

    # Generic training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--num-demos", type=int, default=None,
                    help="Cap demos per suite; None = full suite.")
    return p.parse_args()


def load_model_and_processor(checkpoint: str):
    """OpenVLA loads via AutoModelForVision2Seq / AutoProcessor with
    trust_remote_code, matching the openvla/openvla-7b model card. Swap this
    function's body for pi0 / Octo's own loaders if targeting those instead
    -- the rest of the script (selection, loop, checkpointing) is
    model-agnostic as long as `model` exposes .named_parameters()."""
    
    import torch.nn as nn
    
    # Sledgehammer fix: Intercept the exact failure point in PyTorch's base 
    # attribute lookup to guarantee the remote Hugging Face class passes 
    # the SDPA compatibility check without throwing an AttributeError.
    if not getattr(nn.Module, "_kappatune_patched", False):
        _original_getattr = nn.Module.__getattr__
        
        def _patched_getattr(self, name):
            if name == "_supports_sdpa":
                return True
            return _original_getattr(self, name)
            
        nn.Module.__getattr__ = _patched_getattr
        nn.Module._kappatune_patched = True

    from transformers import AutoProcessor

    # AutoModelForVision2Seq was deprecated upstream in favor of
    # AutoModelForImageTextToText, and newer transformers releases (this
    # was hit on 4.57.6) no longer export the old name at all -- but
    # OpenVLA's own remote code registers itself against
    # AutoModelForVision2Seq specifically. Try the documented name first.
    try:
        from transformers import AutoModelForVision2Seq as _AutoModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as _AutoModelClass

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = _AutoModelClass.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    
    return model, processor


def configure_trainable_params(model, args) -> dict:
    """Applies the chosen adaptation strategy in place; returns a small dict
    of bookkeeping info (trainable param count, selection summary) that gets
    written to out-dir/selection.json for later auditing -- you want to be
    able to prove after the fact exactly which tensors were touched."""
    if args.method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"method": "full", "n_trainable": n_trainable}

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_targets,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, cfg)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"method": "lora", "n_trainable": n_trainable, "config": cfg.to_dict()}

    # kappatune
    patterns = TensorGroupPatterns()
    budget = {
        "vision": args.budget_vision,
        "language": args.budget_language,
        "action_head": args.budget_action_head,
    }
    selector = select_high_kappa_tensors if args.inverse_kappa else select_low_kappa_tensors
    selection = selector(model, budget, patterns)
    n_trainable = apply_selection(model, selection)
    print(summarize_selection(selection))
    return {
        "method": "kappatune",
        "inverse_kappa": args.inverse_kappa,
        "budget_per_group": budget,
        "n_trainable": n_trainable,
        "selected_tensors": sorted(selection.trainable),
    }


# --------------------------------------------------------------------------- #
# LeRobot sample -> OpenVLA training batch
# --------------------------------------------------------------------------- #

def _base_dataset(ds):
    """Unwrap torch Subset (used when --num-demos is set)."""
    return ds.dataset if (hasattr(ds, "dataset") and not hasattr(ds, "meta")) else ds


def compute_action_bounds(ds):
    """1st/99th percentile per action dim, used to map actions to [-1, 1]
    the way OpenVLA does before discretizing into 256 bins."""
    base = _base_dataset(ds)
    try:
        st = base.meta.stats["action"]
        if "q01" in st and "q99" in st:
            return (np.asarray(st["q01"], dtype=np.float32),
                    np.asarray(st["q99"], dtype=np.float32))
    except Exception:
        pass
    actions = np.asarray(base.hf_dataset["action"], dtype=np.float32)
    return np.percentile(actions, 1, axis=0), np.percentile(actions, 99, axis=0)


def make_collate_fn(processor, q01, q99, model_dtype=torch.bfloat16, n_bins=256):
    """LeRobotDataset yields {observation.images.image, task, action, ...}.
    OpenVLA wants pixel_values + input_ids + labels, where the 7 action dims
    are discretized into 256 bins and written into the LAST 256 token ids of
    the Llama vocabulary (id = vocab_size - digitize(a, bins)), followed by
    EOS; prompt tokens are masked with -100 so loss covers only the action."""
    from PIL import Image

    tokenizer = processor.tokenizer
    bins = np.linspace(-1, 1, n_bins)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id
    span = np.maximum(q99 - q01, 1e-8)

    def collate(samples):
        pvs, ids_l, lab_l = [], [], []
        for s in samples:
            img = s["observation.images.image"]
            if torch.is_tensor(img):
                if img.dtype.is_floating_point:
                    img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
                if img.shape[0] in (1, 3):          # CHW -> HWC
                    img = img.permute(1, 2, 0)
                img = img.cpu().numpy()
            pil = Image.fromarray(img).convert("RGB")

            task = s["task"]
            prompt = f"In: What action should the robot take to {task.lower()}?\nOut:"
            prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
            if prompt_ids[-1] != 29871:              # same trailing token OpenVLA adds at inference
                prompt_ids.append(29871)

            a = np.asarray(s["action"], dtype=np.float32).reshape(-1)[:7]
            a = np.clip(2.0 * (a - q01) / span - 1.0, -1.0, 1.0)
            action_ids = (tokenizer.vocab_size - np.digitize(a, bins)).tolist()

            ids_l.append(prompt_ids + action_ids + [eos_id])
            lab_l.append([-100] * len(prompt_ids) + action_ids + [eos_id])
            pvs.append(processor(prompt, pil, return_tensors="pt")["pixel_values"][0])

        B, L = len(samples), max(len(x) for x in ids_l)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(ids_l, lab_l)):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            labels[i, :len(lab)] = torch.tensor(lab)
            attn[i, :len(ids)] = 1
        return {"pixel_values": torch.stack(pvs).to(model_dtype),
                "input_ids": input_ids, "attention_mask": attn, "labels": labels}

    return collate


def train_one_stage(model, processor, dataloader, args) -> None:
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    model.train()
    step = 0
    t0 = time.time()
    
    # 1. Verify dataloader is not empty
    num_batches = len(dataloader)
    print(f"Starting training: {args.epochs} epochs, {num_batches} batches per epoch.")
    if num_batches == 0:
        raise ValueError("Dataloader returned 0 batches! Check dataset path and num_demos setting.")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v
                      for k, v in batch.items()}
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss / args.grad_accum
            loss.backward()
            step += 1
            
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optim.step()
                optim.zero_grad()
                
            # 2. Log on Step 1 and every 10 steps thereafter
            if step == 1 or step % 5 == 0:
                elapsed = time.time() - t0
                current_loss = loss.item() * args.grad_accum
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {step} | Batch {batch_idx+1}/{num_batches} | Loss: {current_loss:.4f} ({elapsed:.0f}s elapsed)")
                
        print(f"--- Epoch {epoch+1} finished ---")


def train_one_stage_BKP(model, processor, dataloader, args) -> None:
    optim = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in dataloader:
            batch = {k: v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v
                      for k, v in batch.items()}
            out = model(
                pixel_values=batch["image"],
                input_ids=batch["language_instruction_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["action_labels"],  # discretized-action tokens for OpenVLA;
                                                 # swap for a regression/flow loss on
                                                 # continuous-action VLAs (pi0-style).
            )
            loss = out.loss / args.grad_accum
            loss.backward()
            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optim.step()
                optim.zero_grad()
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {step} loss {loss.item()*args.grad_accum:.4f} "
                      f"({elapsed:.0f}s elapsed)")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(args.init_checkpoint)
    selection_info = configure_trainable_params(model, args)
    (out_dir / "selection.json").write_text(json.dumps(selection_info, indent=2))
    print(f"Trainable parameters: {selection_info['n_trainable']:,}")

    spec = SuiteSpec(
        name=args.suite,
        repo_id=f"{args.hf_namespace}/{args.suite}_image",
        num_demos=args.num_demos,
    )

    # Fail loudly and informatively if the dataset is empty / unreadable,
    # instead of silently "training" on nothing.
    dataset = load_suite_dataset(spec)
    n_frames = len(dataset)
    print(f"Dataset {spec.repo_id}: {n_frames:,} frames")
    if n_frames == 0:
        raise ValueError(f"Dataset {spec.repo_id} is empty.")
    print("Sample keys:", sorted(dataset[0].keys()))

    q01, q99 = compute_action_bounds(dataset)
    print("Action q01:", np.round(q01, 3), "| q99:", np.round(q99, 3))

    # Persist a PER-SUITE stats registry, merged forward from the init
    # checkpoint's own registry (if any) so earlier suites' entries survive.
    # This used to not be written at all, which meant eval_forgetting.py's
    # load_action_stats() always fell back to recomputing stats from a
    # single hardcoded suite (libero_spatial) for every suite it evaluated --
    # silently corrupting de-normalization (and the inferred gripper
    # convention) for every OTHER suite, including the one just trained on.
    # Different suites can have genuinely different action ranges/gripper
    # encodings, so each suite needs ITS OWN entry, not one shared pair.
    stats_path = Path(args.init_checkpoint) / "action_stats.json"
    registry = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    registry[args.suite] = {"q01": q01.tolist(), "q99": q99.tolist()}
    (out_dir / "action_stats.json").write_text(json.dumps(registry, indent=2))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=make_collate_fn(processor, q01, q99),
    )

    train_one_stage(model, processor, dataloader, args)

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Saved stage checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
