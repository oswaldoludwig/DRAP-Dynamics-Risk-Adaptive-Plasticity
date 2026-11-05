"""
kappatune_vla.py

Extends KappaTune (Ludwig, 2025 -- https://github.com/oswaldoludwig/kappaTune)
from text/audio LLMs to Vision-Language-Action (VLA) models.

Original KappaTune scores each eligible weight tensor W by its condition number
    kappa(W) = sigma_max(W) / sigma_min(W)
and unfreezes a fixed budget of the *lowest*-kappa tensors, on the hypothesis
(formalized in the KappaTune paper via the entropy bound under a fixed
Frobenius-norm constraint) that low-kappa tensors are less specialized /
less committed to pre-trained knowledge, so adapting them causes less
catastrophic forgetting.

The extension here is architectural, not algorithmic: a VLA model has three
structurally distinct tensor populations that the original paper never
scored together:

  1. vision backbone tensors      (e.g. SigLIP / DINOv2 conv & attention weights)
  2. language backbone tensors    (e.g. Llama-2/3 attention & MLP weights)
  3. action-head tensors          (discretized-action classifier head, OR
                                   a diffusion / flow-matching MLP head)

Group (3) is the open question: it did not exist in the original KappaTune
experiments (text classification, ASR adaptor). Action heads are typically
much lower-rank and trained from a different initialization scheme than
attention/MLP blocks, so their condition-number distribution -- and thus
whether "low kappa" still predicts "safe to adapt" -- needs its own empirical
check. This module supports scoring/selecting each group independently so
that hypothesis can be tested group-by-group instead of assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Tensor grouping
# --------------------------------------------------------------------------- #

@dataclass
class TensorGroupPatterns:
    """Regexes used to bucket named parameters into architectural groups.

    Defaults target OpenVLA's Prismatic backbone (SigLIP + DINOv2 fused vision
    encoder, Llama-2 7B language backbone, discretized-action de-tokenizer
    head reusing the LM head). Adjust for other VLAs (e.g. pi0's flow-matching
    action expert has its own transformer -- point `action_head` at it).
    """
    vision: str = r"(vision_backbone|featurizer|patch_embed|siglip|dinov2)"
    language: str = r"(language_model|llm_backbone|llama)"
    action_head: str = r"(action_head|action_decoder|flow_matching|diffusion_head|lm_head)"


# --------------------------------------------------------------------------- #
# Eligibility: which tensors are even candidates for scoring
# --------------------------------------------------------------------------- #

def is_eligible(name: str, param: torch.Tensor, min_dim: int = 2) -> bool:
    """Mirrors KappaTune's eligibility rule: biases, norm scalars (1-D
    tensors), and embeddings' padding rows are excluded. SVD requires >=2D,
    and we reshape conv/patch-embed kernels to 2D before scoring."""
    if param.dim() < min_dim:
        return False
    if any(tok in name for tok in ("bias", "norm", "ln_", "layernorm")):
        return False
    return True


def group_of(name: str, patterns: TensorGroupPatterns) -> str:
    if re.search(patterns.action_head, name, flags=re.IGNORECASE):
        return "action_head"
    if re.search(patterns.vision, name, flags=re.IGNORECASE):
        return "vision"
    if re.search(patterns.language, name, flags=re.IGNORECASE):
        return "language"
    return "other"


# --------------------------------------------------------------------------- #
# Condition number computation
# --------------------------------------------------------------------------- #

def reshape_to_2d(w: torch.Tensor) -> torch.Tensor:
    """Conv / patch-embed kernels (out_ch, in_ch, kh, kw) -> (out_ch, in_ch*kh*kw),
    matching the reshape rule KappaTune specifies for multi-dim tensors."""
    if w.dim() == 2:
        return w
    return w.reshape(w.shape[0], -1)


@torch.no_grad()
def condition_number(w: torch.Tensor, eps: float = 1e-8) -> float:
    """kappa(W) = sigma_max / sigma_min, computed on CPU in fp32 for numerical
    stability (SVD on large bf16 CUDA tensors is both slow and ill-behaved)."""
    w2d = reshape_to_2d(w).detach().to(dtype=torch.float32, device="cpu")
    # svdvals is cheaper than full SVD when only singular values are needed.
    svals = torch.linalg.svdvals(w2d)
    sigma_max = svals[0]
    sigma_min = svals[svals > eps][-1] if (svals > eps).any() else svals[-1] + eps
    return (sigma_max / (sigma_min + eps)).item()


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

@dataclass
class SelectionResult:
    trainable: set[str] = field(default_factory=set)
    kappa_by_name: dict[str, float] = field(default_factory=dict)
    group_by_name: dict[str, str] = field(default_factory=dict)


def select_low_kappa_tensors(
    model: nn.Module,
    budget_per_group: dict[str, int],
    patterns: TensorGroupPatterns | None = None,
    min_dim: int = 2,
) -> SelectionResult:
    """Score every eligible tensor's condition number, bucket by architectural
    group, and select the `budget_per_group[g]` lowest-kappa tensors *within
    each group independently*.

    Scoring per-group (rather than globally, as the original single-backbone
    KappaTune does) is the key methodological change: action-head tensors are
    usually much smaller than attention/MLP tensors, so a global top-K by
    kappa would almost never select any action-head tensor, silently
    reproducing "freeze the whole action head" as a special case rather than
    testing the hypothesis on it.
    """
    patterns = patterns or TensorGroupPatterns()
    kappa_by_name: dict[str, float] = {}
    group_by_name: dict[str, str] = {}
    by_group: dict[str, list[tuple[str, float]]] = {}

    for name, param in model.named_parameters():
        if not is_eligible(name, param, min_dim=min_dim):
            continue
        g = group_of(name, patterns)
        k = condition_number(param)
        kappa_by_name[name] = k
        group_by_name[name] = g
        by_group.setdefault(g, []).append((name, k))

    trainable: set[str] = set()
    for g, tensors in by_group.items():
        k_budget = budget_per_group.get(g, 0)
        if k_budget <= 0:
            continue
        tensors_sorted = sorted(tensors, key=lambda t: t[1])  # ascending kappa
        trainable.update(name for name, _ in tensors_sorted[:k_budget])

    return SelectionResult(trainable=trainable, kappa_by_name=kappa_by_name, group_by_name=group_by_name)


def apply_selection(model: nn.Module, selection: SelectionResult) -> int:
    """Freeze everything, then unfreeze exactly the selected tensors.
    Returns the number of trainable parameters (for logging / matching
    budgets against a LoRA baseline)."""
    n_trainable = 0
    for name, param in model.named_parameters():
        if name in selection.trainable:
            param.requires_grad_(True)
            n_trainable += param.numel()
        else:
            param.requires_grad_(False)
    return n_trainable


def summarize_selection(selection: SelectionResult) -> str:
    from collections import Counter
    counts = Counter(selection.group_by_name[n] for n in selection.trainable)
    lines = [f"Selected {len(selection.trainable)} tensors:"]
    for g, c in counts.items():
        lines.append(f"  {g}: {c} tensors")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# INVERSE-kappa ablation (control condition, per KappaTune Fig. 1)
# --------------------------------------------------------------------------- #

def select_high_kappa_tensors(
    model: nn.Module,
    budget_per_group: dict[str, int],
    patterns: TensorGroupPatterns | None = None,
    min_dim: int = 2,
) -> SelectionResult:
    """Mirror of select_low_kappa_tensors but takes the HIGHEST-kappa tensors
    per group. Always run this as a control: the KappaTune paper's strongest
    evidence for the mechanism (not just the outcome) was that inverting the
    selection reproduces forgetting -- the same check should be run here
    before trusting any VLA-specific result."""
    patterns = patterns or TensorGroupPatterns()
    base = select_low_kappa_tensors(model, {}, patterns, min_dim)  # scores only
    by_group: dict[str, list[tuple[str, float]]] = {}
    for name, k in base.kappa_by_name.items():
        by_group.setdefault(base.group_by_name[name], []).append((name, k))

    trainable: set[str] = set()
    for g, tensors in by_group.items():
        k_budget = budget_per_group.get(g, 0)
        if k_budget <= 0:
            continue
        tensors_sorted = sorted(tensors, key=lambda t: t[1], reverse=True)  # descending kappa
        trainable.update(name for name, _ in tensors_sorted[:k_budget])

    return SelectionResult(trainable=trainable, kappa_by_name=base.kappa_by_name, group_by_name=base.group_by_name)
