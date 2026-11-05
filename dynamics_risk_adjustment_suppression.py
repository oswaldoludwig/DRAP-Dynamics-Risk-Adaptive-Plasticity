"""Dynamics-Risk-Weighted Adjustment Suppression (DRW-AS).

The optimizer uses cumulative past-task importance only.  For tensor T,

    lr_T = base_lr / (1 + beta * Ibar_T) ** (1 / p)

where Ibar is cumulative importance normalized by its positive median.  A
CLASSP-style elementwise threshold can suppress updates whose accumulated
current gradient satisfies grad**2 <= threshold.  threshold=0.0 is the fully
relaxed default and therefore isolates denominator-based suppression.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple
import math
import torch


@dataclass
class SuppressionStats:
    n_tensors: int
    importance_positive_median: float
    scale_min: float
    scale_p50: float
    scale_p95: float
    scale_max: float
    effective_lr_min: float
    effective_lr_p50: float
    effective_lr_max: float
    threshold: float
    threshold_enabled: bool
    power: float
    beta: float

    def to_dict(self):
        return asdict(self)


def _positive_median(values: Iterable[float], eps: float) -> float:
    positive = torch.tensor(
        [float(v) for v in values if math.isfinite(float(v)) and float(v) > 0.0],
        dtype=torch.float64,
    )
    if positive.numel() == 0:
        return 1.0
    return max(float(torch.median(positive).item()), eps)


def normalized_importance(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    importance: Dict[str, float],
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], float]:
    """Normalize cumulative importance over the currently trainable tensors."""
    raw = {name: max(0.0, float(importance.get(name, 0.0))) for name, _ in named_parameters}
    median = _positive_median(raw.values(), eps)
    return {name: value / median for name, value in raw.items()}, median


def suppression_scales(
    named_parameters: List[Tuple[str, torch.nn.Parameter]],
    importance: Dict[str, float],
    beta: float,
    power: float,
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], float]:
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if power <= 0.0:
        raise ValueError("power must be strictly positive")
    normalized, median = normalized_importance(named_parameters, importance, eps)
    exponent = -1.0 / float(power)
    scales = {
        name: float((1.0 + float(beta) * normalized[name]) ** exponent)
        for name, _ in named_parameters
    }
    return scales, median


class DynamicsRiskSuppressedAdamW(torch.optim.AdamW):
    """AdamW with per-tensor LR suppression and optional CLASSP thresholding.

    Each selected tensor has its own parameter group, so importance scales the
    actual AdamW step, not the raw gradient.  This avoids Adam's approximate
    cancellation of a constant raw-gradient scale through its first and second
    moments.
    """

    def __init__(
        self,
        named_parameters: List[Tuple[str, torch.nn.Parameter]],
        importance: Dict[str, float],
        lr: float,
        beta: float = 1.0,
        power: float = 2.0,
        threshold: float = 0.0,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        self.parameter_names = {id(parameter): name for name, parameter in named_parameters}
        self.threshold = float(threshold)
        self.threshold_enabled = self.threshold > 0.0
        self.scales, importance_median = suppression_scales(
            named_parameters, importance, beta=beta, power=power
        )
        groups = []
        for name, parameter in named_parameters:
            scale = self.scales[name]
            groups.append({
                "params": [parameter],
                "lr": float(lr) * scale,
                "base_lr": float(lr),
                "suppression_scale": scale,
                "tensor_name": name,
            })
        super().__init__(
            groups, lr=float(lr), betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
        values = torch.tensor(list(self.scales.values()), dtype=torch.float64)
        effective = values * float(lr)
        self.suppression_stats = SuppressionStats(
            n_tensors=len(named_parameters),
            importance_positive_median=importance_median,
            scale_min=float(values.min()) if values.numel() else 1.0,
            scale_p50=float(torch.quantile(values, 0.50)) if values.numel() else 1.0,
            scale_p95=float(torch.quantile(values, 0.95)) if values.numel() else 1.0,
            scale_max=float(values.max()) if values.numel() else 1.0,
            effective_lr_min=float(effective.min()) if effective.numel() else float(lr),
            effective_lr_p50=float(torch.quantile(effective, 0.50)) if effective.numel() else float(lr),
            effective_lr_max=float(effective.max()) if effective.numel() else float(lr),
            threshold=self.threshold,
            threshold_enabled=self.threshold_enabled,
            power=float(power),
            beta=float(beta),
        )
        self._last_threshold_report = {
            "gradient_elements": 0,
            "kept_elements": 0,
            "suppressed_elements": 0,
            "kept_fraction": 1.0,
        }

    @torch.no_grad()
    def apply_threshold(self):
        """Mask accumulated gradients before clipping and AdamW moment updates.

        This is called after gradient accumulation.  At threshold=0.0 no mask is
        applied, which is the intended denominator-only baseline.
        """
        if not self.threshold_enabled:
            self._last_threshold_report = {
                "gradient_elements": 0,
                "kept_elements": 0,
                "suppressed_elements": 0,
                "kept_fraction": 1.0,
            }
            return self._last_threshold_report

        total = kept = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                mask = grad.detach().float().square().gt(self.threshold)
                total += mask.numel()
                kept += int(mask.sum().item())
                grad.mul_(mask.to(dtype=grad.dtype))
        self._last_threshold_report = {
            "gradient_elements": total,
            "kept_elements": kept,
            "suppressed_elements": total - kept,
            "kept_fraction": (kept / total) if total else 1.0,
        }
        return self._last_threshold_report

    def get_suppression_report(self):
        report = self.suppression_stats.to_dict()
        report.update(self._last_threshold_report)
        return report
