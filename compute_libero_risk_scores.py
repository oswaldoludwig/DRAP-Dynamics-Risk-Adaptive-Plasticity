#!/usr/bin/env python3
"""Offline, policy-independent LIBERO dynamics-risk scoring.

Fits a local ridge-regression model in a sliding window around each transition:

    z_{t+1} = A_t z_t + B_t u_t + c_t

where z is a robustly standardized state and u is a robustly standardized
action. The per-frame risk is the one-step spectral norm sigma_max(A_t).
Risk is normalized per suite so the median valid score equals 1.0.

The output contains frame_index, episode_index, raw spectral norm, normalized
risk, window bounds, fit quality, and validity. No simulator or policy model is
used.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_BY_SUITE = {
    "libero_spatial": "lerobot/libero_spatial_image",
    "libero_object": "lerobot/libero_object_image",
    "libero_goal": "lerobot/libero_goal_image",
    "libero_long": "lerobot/libero_10_image",
}


@dataclass
class Config:
    suite: str
    repo_id: str
    state_key: str
    action_key: str
    episode_key: str
    frame_key: str
    window: int
    ridge: float
    min_transitions: int
    center_window: bool
    normalization: str
    clip_low: float
    clip_high: float


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _column(dataset, key: str) -> np.ndarray:
    """Read one LeRobot/HF column without decoding image columns."""
    base = getattr(dataset, "hf_dataset", dataset)
    try:
        values = base[key]
        return _to_numpy(values)
    except Exception:
        values = []
        for index in range(len(dataset)):
            item = dataset[index]
            if key not in item:
                raise KeyError(
                    f"Missing key {key!r}; sample keys are {sorted(item.keys())}"
                )
            values.append(_to_numpy(item[key]))
        return np.asarray(values)


def robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median/IQR scale, with standard-deviation and unit fallbacks."""
    center = np.nanmedian(x, axis=0)
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(x, axis=0)
    scale = np.where(scale > 1e-8, scale, std)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def fit_local_dynamics(
    state: np.ndarray,
    action: np.ndarray,
    transition_indices: np.ndarray,
    ridge: float,
) -> tuple[float, float]:
    """Return sigma_max(A) and normalized one-step residual RMSE."""
    current = state[transition_indices]
    control = action[transition_indices]
    target = state[transition_indices + 1]

    design = np.concatenate(
        [current, control, np.ones((len(transition_indices), 1))], axis=1
    )
    gram = design.T @ design
    penalty = ridge * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0  # do not penalize intercept
    coefficients = np.linalg.solve(gram + penalty, design.T @ target)

    state_dim = state.shape[1]
    # coefficients maps row vectors: next = current @ A.T + ...
    A = coefficients[:state_dim].T
    sigma_max = float(np.linalg.svd(A, compute_uv=False)[0])

    prediction = design @ coefficients
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    return sigma_max, rmse


def episode_ranges(episode_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous [start, end) ranges; validates episode contiguity."""
    if len(episode_ids) == 0:
        return []
    boundaries = np.flatnonzero(np.r_[True, episode_ids[1:] != episode_ids[:-1], True])
    ranges = []
    seen = set()
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        episode = int(episode_ids[left])
        if episode in seen:
            raise ValueError(f"Episode {episode} is not contiguous in the dataset")
        seen.add(episode)
        ranges.append((episode, int(left), int(right)))
    return ranges


def compute_scores(
    state: np.ndarray,
    action: np.ndarray,
    episode_ids: np.ndarray,
    *,
    window: int,
    ridge: float,
    min_transitions: int,
    center_window: bool,
) -> dict[str, np.ndarray]:
    n = len(state)
    raw = np.full(n, np.nan, dtype=np.float64)
    rmse = np.full(n, np.nan, dtype=np.float64)
    window_start = np.full(n, -1, dtype=np.int64)
    window_end = np.full(n, -1, dtype=np.int64)

    for episode, start, end in episode_ranges(episode_ids):
        # The final frame has no within-episode next state.
        transition_start, transition_end = start, end - 1
        n_transitions = transition_end - transition_start
        if n_transitions < min_transitions:
            continue

        for frame in range(start, end):
            anchor = min(frame, end - 2)
            if center_window:
                left = anchor - window // 2
            else:
                left = anchor - window + 1
            left = max(transition_start, left)
            right = min(transition_end, left + window)
            left = max(transition_start, right - window)
            indices = np.arange(left, right, dtype=np.int64)
            if len(indices) < min_transitions:
                continue
            raw[frame], rmse[frame] = fit_local_dynamics(
                state, action, indices, ridge
            )
            window_start[frame] = left
            window_end[frame] = right

    return {
        "risk_raw": raw,
        "fit_rmse": rmse,
        "window_start": window_start,
        "window_end": window_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=sorted(REPO_BY_SUITE))
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--episode-key", default="episode_index")
    parser.add_argument("--frame-key", default="index")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--min-transitions", type=int, default=16)
    parser.add_argument("--causal-window", action="store_true")
    parser.add_argument("--clip-low", type=float, default=0.25)
    parser.add_argument("--clip-high", type=float, default=4.0)
    parser.add_argument("--hf-home", default=None)
    args = parser.parse_args()

    if args.window < args.min_transitions:
        raise ValueError("--window must be >= --min-transitions")
    if args.ridge < 0:
        raise ValueError("--ridge must be nonnegative")

    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_id = args.repo_id or REPO_BY_SUITE[args.suite]
    print(f"Loading {repo_id}", flush=True)
    dataset = LeRobotDataset(repo_id)

    state = _column(dataset, args.state_key).astype(np.float64)
    action = _column(dataset, args.action_key).astype(np.float64)
    episode_ids = _column(dataset, args.episode_key).reshape(-1).astype(np.int64)
    try:
        frame_indices = _column(dataset, args.frame_key).reshape(-1).astype(np.int64)
    except KeyError:
        frame_indices = np.arange(len(state), dtype=np.int64)

    if state.ndim != 2 or action.ndim != 2:
        raise ValueError(f"Expected 2D state/action, got {state.shape} and {action.shape}")
    if not (len(state) == len(action) == len(episode_ids)):
        raise ValueError("state/action/episode columns have inconsistent lengths")

    state_center, state_scale = robust_center_scale(state)
    action_center, action_scale = robust_center_scale(action)
    state_scaled = (state - state_center) / state_scale
    action_scaled = (action - action_center) / action_scale

    scores = compute_scores(
        state_scaled,
        action_scaled,
        episode_ids,
        window=args.window,
        ridge=args.ridge,
        min_transitions=args.min_transitions,
        center_window=not args.causal_window,
    )
    valid = np.isfinite(scores["risk_raw"])
    if not valid.any():
        raise RuntimeError("No valid local dynamics fits were produced")

    median = float(np.median(scores["risk_raw"][valid]))
    if median <= 0 or not np.isfinite(median):
        raise RuntimeError(f"Invalid risk median: {median}")
    normalized = scores["risk_raw"] / median
    normalized = np.clip(normalized, args.clip_low, args.clip_high)
    normalized[~valid] = 1.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frame_index=frame_indices,
        episode_index=episode_ids,
        risk_raw=scores["risk_raw"].astype(np.float32),
        risk_normalized=normalized.astype(np.float32),
        fit_rmse=scores["fit_rmse"].astype(np.float32),
        valid=valid,
        window_start=scores["window_start"],
        window_end=scores["window_end"],
        state_center=state_center.astype(np.float32),
        state_scale=state_scale.astype(np.float32),
        action_center=action_center.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
    )

    config = Config(
        suite=args.suite,
        repo_id=repo_id,
        state_key=args.state_key,
        action_key=args.action_key,
        episode_key=args.episode_key,
        frame_key=args.frame_key,
        window=args.window,
        ridge=args.ridge,
        min_transitions=args.min_transitions,
        center_window=not args.causal_window,
        normalization="robust state/action scaling; risk divided by valid suite median",
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )
    summary = {
        "config": asdict(config),
        "n_frames": int(len(state)),
        "n_episodes": int(len(np.unique(episode_ids))),
        "valid_fraction": float(valid.mean()),
        "risk_raw": {
            "median": median,
            "p90": float(np.quantile(scores["risk_raw"][valid], 0.90)),
            "p95": float(np.quantile(scores["risk_raw"][valid], 0.95)),
            "p99": float(np.quantile(scores["risk_raw"][valid], 0.99)),
            "max": float(np.max(scores["risk_raw"][valid])),
        },
        "risk_normalized": {
            "median": float(np.median(normalized[valid])),
            "p95": float(np.quantile(normalized[valid], 0.95)),
            "max": float(np.max(normalized[valid])),
        },
        "npz": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
