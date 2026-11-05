"""Dataset wiring for precomputed dynamics-risk scores."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RiskMetadata:
    path: str
    n_frames: int
    median: float
    p95: float
    maximum: float


class RiskScoreStore:
    """O(log N) lookup of suite-normalized risk by persistent frame index."""

    def __init__(self, path: str | Path):
        path = Path(path)
        payload = np.load(path)
        frame_index = payload["frame_index"].astype(np.int64)
        risk = payload["risk_normalized"].astype(np.float32)
        if len(frame_index) != len(risk):
            raise ValueError("frame_index and risk_normalized lengths differ")
        if len(np.unique(frame_index)) != len(frame_index):
            raise ValueError("risk file contains duplicate frame indices")
        order = np.argsort(frame_index)
        self.frame_index = frame_index[order]
        self.risk = risk[order]
        self.metadata = RiskMetadata(
            path=str(path),
            n_frames=len(risk),
            median=float(np.median(risk)),
            p95=float(np.quantile(risk, 0.95)),
            maximum=float(np.max(risk)),
        )

    def lookup(self, frame_index: int) -> float:
        position = int(np.searchsorted(self.frame_index, frame_index))
        if position >= len(self.frame_index) or self.frame_index[position] != frame_index:
            raise KeyError(f"No risk score for global frame index {frame_index}")
        return float(self.risk[position])


class RiskAugmentedDataset(Dataset):
    """Attach `dynamics_risk` using the sample's persistent global index."""

    def __init__(self, dataset: Dataset, risk_path: str | Path, index_key: str = "index"):
        self.dataset = dataset
        self.risk_store = RiskScoreStore(risk_path)
        self.index_key = index_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        if self.index_key not in sample:
            raise KeyError(
                f"Sample at wrapper position {index} lacks persistent index key "
                f"{self.index_key!r}; refusing positional fallback because it could "
                "silently misalign risk scores after filtering or subsetting. "
                f"Available keys: {sorted(sample)}"
            )
        frame_index = sample[self.index_key]
        if torch.is_tensor(frame_index):
            frame_index = int(frame_index.item())
        sample["dynamics_risk"] = torch.tensor(
            self.risk_store.lookup(int(frame_index)), dtype=torch.float32
        )
        return sample
