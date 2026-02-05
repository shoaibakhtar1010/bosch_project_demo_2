from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class Sample:
    iris: torch.Tensor
    fingerprint: torch.Tensor
    label: torch.Tensor


def collate_samples(batch: list[Sample]) -> Sample:
    """Collate a list of `Sample` objects into a single batched `Sample`.

    Torch's default collate doesn't handle dataclasses, so any DataLoader
    reading this dataset must use this `collate_fn`.
    """

    if not batch:
        raise ValueError("Empty batch")

    iris = torch.stack([s.iris for s in batch], dim=0)
    fingerprint = torch.stack([s.fingerprint for s in batch], dim=0)
    labels = torch.stack([s.label for s in batch], dim=0)
    return Sample(iris=iris, fingerprint=fingerprint, label=labels)


class MultimodalBiometricDataset(Dataset[Sample]):
    """Dataset returning (iris_tensor, fingerprint_tensor, label)."""

    def __init__(
        self,
        manifest_path: Path,
        iris_transform: Transform,
        fingerprint_transform: Transform,
        label_to_index: Dict[str, int],
    ) -> None:
        self.df = pd.read_parquet(manifest_path)
        self.iris_transform = iris_transform
        self.fingerprint_transform = fingerprint_transform
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        iris_img = Image.open(row["iris_path"]).convert("RGB")
        fp_img = Image.open(row["fingerprint_path"]).convert("RGB")
        y = torch.tensor(self.label_to_index[str(row["subject_id"])], dtype=torch.long)
        return Sample(
            iris=self.iris_transform(iris_img),
            fingerprint=self.fingerprint_transform(fp_img),
            label=y,
        )
