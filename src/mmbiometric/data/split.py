from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitPaths:
    train_manifest: Path
    val_manifest: Path


def split_manifest(
    manifest_path: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> SplitPaths:
    """
    Split manifest into train/val *within each subject_id*.

    Important:
      - We are training a classifier over subject_id.
      - If you split by subject_id, val may contain identities not in train -> KeyError.
      - Therefore we split *within each subject* (hold out some samples per subject).

    Windows/Pandas note:
      - Pandas Index -> to_numpy() can be read-only.
      - np.random.Generator.shuffle() shuffles in-place and fails on read-only arrays.
      - Use rng.permutation() to get a shuffled *copy* safely.
    """
    df = pd.read_parquet(manifest_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_manifest.parquet"
    val_path = out_dir / "val_manifest.parquet"

    if len(df) == 0:
        df.to_parquet(train_path, index=False)
        df.to_parquet(val_path, index=False)
        return SplitPaths(train_manifest=train_path, val_manifest=val_path)

    if "subject_id" not in df.columns:
        raise ValueError("Manifest missing required column: subject_id")

    if not (0.0 <= float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1). Got: {val_fraction}")

    rng = np.random.default_rng(seed)
    subj = df["subject_id"].astype(str)

    train_parts = []
    val_parts = []

    for _sid, g in df.groupby(subj, sort=True):
        # g.index.to_numpy() can be read-only -> do NOT shuffle in-place.
        idx = g.index.to_numpy()
        idx = rng.permutation(idx)  # safe shuffled copy

        n = len(idx)
        if val_fraction <= 0.0 or n < 2:
            train_parts.append(df.loc[idx])
            continue

        n_val = int(np.floor(n * float(val_fraction)))
        n_val = max(1, n_val)
        n_val = min(n_val, n - 1)  # keep at least 1 in train

        val_parts.append(df.loc[idx[:n_val]])
        train_parts.append(df.loc[idx[n_val:]])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0]
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else df.iloc[0:0]

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    return SplitPaths(train_manifest=train_path, val_manifest=val_path)
