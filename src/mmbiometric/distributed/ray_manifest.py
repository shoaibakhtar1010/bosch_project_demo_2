"""Distributed manifest builder (Ray).

This project uses the Kaggle dataset "multimodal-iris-fingerprint-biometric-data".
The folder structure (inside the KaggleHub download directory) looks like:

    <DATASET_DIR>/IRIS and FINGERPRINT DATASET/
        1/
          Fingerprint/  (10 fingerprint images)
          left/         (5 iris images)
          right/        (5 iris images)
        2/
          ...

The earlier implementation guessed modality by substring matching against the full
absolute path. That breaks because the dataset root folder contains both words
"iris" and "fingerprint"; so everything was classified as iris and the manifest
became empty. This module fixes that by:
  - Resolving the true dataset root: <DATASET_DIR>/IRIS and FINGERPRINT DATASET
  - Classifying modality from RELATIVE path parts (Fingerprint vs left/right)
  - Extracting subject id from the first numeric folder in the RELATIVE path

It produces a manifest parquet with columns:
    subject_id, iris_path, fingerprint_path

To match the Kaggle notebook behaviour and to increase sample count, we create
the Cartesian product per subject: for each subject, every iris image is paired
with every fingerprint image.
"""

# fmt: off
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import ray


_IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class _Partial:
    subject_id: str
    modality: str  # "left_iris" | "right_iris" | "fingerprint"
    path: str


def _resolve_dataset_root(dataset_dir: Path) -> Path:
    """Return the folder that directly contains subject-id folders (1..45)."""
    # KaggleHub downloads a version folder that contains this dataset folder.
    candidate = dataset_dir / "IRIS and FINGERPRINT DATASET"
    if candidate.exists() and candidate.is_dir():
        return candidate

    # Sometimes users may pass the dataset folder itself.
    if dataset_dir.name.lower() == "iris and fingerprint dataset" and dataset_dir.is_dir():
        return dataset_dir

    # Fallback: find a child directory that contains both tokens.
    for child in dataset_dir.iterdir():
        if child.is_dir():
            name = child.name.lower()
            if "iris" in name and "fingerprint" in name:
                return child

    # Last resort: assume the provided path is the dataset root.
    return dataset_dir


def _extract_subject_id(rel_parts: tuple[str, ...]) -> str | None:
    """Extract the subject id from relative path parts."""
    for part in rel_parts:
        if part.isdigit():
            return part
    return None


def _guess_modality_from_relative(rel_parts: tuple[str, ...], stem_lower: str) -> str | None:
    """Classify modality using only RELATIVE parts (no absolute-path leakage)."""
    parts = [p.lower() for p in rel_parts]

    # Fingerprint images live under a folder named "Fingerprint".
    if any(p == "fingerprint" or "fingerprint" in p for p in parts):
        return "fingerprint"

    # Iris images live under "left" or "right".
    if any(p == "left" for p in parts):
        return "left_iris"
    if any(p == "right" for p in parts):
        return "right_iris"

    # Fallback heuristics (safe because rel parts won't include dataset root name).
    if "finger" in stem_lower:
        return "fingerprint"
    if "left" in stem_lower:
        return "left_iris"
    if "right" in stem_lower:
        return "right_iris"


    return None


@ray.remote
def _process_one(path_str: str, dataset_root_str: str) -> _Partial | None:
    p = Path(path_str)

    if not p.is_file():
        return None

    if p.suffix.lower() not in _IMAGE_EXTS:
        return None

    root = Path(dataset_root_str)

    try:
        rel = p.relative_to(root)
        rel_parts = rel.parts
    except Exception:
        # If rel fails (e.g., different drive), use basename-only heuristics.
        rel_parts = p.parts

    sid = _extract_subject_id(rel_parts)
    if sid is None:
        return None

    modality = _guess_modality_from_relative(rel_parts, p.stem.lower())
    if modality is None:
        return None

    return _Partial(subject_id=str(sid), modality=modality, path=str(p))


def build_manifest_distributed(
    dataset_dir: str,
    output_dir: str,
    num_cpus: int = 4,
) -> pd.DataFrame:
    """Scan dataset_dir, build manifest, and save it under output_dir."""
    dataset_root = _resolve_dataset_root(Path(dataset_dir)).resolve()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidate files locally (fast enough; only ~1k files).
    files = [str(p) for p in dataset_root.rglob("*") if p.is_file()]

    # Dispatch lightweight parsing to Ray.
    futures = [
        _process_one.options(num_cpus=0.1).remote(f, str(dataset_root)) for f in files
    ]
    partials = [p for p in ray.get(futures) if p is not None]

    left_by_sid: dict[str, list[str]] = {}
    right_by_sid: dict[str, list[str]] = {}
    fp_by_sid: dict[str, list[str]] = {}

    for item in partials:
        if item.modality == "left_iris":
            left_by_sid.setdefault(item.subject_id, []).append(item.path)
        elif item.modality == "right_iris":
            right_by_sid.setdefault(item.subject_id, []).append(item.path)
        elif item.modality == "fingerprint":
            fp_by_sid.setdefault(item.subject_id, []).append(item.path)

        rows: list[dict[str, str]] = []
        for sid in sorted(set(left_by_sid) | set(right_by_sid) | set(fp_by_sid), key=lambda x: int(x) if x.isdigit() else x):
            left_list = sorted(left_by_sid.get(sid, []))
            right_list = sorted(right_by_sid.get(sid, []))
            fp_list = sorted(fp_by_sid.get(sid, []))
            if not left_list or not right_list or not fp_list:
                continue

            # best coverage: Cartesian product of all three
            for left_path in left_list:
                for right_path in right_list:
                    for fp_path in fp_list:
                        rows.append(
                            {
                                "subject_id": sid,
                                "left_iris_path": left_path,
                                "right_iris_path": right_path,
                                "fingerprint_path": fp_path,
                            }
                        )

    df = pd.DataFrame(rows, columns=["subject_id", "left_iris_path", "right_iris_path", "fingerprint_path"])


    # Fail fast if we didn't find any usable samples.
    if df.empty:
        raise RuntimeError(
            "No (iris, fingerprint) pairs found. "
            f"dataset_root={dataset_root} ; output_dir={out_dir}. "
            "Verify that your DATASET_DIR points to the KaggleHub download folder "
            "(versions/1) and that it contains 'IRIS and FINGERPRINT DATASET/1..45'."
        )

    df_path = out_dir / "manifest.parquet"
    df.to_parquet(df_path, index=False)
    return df
