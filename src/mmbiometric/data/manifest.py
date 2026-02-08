from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ManifestRow:
    subject_id: str
    iris_path: Path
    fingerprint_path: Path


def _guess_modality(p: Path) -> str | None:
    """Best-effort modality inference from the *path structure*.

    Important Windows/KaggleHub pitfall:
      This Kaggle dataset has a folder named "IRIS and FINGERPRINT DATASET".
      If we search the *entire path string* for the substring "fingerprint",
      we incorrectly classify **iris** images as fingerprint too (because that
      substring exists in the dataset root folder name). That makes `iris_by_subject`
      empty and the final manifest ends up with 0 rows.

    So we prefer *exact folder names* (parent directory) and only fall back to
    filename tokens.
    """
    parts = [part.lower() for part in p.parts]
    parent = p.parent.name.lower()
    name = p.name.lower()

    # 1) Strong signals: explicit folders
    if parent in {"fingerprint", "fingerprints", "fp"}:
        return "fingerprint"
    if parent in {"left", "right"}:
        return "iris"

    # 2) Folder tokens (exact matches only)
    if "fingerprint" in parts or "fp" in parts:
        return "fingerprint"
    if "iris" in parts or "left" in parts or "right" in parts:
        return "iris"

    # 3) Filename fallback
    if "finger" in name or "fingerprint" in name:
        return "fingerprint"
    if "iris" in name:
        return "iris"

    return None


def _extract_subject_id(p: Path, dataset_root: Path, subject_regex: str) -> str:
    """Extract a subject id while avoiding digits from the *dataset root*.

    KaggleHub paths often end with ".../versions/1". If we run a naive regex
    (e.g. r"(\\d+)") against the full absolute path, the first match can be that
    trailing "1", causing every file to get the same subject_id.

    Strategy:
      0) Prefer a directory component that fully matches the regex (deepest upward)
      1) Try regex against filename
      2) Try regex against the path relative to dataset_root
      3) Fallback to parent folder name
    """
    # 0) Prefer directory components (relative to dataset_root)
    try:
        rel_parts = p.resolve().relative_to(dataset_root.resolve()).parts
    except Exception:
        rel_parts = p.parts

    for comp in reversed(rel_parts[:-1]):  # exclude filename
        m = re.fullmatch(subject_regex, comp)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    # 1) filename only
    m = re.search(subject_regex, p.name)
    if m:
        return m.group(1) if m.groups() else m.group(0)

    # 2) relative path (exclude dataset_root digits like versions/1)
    try:
        rel = str(p.resolve().relative_to(dataset_root.resolve()))
    except Exception:
        rel = str(p)
    m = re.search(subject_regex, rel)
    if m:
        return m.group(1) if m.groups() else m.group(0)

    # 3) fallback
    return p.parent.name


def build_manifest(
    dataset_dir: Path,
    output_path: Path,
    subject_regex: str = r"(\d+)",
) -> Path:
    """Build a (subject_id, iris_path, fingerprint_path) manifest by pairing modalities."""
    dataset_dir = dataset_dir.resolve()

    files: list[Path] = [
        p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ]

    iris_by_subject: dict[str, list[Path]] = {}
    fp_by_subject: dict[str, list[Path]] = {}

    for p in tqdm(files, desc="Scanning images"):
        modality = _guess_modality(p)
        if modality is None:
            continue
        sid = _extract_subject_id(p, dataset_dir, subject_regex)
        if modality == "iris":
            iris_by_subject.setdefault(sid, []).append(p)
        else:
            fp_by_subject.setdefault(sid, []).append(p)

    rows: list[ManifestRow] = []
    for sid, iris_list in iris_by_subject.items():
        fp_list = fp_by_subject.get(sid, [])
        if not fp_list:
            continue

        iris_list = sorted(iris_list)
        fp_list = sorted(fp_list)
        n = min(len(iris_list), len(fp_list))

        for i in range(n):
            rows.append(
                ManifestRow(
                    subject_id=sid,
                    iris_path=iris_list[i],
                    fingerprint_path=fp_list[i],
                )
            )

    df = pd.DataFrame(
        {
            "subject_id": [r.subject_id for r in rows],
            "iris_path": [str(r.iris_path) for r in rows],
            "fingerprint_path": [str(r.fingerprint_path) for r in rows],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path
