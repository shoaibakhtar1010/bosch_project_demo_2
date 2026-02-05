"""Generate a tiny synthetic multimodal dataset for CI / smoke tests.

The project expects a directory structure that contains modality hints in the path
("iris" and "finger" substrings) and filenames containing a subject id.

This script creates something like:

  <out_dir>/iris/subject_0001_iris.png
  <out_dir>/fingerprint/subject_0001_finger.png
  ...

The images are random noise, so accuracy is meaningless; we only want to validate
that the pipeline runs end-to-end.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def _save_rand_image(path: Path, size: int) -> None:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output dataset directory")
    ap.add_argument("--num-subjects", type=int, default=6)
    ap.add_argument("--samples-per-subject", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    iris_dir = out_dir / "iris"
    finger_dir = out_dir / "fingerprint"

    # Create paired samples per subject.
    for sid in range(1, args.num_subjects + 1):
        for j in range(args.samples_per_subject):
            stem = f"subject_{sid:04d}_{j:02d}"
            _save_rand_image(iris_dir / f"{stem}_iris.png", args.image_size)
            _save_rand_image(finger_dir / f"{stem}_finger.png", args.image_size)

    print(str(out_dir.resolve()))


if __name__ == "__main__":
    main()
