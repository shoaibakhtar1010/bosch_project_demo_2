# fmt: off
"""Single-machine pipeline runner (no Ray).

Why this exists:
- Ray can be painful on Windows (socket/IFNAME/Gloo) and in containers (artifacts paths).
- For most development and CI, a deterministic single-machine pipeline is enough.

This CLI exposes *explicit stages* so you can run them independently:
- manifest  -> writes manifest.parquet
- split     -> writes train_manifest.parquet and val_manifest.parquet
- train     -> trains a model and writes best.pt + labels.json + model_metadata.json
- predict-one / predict-batch -> inference using the artifacts from train

Under the hood it reuses the project’s existing non-Ray building blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mmbiometric.cli import build_manifest_main, train_main
from mmbiometric.cli_infer import predict_batch_main, predict_one_main
from mmbiometric.data.split import split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="mmbiometric-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Stage: manifest (delegates to existing CLI)
    sub.add_parser("manifest", help="Build manifest.parquet (delegates to mmbiometric-build-manifest)")

    # Stage: split
    p_split = sub.add_parser("split", help="Split a manifest.parquet into train/val manifests")
    p_split.add_argument("--manifest", required=True, help="Path to manifest.parquet")
    p_split.add_argument("--out-dir", required=True, help="Output directory for train_manifest.parquet and val_manifest.parquet")
    p_split.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    p_split.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")

    # Stage: train (delegates to existing CLI)
    sub.add_parser("train", help="Train model (delegates to mmbiometric-train)")

    # Stage: inference
    sub.add_parser("predict-one", help="Single-pair inference")
    sub.add_parser("predict-batch", help="Batch inference from manifest")

    # Parse only the first token, then re-dispatch to the real sub-CLI.
    args, rest = parser.parse_known_args()

    if args.cmd == "manifest":
        # Delegate to the existing CLI so we don't duplicate flags.
        build_manifest_main()
        return

    if args.cmd == "train":
        train_main()
        return

    if args.cmd == "predict-one":
        predict_one_main()
        return

    if args.cmd == "predict-batch":
        predict_batch_main()
        return

    if args.cmd == "split":
        manifest_path = Path(args.manifest).resolve()
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "train_manifest.parquet"
        val_path = out_dir / "val_manifest.parquet"
        split_manifest(
            in_path=manifest_path,
            train_path=train_path,
            val_path=val_path,
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
        )
        print(f"[OK] Wrote: {train_path}")
        print(f"[OK] Wrote: {val_path}")
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")
