"""Local (no-Ray) inference CLIs.

These commands use artifacts produced by the local training CLI:
- best.pt
- labels.json (index -> subject_id)
- model_metadata.json

They are intentionally lightweight so they work in Docker/WSL2 and CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mmbiometric.inference.predictor import Predictor


def _load_artifacts(run_dir: Path, checkpoint: Path | None) -> tuple[Path, dict[int, str], dict]:
    run_dir = run_dir.resolve()
    ckpt_path = (checkpoint or (run_dir / "best.pt")).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Expected best.pt in {run_dir} or pass --checkpoint"
        )

    labels_path = run_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.json not found in {run_dir}. Re-run training (it should write labels.json)."
        )
    idx_to_label_raw = json.loads(labels_path.read_text(encoding="utf-8"))
    # JSON keys are strings
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}

    meta_path = run_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"model_metadata.json not found in {run_dir}. Re-run training (it should write model_metadata.json)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return ckpt_path, idx_to_label, meta


def predict_one_main() -> None:
    p = argparse.ArgumentParser(description="Run single-pair inference (iris + fingerprint).")
    p.add_argument("--run-dir", required=True, help="Training run dir containing best.pt, labels.json, model_metadata.json")
    p.add_argument("--iris", required=True, help="Path to iris image")
    p.add_argument("--fingerprint", required=True, help="Path to fingerprint image")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path (defaults to <run-dir>/best.pt)")

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path, idx_to_label, meta = _load_artifacts(run_dir, Path(args.checkpoint) if args.checkpoint else None)

    predictor = Predictor.load(
        ckpt_path=ckpt_path,
        idx_to_label=idx_to_label,
        backbone=meta["backbone"],
        embedding_dim=int(meta["embedding_dim"]),
        dropout=float(meta["dropout"]),
        image_size=int(meta["image_size"]),
        device=meta.get("device", "cpu"),
    )

    pred = predictor.predict(iris_path=Path(args.iris), fingerprint_path=Path(args.fingerprint))
    print(json.dumps({"predicted_subject_id": pred}, indent=2))


def predict_batch_main() -> None:
    p = argparse.ArgumentParser(description="Run batch inference from a manifest.parquet file.")
    p.add_argument("--run-dir", required=True, help="Training run dir containing best.pt, labels.json, model_metadata.json")
    p.add_argument("--manifest", required=True, help="Path to manifest.parquet with columns: modality, subject_id, filepath")
    p.add_argument("--out", required=True, help="Output parquet path for predictions")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint path (defaults to <run-dir>/best.pt)")

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path, idx_to_label, meta = _load_artifacts(run_dir, Path(args.checkpoint) if args.checkpoint else None)

    predictor = Predictor.load(
        ckpt_path=ckpt_path,
        idx_to_label=idx_to_label,
        backbone=meta["backbone"],
        embedding_dim=int(meta["embedding_dim"]),
        dropout=float(meta["dropout"]),
        image_size=int(meta["image_size"]),
        device=meta.get("device", "cpu"),
    )

    df = pd.read_parquet(args.manifest)
    required = {"modality", "subject_id", "filepath"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    # Build pairs by subject_id. For each subject, pick the first iris and first fingerprint.
    iris_df = df[df["modality"] == "iris"].sort_values("filepath")
    fin_df = df[df["modality"] == "fingerprint"].sort_values("filepath")

    iris_pick = iris_df.groupby("subject_id", as_index=False).first()[["subject_id", "filepath"]].rename(
        columns={"filepath": "iris_path"}
    )
    fin_pick = fin_df.groupby("subject_id", as_index=False).first()[["subject_id", "filepath"]].rename(
        columns={"filepath": "fingerprint_path"}
    )

    pairs = iris_pick.merge(fin_pick, on="subject_id", how="inner")
    if pairs.empty:
        raise ValueError("No subject_id had both iris and fingerprint entries. Check your manifest.")

    preds = []
    for row in pairs.itertuples(index=False):
        pred = predictor.predict(iris_path=Path(row.iris_path), fingerprint_path=Path(row.fingerprint_path))
        preds.append({
            "subject_id": row.subject_id,
            "iris_path": row.iris_path,
            "fingerprint_path": row.fingerprint_path,
            "predicted_subject_id": pred,
        })

    out_df = pd.DataFrame(preds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote predictions: {out_path}")
