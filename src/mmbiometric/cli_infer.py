"""Local (no-Ray) inference CLIs.

Artifacts expected inside --run-dir:
- best.pt
- labels.json (index -> subject_id)
- model_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mmbiometric.inference.predictor import Predictor


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    """
    If checkpoint is relative, interpret it relative to run_dir (NOT repo root).
    """
    run_dir = run_dir.resolve()
    if checkpoint is None:
        ckpt = run_dir / "best.pt"
    else:
        checkpoint = Path(checkpoint)
        ckpt = (run_dir / checkpoint) if not checkpoint.is_absolute() else checkpoint
    return ckpt.resolve()


def _load_artifacts(run_dir: Path, checkpoint: Path | None) -> tuple[Path, dict[int, str], dict]:
    run_dir = run_dir.resolve()
    ckpt_path = _resolve_checkpoint(run_dir, checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Expected best.pt in {run_dir} or pass --checkpoint"
        )

    labels_path = run_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found in {run_dir}.")

    idx_to_label_raw = json.loads(labels_path.read_text(encoding="utf-8"))
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}

    meta_path = run_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"model_metadata.json not found in {run_dir}.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ckpt_path, idx_to_label, meta


def predict_one_main() -> None:
    p = argparse.ArgumentParser(description="Run single-pair inference (iris + fingerprint).")
    p.add_argument("--run-dir", required=True, help="Training run dir containing best.pt, labels.json, model_metadata.json")
    p.add_argument("--iris", required=True, help="Path to iris image")
    p.add_argument("--fingerprint", required=True, help="Path to fingerprint image")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint (relative to run-dir if not absolute)")
    p.add_argument("--topk", type=int, default=0, help="If >0, also print top-k probabilities")

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

    out = {"predicted_subject_id": pred}
    if args.topk and args.topk > 0:
        out["topk"] = predictor.predict_topk(Path(args.iris), Path(args.fingerprint), k=args.topk)

    print(json.dumps(out, indent=2))


def predict_batch_main() -> None:
    p = argparse.ArgumentParser(description="Run batch inference from a manifest.parquet file.")
    p.add_argument("--run-dir", required=True, help="Training run dir containing best.pt, labels.json, model_metadata.json")
    p.add_argument("--manifest", required=True, help="manifest.parquet with columns: modality, subject_id, filepath")
    p.add_argument("--out", required=True, help="Output parquet path for predictions")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint (relative to run-dir if not absolute)")

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
        preds.append(
            {
                "subject_id": row.subject_id,
                "iris_path": row.iris_path,
                "fingerprint_path": row.fingerprint_path,
                "predicted_subject_id": pred,
            }
        )

    out_df = pd.DataFrame(preds)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[OK] Wrote predictions: {out_path}")
