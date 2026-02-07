from __future__ import annotations

# Standard library imports
import argparse
import json
from pathlib import Path
from typing import Any

# Third‑party imports
import pandas as pd

# Local application imports
from mmbiometric.inference.predictor import Predictor


def _p(path_str: str) -> Path:
    return Path(path_str)


def predict_one_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Training run dir (contains best.pt, labels.json, model_metadata.json)")
    parser.add_argument("--iris", required=True, help="Path to iris image")
    parser.add_argument("--fingerprint", required=True, help="Path to fingerprint image")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint filename/path (default: best.pt in run-dir)")
    parser.add_argument("--topk", type=int, default=None, help="Optional top-k predictions to include")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args()

    run_dir = _p(args.run_dir)
    iris_path = _p(args.iris)
    fp_path = _p(args.fingerprint)

    predictor = Predictor.load(
        run_dir=run_dir,
        checkpoint=_p(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )

    out: dict[str, Any] = {"predicted_subject_id": predictor.predict(iris_path, fp_path)}
    if args.topk is not None and args.topk > 1:
        out["topk"] = predictor.predict_topk(iris_path, fp_path, k=args.topk)

    print(json.dumps(out, indent=2))
    return 0


def _predict_batch_from_paired_manifest(
    predictor: Predictor,
    df: pd.DataFrame,
    topk: int | None,
) -> pd.DataFrame:
    required = {"iris_path", "fingerprint_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns for paired format: {sorted(missing)}")

    preds: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        iris_path = Path(str(row["iris_path"]))
        fp_path = Path(str(row["fingerprint_path"]))
        rec: dict[str, Any] = {}

        if "subject_id" in df.columns:
            rec["subject_id"] = str(row["subject_id"])

        rec["predicted_subject_id"] = predictor.predict(iris_path, fp_path)

        # For parquet compatibility, store topk as JSON text
        if topk is not None and topk > 1:
            rec["topk_json"] = json.dumps(predictor.predict_topk(iris_path, fp_path, k=topk))

        preds.append(rec)

    return pd.DataFrame(preds)


def _predict_batch_from_long_manifest(
    predictor: Predictor,
    df: pd.DataFrame,
    topk: int | None,
) -> pd.DataFrame:
    """
    Supports a 'long' manifest with columns:
      - subject_id
      - filepath
      - modality  (values like iris / fingerprint)
    We pair first iris + first fingerprint per subject_id.
    """
    required = {"subject_id", "filepath", "modality"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns for long format: {sorted(missing)}")

    # Normalize modality strings
    tmp = df.copy()
    tmp["modality"] = tmp["modality"].astype(str).str.lower()

    preds: list[dict[str, Any]] = []
    for subject_id, g in tmp.groupby("subject_id"):
        iris_rows = g[g["modality"].str.contains("iris", na=False)]
        fp_rows = g[g["modality"].str.contains("finger", na=False)]

        if len(iris_rows) == 0 or len(fp_rows) == 0:
            continue

        iris_path = Path(str(iris_rows.iloc[0]["filepath"]))
        fp_path = Path(str(fp_rows.iloc[0]["filepath"]))

        rec: dict[str, Any] = {"subject_id": str(subject_id)}
        rec["predicted_subject_id"] = predictor.predict(iris_path, fp_path)

        if topk is not None and topk > 1:
            rec["topk_json"] = json.dumps(predictor.predict_topk(iris_path, fp_path, k=topk))

        preds.append(rec)

    return pd.DataFrame(preds)


def predict_batch_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True, help="Manifest parquet/csv")
    parser.add_argument("--out", required=True, help="Output parquet/csv")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    run_dir = _p(args.run_dir)
    manifest_path = _p(args.manifest)
    out_path = _p(args.out)

    predictor = Predictor.load(
        run_dir=run_dir,
        checkpoint=_p(args.checkpoint) if args.checkpoint else None,
        device=args.device,
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(manifest_path)
    elif manifest_path.suffix.lower() in {".csv"}:
        df = pd.read_csv(manifest_path)
    else:
        raise ValueError("Manifest must be .parquet or .csv")

    cols = set(df.columns)
    if {"iris_path", "fingerprint_path"}.issubset(cols):
        out_df = _predict_batch_from_paired_manifest(predictor, df, args.topk)
    elif {"subject_id", "filepath", "modality"}.issubset(cols):
        out_df = _predict_batch_from_long_manifest(predictor, df, args.topk)
    else:
        raise ValueError(
            "Unsupported manifest schema.\n"
            "Supported:\n"
            "  A) paired: subject_id, iris_path, fingerprint_path\n"
            "  B) long:   subject_id, filepath, modality\n"
            f"Found columns: {sorted(cols)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        out_df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        out_df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")

    print(f"[OK] wrote predictions: {out_path}  rows={len(out_df)}")
    return 0
