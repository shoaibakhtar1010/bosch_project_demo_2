from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmbiometric.inference.predict_one import predict_one


def predict_batch_from_manifest(
    checkpoint_path: Path,
    labels_json: Path,
    model_metadata_json: Path,
    manifest_path: Path,
    out_csv: Path,
    device: str = "cpu",
    topk: int = 1,
) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    required = {"subject_id", "iris_path", "fingerprint_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    preds = []
    for row in df.itertuples(index=False):
        iris = Path(getattr(row, "iris_path"))
        fp = Path(getattr(row, "fingerprint_path"))
        gt = str(getattr(row, "subject_id"))

        p = predict_one(
            checkpoint_path=checkpoint_path,
            labels_json=labels_json,
            model_metadata_json=model_metadata_json,
            iris_img=iris,
            fp_img=fp,
            device=device,
            topk=max(topk, 1),
        )

        rec = {
            "gt_subject_id": gt,
            "pred_subject_id": p.predicted_subject_id,
        }
        if topk > 1:
            rec["topk"] = p.topk
        preds.append(rec)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(out_csv, index=False)
