from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from mmbiometric.inference.predictor import Predictor


def main() -> None:
    run_dir = Path("runs/train_ray")
    ckpt = run_dir / "best.pt"
    labels_path = run_dir / "labels.json"
    meta_path = run_dir / "model_metadata.json"

    manifest_path = run_dir / "manifest.parquet"  # or a custom manifest path
    out_path = run_dir / "predictions.parquet"

    idx_to_label = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    predictor = Predictor.load(
        ckpt_path=ckpt,
        backbone=meta.get("backbone", "resnet18"),
        embedding_dim=int(meta.get("embedding_dim", 256)),
        dropout=float(meta.get("dropout", 0.1)),
        idx_to_label=idx_to_label,
        image_size=int(meta.get("image_size", 224)),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    df = pd.read_parquet(manifest_path)

    preds: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        iris = Image.open(row["iris_path"])
        fp = Image.open(row["fingerprint_path"])
        preds.append(predictor.predict_one(iris, fp))

    df_out = df.copy()
    df_out["prediction"] = preds
    df_out.to_parquet(out_path, index=False)
    print("wrote:", out_path)


if __name__ == "__main__":
    main()
