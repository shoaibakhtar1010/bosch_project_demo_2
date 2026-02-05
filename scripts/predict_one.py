from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from mmbiometric.inference.predictor import Predictor


def main() -> None:
    run_dir = Path("runs/train_ray")  # change to your run directory
    ckpt = run_dir / "best.pt"
    labels_path = run_dir / "labels.json"
    meta_path = run_dir / "model_metadata.json"

    idx_to_label = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}

    # Prefer reading metadata so inference does not hardcode hyperparams.
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    backbone = meta.get("backbone", "resnet18")
    embedding_dim = int(meta.get("embedding_dim", 256))
    dropout = float(meta.get("dropout", 0.1))
    image_size = int(meta.get("image_size", 224))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor = Predictor.load(
        ckpt_path=ckpt,
        backbone=backbone,
        embedding_dim=embedding_dim,
        dropout=dropout,
        idx_to_label=idx_to_label,
        image_size=image_size,
        device=device,
    )

    iris_img = Image.open("path/to/iris.png")
    fp_img = Image.open("path/to/fingerprint.png")

    pred = predictor.predict_one(iris_img, fp_img)
    print("prediction:", pred)


if __name__ == "__main__":
    main()
