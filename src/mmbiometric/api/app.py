"""FastAPI app exposing the multimodal biometric inference endpoint.

The import order in this module follows Ruff's recommended grouping:
  1. Standard library imports
  2. Third‑party imports
  3. Local application imports

Using ``Annotated`` avoids calling FastAPI dependency functions (like
``File()``) at import time, which Ruff's B008 rule flags. See the
``predict`` route definition below for an example.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from mmbiometric.inference.predict_one import Predictor

app = FastAPI(title="mmbiometric inference")

PREDICTOR: Predictor | None = None

# Define FastAPI dependencies at module level to avoid calling ``File()``
# in default parameter values.  This satisfies Ruff's B008 rule.
_left_iris_file: File = File(...)
_right_iris_file: File = File(...)
_fp_file: File = File(...)



def _load_predictor() -> Predictor:
    model_dir = Path(os.environ.get("MODEL_DIR", "/model"))
    ckpt = model_dir / "best.pt"
    labels = model_dir / "labels.json"
    meta_path = model_dir / "model_metadata.json"

    if not ckpt.exists() or not labels.exists():
        raise RuntimeError(f"Missing model files in {model_dir}")

    idx_to_label_raw: dict[str, Any] = json.loads(labels.read_text(encoding="utf-8"))
    idx_to_label = {int(k): str(v) for k, v in idx_to_label_raw.items()}

    # Prefer model metadata (written by the training CLI) so deployments are
    # config-free. Fallback to environment variables for backwards compatibility.
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    backbone = str(meta.get("backbone") or os.environ.get("BACKBONE", "resnet18"))
    embedding_dim = int(meta.get("embedding_dim") or os.environ.get("EMBEDDING_DIM", "256"))
    dropout = float(meta.get("dropout") or os.environ.get("DROPOUT", "0.1"))
    image_size = int(meta.get("image_size") or os.environ.get("IMAGE_SIZE", "224"))

    # Determine device lazily; pass as string to Predictor.load for compatibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Predictor.load(
        ckpt_path=ckpt,
        idx_to_label=idx_to_label,
        backbone=backbone,
        embedding_dim=embedding_dim,
        dropout=dropout,
        image_size=image_size,
        device=device,
    )


@app.on_event("startup")
def startup() -> None:
    global PREDICTOR
    PREDICTOR = _load_predictor()


@app.post("/predict")
async def predict(
    left_iris: Annotated[UploadFile, _left_iris_file],
    right_iris: Annotated[UploadFile, _right_iris_file],
    fingerprint: Annotated[UploadFile, _fp_file],
) -> dict[str, str]:

    """Predict a subject ID given an iris and fingerprint image.

    The ``File`` dependency is provided via ``Annotated`` rather than as a
    default argument, which avoids triggering Ruff's B008 rule.  See:
    https://docs.astral.sh/ruff/rules/function-call-in-default-argument/
    """
    if PREDICTOR is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")

    # Read uploaded images into PIL.Image
    left_img = Image.open(io.BytesIO(await left_iris.read()))
    right_img = Image.open(io.BytesIO(await right_iris.read()))
    fp_img = Image.open(io.BytesIO(await fingerprint.read()))
    pred = PREDICTOR.predict_one(left_img, right_img, fp_img)

    return {"prediction": pred}
