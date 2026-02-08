"""Inference helper for predicting a single (iris, fingerprint) pair.

This module defines a lightweight ``Predictor`` class used by the HTTP API. It
wraps the core ``MultimodalNet`` model from :mod:`mmbiometric.models.multimodal_net`
and exposes methods for loading checkpoints and running inference on PIL images.
Prior implementations referenced a non-existent ``MultimodalFusionNet`` and expected
custom normalization; these bugs have been fixed so that the predictor now uses
the same model architecture and transforms as training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet


@dataclass
class Predictor:
    # Use the core MultimodalNet for inference
    model: MultimodalNet
    idx_to_label: dict[int, str]
    device: torch.device
    image_size: int

    @classmethod
    def load(
        cls,
        ckpt_path: Path,
        idx_to_label: dict[int, str],
        backbone: str,
        embedding_dim: int,
        dropout: float,
        image_size: int,
        device: str | torch.device = "cpu",
    ) -> Predictor:
        """Load a :class:`MultimodalNet` checkpoint and return a Predictor instance.

        Parameters
        ----------
        ckpt_path:
            Path to the saved model weights (a ``.pt`` file).  The state is expected to
            be the raw ``state_dict`` saved during training (no wrapper dict) or a
            wrapper containing ``model_state_dict``.
        idx_to_label:
            Mapping from class indices to the original subject identifiers.  The
            number of classes inferred from this mapping must match the model.
        backbone:
            Backbone name used during training (e.g., ``"resnet18"``).  Must be
            supported by :func:`mmbiometric.models.backbones.build_backbone`.
        embedding_dim:
            Size of the intermediate embedding layer in the classifier head.
        dropout:
            Dropout probability applied in the classifier head.
        image_size:
            Input image size expected by the model.  Determines the resize
            operation applied during preprocessing.
        device:
            Device on which to run inference (e.g., ``"cpu"`` or ``"cuda"``).  Can
            be a string or a :class:`torch.device` instance.
        """
        # Normalize device to a torch.device
        device_t = torch.device(device)

        # Instantiate the model using the same hyperparameters as training
        model = MultimodalNet(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=len(idx_to_label),
            dropout=dropout,
        )

        # Load the state dict (support both bare and wrapped formats)
        state = torch.load(ckpt_path, map_location=device_t)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Strip any DataParallel prefix
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state, strict=True)

        model.to(device_t)
        model.eval()

        return cls(model=model, idx_to_label=idx_to_label, device=device_t, image_size=image_size)

    def _image_transform(self):
        """Return the image transform used for preprocessing.

        This reuses the project's default transform to ensure consistency with
        training.  It resizes the image to ``(image_size, image_size)`` and
        converts it to a tensor without additional normalization.  If your
        training pipeline uses a different transform, modify this accordingly.
        """
        return default_image_transform(image_size=self.image_size)

    def _preprocess(self, iris_img: Image.Image, fingerprint_img: Image.Image):
        tfm = self._image_transform()

        iris_t = tfm(iris_img).unsqueeze(0).to(self.device)
        fp_t = tfm(fingerprint_img).unsqueeze(0).to(self.device)
        return iris_t, fp_t

    def predict_one(self, iris_img: Image.Image, fingerprint_img: Image.Image) -> str:
        """
        Predict a subject label for a single (iris, fingerprint) pair, given PIL Images.
        Returns the predicted subject_id (label string).
        """
        iris_t, fp_t = self._preprocess(iris_img, fingerprint_img)
        with torch.no_grad():
            logits = self.model(iris_t, fp_t)
            pred_idx = int(torch.argmax(logits, dim=1).item())
        return self.idx_to_label[pred_idx]

    def predict(self, iris_path: Path, fingerprint_path: Path) -> str:
        """
        Convenience wrapper used by the CLI:
        takes file paths, loads images, and calls predict_one().
        """
        iris_img = Image.open(iris_path).convert("RGB")
        fp_img = Image.open(fingerprint_path).convert("RGB")
        return self.predict_one(iris_img=iris_img, fingerprint_img=fp_img)


# -----------------------------------------------------------------------------
# Module-level convenience API
# -----------------------------------------------------------------------------

@dataclass
class PredictOneOutput:
    """Return type for the module-level :func:`predict_one` function.

    Parameters
    ----------
    predicted_subject_id:
        The most likely subject identifier predicted by the model.
    topk:
        Optional list of top-k predictions, where each entry contains
        ``"subject_id"`` and ``"prob"`` keys.  Only populated when ``topk > 1``.
    """

    predicted_subject_id: str
    topk: list[dict[str, Any]] | None = None


def predict_one(
    checkpoint_path: Path,
    labels_json: Path,
    model_metadata_json: Path,
    iris_img: Path | Image.Image,
    fp_img: Path | Image.Image,
    device: str = "cpu",
    topk: int = 1,
) -> PredictOneOutput:
    """Predict the subject ID for a single (iris, fingerprint) pair.

    This convenience function is used by legacy batch prediction scripts.  It
    loads the model checkpoint and metadata, constructs a predictor, runs
    inference on the given images, and returns both the top-1 and (optionally)
    top-k predictions.

    Parameters
    ----------
    checkpoint_path:
        Path to the model weights file (``.pt``).
    labels_json:
        Path to ``labels.json`` mapping class indices to subject IDs.
    model_metadata_json:
        Path to ``model_metadata.json`` containing hyperparameters such as
        backbone, embedding dimension, dropout, and image size.  If missing,
        sensible defaults are used.
    iris_img, fp_img:
        Either file paths or already loaded :class:`PIL.Image.Image` objects for
        the iris and fingerprint images.  If paths are provided, the images
        will be opened with ``Image.open``.
    device:
        Device for inference (``"cpu"`` or ``"cuda"``).
    topk:
        Number of top predictions to include.  Must be >= 1.  When ``topk`` is
        1, only the top prediction is returned and the ``topk`` field in the
        result is ``None``.
    """
    # Load the label mapping (keys may be strings or ints); convert to int->str
    with open(labels_json, encoding="utf-8") as f:
        lbl_data = json.load(f)
    # Handle wrapped formats
    if isinstance(lbl_data, dict) and "idx_to_label" in lbl_data:
        lbl_data = lbl_data["idx_to_label"]
    idx_to_label = {int(k): str(v) for k, v in lbl_data.items()}

    # Load model metadata if available
    meta: dict[str, Any] = {}
    try:
        with open(model_metadata_json, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    backbone = meta.get("backbone", "resnet18")
    embedding_dim = int(meta.get("embedding_dim", 256))
    dropout = float(meta.get("dropout", 0.0))
    image_size = int(meta.get("image_size", 224))

    # Instantiate predictor
    predictor = Predictor.load(
        ckpt_path=checkpoint_path,
        idx_to_label=idx_to_label,
        backbone=backbone,
        embedding_dim=embedding_dim,
        dropout=dropout,
        image_size=image_size,
        device=device,
    )

    # Load images if paths provided
    if isinstance(iris_img, (str, Path)):
        iris_img_obj = Image.open(Path(iris_img)).convert("RGB")
    else:
        iris_img_obj = iris_img
    if isinstance(fp_img, (str, Path)):
        fp_img_obj = Image.open(Path(fp_img)).convert("RGB")
    else:
        fp_img_obj = fp_img

    # Top-1 prediction
    pred_id = predictor.predict_one(iris_img_obj, fp_img_obj)
    result = PredictOneOutput(predicted_subject_id=str(pred_id))

    # Top-k predictions (if requested)
    k = max(1, int(topk))
    if k > 1:
        # Use logits and softmax to compute probabilities
        # Preprocess once to avoid repeated conversions
        iris_t, fp_t = predictor._preprocess(iris_img_obj, fp_img_obj)
        with torch.no_grad():
            logits = predictor.model(iris_t, fp_t).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            k = min(k, probs.numel())
            vals, idxs = torch.topk(probs, k=k)
            topk_list: list[dict[str, Any]] = []
            for score, idx in zip(vals.tolist(), idxs.tolist(), strict=False):
                topk_list.append(
                    {
                        "subject_id": str(idx_to_label.get(int(idx), str(idx))),
                        "prob": float(score),
                    }
                )
            result.topk = topk_list

    return result
