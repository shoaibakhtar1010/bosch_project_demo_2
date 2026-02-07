from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import torch

from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Handle checkpoints saved under DataParallel/DistributedDataParallel (module.* keys)."""
    if not state:
        return state
    keys = list(state.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state.items()}
    return state


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_idx_to_label(labels_path: Path) -> dict[int, str]:
    """
    Accept multiple label formats:
      A) Flat mapping written by your training: {"0": "1", "1": "2", ...}
      B) Wrapped: {"idx_to_label": {...}}
      C) Inverse: {"label_to_idx": {...}}
      D) List: ["1","2",...]
      E) {"classes": [...]} or {"class_names": [...]}
    """
    obj = _read_json(labels_path)

    # B) wrapped
    if isinstance(obj, dict) and "idx_to_label" in obj:
        obj = obj["idx_to_label"]

    # C) inverse
    if isinstance(obj, dict) and "label_to_idx" in obj:
        inv = obj["label_to_idx"]
        if not isinstance(inv, dict) or not inv:
            return {}
        return {int(v): str(k) for k, v in inv.items()}

    # D/E) list styles
    if isinstance(obj, dict) and ("classes" in obj or "class_names" in obj):
        arr = obj.get("classes") or obj.get("class_names")
        if isinstance(arr, list) and arr:
            return {i: str(v) for i, v in enumerate(arr)}

    if isinstance(obj, list):
        return {i: str(v) for i, v in enumerate(obj)}

    # A) flat dict mapping index->label (keys are usually "0","1",...)
    if isinstance(obj, dict) and obj:
        # if keys look like integers
        if all(str(k).isdigit() for k in obj.keys()):
            return {int(k): str(v) for k, v in obj.items()}

    return {}


def _load_checkpoint_state(ckpt_path: Path, device: str) -> dict[str, torch.Tensor]:
    payload = torch.load(ckpt_path, map_location=device)

    # Training saves {"model_state_dict": ..., "val_acc": ...}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload

    if not isinstance(state, dict):
        raise ValueError(
            f"Checkpoint at {ckpt_path} is not a valid state_dict or wrapper dict."
        )

    return _strip_module_prefix(state)


@dataclass(frozen=True)
class PredictorArtifacts:
    run_dir: Path
    checkpoint_path: Path
    idx_to_label: dict[int, str]
    meta: dict[str, Any]


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        transform,
        idx_to_label: dict[int, str],
        device: str,
    ) -> None:
        self.model = model
        self.transform = transform
        self.idx_to_label = idx_to_label
        self.device = device

    @staticmethod
    def load(
        run_dir: Path,
        checkpoint: Path | None = None,
        device: str | None = None,
    ) -> Predictor:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve checkpoint
        if checkpoint is None:
            ckpt_path = run_dir / "best.pt"
        else:
            checkpoint = Path(checkpoint)
            ckpt_path = checkpoint if checkpoint.is_absolute() else (run_dir / checkpoint)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Expected best.pt in {run_dir} or pass --checkpoint."
            )

        # Labels
        labels_path = run_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found in run_dir: {labels_path}")

        idx_to_label = _load_idx_to_label(labels_path)
        if not idx_to_label:
            raise ValueError(
                "labels.json produced an empty idx_to_label mapping; cannot run inference.\n"
                f"labels.json path: {labels_path}\n"
                "Expected either a flat mapping like {\"0\":\"1\",...} or wrapped {\"idx_to_label\":{...}}."
            )

        # Metadata (optional but recommended)
        meta_path = run_dir / "model_metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            meta = _read_json(meta_path)

        # Read model hyperparameters from metadata, falling back to sensible defaults
        # If training saved these values in model_metadata.json they should be used for
        # inference to ensure the architecture matches the checkpoint.  Otherwise we
        # assume the defaults used in `configs/default.yaml`.
        image_size = int(meta.get("image_size", 224))
        num_classes = int(meta.get("num_classes", len(idx_to_label)))
        backbone = meta.get("backbone", "resnet18")
        # Embedding dimension and dropout may not be present in older checkpoints;
        # default to 256 and 0.0 if missing.  These values mirror the defaults
        # defined in configs/default.yaml.  Casting to appropriate types ensures
        # JSON strings are handled gracefully.
        embedding_dim = int(meta.get("embedding_dim", 256))
        # Dropout may be stored as string or number; convert to float.
        dropout = meta.get("dropout", 0.0)
        try:
            dropout = float(dropout)
        except (TypeError, ValueError):
            dropout = 0.0

        transform = default_image_transform(image_size=image_size)

        # Build the model using the same architecture as training
        model = MultimodalNet(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
        ).to(device)

        # Load the checkpoint state into the model
        state = _load_checkpoint_state(ckpt_path, device=device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            # Provide a detailed error to help users diagnose mismatched architectures
            raise RuntimeError(
                "Failed to load model weights. This usually means the inference model "
                "architecture doesn't match the training architecture OR you're loading the wrong checkpoint.\n"
                f"Checkpoint: {ckpt_path}\n"
                f"Error: {e}"
            ) from e

        model.eval()
        return Predictor(model=model, transform=transform, idx_to_label=idx_to_label, device=device)

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        t = self.transform(img)  # C,H,W
        return t

    def _prepare_inputs(self, iris_path: Path, fp_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        iris = self._load_image_tensor(iris_path).unsqueeze(0).to(self.device)
        fp = self._load_image_tensor(fp_path).unsqueeze(0).to(self.device)
        return iris, fp

    @torch.inference_mode()
    def predict_logits(self, iris_path: Path, fingerprint_path: Path) -> torch.Tensor:
        iris, fp = self._prepare_inputs(iris_path, fingerprint_path)
        logits = self.model(iris, fp)  # (1, num_classes)
        return logits.squeeze(0)

    @torch.inference_mode()
    def predict(self, iris_path: Path, fingerprint_path: Path) -> str:
        logits = self.predict_logits(iris_path, fingerprint_path)
        pred_idx = int(torch.argmax(logits).item())
        return str(self.idx_to_label.get(pred_idx, str(pred_idx)))

    @torch.inference_mode()
    def predict_topk(self, iris_path: Path, fingerprint_path: Path, k: int = 5) -> list[dict[str, Any]]:
        logits = self.predict_logits(iris_path, fingerprint_path)
        probs = torch.softmax(logits, dim=-1)

        k = max(1, min(int(k), probs.numel()))
        vals, idxs = torch.topk(probs, k=k)

        out: list[dict[str, Any]] = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            out.append(
                {
                    "subject_id": str(self.idx_to_label.get(int(idx), str(idx))),
                    "prob": float(score),
                }
            )
        return out
