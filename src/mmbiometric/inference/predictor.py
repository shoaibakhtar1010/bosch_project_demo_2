from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type
import importlib

import torch
import torch.nn.functional as F
from PIL import Image

from mmbiometric.data.transforms import default_image_transform


def _resolve_model_class() -> Type[torch.nn.Module]:
    """
    Tries multiple model locations to be resilient to refactors.
    Update candidates if your repo uses different module/class names.
    """
    candidates = [
        ("mmbiometric.models.multimodal_net", "MultimodalNet"),
        ("mmbiometric.models.fusion_net", "FusionNet"),
        ("mmbiometric.models", "FusionNet"),
        ("mmbiometric.models", "MultimodalNet"),
    ]
    last_err: Exception | None = None
    for module_path, cls_name in candidates:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            if isinstance(cls, type):
                return cls  # type: ignore[return-value]
        except Exception as e:
            last_err = e

    raise ImportError(
        "Could not locate the model class. Tried:\n"
        + "\n".join([f"  - {m}:{c}" for m, c in candidates])
        + "\nFix by ensuring your model class exists in one of those modules, "
          "or update _resolve_model_class() to match your repo."
    ) from last_err


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats:
      - raw state_dict (dict[str, Tensor])
      - {"model_state_dict": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - {"model": state_dict, ...}
    Also strips 'module.' prefix if present (DDP).
    """
    if not isinstance(ckpt_obj, dict):
        raise ValueError("Unsupported checkpoint type; expected a dict or a state_dict dict.")

    # case 1: wrapper dict
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            sd = ckpt_obj[key]
            break
    else:
        # case 2: maybe it already IS a state_dict (heuristic: tensor-ish values)
        # If values are tensors, assume state_dict.
        if all(hasattr(v, "shape") for v in ckpt_obj.values()):
            sd = ckpt_obj
        else:
            raise ValueError(
                f"Checkpoint dict does not contain a state_dict under keys "
                f"model_state_dict/state_dict/model. Keys={list(ckpt_obj.keys())[:20]}"
            )

    # Strip "module." prefix if trained under DDP
    if sd and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd


@dataclass
class Predictor:
    model: torch.nn.Module
    idx_to_label: Dict[int, str]
    transform: Any
    device: torch.device

    @classmethod
    def load(
        cls,
        ckpt_path: Path,
        idx_to_label: Dict[int, str],
        backbone: str,
        embedding_dim: int,
        dropout: float,
        image_size: int,
        device: str = "cpu",
    ) -> "Predictor":
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model_cls = _resolve_model_class()

        num_classes = len(idx_to_label)
        model = model_cls(
            backbone=backbone,
            embedding_dim=embedding_dim,
            dropout=dropout,
            num_classes=num_classes,
        )

        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt_obj)

        # strict=True is good — fail fast if architecture truly differs.
        model.load_state_dict(state_dict, strict=True)

        torch_device = torch.device(device)
        model.to(torch_device)
        model.eval()

        transform = default_image_transform(image_size=image_size, train=False)

        return cls(
            model=model,
            idx_to_label=idx_to_label,
            transform=transform,
            device=torch_device,
        )

    def _load_img(self, path: Path) -> torch.Tensor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)  # [C,H,W]
        return x.unsqueeze(0).to(self.device)  # [1,C,H,W]

    @torch.inference_mode()
    def predict(self, iris_path: Path, fingerprint_path: Path) -> str:
        iris = self._load_img(iris_path)
        fp = self._load_img(fingerprint_path)

        logits = self.model(iris, fp)  # [1, num_classes]
        pred_idx = int(torch.argmax(logits, dim=1).item())
        return self.idx_to_label.get(pred_idx, str(pred_idx))

    @torch.inference_mode()
    def predict_topk(self, iris_path: Path, fingerprint_path: Path, k: int = 5) -> list[dict]:
        iris = self._load_img(iris_path)
        fp = self._load_img(fingerprint_path)

        logits = self.model(iris, fp)
        probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        k = min(k, probs.numel())
        vals, idxs = torch.topk(probs, k=k)

        out = []
        for score, idx in zip(vals.tolist(), idxs.tolist()):
            out.append(
                {
                    "class_index": int(idx),
                    "subject_id": self.idx_to_label.get(int(idx), str(idx)),
                    "prob": float(score),
                }
            )
        return out
