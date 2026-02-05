from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image

from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet


@dataclass(frozen=True)
class Predictor:
    model: MultimodalNet
    idx_to_label: Dict[int, str]
    transform: object
    device: torch.device

    @classmethod
    def load(
        cls,
        ckpt_path: Path,
        backbone: str,
        embedding_dim: int,
        dropout: float,
        idx_to_label: Dict[int, str],
        image_size: int,
        device: torch.device,
    ) -> "Predictor":
        model = MultimodalNet(backbone=backbone, embedding_dim=embedding_dim,
                              num_classes=len(idx_to_label), dropout=dropout)
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
        model.eval()
        tfm = default_image_transform(image_size)
        return cls(model=model, idx_to_label=idx_to_label, transform=tfm, device=device)

    @torch.no_grad()
    def predict_one(self, iris_img: Image.Image, fp_img: Image.Image) -> str:
        iris = self.transform(iris_img.convert("RGB")).unsqueeze(0).to(self.device)
        fp = self.transform(fp_img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(iris, fp)
        pred = int(torch.argmax(logits, dim=1).item())
        return self.idx_to_label[pred]
