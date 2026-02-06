from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

from mmbiometric.models.fusion import MultimodalFusionNet


@dataclass
class Predictor:
    model: MultimodalFusionNet
    idx_to_label: Dict[int, str]
    device: torch.device
    image_size: int

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
        device_t = torch.device(device)

        model = MultimodalFusionNet(
            backbone=backbone,
            embedding_dim=embedding_dim,
            num_classes=len(idx_to_label),
            dropout=dropout,
        )
        state = torch.load(ckpt_path, map_location=device_t)
        model.load_state_dict(state)
        model.to(device_t)
        model.eval()

        return cls(model=model, idx_to_label=idx_to_label, device=device_t, image_size=image_size)

    def _image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

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
