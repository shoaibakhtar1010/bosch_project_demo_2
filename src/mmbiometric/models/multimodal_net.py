from __future__ import annotations

import torch
import torch.nn as nn

from mmbiometric.models.backbones import build_backbone


class MultimodalNet(nn.Module):
    """Two-tower model: iris encoder + fingerprint encoder, fused for classification."""

    def __init__(self, backbone: str, embedding_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        iris_backbone, iris_dim = build_backbone(backbone)
        fp_backbone, fp_dim = build_backbone(backbone)
        self.iris = iris_backbone
        self.fingerprint = fp_backbone
        fused_dim = (2 * iris_dim) + fp_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(
        self,
        left_iris: torch.Tensor,
        right_iris: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        left_feat = self.iris(left_iris)
        right_feat = self.iris(right_iris)
        fp_feat = self.fingerprint(fingerprint)
        fused = torch.cat([left_feat, right_feat, fp_feat], dim=1)
        return self.head(fused)
