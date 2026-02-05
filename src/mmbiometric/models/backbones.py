from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_backbone(name: str) -> Tuple[nn.Module, int]:
    """Return (backbone_without_head, feature_dim)."""
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    raise ValueError(f"Unsupported backbone: {name}")
