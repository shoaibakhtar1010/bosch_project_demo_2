from __future__ import annotations

import torch

from mmbiometric.models.multimodal_net import MultimodalNet


def test_model_forward_shape() -> None:
    model = MultimodalNet(backbone="resnet18", embedding_dim=64, num_classes=5, dropout=0.1)
    iris = torch.randn(2, 3, 224, 224)
    fp = torch.randn(2, 3, 224, 224)
    out = model(iris, fp)
    assert out.shape == (2, 5)
