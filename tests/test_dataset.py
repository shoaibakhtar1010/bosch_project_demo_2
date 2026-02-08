# fmt: off
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from mmbiometric.data.dataset import MultimodalBiometricDataset
from mmbiometric.data.transforms import default_image_transform


def _mk_img(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (16, 16))
    img.save(p)


def test_dataset_returns_tensors(tmp_path: Path) -> None:
    iris = tmp_path / "iris" / "001.png"
    fp = tmp_path / "fp" / "001.png"
    _mk_img(iris)
    _mk_img(fp)

    df = pd.DataFrame({"subject_id": ["001"], "iris_path": [str(iris)], "fingerprint_path": [str(fp)]})
    manifest = tmp_path / "m.parquet"
    df.to_parquet(manifest, index=False)

    tfm = default_image_transform(32)
    ds = MultimodalBiometricDataset(manifest, tfm, tfm, {"001": 0})
    sample = ds[0]
    assert isinstance(sample.iris, torch.Tensor)
    assert isinstance(sample.fingerprint, torch.Tensor)
    assert sample.iris.shape[1:] == (32, 32)
    assert sample.label.item() == 0
