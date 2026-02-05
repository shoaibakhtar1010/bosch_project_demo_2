from __future__ import annotations

from pathlib import Path

from PIL import Image

from mmbiometric.data.manifest import build_manifest


def _mk_img(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (16, 16))
    img.save(p)


def test_build_manifest_pairs_by_subject(tmp_path: Path) -> None:
    # Create a tiny fake dataset layout
    _mk_img(tmp_path / "iris" / "001_iris.png")
    _mk_img(tmp_path / "fingerprint" / "001_fp.png")
    _mk_img(tmp_path / "iris" / "002_iris.png")
    _mk_img(tmp_path / "fingerprint" / "002_fp.png")

    out = tmp_path / "manifest.parquet"
    build_manifest(tmp_path, out, subject_regex=r"(\d{3})")
    assert out.exists()
