from __future__ import annotations

# Standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Third‑party imports
import yaml


DeviceChoice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: Path | None
    manifest_path: Path | None
    image_size: int
    batch_size: int
    num_workers: int
    val_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    device: DeviceChoice
    log_every: int


@dataclass(frozen=True)
class AppConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> AppConfig:
    """Load YAML config into strongly-typed dataclasses.

    Ray workers often receive the config path as a plain string (e.g. because
    `train_loop_config` is JSON-serialised). Accept both `str` and `Path`.
    """

    p = Path(path).expanduser()
    if p.is_dir():
        raise IsADirectoryError(f"Config path is a directory, expected file: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    data = raw["data"]
    model = raw["model"]
    train = raw["train"]
    return AppConfig(
        seed=int(raw["seed"]),
        data=DataConfig(
            dataset_dir=Path(data["dataset_dir"]) if data.get("dataset_dir") else None,
            manifest_path=Path(data["manifest_path"]) if data.get("manifest_path") else None,
            image_size=int(data["image_size"]),
            batch_size=int(data["batch_size"]),
            num_workers=int(data["num_workers"]),
            val_fraction=float(data["val_fraction"]),
            test_fraction=float(data["test_fraction"]),
        ),
        model=ModelConfig(
            backbone=str(model["backbone"]),
            embedding_dim=int(model["embedding_dim"]),
            dropout=float(model["dropout"]),
        ),
        train=TrainConfig(
            epochs=int(train["epochs"]),
            lr=float(train["lr"]),
            weight_decay=float(train["weight_decay"]),
            device=str(train["device"]),
            log_every=int(train["log_every"]),
        ),
    )
