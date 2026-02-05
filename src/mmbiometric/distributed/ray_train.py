"""Distributed training with Ray Train.

This implementation mirrors the single-process training loop in
`mmbiometric/cli.py`, but runs it under Ray Train's `TorchTrainer`.

It is intentionally defensive:
  * Uses the real `MultimodalBiometricDataset` constructor signature
    (`manifest_path=...`), not a `manifest=` keyword.
  * Supplies a custom `collate_fn` because the dataset returns a `Sample`
    dataclass, which PyTorch's `default_collate` cannot batch.
  * Avoids double `ray.init()` calls (common when CLI already initialises Ray).
  * Only rank-0 writes files to `output_dir`.

If you're on Windows and see c10d socket warnings, they're usually harmless.
If you get a hang, export `MASTER_ADDR` to your LAN IP (same as Ray head) and
set `GLOO_SOCKET_IFNAME` to your active NIC.
"""

from __future__ import annotations

import json
import os

if os.name == "nt":
    # Work around PyTorch distributed rendezvous issues on Windows.
    # See https://github.com/pytorch/pytorch/issues/150381
    os.environ.setdefault("USE_LIBUV", "0")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from mmbiometric.config import AppConfig, load_config
from mmbiometric.data.dataset import MultimodalBiometricDataset, Sample
from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet


# -----------------------------
# Public API
# -----------------------------


@dataclass(frozen=True)
class RayTrainArgs:
    """Arguments passed from `mmbiometric/cli_ray.py`."""

    config_path: str
    dataset_dir: str
    output_dir: str
    subject_regex: str = r"^(\d+)$"

    num_workers: int = 2
    cpus_per_worker: int = 2
    use_gpu: bool = False
    master_addr: str | None = None
    gloo_socket_ifname: str | None = None


def _apply_torch_dist_env(args: RayTrainArgs) -> None:
    if args.master_addr:
        os.environ.setdefault("MASTER_ADDR", args.master_addr)
    if args.gloo_socket_ifname:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", args.gloo_socket_ifname)

    if os.name != "nt":
        return

    if os.environ.get("MASTER_ADDR"):
        return

    address = os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")
    if address and ":" in address:
        os.environ.setdefault("MASTER_ADDR", address.split(":", 1)[0])
        return

    try:
        from ray._private.services import get_node_ip_address

        os.environ.setdefault("MASTER_ADDR", get_node_ip_address())
    except Exception:
        pass


def train_distributed(args: RayTrainArgs) -> "ray.train.Result":
    """Entry point used by the CLI.

    Assumes preprocess has already generated:
      - {output_dir}/manifest.parquet
      - {output_dir}/splits/train_manifest.parquet
      - {output_dir}/splits/val_manifest.parquet
    """

    # Avoid calling ray.init() twice. CLI usually calls it already.
    if not ray.is_initialized():
        # Prefer env var (RAY_ADDRESS) if user started a head.
        address = os.environ.get("RAY_ADDRESS")
        ray.init(address=address or "auto", ignore_reinit_error=True)

    app_cfg = load_config(args.config_path)

    _apply_torch_dist_env(args)

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_per_worker,
        train_loop_config={
            "config_path": args.config_path,
            "output_dir": args.output_dir,
        },
        scaling_config=ScalingConfig(
            num_workers=int(args.num_workers),
            use_gpu=bool(args.use_gpu),
            resources_per_worker={"CPU": int(args.cpus_per_worker)},
        ),
        run_config=train.RunConfig(
            name="mmbiometric_ray_train",
            storage_path=str(Path(args.output_dir).resolve()),
        ),
    )

    return trainer.fit()


# -----------------------------
# Helpers
# -----------------------------


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate_samples(batch: list[Sample]) -> Sample:
    """Batch a list of `Sample` into a single `Sample` of stacked tensors."""

    iris = torch.stack([b.iris for b in batch], dim=0)
    fingerprint = torch.stack([b.fingerprint for b in batch], dim=0)
    label = torch.stack([b.label for b in batch], dim=0)
    return Sample(iris=iris, fingerprint=fingerprint, label=label)


def _read_split_manifests(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splits_dir = output_dir / "splits"
    train_path = splits_dir / "train_manifest.parquet"
    val_path = splits_dir / "val_manifest.parquet"

    if train_path.exists() and val_path.exists():
        return pd.read_parquet(train_path), pd.read_parquet(val_path)

    # Fallback: split the main manifest if splits weren't created.
    manifest_path = output_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run mmbiometric-ray-preprocess first."
        )

    df = pd.read_parquet(manifest_path)
    if len(df) == 0:
        raise RuntimeError(
            f"Manifest is empty at {manifest_path}. Check --dataset-dir and preprocessing filters."
        )

    # Subject-wise split (so train/val contain disjoint subjects).
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    subjects = sorted(df["subject_id"].unique().tolist())
    rng = np.random.default_rng(42)
    rng.shuffle(subjects)
    # default 20% validation
    n_val = max(1, int(0.2 * len(subjects)))
    val_subjects = set(subjects[:n_val])
    tr_df = df[~df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    va_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    return tr_df, va_df


def _build_label_map(train_df: pd.DataFrame) -> Dict[str, int]:
    train_df = train_df.copy()
    train_df["subject_id"] = train_df["subject_id"].astype(str)
    subjects = sorted(train_df["subject_id"].unique().tolist())
    return {sid: i for i, sid in enumerate(subjects)}


def _write_split_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _train_loop_per_worker(cfg: Dict) -> None:
    """Ray Train worker function."""

    ctx = train.get_context()
    rank = ctx.get_world_rank()

    config_path = str(cfg["config_path"])
    output_dir = Path(cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # `train_loop_config` is a plain dict (often JSON-like), so paths may arrive
    # as strings even if the driver passed a `Path`.
    app_cfg: AppConfig = load_config(Path(config_path))
    # `seed` is defined on AppConfig (top-level), not on TrainConfig.
    # Keep a safe default for older configs.
    _seed_everything(int(getattr(app_cfg, "seed", 42)))

    # Load split manifests.
    train_df, val_df = _read_split_manifests(output_dir)
    if len(train_df) == 0:
        raise RuntimeError(
            "Training split is empty. Check preprocess output and dataset-dir."
        )
    if len(val_df) == 0:
        raise RuntimeError(
            "Validation split is empty. Check preprocess output and dataset-dir."
        )

    # Persist splits on rank 0 if they were generated dynamically.
    if rank == 0:
        splits_dir = output_dir / "splits"
        if not (splits_dir / "train_manifest.parquet").exists():
            _write_split_manifest(train_df, splits_dir / "train_manifest.parquet")
        if not (splits_dir / "val_manifest.parquet").exists():
            _write_split_manifest(val_df, splits_dir / "val_manifest.parquet")

    label_to_idx = _build_label_map(train_df)
    num_classes = len(label_to_idx)

    image_size = int(getattr(app_cfg.data, "image_size", 128))
    tfm = default_image_transform(image_size)

    # Create *per-worker* local parquet files for datasets.
    # This is important because `MultimodalBiometricDataset` reads a parquet from disk.
    work_dir = output_dir / "_worker_manifests" / f"rank_{rank}"
    work_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = work_dir / "train.parquet"
    val_manifest_path = work_dir / "val.parquet"
    train_df.to_parquet(train_manifest_path, index=False)
    val_df.to_parquet(val_manifest_path, index=False)

    train_ds = MultimodalBiometricDataset(
        manifest_path=train_manifest_path,
        iris_transform=tfm,
        fingerprint_transform=tfm,
        label_to_index=label_to_idx,
    )
    val_ds = MultimodalBiometricDataset(
        manifest_path=val_manifest_path,
        iris_transform=tfm,
        fingerprint_transform=tfm,
        label_to_index=label_to_idx,
    )

    # batch size lives under app_cfg.data (not app_cfg.train)
    batch_size = int(getattr(app_cfg.data, "batch_size", 16))
    num_workers = int(getattr(app_cfg.data, "num_workers", 0))
    if os.name == "nt":
        # Nested multiprocessing (DataLoader workers inside Ray workers) is
        # a common source of flakiness on Windows. Default to single-process
        # data loading; you can set data.num_workers=0 explicitly too.
        num_workers = 0

    # NOTE: Windows + Ray + DataLoader workers can be fragile; if you hit hangs,
    # set `data.num_workers: 0` in configs/default.yaml.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_samples,
    )

    # Let Ray wrap model & loader for DDP.
    device = train.torch.get_device()

    model = MultimodalNet(
        backbone=str(app_cfg.model.backbone),
        embedding_dim=int(app_cfg.model.embedding_dim),
        num_classes=num_classes,
        dropout=float(app_cfg.model.dropout),
    )
    model.to(device)
    model = train.torch.prepare_model(model)
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=float(app_cfg.train.lr),
        weight_decay=float(app_cfg.train.weight_decay),
    )

    best_val_acc = -1.0
    best_path = output_dir / "best.pt"

    if rank == 0:
        idx_to_label = {i: sid for sid, i in label_to_idx.items()}
        (output_dir / "labels.json").write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

        meta = {
            "num_classes": num_classes,
            "backbone": str(app_cfg.model.backbone),
            "embedding_dim": int(app_cfg.model.embedding_dim),
            "dropout": float(app_cfg.model.dropout),
            "image_size": image_size,
            "config_path": str(Path(config_path).resolve()),
        }
        (output_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for epoch in range(int(app_cfg.train.epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            iris = batch.iris.to(device)
            fp = batch.fingerprint.to(device)
            labels = batch.label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(iris, fp)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

        # Aggregate train metrics across workers.
        loss_tensor = torch.tensor([running_loss, total, correct], device=device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        running_loss, total, correct = loss_tensor.tolist()

        train_loss = float(running_loss / max(1.0, total))
        train_acc = float(correct / max(1.0, total))

        # Validation (each worker evaluates its full val set; report from rank 0).
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                iris = batch.iris.to(device)
                fp = batch.fingerprint.to(device)
                labels = batch.label.to(device)
                logits = model(iris, fp)
                loss = criterion(logits, labels)
                val_loss_sum += float(loss.item()) * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.size(0))

        val_loss_tensor = torch.tensor([val_loss_sum, val_total, val_correct], device=device, dtype=torch.float64)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        val_loss_sum, val_total, val_correct = val_loss_tensor.tolist()

        val_loss = float(val_loss_sum / max(1.0, val_total))
        val_acc = float(val_correct / max(1.0, val_total))

        if rank == 0:
            # Save best model.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model_state_dict": getattr(model, "module", model).state_dict(),
                        "val_acc": best_val_acc,
                    },
                    best_path,
                )

            train.report(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                }
            )
