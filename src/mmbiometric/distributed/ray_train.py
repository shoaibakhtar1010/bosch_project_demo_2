# src/mmbiometric/distributed/ray_train.py
from __future__ import annotations

import json
import os
import platform
import socket
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import ray
from ray import train
import torch
import torch.distributed as dist
import yaml
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.torch import get_device, prepare_data_loader, prepare_model
from ray.train.torch.config import TorchConfig, _TorchBackend, _setup_torch_process_group
from torch.utils.data import DataLoader

from mmbiometric.data.dataset import MultimodalBiometricDataset, collate_samples
from mmbiometric.data.manifest import build_manifest
from mmbiometric.data.split import split_manifest
from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet
from mmbiometric.utils.seed import seed_everything


def _to_storage_uri(path_str: str) -> str:
    # Keep as-is; your project likely expects local paths on single-node runs.
    # If you later move to shared storage (S3/NFS), you can adapt this.
    return str(path_str)


def _get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 2
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    save_every: int = 1


def train_loop_per_worker(config: dict):
    cfg = TrainConfig(**config)

    seed_everything(cfg.seed)

    # Ray Train v2 way to get trial dir (replaces ray.train.session.get_trial_dir()).
    trial_dir = Path(train.get_context().get_trial_dir())
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Load / prepare data
    dataset_dir = Path(config["dataset_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Manifest and splits (your pipeline)
    manifest = build_manifest(dataset_dir)
    train_manifest, val_manifest = split_manifest(manifest, seed=cfg.seed)

    # Dataset + loaders
    transform = default_image_transform()
    train_ds = MultimodalBiometricDataset(train_manifest, transform=transform)
    val_ds = MultimodalBiometricDataset(val_manifest, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_samples,
        pin_memory=False,
    )

    # Prepare for Ray Train (wraps loaders/model for distributed)
    train_loader = prepare_data_loader(train_loader)
    val_loader = prepare_data_loader(val_loader)

    device = get_device()
    model = MultimodalNet()
    model = prepare_model(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Rank info (after Ray Train sets up process group)
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    if rank == 0:
        print(f"[train] trial_dir={trial_dir}")
        print(f"[train] host={_get_hostname()} rank={rank} world_size={world_size}")
        print(f"[train] device={device}")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for batch in train_loader:
            # Adjust this according to your dataset batch structure.
            # Assuming collate_samples returns (x, y) or dict-like.
            if isinstance(batch, dict):
                x = batch["x"].to(device)
                y = batch["y"].to(device)
            else:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.shape[0]
            train_loss_sum += float(loss.detach().item()) * bs
            n_train += bs

        # Validation
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                else:
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)

                bs = y.shape[0]
                val_loss_sum += float(loss.detach().item()) * bs
                n_val += bs

                preds = torch.argmax(logits, dim=1)
                correct += int((preds == y).sum().item())

        train_loss = train_loss_sum / max(1, n_train)
        val_loss = val_loss_sum / max(1, n_val)
        val_acc = correct / max(1, n_val)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }

        # Save checkpoint periodically (rank 0 only)
        if (epoch + 1) % cfg.save_every == 0 and rank == 0:
            ckpt_dir = trial_dir / f"checkpoint_epoch_{epoch + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

            with open(ckpt_dir / "epoch.json", "w", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, **metrics}, f, indent=2)

            # (Optional) copy to your requested output_dir for convenience
            out_ckpt_dir = output_dir / f"checkpoint_epoch_{epoch + 1}"
            if out_ckpt_dir.exists():
                shutil.rmtree(out_ckpt_dir)
            shutil.copytree(ckpt_dir, out_ckpt_dir)

        # Report metrics to Ray Train (rank 0 only)
        if rank == 0:
            # Build a checkpoint when we saved one this epoch; otherwise report metrics only.
            try:
                # Prefer Ray Train v2 `train.report`; fall back to Ray AIR `session.report` if needed.
                report_fn = getattr(train, "report", None)
                if report_fn is None:
                    from ray.air import session as air_session  # type: ignore
                    report_fn = getattr(air_session, "report", None)

                checkpoint_obj = None
                if (epoch + 1) % cfg.save_every == 0 and rank == 0:
                    # ckpt_dir was created above in the save block
                    CheckpointCls = getattr(train, "Checkpoint", None)
                    if CheckpointCls is None:
                        from ray.air.checkpoint import Checkpoint as CheckpointCls  # type: ignore
                    checkpoint_obj = CheckpointCls.from_directory(str(ckpt_dir))

                if report_fn is None:
                    print(f"[metrics] {metrics}")
                elif checkpoint_obj is not None:
                    report_fn(metrics, checkpoint=checkpoint_obj)
                else:
                    report_fn(metrics)
            except Exception as e:
                # Don't fail training just because metric reporting/checkpointing failed.
                print(f"[warn] Ray report/checkpoint failed: {e}")
                print(f"[metrics] {metrics}")


def train_distributed(args):
    # Your CLI passes these in; keep compatible.
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Merge CLI overrides into train config
    train_cfg = cfg.get("train", {})
    train_cfg["dataset_dir"] = args.dataset_dir
    train_cfg["output_dir"] = args.output_dir

    ray.init(ignore_reinit_error=True)

    # Build backend init method (your args already include master addr/port)
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    backend = _TorchBackend(process_group_init_method=init_method)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=train_cfg,
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=False,
            resources_per_worker={"CPU": args.cpus_per_worker},
        ),
        run_config=RunConfig(name="mmbiometric_ray_train"),
        backend_config=TorchConfig(backend=backend),
    )

    result = trainer.fit()
    return result
