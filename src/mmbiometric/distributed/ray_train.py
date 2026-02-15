from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import ipaddress
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import ray
import torch
import torch.distributed as dist
import yaml
from ray import train
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


def _is_ipv4(addr: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(addr), ipaddress.IPv4Address)
    except Exception:
        return False


def _windows_iface_alias_to_ipv4(alias: str) -> str | None:
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f'(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "{alias}" | '
            "Select-Object -First 1 -ExpandProperty IPAddress)",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        return out or None
    except Exception:
        return None


def _to_storage_uri(path_str: str) -> str:
    p = Path(path_str).expanduser()
    p = p if p.is_absolute() else (Path.cwd() / p)
    return p.as_uri()


def _sanitize_master_addr(addr: str) -> str:
    a = (addr or "").strip()
    if not a:
        return "127.0.0.1"
    if a.lower() in {"kubernetes.docker.internal", "localhost"}:
        return "127.0.0.1"
    return a


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return int(s.getsockname()[1])


def _set_env(
    master_addr: str,
    master_port: int,
    gloo_ifname: str,
    gloo_family: str,
    use_libuv: int,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["GLOO_SOCKET_FAMILY"] = gloo_family

    gloo_val = (gloo_ifname or "").strip()

    if platform.system().lower().startswith("win"):
        if not gloo_val and _is_ipv4(master_addr):
            gloo_val = master_addr
        if gloo_val and not _is_ipv4(gloo_val):
            ip = _windows_iface_alias_to_ipv4(gloo_val)
            if ip:
                gloo_val = ip

    if gloo_val:
        os.environ["GLOO_SOCKET_IFNAME"] = gloo_val
    else:
        os.environ.pop("GLOO_SOCKET_IFNAME", None)

    os.environ["USE_LIBUV"] = str(int(use_libuv))
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")


def _setup_pg_debug(
    backend: str,
    world_rank: int,
    world_size: int,
    init_method: str,
    timeout_s: int,
):
    env_keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "GLOO_SOCKET_IFNAME",
        "GLOO_SOCKET_FAMILY",
        "USE_LIBUV",
        "TORCH_DISTRIBUTED_DEBUG",
        "HOSTNAME",
        "COMPUTERNAME",
    ]
    env_view = {k: os.environ.get(k) for k in env_keys}
    print(f"[pg-debug] rank={world_rank} world_size={world_size} init_method={init_method}")
    print(f"[pg-debug] env={env_view}")

    try:
        ma = os.environ.get("MASTER_ADDR", "")
        if ma:
            infos = socket.getaddrinfo(ma, None)
            print(f"[pg-debug] MASTER_ADDR getaddrinfo={infos[:3]}")
    except Exception as e:
        print(f"[pg-debug] MASTER_ADDR getaddrinfo failed: {e}")

    return _setup_torch_process_group(
        backend=backend,
        world_rank=world_rank,
        world_size=world_size,
        init_method=init_method,
        timeout_s=timeout_s,
    )


@dataclass
class PatchedTorchConfig(TorchConfig):
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    gloo_socket_ifname: str = ""
    gloo_socket_family: str = "AF_INET"
    use_libuv: int = 0

    @property
    def backend_cls(self):
        return _PatchedTorchBackend


class _PatchedTorchBackend(_TorchBackend):
    def on_start(self, worker_group, backend_config: PatchedTorchConfig):
        master_addr = _sanitize_master_addr(backend_config.master_addr)
        master_port = (
            int(backend_config.master_port) if backend_config.master_port else _pick_free_port()
        )
        init_method = backend_config.init_method or f"tcp://{master_addr}:{master_port}"

        worker_group.execute(
            _set_env,
            master_addr=master_addr,
            master_port=master_port,
            gloo_ifname=(backend_config.gloo_socket_ifname or ""),
            gloo_family=(backend_config.gloo_socket_family or "AF_INET"),
            use_libuv=int(backend_config.use_libuv),
        )

        setup_futures = []
        world_size = getattr(worker_group, "num_workers", None)
        if world_size is None:
            try:
                world_size = len(worker_group)
            except Exception:
                world_size = len(worker_group.get_workers())

        for rank in range(world_size):
            setup_futures.append(
                worker_group.execute_single_async(
                    rank,
                    _setup_pg_debug,
                    backend=backend_config.backend,
                    world_rank=rank,
                    world_size=world_size,
                    init_method=init_method,
                    timeout_s=backend_config.timeout_s,
                )
            )
        ray.get(setup_futures)


@dataclass
class RayTrainArgs:
    config_path: str
    dataset_dir: str
    output_dir: str
    subject_regex: str = r"(\d+)"
    num_workers: int = 2
    cpus_per_worker: int = 2
    use_gpu: bool = False

    master_addr: str = "127.0.0.1"
    master_port: int = 29500

    gloo_ifname: str = ""
    gloo_family: str = "AF_INET"
    use_libuv: int = 0

    ray_address: str = ""
    run_name: str = "mmbiometric_ray_train"
    fresh_run: bool = False


def _ensure_ray_connected(address: str) -> None:
    if ray.is_initialized():
        return

    addr = (address or "").strip()
    if addr:
        ray.init(address=addr, ignore_reinit_error=True, log_to_driver=True)
    else:
        ray.init(ignore_reinit_error=True, log_to_driver=True)


def _load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _train_loop_per_worker(config: dict):
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    try:
        trial_dir = Path(train.get_trial_dir())
    except Exception:
        trial_dir = Path.cwd()
    out_dir = trial_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    data_cfg = config.get("data", {}) or {}
    model_cfg = config.get("model", {}) or {}
    train_cfg = config.get("train", {}) or {}

    dataset_dir = Path(str(config.get("_dataset_dir") or data_cfg.get("dataset_dir"))).expanduser()
    if not dataset_dir:
        raise ValueError("dataset_dir is required (pass --dataset-dir)")
    dataset_dir = dataset_dir.resolve()

    subject_regex = str(config.get("_subject_regex") or r"(\d+)")

    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    val_fraction = float(data_cfg.get("val_fraction", 0.2))

    num_workers = int(data_cfg.get("num_workers", 2))
    if platform.system().lower().startswith("win"):
        num_workers = 0

    epochs = int(train_cfg.get("epochs", 3))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    log_every = int(train_cfg.get("log_every", 20))

    backbone = str(model_cfg.get("backbone", "resnet18"))
    embedding_dim = int(model_cfg.get("embedding_dim", 256))
    dropout = float(model_cfg.get("dropout", 0.1))

    seed_everything(seed + rank)

    manifest_path = out_dir / "manifest.parquet"
    if rank == 0 and not manifest_path.exists():
        build_manifest(
            dataset_dir=dataset_dir, output_path=manifest_path, subject_regex=subject_regex
        )
    if world > 1 and dist.is_initialized():
        dist.barrier()

    splits_dir = out_dir / "splits"
    train_manifest = splits_dir / "train_manifest.parquet"
    val_manifest = splits_dir / "val_manifest.parquet"
    if rank == 0 and (not train_manifest.exists() or not val_manifest.exists()):
        split_manifest(manifest_path, out_dir=splits_dir, val_fraction=val_fraction, seed=seed)
    if world > 1 and dist.is_initialized():
        dist.barrier()

    train_df = pd.read_parquet(train_manifest)
    if len(train_df) == 0:
        raise ValueError(
            "Train split is empty. Subject_id extraction likely collapsed. "
            f"Try a different --subject-regex. manifest={manifest_path}"
        )
    labels = sorted(train_df["subject_id"].astype(str).unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    num_classes = len(label_to_idx)

    tfm = default_image_transform(image_size)
    train_ds = MultimodalBiometricDataset(train_manifest, tfm, tfm, label_to_idx)
    val_ds = MultimodalBiometricDataset(val_manifest, tfm, tfm, label_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )

    train_loader = prepare_data_loader(train_loader)
    val_loader = prepare_data_loader(val_loader)

    model = MultimodalNet(
        backbone=backbone,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    model = prepare_model(model)

    device = get_device()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = out_dir / "best.pt"

    def _reduce_mean(x: torch.Tensor) -> torch.Tensor:
        if world <= 1 or not dist.is_initialized():
            return x
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= float(world)
        return y

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            iris = batch.iris.to(device)
            fp = batch.fingerprint.to(device)
            y = batch.label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(iris, fp)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            n_batches += 1

            if rank == 0 and log_every > 0 and (step + 1) % log_every == 0:
                print(
                    f"[train] epoch={epoch} step={step + 1} loss={loss_sum / max(1, n_batches):.4f}"
                )

        train_loss = torch.tensor(loss_sum / max(1, n_batches), device=device)
        train_loss = _reduce_mean(train_loss)

        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)

        with torch.no_grad():
            for batch in val_loader:
                iris = batch.iris.to(device)
                fp = batch.fingerprint.to(device)
                y = batch.label.to(device)

                logits = model(iris, fp)
                loss = criterion(logits, y)

                bs = float(y.numel())
                val_loss_sum += loss.detach() * bs
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).float().sum()
                total += torch.tensor(bs, device=device)

        if world > 1 and dist.is_initialized():
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        val_loss = (val_loss_sum / torch.clamp(total, min=1.0)).item()
        val_acc = (correct / torch.clamp(total, min=1.0)).item()

        if rank == 0 and val_acc > best_acc:
            best_acc = float(val_acc)
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model_state_dict": state, "val_acc": best_acc}, best_path)

            (out_dir / "labels.json").write_text(
                json.dumps(idx_to_label, indent=2, sort_keys=True), encoding="utf-8"
            )
            (out_dir / "model_metadata.json").write_text(
                json.dumps(
                    {
                        "backbone": backbone,
                        "embedding_dim": embedding_dim,
                        "dropout": dropout,
                        "image_size": image_size,
                        "num_classes": num_classes,
                        "best_val_acc": best_acc,
                        "best_ckpt_path": str(best_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        # Corrected metrics reporting: construct a metrics dict and call train.report.
        if rank == 0:
            metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "best_val_acc": float(best_acc),
            }
            try:
                train.report(metrics)
            except Exception:
                try:
                    from ray.air import session as air_session  # type: ignore

                    air_session.report(metrics)
                except Exception:
                    print(f"[metrics] {metrics}")


def train_distributed(args: RayTrainArgs):
    run_dir = os.path.join(args.output_dir, args.run_name)
    if args.fresh_run and os.path.exists(run_dir):
        shutil.rmtree(run_dir, ignore_errors=True)

    cfg = _load_yaml(args.config_path)
    cfg["_dataset_dir"] = str(Path(args.dataset_dir).expanduser())
    cfg["_subject_regex"] = str(args.subject_regex)

    _ensure_ray_connected(args.ray_address)

    master_addr = _sanitize_master_addr(args.master_addr)
    master_port = int(args.master_port) if args.master_port else _pick_free_port()

    torch_config = PatchedTorchConfig(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        timeout_s=1800,
        master_addr=master_addr,
        master_port=master_port,
        gloo_socket_ifname=(args.gloo_ifname or ""),
        gloo_socket_family=(args.gloo_family or "AF_INET"),
        use_libuv=int(args.use_libuv),
    )

    scaling_config = ScalingConfig(
        num_workers=int(args.num_workers),
        use_gpu=bool(args.use_gpu),
        resources_per_worker={"CPU": int(args.cpus_per_worker)},
    )

    storage_uri = _to_storage_uri(args.output_dir)

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop_per_worker,
        train_loop_config=cfg,
        scaling_config=scaling_config,
        run_config=RunConfig(
            name=args.run_name,
            storage_path=storage_uri,
        ),
        torch_config=torch_config,
    )
    return trainer.fit()
