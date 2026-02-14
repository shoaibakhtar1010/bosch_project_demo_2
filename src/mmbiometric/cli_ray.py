# src/mmbiometric/cli_ray.py
"""Ray-based CLIs.

Console scripts (see pyproject.toml):
  - mmbiometric-ray-preprocess -> preprocess_main
  - mmbiometric-ray-train      -> train_main

Important:
  console_scripts call the referenced function with *no arguments*.
  Therefore `preprocess_main()` and `train_main()` must be no-arg entrypoints
  that parse sys.argv internally.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from mmbiometric.data.manifest import build_manifest
from mmbiometric.utils.logging import get_logger

logger = get_logger(__name__)


def _maybe_init_ray(ray_address: str, num_cpus: int) -> None:
    """Init Ray safely.

    - If address is provided (arg or env RAY_ADDRESS), CONNECT to that cluster.
      (Do NOT pass num_cpus / num_gpus when connecting.)
    - Else start LOCAL Ray with num_cpus.
    """
    try:
        import ray  # type: ignore
    except Exception:
        return

    if ray.is_initialized():
        return

    addr = (ray_address or os.environ.get("RAY_ADDRESS", "")).strip()

    if addr:
        # CONNECT mode
        ray.init(address=addr, ignore_reinit_error=True, log_to_driver=True)
    else:
        # LOCAL mode
        ray.init(num_cpus=int(num_cpus), ignore_reinit_error=True, include_dashboard=False)

def _preprocess_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmbiometric-ray-preprocess")
    p.add_argument("--dataset-dir", required=True, help="Dataset root directory")
    p.add_argument("--output-dir", required=True, help="Output directory")
    p.add_argument("--subject-regex", default=r"(\d+)", help="Regex to extract subject id")
    p.add_argument(
        "--num-cpus",
        type=int,
        default=6,
        help="Local Ray CPUs if starting a local Ray runtime",
    )
    p.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", ""),
        help='Ray address like "192.168.0.101:6379" (optional).',
    )
    p.add_argument(
        "--distributed",
        action="store_true",
        help="If set, uses Ray tasks to build the manifest (otherwise local scan).",
    )
    return p


def preprocess_main(argv: Sequence[str] | None = None) -> None:
    """Build manifest.parquet under --output-dir.

    This is intentionally a no-arg console entrypoint.
    """
    ns = _preprocess_parser().parse_args(list(argv) if argv is not None else None)
    dataset_dir = Path(ns.dataset_dir).expanduser().resolve()
    out_dir = Path(ns.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.parquet"

    # Dataset is small (~900 images) so local scanning is fine.
    # Use distributed mode only if you explicitly request it.
    if bool(ns.distributed):
        _maybe_init_ray(str(ns.ray_address), int(ns.num_cpus))
        from mmbiometric.distributed.ray_manifest import build_manifest_distributed

        build_manifest_distributed(
            dataset_dir=str(dataset_dir),
            output_dir=str(out_dir),
            num_cpus=int(ns.num_cpus),
        )
        logger.info("manifest written (distributed): %s", manifest_path)
    else:
        build_manifest(
            dataset_dir=dataset_dir,
            output_path=manifest_path,
            subject_regex=str(ns.subject_regex),
        )
        logger.info("manifest written: %s", manifest_path)

    print(f"[OK] Wrote manifest: {manifest_path}")


def _train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mmbiometric-ray-train")
    p.add_argument("--config", dest="config_path", required=True, help="Path to YAML config")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject-regex", default=r"(\d+)")

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cpus-per-worker", type=int, default=2)
    p.add_argument("--use-gpu", action="store_true", default=False)

    # Distributed rendezvous settings
    p.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    p.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500")))

    p.add_argument(
        "--gloo-ifname",
        default=os.environ.get("GLOO_SOCKET_IFNAME", ""),
        help="Optional. If set wrong, Gloo will fail. Prefer leaving unset on Windows.",
    )
    p.add_argument(
        "--gloo-family",
        default=os.environ.get("GLOO_SOCKET_FAMILY", "AF_INET"),
        choices=["AF_INET", "AF_INET6"],
    )
    p.add_argument(
        "--use-libuv",
        type=int,
        default=int(os.environ.get("USE_LIBUV", "0")),
        choices=[0, 1],
        help="Torch TCPStore: 0 disables libuv (often safer on Windows).",
    )

    p.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", ""),
        help='Ray address like "192.168.0.101:6379". If empty, uses "auto".',
    )
    p.add_argument("--run-name", default="mmbiometric_ray_train")
    p.add_argument("--fresh-run", action="store_true")
    return p


def train_main(argv: Sequence[str] | None = None) -> None:
    """Train using Ray Train + torch.distributed (DDP).

    This is intentionally a no-arg console entrypoint.
    """
    ns = _train_parser().parse_args(list(argv) if argv is not None else None)
    from mmbiometric.distributed.ray_train import RayTrainArgs, train_distributed

    ray_args = RayTrainArgs(
        config_path=str(ns.config_path),
        dataset_dir=str(ns.dataset_dir),
        output_dir=str(ns.output_dir),
        subject_regex=str(ns.subject_regex),
        num_workers=int(ns.num_workers),
        cpus_per_worker=int(ns.cpus_per_worker),
        use_gpu=bool(ns.use_gpu),
        master_addr=str(ns.master_addr),
        master_port=int(ns.master_port),
        gloo_ifname=(ns.gloo_ifname or ""),
        gloo_family=str(ns.gloo_family or "AF_INET"),
        use_libuv=int(ns.use_libuv),
        ray_address=(ns.ray_address or ""),
        run_name=str(ns.run_name),
        fresh_run=bool(ns.fresh_run),
    )
    train_distributed(ray_args)
