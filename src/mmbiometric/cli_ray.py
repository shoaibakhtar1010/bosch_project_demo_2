from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray

from mmbiometric.distributed.ray_manifest import build_manifest_distributed
from mmbiometric.distributed.ray_train import RayTrainArgs, train_distributed


def _ray_init() -> None:
    """Initialize Ray.

    Precedence:
      1) If RAY_ADDRESS or RAY_HEAD_ADDRESS is set, connect to that cluster.
      2) Otherwise, start a *fresh local* Ray instance.

    On Windows, Ray + PyTorch distributed can trip over hostname/IP resolution
    when attaching to a previously-started cluster. Starting a fresh local
    instance with an explicit loopback node IP avoids the common
    `makeDeviceForHostname(): unsupported gloo device` failure.
    """

    address = os.environ.get("RAY_ADDRESS") or os.environ.get("RAY_HEAD_ADDRESS")

    if ray.is_initialized():
        return

    if address:
        ray.init(address=address)
        return

    # Start a new local instance even if one is already running.
    # See Ray docs for `address="local"` and `_node_ip_address`.
    if os.name == "nt":
        ray.init(address="local", _node_ip_address="127.0.0.1")
    else:
        ray.init()


def preprocess_main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: build a manifest.parquet from the raw dataset."""
    parser = argparse.ArgumentParser(prog="mmbiometric-ray-preprocess")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where manifest.parquet will be written",
    )
    parser.add_argument("--num-cpus", type=int, default=4, help="CPUs to use for preprocessing")
    args = parser.parse_args(argv)

    _ray_init()

    build_manifest_distributed(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_cpus=args.num_cpus,
    )

    out_manifest = Path(args.output_dir) / "manifest.parquet"
    print(f"[OK] Wrote manifest: {out_manifest}")


def train_main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: distributed training using Ray Train."""
    parser = argparse.ArgumentParser(prog="mmbiometric-ray-train")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root directory")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory (must contain manifest.parquet or splits/*.parquet)",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="Number of Ray Train workers")
    parser.add_argument("--cpus-per-worker", type=int, default=2, help="CPUs per worker")
    parser.add_argument("--use-gpu", action="store_true", help="Use 1 GPU per worker")
    parser.add_argument(
        "--master-addr",
        help="Override MASTER_ADDR for torch distributed (useful on Windows multi-worker).",
    )
    parser.add_argument(
        "--gloo-ifname",
        help="Override GLOO_SOCKET_IFNAME for torch distributed (e.g. 'Wi-Fi').",
    )
    parser.add_argument(
        "--subject-regex",
        default=r"^(\d+)$",
        help="Regex to extract subject_id from folder names (defaults to numeric folders).",
    )

    args = parser.parse_args(argv)

    config_value = args.config.strip()
    if not config_value:
        raise SystemExit("Config path is empty. Set --config to a valid YAML file.")
    config_path = Path(config_value).expanduser()
    if config_path.is_dir():
        raise SystemExit(f"Config path must be a file, got directory: {config_path}")

    _ray_init()

    ray_args = RayTrainArgs(
        config_path=str(config_path),
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        subject_regex=args.subject_regex,
        num_workers=args.num_workers,
        cpus_per_worker=args.cpus_per_worker,
        use_gpu=bool(args.use_gpu),
        master_addr=args.master_addr,
        gloo_socket_ifname=args.gloo_ifname,
    )
    train_distributed(ray_args)


def main(argv: list[str] | None = None) -> None:
    """Optional combined CLI: `mmbiometric-ray preprocess|train ...`"""
    parser = argparse.ArgumentParser(prog="mmbiometric-ray")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preprocess", help="Build manifest.parquet")
    p_pre.add_argument("--dataset-dir", required=True)
    p_pre.add_argument("--output-dir", required=True)
    p_pre.add_argument("--num-cpus", type=int, default=4)

    p_tr = sub.add_parser("train", help="Train with Ray Train")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument("--dataset-dir", required=True)
    p_tr.add_argument("--output-dir", required=True)
    p_tr.add_argument("--num-workers", type=int, default=2)
    p_tr.add_argument("--cpus-per-worker", type=int, default=2)
    p_tr.add_argument("--use-gpu", action="store_true")
    p_tr.add_argument("--master-addr")
    p_tr.add_argument("--gloo-ifname")
    p_tr.add_argument("--subject-regex", default=r"^(\d+)$")

    ns = parser.parse_args(argv)

    if ns.cmd == "preprocess":
        preprocess_main(
            [
                "--dataset-dir",
                ns.dataset_dir,
                "--output-dir",
                ns.output_dir,
                "--num-cpus",
                str(ns.num_cpus),
            ]
        )
    elif ns.cmd == "train":
        argv2 = [
            "--config",
            ns.config,
            "--dataset-dir",
            ns.dataset_dir,
            "--output-dir",
            ns.output_dir,
            "--num-workers",
            str(ns.num_workers),
            "--cpus-per-worker",
            str(ns.cpus_per_worker),
            "--subject-regex",
            ns.subject_regex,
        ]
        if ns.use_gpu:
            argv2.append("--use-gpu")
        if ns.master_addr:
            argv2.extend(["--master-addr", ns.master_addr])
        if ns.gloo_ifname:
            argv2.extend(["--gloo-ifname", ns.gloo_ifname])
        train_main(argv2)
    else:
        raise SystemExit(f"Unknown command: {ns.cmd}")


if __name__ == "__main__":
    main()
