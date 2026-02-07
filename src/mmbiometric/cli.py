from __future__ import annotations

# Standard library imports
import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

# Third‑party imports
import torch
from torch.utils.data import DataLoader
import yaml

# Local application imports
from mmbiometric.config import load_config
from mmbiometric.data.dataset import MultimodalBiometricDataset, collate_samples
from mmbiometric.data.manifest import build_manifest
from mmbiometric.data.split import split_manifest
from mmbiometric.data.transforms import default_image_transform
from mmbiometric.models.multimodal_net import MultimodalNet
from mmbiometric.training.loops import fit
from mmbiometric.utils.logging import get_logger
from mmbiometric.utils.seed import seed_everything


logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset-dir", type=str, required=False)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--subject-regex", type=str, default=r"(\d+)")
    return p.parse_args()


def build_manifest_main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config))
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.data.dataset_dir
    if dataset_dir is None:
        raise SystemExit("dataset-dir is required")
    out_dir = Path(args.output_dir)
    manifest = build_manifest(dataset_dir=dataset_dir, output_path=out_dir / "manifest.parquet",
                              subject_regex=args.subject_regex)
    logger.info("manifest written: %s", manifest)


def train_main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config))

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.data.dataset_dir
    if dataset_dir is None:
        raise SystemExit("dataset-dir is required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    # 1) manifest
    manifest_path = cfg.data.manifest_path or (out_dir / "manifest.parquet")
    if not manifest_path.exists():
        build_manifest(dataset_dir=dataset_dir, output_path=manifest_path, subject_regex=args.subject_regex)

    # 2) split by subject
    splits = split_manifest(manifest_path, out_dir=out_dir / "splits", val_fraction=cfg.data.val_fraction, seed=cfg.seed)

    # 3) label mapping from train subjects
    import pandas as pd
    train_df = pd.read_parquet(splits.train_manifest)
    if len(train_df) == 0:
        # This almost always means the subject extraction collapsed to a single subject
        # and the split logic put that subject entirely into the validation split.
        manifest_df = pd.read_parquet(manifest_path)
        uniq_subj = sorted(manifest_df["subject_id"].astype(str).unique().tolist())
        raise ValueError(
            "Train split is empty (0 samples).\n"
            f"Manifest rows={len(manifest_df)} unique_subjects={len(uniq_subj)}\n"
            "This is usually caused by the subject-id regex matching a constant token in the path.\n"
            "Fix options:\n"
            "  1) Re-run with a better --subject-regex that targets the subject folder / filename.\n"
            "     Example: --subject-regex '(?:subject|person|id)[^0-9]*([0-9]+)'\n"
            "  2) Ensure you're passing the correct --dataset-dir (the directory *containing* the images).\n"
            "  3) Delete output-dir/splits and re-run.\n"
            f"First 10 detected subject_ids: {uniq_subj[:10]}"
        )
    labels = sorted(train_df["subject_id"].astype(str).unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    num_classes = len(label_to_idx)

    tfm = default_image_transform(cfg.data.image_size)

    train_ds = MultimodalBiometricDataset(splits.train_manifest, tfm, tfm, label_to_idx)
    val_ds = MultimodalBiometricDataset(splits.val_manifest, tfm, tfm, label_to_idx)

    num_workers = cfg.data.num_workers
    if platform.system().lower().startswith("win") and num_workers > 0:
        # Windows uses spawn for DataLoader workers; keeping this at 0 avoids pickling issues.
        logger.warning("Windows detected: forcing DataLoader num_workers=0 (was %s)", num_workers)
        num_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )

    # 4) model
    model = MultimodalNet(
        backbone=cfg.model.backbone,
        embedding_dim=cfg.model.embedding_dim,
        num_classes=len(label_to_idx),
        dropout=cfg.model.dropout,
    )

    # 5) device
    if cfg.train.device == "cpu":
        device = torch.device("cpu")
    elif cfg.train.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # 6) fit
    res = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.train.epochs,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        device=device,
        out_dir=out_dir,
        log_every=cfg.train.log_every,
    )

    # Save label mapping alongside checkpoint for inference
    labels_path = out_dir / "labels.json"
    labels_path.write_text(json.dumps(idx_to_label, indent=2, sort_keys=True), encoding="utf-8")

    # Persist the resolved config and minimal model metadata so inference can be
    # run without having to re-specify architecture flags.
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8"
    )
    (out_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "backbone": cfg.model.backbone,
                "embedding_dim": cfg.model.embedding_dim,
                "dropout": cfg.model.dropout,
                "image_size": cfg.data.image_size,
                "num_classes": num_classes,
                "best_val_acc": float(res.best_val_acc),
                "best_ckpt_path": str(res.best_ckpt_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(device),
                "seed": int(cfg.seed),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("best_val_acc=%.4f ckpt=%s", res.best_val_acc, res.best_ckpt_path)
