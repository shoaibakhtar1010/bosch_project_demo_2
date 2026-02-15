from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmbiometric.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainResult:
    best_val_acc: float
    best_ckpt_path: Path


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_every: int,
) -> float:
    model.train()
    running = 0.0
    for step, batch in enumerate(loader):
        left = batch.left_iris.to(device)
        right = batch.right_iris.to(device)
        fp = batch.fingerprint.to(device)
        y = batch.label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(left, right, fp)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        if log_every > 0 and (step + 1) % log_every == 0:
            logger.info("step=%s loss=%.4f", step + 1, running / (step + 1))
    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for batch in loader:
        left = batch.left_iris.to(device)
        right = batch.right_iris.to(device)
        fp = batch.fingerprint.to(device)
        y = batch.label.to(device)
        logits = model(left, right, fp)
        loss = criterion(logits, y)
        loss_sum += float(loss.item())
        acc_sum += accuracy(logits, y)
        n += 1
    return loss_sum / max(1, n), acc_sum / max(1, n)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    out_dir: Path,
    log_every: int,
) -> TrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, log_every)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            val_loss,
            val_acc,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": best_acc}, best_path)

    return TrainResult(best_val_acc=best_acc, best_ckpt_path=best_path)
