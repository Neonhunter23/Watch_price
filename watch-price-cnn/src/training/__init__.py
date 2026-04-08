"""Training loop, early stopping, and optimizer/scheduler factories.

V4: Handles (images, brand_idxs, text_features, targets) batches.
    Uses MSE loss instead of Huber for better high-price gradients.
"""

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def create_optimizer(model, config):
    tc = config["training"]
    name = tc.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=tc["learning_rate"],
            momentum=0.9,
            weight_decay=tc["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(optimizer, config):
    tc = config["training"]
    name = tc.get("scheduler", "cosine").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=tc["epochs"] - tc.get("warmup_epochs", 5),
            eta_min=tc.get("min_lr", 1e-6),
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=10,
            factor=0.5,
        )
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss, total_mae, n = 0.0, 0.0, 0

    for images, brand_idxs, text_feats, targets in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        brand_idxs = brand_idxs.to(device)
        text_feats = text_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images, brand_idxs, text_feats)
        loss = criterion(preds, targets)
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_mae += torch.abs(preds - targets).sum().item()
        n += bs

    return {"loss": total_loss / n, "mae": total_mae / n}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0

    for images, brand_idxs, text_feats, targets in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        brand_idxs = brand_idxs.to(device)
        text_feats = text_feats.to(device)
        targets = targets.to(device)

        preds = model(images, brand_idxs, text_feats)
        loss = criterion(preds, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_mae += torch.abs(preds - targets).sum().item()
        n += bs

    return {"loss": total_loss / n, "mae": total_mae / n}


def train(model, train_loader, val_loader, config, device):
    tc = config["training"]
    epochs = tc["epochs"]

    # V4: MSE loss — gives stronger gradients for expensive watches
    loss_name = tc.get("loss", "mse").lower()
    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        criterion = nn.MSELoss()

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        patience=tc["early_stopping"]["patience"],
        min_delta=tc["early_stopping"]["min_delta"],
    )

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": [], "lr": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=tc.get("gradient_clip"),
        )
        val_metrics = validate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["lr"].append(lr)

        print(
            f"  Train Loss: {train_metrics['loss']:.4f} | MAE: {train_metrics['mae']:.4f}"
            f"  Val Loss: {val_metrics['loss']:.4f} | MAE: {val_metrics['mae']:.4f}"
            f"  LR: {lr:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_dir / "best_model.pt",
            )
            print(f"  New best model saved (val_loss={best_val_loss:.4f})")

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        if early_stopping.step(val_metrics["loss"]):
            print(f"\n Early stopping at epoch {epoch} (patience={early_stopping.patience})")
            break

    return history
