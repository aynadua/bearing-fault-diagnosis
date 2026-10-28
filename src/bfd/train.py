"""CNN training, shared by the raw-waveform and order-tracked experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bfd.config import Config
from bfd.models import VibrationCNN, count_parameters
from bfd.simulate import CLASSES


@dataclass
class TrainResult:
    model: VibrationCNN
    history: list[dict[str, float]]
    best_val_accuracy: float
    train_seconds: float


def _as_tensor(x: np.ndarray) -> torch.Tensor:
    """Add the channel axis the convolutions expect."""
    return torch.from_numpy(x.astype(np.float32)).unsqueeze(1)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(model(_as_tensor(x[start : start + batch_size])).numpy())
    return np.concatenate(chunks).argmax(axis=1)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    verbose: bool = True,
) -> TrainResult:
    """Fit the CNN, keeping the best weights by validation accuracy."""
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    model = VibrationCNN(n_classes=len(CLASSES), width=cfg.train.width)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.train.epochs)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(_as_tensor(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.train.batch_size,
        shuffle=True,
    )

    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    start = time.perf_counter()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(yb)
            seen += len(yb)
        scheduler.step()

        val_accuracy = float(np.mean(predict(model, x_val) == y_val))
        history.append(
            {"epoch": epoch, "train_loss": total_loss / seen, "val_accuracy": val_accuracy}
        )
        if verbose:
            print(
                f"epoch {epoch:3d}  loss {total_loss / seen:.4f}  val_acc {val_accuracy:.4f}",
                flush=True,
            )

        if val_accuracy > best_accuracy:
            best_accuracy, best_state, stale_epochs = val_accuracy, {
                k: v.clone() for k, v in model.state_dict().items()
            }, 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.train.patience:
                if verbose:
                    print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        history=history,
        best_val_accuracy=best_accuracy,
        train_seconds=time.perf_counter() - start,
    )


def save_checkpoint(
    result: TrainResult, cfg: Config, representation: str, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "config": cfg.to_dict(),
            "representation": representation,
            "classes": list(CLASSES),
            "n_parameters": count_parameters(result.model),
            "best_val_accuracy": result.best_val_accuracy,
            "history": result.history,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> tuple[VibrationCNN, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    width = blob["config"]["train"]["width"]
    model = VibrationCNN(n_classes=len(blob["classes"]), width=width)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob
