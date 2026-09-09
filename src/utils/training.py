from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across torch/numpy/python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    state: dict,
    path: str | Path,
) -> None:
    """Save a training checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load a training checkpoint from disk."""
    return torch.load(str(path), map_location=map_location)


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    min_lr: float = 0.0,
):
    """Cosine annealing LR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=min_lr
    )
