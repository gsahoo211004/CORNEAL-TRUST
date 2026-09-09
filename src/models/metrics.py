from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    """Intersection-over-Union (Jaccard) score for binary segmentation.

    Args:
        pred: (B, 1, H, W) logits.
        target: (B, 1, H, W) binary mask.
    Returns:
        Mean IoU over the batch (float tensor).
    """
    p = (torch.sigmoid(pred) > threshold).float()
    t = target.float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6
) -> torch.Tensor:
    """Dice score for binary segmentation."""
    p = (torch.sigmoid(pred) > threshold).float()
    t = target.float()
    inter = (p * t).sum(dim=(1, 2, 3))
    total = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (total + eps)
    return dice.mean()


def accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Pixel accuracy for binary segmentation."""
    p = (torch.sigmoid(pred) > threshold).float()
    correct = (p == target.float()).float().mean(dim=(1, 2, 3))
    return correct.mean()


class SegmentationMetrics:
    """Accumulates segmentation metrics across batches."""

    def __init__(self) -> None:
        self.ious: list[float] = []
        self.dices: list[float] = []
        self.accs: list[float] = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        self.ious.append(float(iou_score(pred, target).item()))
        self.dices.append(float(dice_score(pred, target).item()))
        self.accs.append(float(accuracy(pred, target).item()))

    def results(self) -> dict[str, float]:
        return {
            "iou": float(np.mean(self.ious)) if self.ious else 0.0,
            "dice": float(np.mean(self.dices)) if self.dices else 0.0,
            "accuracy": float(np.mean(self.accs)) if self.accs else 0.0,
        }

    def reset(self) -> None:
        self.ious.clear()
        self.dices.clear()
        self.accs.clear()
