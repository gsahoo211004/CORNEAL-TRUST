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


def accuracy_score(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Top-1 accuracy for classification logits."""
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean()


def macro_f1_score(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Macro-averaged F1 for classification logits (per-batch).

    Averaging is performed over classes that actually appear in the truth or
    prediction, so a perfect prediction scores 1.0 even if some classes are
    absent in the batch.
    """
    pred = logits.argmax(dim=1)
    active = sorted(set(target.tolist()) | set(pred.tolist()))
    if not active:
        return torch.tensor(0.0)
    f1s = []
    for c in active:
        tp = ((pred == c) & (target == c)).sum().float()
        fp = ((pred == c) & (target != c)).sum().float()
        fn = ((pred != c) & (target == c)).sum().float()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1s.append(2 * precision * recall / (precision + recall + 1e-6))
    return torch.stack(f1s).mean()


def ece_score(
    probs: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for classification probabilities.

    Bins samples by max predicted probability and compares mean bin
    confidence to mean bin accuracy. Returns a scalar float in [0, 1]
    (0 = perfectly calibrated).
    """
    conf, pred = probs.max(dim=1)
    acc = (pred == target).float()
    n = conf.numel()
    if n == 0:
        return 0.0
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = 0.0
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= low) & (conf <= high)
        else:
            in_bin = (conf >= low) & (conf < high)
        size = int(in_bin.sum().item())
        if size == 0:
            continue
        ece += float((size / n) * (conf[in_bin].mean() - acc[in_bin].mean()).abs())
    return ece


class ClassificationMetrics:
    """Accumulates classification metrics (accuracy, macro-F1) across batches."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.accs: list[float] = []
        self.f1s: list[float] = []

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.accs.append(float(accuracy_score(logits, target).item()))
        self.f1s.append(float(macro_f1_score(logits, target, self.num_classes).item()))

    def results(self) -> dict[str, float]:
        return {
            "accuracy": float(np.mean(self.accs)) if self.accs else 0.0,
            "macro_f1": float(np.mean(self.f1s)) if self.f1s else 0.0,
        }

    def reset(self) -> None:
        self.accs.clear()
        self.f1s.clear()
