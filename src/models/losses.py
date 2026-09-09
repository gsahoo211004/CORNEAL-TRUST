from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0
) -> torch.Tensor:
    """Dice loss for binary segmentation.

    Args:
        pred: (B, 1, H, W) logits.
        target: (B, 1, H, W) binary mask in {0, 1}.
    """
    p = torch.sigmoid(pred)
    p = p.view(p.size(0), -1)
    t = target.view(target.size(0), -1)
    intersection = (p * t).sum(1)
    union = p.sum(1) + t.sum(1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class AMCLLoss(nn.Module):
    """Adaptive Multi-Scale Corneal Loss (L_AMCL).

    L_AMCL = alpha * L_nerve + beta * L_topology + gamma * L_uncert + delta * L_triage

    For Phase 2 we train the segmentation head with:
      - L_nerve: combined BCE + Dice (nerve binarization error)
      - L_topology: topology-preservation term using a soft skeleton
        proxy (grouped-min pooling promotes thin connected structure)
      - L_uncert / L_triage: included as placeholders (weighted 0 for now)
        since uncertainty and triage heads arrive in later phases.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 1.0,
        delta: float = 1.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        topology_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.topology_kernel = topology_kernel

    def _nerve_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice_loss(pred, target)

    def _topology_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Topology-preservation penalty.

        Uses max pooling on the target and a soft proxy on the prediction
        to reward preserving thin, connected nerve structure. A larger
        mismatch between the pooled target and pooled prediction indicates
        structural degradation.
        """
        pool = F.max_pool2d
        k = self.topology_kernel
        pooled_target = pool(target, kernel_size=k, stride=1, padding=k // 2)
        p = torch.sigmoid(pred)
        pooled_pred = pool(p, kernel_size=k, stride=1, padding=k // 2)
        return F.mse_loss(pooled_pred, pooled_target)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        """Compute the joint loss.

        Returns sum of weighted terms. Batch-average of L_nerve is
        reported as the dominant training signal in Phase 2.
        """
        nerve = self._nerve_loss(pred, target)
        topology = self._topology_loss(pred, target)
        loss = self.alpha * nerve + self.beta * topology
        return loss

    def component_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return individual loss components for logging."""
        nerve = self._nerve_loss(pred, target)
        topology = self._topology_loss(pred, target)
        return {
            "nerve": nerve,
            "topology": topology,
        }
