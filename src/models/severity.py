from __future__ import annotations

import torch
import torch.nn as nn

from .unet import DoubleConv, Down


class SeverityNet(nn.Module):
    """Lightweight 4-class corneal neuropathy severity classifier.

    Reuses the U-Net encoder (same ``DoubleConv``/``Down`` blocks as
    ``NerveUnet``) followed by a global-average-pool classification head.
    Outputs raw class logits for CrossEntropyLoss.

    Args:
        in_channels: input image channels (1 for grayscale).
        num_classes: number of severity grades (CORN1500/CORN-3: 4).
        base_channels: first-layer channels (encoder width).
        depth: encoder depth (number of spatial scales).
        dropout: dropout applied in encoder + head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)
        channels = [base_channels * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList(
            [
                Down(channels[i], channels[i + 1], dropout=dropout)
                for i in range(depth - 1)
            ]
        )
        self.bottleneck = Down(channels[-1], channels[-1] * 2, dropout=dropout)

        feat = channels[-1] * 2
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat, feat // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        x = self.bottleneck(x)
        return self.head(x)


def build_severity_net(cfg: dict | None = None) -> SeverityNet:
    """Build a SeverityNet from config dict (or defaults)."""
    if cfg is None:
        cfg = {}
    sev_cfg = cfg.get("severity", {})
    return SeverityNet(
        in_channels=cfg.get("data", {}).get("channels", 1),
        num_classes=sev_cfg.get("num_classes", 4),
        base_channels=sev_cfg.get("base_channels", 32),
        depth=sev_cfg.get("depth", 4),
        dropout=sev_cfg.get("dropout", 0.2),
    )