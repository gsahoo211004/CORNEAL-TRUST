from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two sequential conv-bn-relu blocks with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if dropout > 0:
            self.block = nn.Sequential(
                self.block, nn.Dropout2d(p=dropout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling block: maxpool + double conv."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling block: up-conv (in->out) + skip concat + double conv.

    Args:
        in_channels: channels from the decoder path below (bottleneck side).
        out_channels: target channels (also the number of skip channels).
    """

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            out_channels * 2, out_channels, dropout=dropout
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class NerveUnet(nn.Module):
    """Lightweight U-Net for corneal nerve fiber segmentation.

    Configurable base channel count for edge-deployment flexibility.
    Outputs a single-channel logit map (sigmoid applied outside for BCE).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        channels = [base_channels * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList(
            [Down(channels[i], channels[i + 1], dropout=dropout) for i in range(depth - 1)]
        )
        self.bottleneck = Down(channels[-1], channels[-1] * 2, dropout=dropout)

        self.ups = nn.ModuleList()
        prev = channels[-1] * 2
        for i in range(depth - 1, -1, -1):
            self.ups.append(Up(prev, channels[i], dropout=dropout))
            prev = channels[i]

        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = self.bottleneck(skips[-1])
        for up in self.ups:
            x = up(x, skips.pop())
        return self.outc(x)


def build_unet(cfg: dict | None = None) -> NerveUnet:
    """Build a NerveUnet from config dict (or defaults)."""
    if cfg is None:
        cfg = {}
    model_cfg = cfg.get("model", {})
    return NerveUnet(
        in_channels=cfg.get("data", {}).get("channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
        depth=model_cfg.get("depth", 4),
        dropout=model_cfg.get("dropout", 0.0),
    )
