"""Phase 4: CORN-2 image-quality classifier (feeds the CDTI Q_image).

Reuses the exact ``SeverityNet`` encoder/head architecture with a 2-class
head, classifying a raw IVCCM capture as high (class 1) or low (class 0)
quality so that the CDTI can discount predictions made on unusable images.
"""

from __future__ import annotations

from .severity import SeverityNet


class QualityNet(SeverityNet):
    """Binary image-quality classifier (reuses the severity encoder)."""

    pass


def build_quality_net(cfg: dict | None = None) -> QualityNet:
    """Build a 2-class QualityNet from config dict (or defaults)."""
    if cfg is None:
        cfg = {}
    q_cfg = cfg.get("quality", {})
    return QualityNet(
        in_channels=cfg.get("data", {}).get("channels", 1),
        num_classes=int(q_cfg.get("num_classes", 2)),
        base_channels=q_cfg.get("base_channels", 32),
        depth=q_cfg.get("depth", 4),
        dropout=q_cfg.get("dropout", 0.2),
    )