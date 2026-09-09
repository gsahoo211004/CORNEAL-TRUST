from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(cfg: dict[str, Any] | None = None) -> A.Compose:
    """Build training augmentations from config dict."""
    if cfg is None:
        cfg = {}
    aug_cfg = cfg.get("augmentation", {}).get("train", {})

    transforms = [
        A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_prob", 0.5)),
        A.VerticalFlip(p=aug_cfg.get("vertical_flip_prob", 0.3)),
        A.Affine(
            translate_percent=(0.0, 0.1),
            scale=(0.9, 1.1),
            rotate=(-aug_cfg.get("rotation_limit", 15), aug_cfg.get("rotation_limit", 15)),
            border_mode=0,
            p=0.5,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                    contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                    p=1,
                ),
                A.GaussNoise(p=1),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=aug_cfg.get("blur_limit", 3), p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ],
            p=0.3,
        ),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transforms(cfg: dict[str, Any] | None = None) -> A.Compose:
    """Build validation/test transforms (normalize + tensor only)."""
    return A.Compose(
        [
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
