from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from torch.utils.data import Dataset


class Corn1SegmentationDataset(Dataset):
    """CORN-1 dataset for corneal nerve fiber segmentation.

    Loads paired image-mask .tif files from pre-split train/val/test dirs.
    Images are loaded as grayscale (single channel).
    Masks are binarized using a configurable threshold.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 384,
        mask_threshold: int = 127,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.transform = transform

        self.image_dir = self.root / "images" / f"{split}_orig"
        self.label_dir = self.root / "labels" / f"{split}_label"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[dict[str, Path]]:
        image_files = sorted(self.image_dir.glob("*.tif"))
        samples = []
        for img_path in image_files:
            mask_path = self.label_dir / img_path.name
            if mask_path.exists():
                samples.append({"image": img_path, "mask": mask_path})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        mask = self._load_mask(sample["mask"])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            image = np.ascontiguousarray(image)
            mask = (mask > self.mask_threshold).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            mask = np.ascontiguousarray(mask)

        return {
            "image": image,
            "mask": mask,
            "image_path": str(sample["image"]),
        }

    def _load_image(self, path: Path) -> np.ndarray:
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _load_mask(self, path: Path) -> np.ndarray:
        mask = tifffile.imread(str(path))
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask
