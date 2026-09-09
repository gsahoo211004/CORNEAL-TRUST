from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from torch.utils.data import Dataset


class Corn2QualityDataset(Dataset):
    """CORN-2 dataset for image quality classification (high vs low).

    Training set: high-quality/ and low-quality/ subdirectories.
    Test set: low-quality/ only.
    Returns image tensor + binary label (1=high, 0=low).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 384,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[dict[str, Any]]:
        samples = []

        if self.split == "train":
            for quality, label in [("high-quality", 1), ("low-quality", 0)]:
                q_dir = self.root / "train" / quality
                if not q_dir.exists():
                    continue
                for img_path in sorted(q_dir.glob("*.tif")):
                    samples.append({"image": img_path, "label": label})
        elif self.split == "test":
            low_dir = self.root / "test" / "low-quality"
            if low_dir.exists():
                for img_path in sorted(low_dir.glob("*.tif")):
                    samples.append({"image": img_path, "label": 0})
            high_dir = self.root / "test" / "high-quality"
            if high_dir.exists():
                for img_path in sorted(high_dir.glob("*.tif")):
                    samples.append({"image": img_path, "label": 1})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        label = np.float32(sample["label"])

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            image = np.ascontiguousarray(image)

        return {
            "image": image,
            "label": label,
            "image_path": str(sample["image"]),
        }

    def _load_image(self, path: Path) -> np.ndarray:
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
