from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Corn1500SeverityDataset(Dataset):
    """CORN1500 dataset for 4-class severity grading.

    Directory structure: root/level{1,2,3,4}/
    Filenames: {n}_level{i}.jpg
    Labels: 0=level1, 1=level2, 2=level3, 3=level4
    """

    LEVEL_MAP = {"level1": 0, "level2": 1, "level3": 2, "level4": 3}

    def __init__(
        self,
        root: str | Path,
        image_size: int = 384,
        transform: Any = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.transform = transform

        self.samples = self._build_sample_list()

    def _build_sample_list(self) -> list[dict[str, Any]]:
        samples = []
        for level_name, label in self.LEVEL_MAP.items():
            level_dir = self.root / level_name
            if not level_dir.exists():
                continue
            for img_path in sorted(level_dir.glob("*.jpg")):
                samples.append({"image": img_path, "label": label})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        label = np.int64(sample["label"])

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
        img = np.array(Image.open(str(path)).convert("L"))
        return img
