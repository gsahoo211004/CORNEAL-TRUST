from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from torch.utils.data import Dataset


class Corn3SeverityDataset(Dataset):
    """CORN-3 dataset for 4-class severity grading.

    Directory structure: root/grade{1,2,3,4}/
    Labels: 0=grade1, 1=grade2, 2=grade3, 3=grade4
    """

    GRADE_MAP = {"grade1": 0, "grade2": 1, "grade3": 2, "grade4": 3}

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
        for grade_name, label in self.GRADE_MAP.items():
            grade_dir = self.root / grade_name
            if not grade_dir.exists():
                continue
            for img_path in sorted(grade_dir.glob("*.tif")):
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
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
