from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from .corn1 import Corn1SegmentationDataset
from .corn2 import Corn2QualityDataset
from .corn3 import Corn3SeverityDataset
from .corn1500 import Corn1500SeverityDataset
from .transforms import get_train_transforms, get_val_transforms


class CornealDataModule:
    """Unified data module providing dataloaders for all 4 CORNEAL datasets.

    Usage:
        dm = CornealDataModule(cfg)
        train_loader = dm.get_train_loader("corn1")
        val_loader = dm.get_val_loader("corn1")
        test_loader = dm.get_test_loader("corn1")
    """

    DATASET_REGISTRY = {
        "corn1": Corn1SegmentationDataset,
        "corn2": Corn2QualityDataset,
        "corn3": Corn3SeverityDataset,
        "corn1500": Corn1500SeverityDataset,
    }

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.get("data", {})
        self.train_cfg = cfg.get("training", {})

        self.image_size = self.data_cfg.get("image_size", 384)
        self.batch_size = self.train_cfg.get("batch_size", 8)
        self.num_workers = self.train_cfg.get("num_workers", 0)

        self.train_transform = get_train_transforms(cfg)
        self.val_transform = get_val_transforms(cfg)

        self._cache: dict[str, dict[str, Dataset]] = {}

    def _resolve_root(self, dataset_key: str) -> Path:
        """Resolve the root path for a dataset from config.

        Relative paths are resolved relative to the project root
        (parent of CORNEAL_TRUST/), since datasets live as siblings.
        """
        project_root = Path(__file__).resolve().parent.parent.parent
        raw = self.cfg.get(dataset_key, {}).get("root", "")
        root = Path(raw)
        if not root.is_absolute():
            root = project_root / root
        return root

    def _make_dataset(
        self, dataset_key: str, split: str, transform: Any
    ) -> Dataset:
        """Create a dataset instance for the given key and split."""
        root = self._resolve_root(dataset_key)
        cls = self.DATASET_REGISTRY[dataset_key]

        if dataset_key == "corn1":
            return cls(
                root=root,
                split=split,
                image_size=self.image_size,
                mask_threshold=self.cfg.get("corn1", {}).get("mask_threshold", 127),
                transform=transform,
            )
        elif dataset_key == "corn2":
            return cls(
                root=root,
                split=split,
                image_size=self.image_size,
                transform=transform,
            )
        elif dataset_key in ("corn3", "corn1500"):
            return cls(
                root=root,
                image_size=self.image_size,
                transform=transform,
            )
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

    def get_dataset(
        self, dataset_key: str, split: str = "train"
    ) -> Dataset:
        """Get a cached dataset instance."""
        cache_key = f"{dataset_key}_{split}"
        if cache_key not in self._cache:
            transform = (
                self.train_transform if split == "train" else self.val_transform
            )
            self._cache[cache_key] = self._make_dataset(
                dataset_key, split, transform
            )
        return self._cache[cache_key]

    def get_train_loader(self, dataset_key: str) -> DataLoader:
        """Get training dataloader."""
        dataset = self.get_dataset(dataset_key, split="train")
        use_pin = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_pin,
            drop_last=False,
        )

    def get_val_loader(self, dataset_key: str) -> DataLoader:
        """Get validation dataloader."""
        dataset = self.get_dataset(dataset_key, split="val")
        use_pin = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin,
        )

    def get_test_loader(self, dataset_key: str) -> DataLoader:
        """Get test dataloader."""
        dataset = self.get_dataset(dataset_key, split="test")
        use_pin = torch.cuda.is_available()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin,
        )

    def get_combined_severity_loader(
        self, split: str = "train", use_corn3_val: bool = True
    ) -> DataLoader:
        """Get a combined CORN1500 + CORN-3 severity dataloader.

        For training: CORN1500 (primary) + CORN-3 (if available).
        For validation/test: CORN-3 only (held-out validation).
        """
        datasets_list: list[Dataset] = []

        if split == "train":
            corn1500_root = self._resolve_root("corn1500")
            if corn1500_root.exists():
                datasets_list.append(
                    Corn1500SeverityDataset(
                        root=corn1500_root,
                        image_size=self.image_size,
                        transform=self.train_transform,
                    )
                )
        elif split in ("val", "test") and use_corn3_val:
            corn3_root = self._resolve_root("corn3")
            if corn3_root.exists():
                datasets_list.append(
                    Corn3SeverityDataset(
                        root=corn3_root,
                        image_size=self.image_size,
                        transform=self.val_transform,
                    )
                )

        if not datasets_list:
            raise FileNotFoundError(
                f"No severity datasets found for split={split}"
            )

        combined = ConcatDataset(datasets_list)
        use_pin = torch.cuda.is_available()
        return DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            pin_memory=use_pin,
        )

    def summary(self) -> dict[str, dict[str, int]]:
        """Return dataset sizes for all registered datasets."""
        info: dict[str, dict[str, int]] = {}
        for key in self.DATASET_REGISTRY:
            root = self._resolve_root(key)
            if not root.exists():
                info[key] = {"exists": 0}
                continue
            splits: dict[str, int] = {}
            for split in ("train", "val", "test"):
                try:
                    ds = self.get_dataset(key, split)
                    splits[split] = len(ds)
                except (FileNotFoundError, ValueError):
                    splits[split] = 0
            info[key] = splits
        return info
