from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, get_project_root


class TestConfig:
    def test_project_root(self):
        root = get_project_root()
        assert root.exists()
        assert (root / "configs" / "default.yaml").exists()

    def test_load_default_config(self):
        cfg = load_config()
        assert "data" in cfg
        assert "training" in cfg
        assert cfg["data"]["image_size"] == 384
        assert cfg["data"]["channels"] == 1
        assert cfg["data"]["num_classes_severity"] == 4

    def test_load_nonexistent_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestTransforms:
    def test_train_transforms(self):
        from src.data.transforms import get_train_transforms

        t = get_train_transforms()
        assert t is not None

        dummy_img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
        result = t(image=dummy_img)
        assert result["image"].shape == (1, 384, 384)

    def test_val_transforms(self):
        from src.data.transforms import get_val_transforms

        t = get_val_transforms()
        dummy_img = np.random.randint(0, 255, (384, 384), dtype=np.uint8)
        result = t(image=dummy_img)
        assert result["image"].shape == (1, 384, 384)


class TestCorn1Dataset:
    def test_loads_with_real_data(self, default_config):
        from src.data.corn1 import Corn1SegmentationDataset

        cfg = default_config
        root = ROOT.parent / cfg["corn1"]["root"].lstrip("../")
        if not root.exists():
            pytest.skip("CORN-1 data not available")

        ds = Corn1SegmentationDataset(root=root, split="train")
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "mask" in sample
        assert sample["image"].shape == (1, 384, 384)
        assert sample["mask"].shape == (1, 384, 384)


class TestCorn2Dataset:
    def test_loads_with_real_data(self, default_config):
        from src.data.corn2 import Corn2QualityDataset

        cfg = default_config
        root = ROOT.parent / cfg["corn2"]["root"].lstrip("../")
        if not root.exists():
            pytest.skip("CORN-2 data not available")

        ds = Corn2QualityDataset(root=root, split="train")
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (1, 384, 384)
        assert sample["label"] in (0.0, 1.0)


class TestCorn3Dataset:
    def test_loads_with_real_data(self, default_config):
        from src.data.corn3 import Corn3SeverityDataset

        cfg = default_config
        root = ROOT.parent / cfg["corn3"]["root"].lstrip("../")
        if not root.exists():
            pytest.skip("CORN-3 data not available")

        ds = Corn3SeverityDataset(root=root)
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (1, 384, 384)
        assert 0 <= sample["label"] <= 3


class TestCorn1500Dataset:
    def test_loads_with_real_data(self, default_config):
        from src.data.corn1500 import Corn1500SeverityDataset

        cfg = default_config
        root = ROOT.parent / cfg["corn1500"]["root"].lstrip("../")
        if not root.exists():
            pytest.skip("CORN1500 data not available")

        ds = Corn1500SeverityDataset(root=root)
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (1, 384, 384)
        assert 0 <= sample["label"] <= 3


class TestDataModule:
    def test_summary(self, default_config):
        from src.data.datamodule import CornealDataModule

        dm = CornealDataModule(default_config)
        summary = dm.summary()
        assert isinstance(summary, dict)
        assert "corn1" in summary
        assert "corn2" in summary
        assert "corn3" in summary
        assert "corn1500" in summary

    def test_train_loader_corn1(self, default_config):
        from src.data.datamodule import CornealDataModule

        dm = CornealDataModule(default_config)
        root = ROOT.parent / default_config["corn1"]["root"].lstrip("../")
        if not root.exists():
            pytest.skip("CORN-1 data not available")

        loader = dm.get_train_loader("corn1")
        batch = next(iter(loader))
        assert batch["image"].shape[1:] == (1, 384, 384)
        assert batch["mask"].ndim >= 2
