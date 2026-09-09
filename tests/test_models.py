from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestNerveUnet:
    def test_forward_shape(self, default_config):
        from src.models import build_unet

        model = build_unet(default_config)
        x = torch.randn(2, 1, 384, 384)
        out = model(x)
        assert out.shape == (2, 1, 384, 384)

    def test_output_bounded(self):
        from src.models.unet import NerveUnet

        model = NerveUnet()
        x = torch.randn(1, 1, 192, 192)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_varying_depth(self):
        from src.models.unet import NerveUnet

        for depth in (3, 4):
            model = NerveUnet(depth=depth)
            out = model(torch.randn(1, 1, 384, 384))
            assert out.shape == (1, 1, 384, 384)

    def test_small_input(self):
        from src.models.unet import NerveUnet

        model = NerveUnet(depth=3)
        out = model(torch.randn(1, 1, 64, 64))
        assert out.shape == (1, 1, 64, 64)

    def test_build_unet_custom(self):
        from src.models import build_unet

        cfg = {"model": {"base_channels": 16, "depth": 3}}
        cfg.setdefault("data", {"channels": 1})
        model = build_unet(cfg)
        assert model.inc.block[0].out_channels == 16


class TestAMCLLoss:
    def test_loss_finite_and_positive(self):
        from src.models import AMCLLoss

        loss_fn = AMCLLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        from src.models import AMCLLoss

        loss_fn = AMCLLoss(alpha=1.0, beta=0.0)
        target = torch.zeros(2, 1, 64, 64)
        target[:, :, 16:48, 16:48] = 1.0
        # Large logits matching target
        pred = torch.where(target > 0, torch.tensor(5.0), torch.tensor(-5.0))
        loss = loss_fn(pred, target)
        assert loss.item() < 0.05

    def test_component_losses(self):
        from src.models import AMCLLoss

        loss_fn = AMCLLoss()
        pred = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        comp = loss_fn.component_losses(pred, target)
        assert "nerve" in comp and "topology" in comp
        assert torch.isfinite(comp["nerve"])
        assert torch.isfinite(comp["topology"])

    def test_dice_loss_perfect(self):
        from src.models import dice_loss

        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1.0
        pred = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
        assert dice_loss(pred, target).item() < 0.05


class TestMetrics:
    def test_perfect_iou_and_dice(self):
        from src.models import iou_score, dice_score, accuracy

        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1.0
        pred = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
        assert iou_score(pred, target).item() > 0.99
        assert dice_score(pred, target).item() > 0.99
        assert accuracy(pred, target).item() > 0.99

    def test_accumulator(self):
        from src.models import SegmentationMetrics

        m = SegmentationMetrics()
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1.0
        pred = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
        m.update(pred, target)
        res = m.results()
        assert res["iou"] > 0.99
        assert res["dice"] > 0.99


class TestSkeleton:
    def test_skeletonize_line(self):
        from src.utils.skeleton import skeletonize_binary, topological_features

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 10:40] = 1  # horizontal line
        skel = skeletonize_binary(mask)
        topo = topological_features(skel)
        assert topo["endpoints"] == 2
        assert topo["total_length"] > 20

    def test_skeletonize_empty(self):
        from src.utils.skeleton import topological_features

        mask = np.zeros((50, 50), dtype=np.uint8)
        topo = topological_features(mask)
        assert topo["n_junctions"] == 0
        assert topo["total_length"] == 0

    def test_branching_density(self):
        from src.utils.skeleton import branching_density

        mask = np.zeros((50, 50), dtype=np.uint8)
        skel = np.zeros((50, 50), dtype=np.uint8)
        # star shape with a junction
        skel[25, 25] = 1
        skel[25, 20:31] = 1
        skel[20:31, 25] = 1
        density = branching_density(skel, total_area=2500.0)
        assert density >= 0.0


class TestTrainingUtils:
    def test_get_device(self):
        from src.utils.training import get_device

        assert isinstance(get_device(), torch.device)

    def test_set_seed_reproducible(self):
        from src.utils.training import set_seed

        set_seed(7)
        a = torch.randn(3)
        set_seed(7)
        b = torch.randn(3)
        assert torch.equal(a, b)

    def test_checkpoint_roundtrip(self, tmp_path):
        from src.utils.training import save_checkpoint, load_checkpoint

        path = tmp_path / "sub" / "ckpt.pt"
        save_checkpoint({"a": 1, "b": [1, 2, 3]}, path)
        assert path.exists()
        state = load_checkpoint(path)
        assert state["a"] == 1
        assert state["b"] == [1, 2, 3]
