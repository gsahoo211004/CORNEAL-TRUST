from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestCheckpointResume:
    """The train scripts persist optimizer+scheduler state; resume continues
    at state['epoch'] + 1 with the stored best metric."""

    def test_roundtrip_preserves_state(self, tmp_path):
        from src.utils.training import save_checkpoint, load_checkpoint

        path = tmp_path / "ckpt.pt"
        state = {
            "epoch": 7,
            "model_state_dict": {"a": torch.ones(2)},
            "optimizer_state_dict": {"b": 1},
            "scheduler_state_dict": {"last_epoch": 41},
            "best_acc": 0.73,
        }
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)
        assert loaded["epoch"] == 7
        assert loaded["best_acc"] == pytest.approx(0.73)
        assert set(state.keys()) <= set(loaded.keys())

    def test_resume_epoch_convention(self, tmp_path):
        from src.utils.training import save_checkpoint, load_checkpoint

        state = {"epoch": 3, "optimizer_state_dict": {}, "best_acc": 0.5}
        save_checkpoint(state, tmp_path / "x.pt")
        loaded = load_checkpoint(tmp_path / "x.pt")
        # scripts do start_epoch = state['epoch'] + 1
        assert int(loaded["epoch"]) + 1 == 4

    def test_legacy_checkpoint_missing_scheduler_is_graceful(self, tmp_path):
        from src.utils.training import save_checkpoint, load_checkpoint

        save_checkpoint({"epoch": 1, "optimizer_state_dict": {}, "best_acc": 0.0}, tmp_path / "old.pt")
        loaded = load_checkpoint(tmp_path / "old.pt")
        assert loaded.get("scheduler_state_dict") is None


class TestEceScore:
    def test_perfectly_calibrated_zero(self):
        from src.models import ece_score

        # all predictions perfect at confidence 1.0 -> no miscalibration
        probs = torch.eye(3)
        target = torch.tensor([0, 1, 2])
        assert ece_score(probs, target, n_bins=10) == pytest.approx(0.0, abs=1e-4)

    def test_overconfident_high_ece(self):
        from src.models import ece_score

        probs = torch.tensor([[0.99] * 4] + [[0.99, 0.005, 0.005, 0.0]] * 9)
        target = torch.tensor([3] + [2] * 9)  # always wrong, confident
        ece = ece_score(probs, target, n_bins=10)
        assert ece > 0.5

    def test_returns_float(self):
        from src.models import ece_score

        assert isinstance(ece_score(torch.eye(4), torch.tensor([0, 1, 2, 3])), float)