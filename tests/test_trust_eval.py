from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_accuracy_macro_f1_perfect():
    from scripts.evaluate_trust import accuracy_macro_f1

    true = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3])
    acc, f1 = accuracy_macro_f1(pred, true, 4)
    assert acc == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0, abs=1e-3)


def test_accuracy_macro_f1_randomish():
    from scripts.evaluate_trust import accuracy_macro_f1

    rng = np.random.default_rng(0)
    true = rng.integers(0, 4, 200)
    pred = rng.integers(0, 4, 200)
    acc, f1 = accuracy_macro_f1(pred, true, 4)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_confusion_matrix_orientation():
    from scripts.evaluate_trust import confusion_matrix

    true = np.array([0, 1, 1])
    pred = np.array([0, 1, 0])
    cm = confusion_matrix(pred, true, 2)
    assert cm.shape == (2, 2)
    assert cm[1, 0] == 1  # truth=1, pred=0
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1


def test_coverage_curve_bounds():
    from scripts.evaluate_trust import coverage_curve

    rows = [
        {"cdti": 0.9, "pred": 1, "true": 1},
        {"cdti": 0.5, "pred": 1, "true": 0},
        {"cdti": 0.1, "pred": 0, "true": 0},
    ]
    curve = coverage_curve(rows, n=11)
    assert curve[0]["threshold"] == 0.0
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[-1]["threshold"] == 1.0
    assert curve[-1]["coverage"] == 0.0
    # accuracy on accepted never exceeds 100%
    assert all(c["accuracy"] <= 1.0 for c in curve)


def test_coverage_curve_respects_threshold():
    from scripts.evaluate_trust import coverage_curve

    rows = [{"cdti": 0.8, "pred": 1, "true": 1}] * 2 + [{"cdti": 0.2, "pred": 0, "true": 1}] * 2
    curve = coverage_curve(rows, n=6)
    at080 = next(c for c in curve if c["threshold"] == 0.8)
    assert at080["coverage"] == pytest.approx(0.5)
    assert at080["accuracy"] == pytest.approx(1.0)


class TestCorn1500LabelCounts:
    def test_counts_sum_to_len(self, tmp_path):
        from src.data.corn1500 import Corn1500SeverityDataset

        root = tmp_path / "corn1500"
        for i, level in enumerate(["level1", "level2", "level3", "level4"], start=1):
            d = root / level
            d.mkdir(parents=True)
            for n in range(1, i + 2):
                (d / f"{n}_{level}.jpg").touch()

        ds = Corn1500SeverityDataset(root=root, image_size=64)
        counts = ds.label_counts()
        assert sum(counts) == len(ds)
        assert counts[0] == 2 and counts[1] == 3 and counts[2] == 4 and counts[3] == 5