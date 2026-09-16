from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestSeverityNet:
    def test_forward_shape_default(self, default_config):
        from src.models import build_severity_net

        model = build_severity_net(default_config)
        x = torch.randn(2, 1, 384, 384)
        out = model(x)
        assert out.shape == (2, default_config["severity"]["num_classes"])

    def test_forward_custom_dims(self):
        from src.models.severity import SeverityNet

        model = SeverityNet(in_channels=1, num_classes=4, base_channels=16, depth=3)
        out = model(torch.randn(1, 1, 192, 192))
        assert out.shape == (1, 4)
        assert torch.isfinite(out).all()

    def test_small_input(self):
        from src.models.severity import SeverityNet

        model = SeverityNet(num_classes=4, depth=3)
        out = model(torch.randn(1, 1, 64, 64))
        assert out.shape == (1, 4)

    def test_crossentropy_loss_drops(self):
        from src.models.severity import SeverityNet

        torch.manual_seed(0)
        model = SeverityNet(num_classes=4, depth=3)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        crit = torch.nn.CrossEntropyLoss()
        x = torch.randn(8, 1, 64, 64)
        y = torch.randint(0, 4, (8,))
        opt.zero_grad()
        loss0 = crit(model(x), y)
        for _ in range(5):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        loss1 = crit(model(x), y)
        assert loss1.item() < loss0.item() + 1e-6


class TestClassificationMetrics:
    def test_accuracy_and_f1(self):
        from src.models import ClassificationMetrics

        m = ClassificationMetrics(num_classes=4)
        targets = torch.tensor([0, 0, 1, 2])
        logits = torch.zeros(4, 4)
        logits[torch.arange(4), targets] = 10.0
        m.update(logits, targets)
        res = m.results()
        assert res["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert res["macro_f1"] == pytest.approx(1.0, abs=1e-3)

    def test_perfect_module_import(self):
        from src.models import accuracy_score, macro_f1_score

        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.zeros(4, 4)
        logits[torch.arange(4), targets] = 9.0
        assert accuracy_score(logits, targets).item() == pytest.approx(1.0)
        assert macro_f1_score(logits, targets, 4).item() == pytest.approx(1.0, abs=1e-3)