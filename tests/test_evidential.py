from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestEvidentialPredict:
    def test_dirichlet_properties(self):
        from src.models import evidential_predict

        evidence = torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
        out = evidential_predict(evidence)
        assert (out["alpha"] >= 1.0).all()
        # probs sum to 1
        assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(2), atol=1e-5)
        # uncertainty in [0, 1]
        assert (out["uncertainty"] >= 0.0).all()
        assert (out["uncertainty"] <= 1.0).all()
        # zero evidence is maximally vacuous; strong evidence is confident
        u_zero = out["uncertainty"][0].item()
        u_strong = out["uncertainty"][1].item()
        assert u_zero == pytest.approx(1.0, abs=1e-4)
        assert u_strong < u_zero

    def test_predicted_class_highest_prob(self):
        from src.models import evidential_predict

        evidence = torch.tensor([[0.2, 9.0, 0.1, 0.0]])
        out = evidential_predict(evidence)
        assert out["pred"].item() == 1
        assert out["confidence"].item() == pytest.approx(out["probs"][0, 1].item())


class TestEvidentialNet:
    def test_forward_nonnegative(self, default_config):
        from src.models import build_evidential_net

        model = build_evidential_net(default_config)
        evidence = model(torch.randn(2, 1, 64, 64))
        assert evidence.shape == (2, default_config["severity"]["num_classes"])
        assert (evidence > 0).all()
        assert torch.isfinite(evidence).all()

    def test_matches_severity_params_count(self, default_config):
        from src.models import build_evidential_net, build_severity_net

        a = build_evidential_net(default_config)
        b = build_severity_net(default_config)
        assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


class TestKlDirichlet:
    def test_uniform_divergence_zero(self):
        from src.models import kl_dirichlet

        alpha = torch.ones(3, 4)
        kl = kl_dirichlet(alpha, torch.ones(3, 4))
        assert torch.allclose(kl, torch.zeros(3), atol=1e-4)

    def test_nonneg(self):
        from src.models import kl_dirichlet

        alpha = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
        kl = kl_dirichlet(alpha, torch.ones(1, 4))
        assert kl.item() >= 0.0


class TestEvidentialLoss:
    def test_gradient_flow(self):
        from src.models import EvidentialLoss, EvidentialSeverityNet

        torch.manual_seed(0)
        model = EvidentialSeverityNet(num_classes=4, depth=3)
        loss_fn = EvidentialLoss(kl_weight=1.0, kl_anneal_epochs=5)
        x = torch.randn(8, 1, 64, 64)
        y = torch.randint(0, 4, (8,))
        loss0 = loss_fn(model(x), y, epoch=5)
        loss0.backward()
        assert all(p.grad is not None for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(5):
            opt.zero_grad()
            loss = loss_fn(model(x), y, epoch=5)
            loss.backward()
            opt.step()
        assert loss_fn(model(x), y, epoch=5).item() < loss0.item() + 1e-6

    def test_annealing_schedule(self):
        from src.models import annealing_schedule

        assert annealing_schedule(None, 5) == 1.0
        assert annealing_schedule(1, 5) == pytest.approx(0.2)
        assert annealing_schedule(5, 5) == pytest.approx(1.0)
        assert annealing_schedule(20, 5) == pytest.approx(1.0)

    def test_kl_regularizer_adds_penalty(self):
        """For the same evidence, higher kl_weight must raise the loss value,
        proving the KL term actually penalises deviation from the Dir(1) prior."""
        from src.models import EvidentialSeverityNet, EvidentialLoss, evidential_predict

        torch.manual_seed(42)
        model = EvidentialSeverityNet(num_classes=4, depth=3)
        x = torch.randn(8, 1, 64, 64)
        y = torch.tensor([0] * 8)
        evidence = model(x)
        loss_k0 = EvidentialLoss(kl_weight=0.0, kl_anneal_epochs=5)(evidence, y, epoch=5)
        loss_k1 = EvidentialLoss(kl_weight=1.0, kl_anneal_epochs=5)(evidence, y, epoch=5)
        loss_k10 = EvidentialLoss(kl_weight=10.0, kl_anneal_epochs=5)(evidence, y, epoch=5)
        assert loss_k1.item() > loss_k0.item()
        assert loss_k10.item() > loss_k1.item()


class TestQualityNet:
    def test_two_class_output(self, default_config):
        from src.models import build_quality_net

        model = build_quality_net(default_config)
        out = model(torch.randn(2, 1, 64, 64))
        assert out.shape == (2, default_config["quality"]["num_classes"])