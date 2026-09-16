from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestComputeCdti:
    def test_bounds(self):
        from src.models import compute_cdti

        cdti = compute_cdti(confidence=0.9, epi_confidence=0.8, q_image=0.7)
        val = float(cdti.item())
        assert 0.0 <= val <= 1.0

    def test_product_form_reduces_with_weak_factor(self):
        from src.models import compute_cdti

        strong = compute_cdti(confidence=0.9, epi_confidence=0.9, q_image=1.0)
        weak_q = compute_cdti(confidence=0.9, epi_confidence=0.9, q_image=0.1)
        weak_conf = compute_cdti(confidence=0.1, epi_confidence=0.9, q_image=1.0)
        assert weak_q.item() < strong.item()
        assert weak_conf.item() < strong.item()

    def test_identity_when_all_factors_one(self):
        from src.models import compute_cdti

        cdti = compute_cdti(confidence=1.0, epi_confidence=1.0, q_image=1.0)
        assert cdti.item() == pytest.approx(1.0, abs=1e-6)

    def test_weights_change_result(self):
        from src.models import compute_cdti

        plain = compute_cdti(confidence=0.5, epi_confidence=0.5, q_image=1.0)
        weighted = compute_cdti(
            confidence=0.5, epi_confidence=0.5, q_image=1.0,
            q_weight=1.0, conf_weight=0.0, epi_weight=0.0,
        )
        # conf^0 * epi^0 * q^1 = 1.0
        assert weighted.item() == pytest.approx(1.0, abs=1e-6)
        assert weighted.item() != pytest.approx(plain.item())


class TestReferralVerdict:
    def test_threshold_flip(self):
        from src.models import referral_verdict

        below = referral_verdict(0.59, threshold=0.60)
        above = referral_verdict(0.61, threshold=0.60)
        exactly = referral_verdict(0.60, threshold=0.60)
        assert below["trusted"] is False
        assert below["verdict"] == "LowTrust"
        assert above["trusted"] is True
        assert above["verdict"] == "Trusted"
        assert exactly["trusted"] is True


class TestTrustReport:
    def test_defaults(self, default_config):
        from src.models import trust_report

        report = trust_report(confidence=0.9, uncertainty=0.2, q_image=1.0, cfg=default_config)
        assert report["threshold"] == pytest.approx(default_config["trust"]["referral_threshold"])
        assert report["epi_confidence"] == pytest.approx(0.8)
        assert report["cdti"] == pytest.approx(0.9 * 0.8, abs=1e-5)
        assert report["verdict"] == "Trusted"

    def test_low_trust_when_confident_but_uncertain(self, default_config):
        from src.models import trust_report

        # high class confidence but big vacuous uncertainty still tanks CDTI
        report = trust_report(confidence=0.9, uncertainty=0.9, q_image=1.0, cfg=default_config)
        assert report["cdti"] <= 0.1
        assert report["verdict"] == "LowTrust"
        assert report["trusted"] is False

    def test_scalar_tensor_input(self, default_config):
        from src.models import trust_report

        report = trust_report(
            confidence=torch.tensor(0.8), uncertainty=torch.tensor(0.1), q_image=1.0, cfg=default_config
        )
        assert report["epi_confidence"] == pytest.approx(0.9)
        assert 0.0 <= report["cdti"] <= 1.0