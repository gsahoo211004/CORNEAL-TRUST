"""Phase 4: Corneal Diagnostic Trust Index (CDTI).

CDTI is a single scalar in [0, 1] describing how much clinical trust to
place in a severity prediction for a single image. It combines three
factors in product form (any weak factor pulls the index down, which is
conservative and appropriate for medical referral decisions):

    CDTI = Q_image^w_q  *  Conf^w_c  *  EpiConf^w_e

    Q_image   : image-quality confidence from the CORN-2 quality model
                (prob of high quality; 1.0 = crisp sub-basal image)
    Conf      : predictive confidence, max class prob (known probability)
    EpiConf   : epistemic confidence = 1 - vacuous uncertainty u

When CDTI < ``referral_threshold`` (default 0.60) the prediction is
declared **LowTrust** and escalated to a clinician/human review even if
the predicted grade is low, guarding against acting on uncertain or
unusable images.
"""

from __future__ import annotations

from typing import Any

import torch


def compute_cdti(
    confidence: float | torch.Tensor,
    epi_confidence: float | torch.Tensor,
    q_image: float | torch.Tensor = 1.0,
    q_weight: float = 1.0,
    conf_weight: float = 1.0,
    epi_weight: float = 1.0,
) -> torch.Tensor:
    """Compute CDTI in product form, clipped to [0, 1].

    Args:
        confidence: max class probability in [0, 1].
        epi_confidence: 1 - vacuous uncertainty in [0, 1].
        q_image: image-quality confidence in [0, 1] (default 1.0 when no
            quality model / a perfect image is assumed).
        q_weight, conf_weight, epi_weight: optional per-factor exponents.
    """
    conf = torch.clamp(
        torch.as_tensor(confidence, dtype=torch.float32), 0.0, 1.0
    )
    epi = torch.clamp(
        torch.as_tensor(epi_confidence, dtype=torch.float32), 0.0, 1.0
    )
    q = torch.clamp(torch.as_tensor(q_image, dtype=torch.float32), 0.0, 1.0)
    cdti = torch.pow(q, q_weight) * torch.pow(conf, conf_weight) * torch.pow(epi, epi_weight)
    return torch.clamp(cdti, 0.0, 1.0)


def referral_verdict(
    cdti: float | torch.Tensor,
    threshold: float = 0.60,
) -> dict[str, Any]:
    """Classify a CDTI value against the referral threshold.

    Returns dict with cdti, threshold, trusted (bool) and verdict str
    ('Trusted' when cdti >= threshold, 'LowTrust' otherwise).
    """
    val = float(torch.as_tensor(cdti).item())
    trusted = val >= float(threshold)
    return {
        "cdti": val,
        "threshold": float(threshold),
        "trusted": trusted,
        "verdict": "Trusted" if trusted else "LowTrust",
    }


def trust_report(
    confidence: float | torch.Tensor,
    uncertainty: float | torch.Tensor,
    q_image: float | torch.Tensor = 1.0,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Full trust report: CDTI + verdict for a (conf, u, q) triple.

    ``cfg`` may carry a ``trust:`` section with q/conf/epi weights and the
    referral threshold.
    """
    cfg = cfg or {}
    trust_cfg = cfg.get("trust", {})
    epi_confidence = 1.0 - float(torch.as_tensor(uncertainty).item())
    q_weight = float(trust_cfg.get("q_weight", 1.0))
    conf_weight = float(trust_cfg.get("conf_weight", 1.0))
    epi_weight = float(trust_cfg.get("epi_weight", 1.0))
    threshold = float(trust_cfg.get("referral_threshold", 0.60))

    cdti = compute_cdti(
        confidence=confidence,
        epi_confidence=epi_confidence,
        q_image=q_image,
        q_weight=q_weight,
        conf_weight=conf_weight,
        epi_weight=epi_weight,
    )
    verdict = referral_verdict(cdti, threshold=threshold)
    verdict["epi_confidence"] = float(min(max(epi_confidence, 0.0), 1.0))
    verdict["q_image"] = float(torch.as_tensor(q_image).item())
    verdict["confidence"] = float(torch.as_tensor(confidence).item())
    return verdict