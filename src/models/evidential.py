"""Phase 4: Evidential deep learning for calibrated severity prediction.

Implements the Dirichlet-prior evidential classification framework
(Sensoy et al., NeurIPS 2018). Instead of a softmax over class logits the
head outputs non-negative *evidence* per class. A Dirichlet distribution
over class probabilities is formed with concentration parameters
``alpha_k = evidence_k + 1``, giving two useful quantities:

    p_k = alpha_k / S          known-probability mass (class confidence)
    u   = K / S                vacuous uncertainty, S = sum(alpha)

``u`` approaches 1 when the network has no supporting evidence (the model
is "unsure"), unlike a softmax which always asserts a confident output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .severity import SeverityNet


def kl_dirichlet(
    alpha: torch.Tensor, beta: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Kullback-Leibler divergence between two Dirichlet distributions.

    ``KL(Dir(alpha) || Dir(beta))`` in closed form (per sample, shape B).
    """
    alpha0 = alpha.sum(dim=-1)
    beta0 = beta.sum(dim=-1)
    lgamma_a0 = torch.lgamma(alpha0)
    lgamma_a = torch.lgamma(alpha).sum(dim=-1)
    lgamma_b0 = torch.lgamma(beta0)
    lgamma_b = torch.lgamma(beta).sum(dim=-1)
    cross = (
        (alpha - beta) * (torch.digamma(alpha) - torch.digamma(beta0).unsqueeze(-1))
    ).sum(dim=-1)
    return lgamma_a0 - lgamma_a - lgamma_b0 + lgamma_b + cross + eps


def annealing_schedule(epoch: int | None, anneal_epochs: int = 5) -> float:
    """Linear warm-up of the KL regularizer over the first ``anneal_epochs``."""
    if epoch is None:
        return 1.0
    return min(1.0, max(epoch, 1) / max(anneal_epochs, 1))


@torch.no_grad()
def evidential_predict(
    evidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Turn evidential head outputs into a full set of predictions.

    Args:
        evidence: (B, K) non-negative per-class evidence.

    Returns dict with:
        alpha (B,K), alpha0 (B,1), probs (B,K), uncertainty (B,),
        confidence (B,) [max prob], pred (B,) [argmax class].
    """
    alpha = evidence + 1.0
    alpha0 = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / alpha0
    uncertainty = evidence.shape[-1] / alpha0.squeeze(-1)
    return {
        "alpha": alpha,
        "alpha0": alpha0,
        "probs": probs,
        "uncertainty": uncertainty,
        "confidence": probs.max(dim=-1).values,
        "pred": probs.argmax(dim=-1),
        "evidence": evidence,
    }


class EvidentialLoss(nn.Module):
    """Type-II maximum-likelihood loss with annealed Dirichlet regularization.

    ``L = E[(y - p)^2] + var(p)`` reduces to a sum of digamma terms
    (Sensoy et al.); equivalently the negative log marginal likelihood
    ``sum_k y_k (psi(S) - psi(alpha_k))``.  We add the KL regularizer
    ``lambda * KL(Dir(alpha_tilde) || Dir(1))`` that pushes misclassified
    evidence toward zero (raising uncertainty on mistakes).
    """

    def __init__(
        self,
        kl_weight: float = 1.0,
        kl_anneal_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.kl_anneal_epochs = int(kl_anneal_epochs)

    def forward(
        self,
        evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int | None = None,
    ) -> torch.Tensor:
        alpha = evidence + 1.0
        alpha0 = alpha.sum(dim=-1)
        y = F.one_hot(target.long(), num_classes=evidence.shape[-1]).float()

        digamma = torch.digamma(alpha)
        mle = (y * (torch.digamma(alpha0.unsqueeze(-1)) - digamma)).sum(dim=-1)

        alpha_tilde = y + (1.0 - y) * alpha
        kl = kl_dirichlet(alpha_tilde, torch.ones_like(alpha))

        scale = annealing_schedule(epoch, self.kl_anneal_epochs)
        loss = mle + self.kl_weight * scale * kl
        return loss.mean()


class EvidentialSeverityNet(SeverityNet):
    """4-class severity classifier with a Dirichlet evidential head.

    Shares the exact encoder/head architecture of ``SeverityNet`` but
    returns per-class evidence (softplus of the head logits) instead of
    raw logits, enabling uncertainty-aware trust computations.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(super().forward(x))


def build_evidential_net(cfg: dict | None = None) -> EvidentialSeverityNet:
    """Build an EvidentialSeverityNet from config dict (or defaults)."""
    if cfg is None:
        cfg = {}
    sev_cfg = cfg.get("severity", {})
    return EvidentialSeverityNet(
        in_channels=cfg.get("data", {}).get("channels", 1),
        num_classes=sev_cfg.get("num_classes", 4),
        base_channels=sev_cfg.get("base_channels", 32),
        depth=sev_cfg.get("depth", 4),
        dropout=sev_cfg.get("dropout", 0.2),
    )