from __future__ import annotations

from typing import Any

from .graph import CornealKnowledgeGraph

RISK_ORDER = ("Low", "Medium", "High")


class TriageEngine:
    """Deterministic rule engine producing clinical triage recommendations.

    Combines the 4-class severity model output with structural features from
    the Corneal Knowledge Graph. Each structural rule can escalate the
    baseline risk until it cannot be increased further:

    Baseline mapping (severity grade -> risk):
        grade 1 (level1) -> Low | grade 2 -> Medium | grade 3/4 -> High

    Structural rules (evidence-fired):
        - reduced_fiber_density : fiber_density < low threshold
        - severe_fiber_loss     : fiber_density < very_low threshold
        - high_tortuosity       : mean tortuosity > high threshold
        - moderate_tortuosity   : mean tortuosity > medium threshold
        - sparse_branching      : branching_density < low threshold
        - severe_sparsity       : branching_density < very_low threshold
    """

    def __init__(self, kg_cfg: dict[str, Any] | None = None) -> None:
        kg_cfg = kg_cfg or {}
        fd = kg_cfg.get("fiber_density_fraction", {})
        tor = kg_cfg.get("tortuosity", {})
        br = kg_cfg.get("branching_per_fiber", {})
        self.cfg = {
            "fiber_low": float(fd.get("low", 0.05)),
            "fiber_very_low": float(fd.get("very_low", 0.02)),
            "tort_medium": float(tor.get("medium", 1.15)),
            "tort_high": float(tor.get("high", 1.5)),
            "branch_low": float(br.get("low", 0.004)),
            "branch_very_low": float(br.get("very_low", 0.002)),
        }
        self.windows = {
            "Low": int(kg_cfg.get("referral_window_days", {}).get("low", 90)),
            "Medium": int(kg_cfg.get("referral_window_days", {}).get("medium", 14)),
            "High": int(kg_cfg.get("referral_window_days", {}).get("high", 7)),
        }

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------
    def _baseline_risk(self, severity_grade: int) -> str:
        grade = int(severity_grade)
        if grade >= 3:
            return "High"
        if grade == 2:
            return "Medium"
        return "Low"

    def structural_evidence(self, features: dict[str, float]) -> list[str]:
        """Return the names of fired structural rules.

        Rules are only evaluated when the underlying feature is present in
        the features dict, so an empty/missing feature set never triggers
        false escalations.
        """
        fired: list[str] = []

        if "fiber_density" in features:
            fd = float(features.get("fiber_density", 0.0))
            if fd < self.cfg["fiber_very_low"]:
                fired.append("severe_fiber_loss")
            elif fd < self.cfg["fiber_low"]:
                fired.append("reduced_fiber_density")

        if "tortuosity_index" in features:
            tort = float(features.get("tortuosity_index", 1.0))
            if tort > self.cfg["tort_high"]:
                fired.append("high_tortuosity")
            elif tort > self.cfg["tort_medium"]:
                fired.append("moderate_tortuosity")

        if "branching_density" in features:
            bd = float(features.get("branching_density", 0.0))
            if bd < self.cfg["branch_very_low"]:
                fired.append("severe_branching_sparsity")
            elif bd < self.cfg["branch_low"]:
                fired.append("sparse_branching")

        return fired

    def _escalate(self, risk: str, fired: list[str]) -> str:
        """Escalate risk once per fired structural rule (Low->Medium->High)."""
        idx = RISK_ORDER.index(risk)
        for _ in fired:
            idx = min(idx + 1, len(RISK_ORDER) - 1)
        return RISK_ORDER[idx]

    # ------------------------------------------------------------------
    # Public triage API
    # ------------------------------------------------------------------
    def triage(
        self,
        severity_grade: int,
        kg: CornealKnowledgeGraph | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Produce a full clinical triage recommendation.

        Args:
            severity_grade: model severity grade, 1-based (1..4).
            kg: CornealKnowledgeGraph instance, or a features dict.
        """
        if isinstance(kg, CornealKnowledgeGraph):
            features = kg.features
            stats = kg.stats()
        elif isinstance(kg, dict):
            features = dict(kg)
            stats = dict(kg)
        else:
            features = {}
            stats = {}

        risk = self._baseline_risk(severity_grade)
        fired = self.structural_evidence(features)
        risk = self._escalate(risk, fired)

        referral = self._referral_text(risk, severity_grade, fired, features)

        return {
            "risk_level": risk,
            "severity_grade": int(severity_grade),
            "structural_rules_fired": fired,
            "fiber_density": float(features.get("fiber_density", 0.0)),
            "tortuosity_index": float(features.get("tortuosity_index", 1.0)),
            "branching_density": float(features.get("branching_density", 0.0)),
            "n_junctions": float(features.get("n_junctions", 0.0)),
            "referral_window_days": self.windows[risk],
            "referral_text": referral,
            "evidence_path": self._evidence_path(risk, fired, features),
            "graph_stats": stats,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _evidence_path(
        risk: str, fired: list[str], features: dict[str, float]
    ) -> dict[str, Any]:
        labels = {
            "severe_fiber_loss": "Sub-basal density severely reduced",
            "reduced_fiber_density": "Sub-basal density reduction",
            "high_tortuosity": "High fibre tortuosity",
            "moderate_tortuosity": "Moderate fibre tortuosity",
            "severe_branching_sparsity": "Severe branching loss",
            "sparse_branching": "Reduced branching junctions",
        }
        evidence = [labels.get(r, r) for r in fired]
        return {
            "structural_evidence": evidence,
            "pathological_progression": f"{risk}-risk DPN progression",
            "recommended_triage": risk,
        }

    @staticmethod
    def _referral_text(
        risk: str, severity_grade: int, fired: list[str], features: dict[str, float]
    ) -> str:
        if risk == "Low":
            return (
                "Sub-basal nerve morphology within normal limits. Routine "
                "diabetic review recommended; no urgent specialist referral."
            )
        if risk == "Medium":
            parts = ["Referral to endocrinologist and podiatrist within 14 days."]
            if fired:
                parts.insert(0, "Follow-up recommended due to " + ", ".join(fired) + ".")
            return " ".join(parts)
        parts = [
            f"Immediate referral to endocrinologist and podiatrist within 7 days. "
            f"(severity grade {severity_grade}, " + ", ".join(fired) + ")."
        ]
        return " ".join(parts) if fired else (
            f"Immediate referral to endocrinologist and podiatrist within 7 days "
            f"(severity grade {severity_grade})."
        )


def load_triage_engine(cfg: dict[str, Any] | None = None) -> TriageEngine:
    """Build a TriageEngine from config dict (or en empty config -> defaults)."""
    if cfg is None:
        from ..utils.config import load_config

        cfg = load_config().get("knowledge_graph", {})
    return TriageEngine(cfg)