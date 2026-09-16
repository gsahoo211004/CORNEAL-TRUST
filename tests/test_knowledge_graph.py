from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _horizontal_line_50():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 255
    return mask


def _cross_50():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 255
    mask[10:40, 25] = 255
    return mask


def _zigzag_50():
    mask = np.zeros((50, 50), dtype=np.uint8)
    y = 25.0
    for x in range(5, 45):
        ty = 25 + 10 * np.sin(x / 3.0)
        steps = max(1, int(round(abs(ty - y))))
        for s in range(1, steps + 1):
            yy = int(round(y + (ty - y) * s / steps))
            mask[yy, x] = 255
        y = ty
    return mask


class TestClinicalFeatures:
    def test_line_endpoints(self):
        from src.knowledge_graph.features import compute_clinical_features
        features = compute_clinical_features(_horizontal_line_50(), image_area=2500.0)
        assert features["endpoints"] == pytest.approx(2.0)
        assert features["n_junctions"] == pytest.approx(0.0)
        assert features["fiber_density"] > 0.0

    def test_cross_has_junction(self):
        from src.knowledge_graph.features import compute_clinical_features
        features = compute_clinical_features(_cross_50(), image_area=2500.0)
        assert features["n_junctions"] >= 1.0
        assert features["endpoints"] >= 2.0

    def test_empty_mask(self):
        from src.knowledge_graph.features import compute_clinical_features
        features = compute_clinical_features(np.zeros((50, 50), dtype=np.uint8))
        assert features["n_junctions"] == 0.0
        assert features["fiber_density"] == 0.0

    def test_zigzag_higher_tortuosity(self):
        from src.knowledge_graph.features import tortuosity_index
        from src.utils.skeleton import skeletonize_binary
        line = skeletonize_binary(_horizontal_line_50())
        zigzag = skeletonize_binary(_zigzag_50())
        t_line = tortuosity_index(line)
        t_zig = tortuosity_index(zigzag)
        assert t_zig > t_line + 0.1


class TestFiberGraph:
    def test_nodes_and_edges(self):
        from src.knowledge_graph.graph import build_fiber_graph
        from src.utils.skeleton import skeletonize_binary
        graph = build_fiber_graph(skeletonize_binary(_cross_50()))
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert "features" in graph.graph

    def test_empty_skeleton(self):
        from src.knowledge_graph.graph import build_fiber_graph
        graph = build_fiber_graph(np.zeros((50, 50), dtype=np.uint8))
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_line_two_nodes(self):
        from src.knowledge_graph.graph import build_fiber_graph
        from src.utils.skeleton import skeletonize_binary
        graph = build_fiber_graph(skeletonize_binary(_horizontal_line_50()))
        assert graph.number_of_nodes() == 2


class TestCornealKnowledgeGraph:
    def test_from_mask_stats(self):
        from src.knowledge_graph.graph import CornealKnowledgeGraph
        kg = CornealKnowledgeGraph.from_mask(_horizontal_line_50())
        stats = kg.stats()
        assert stats["n_nodes"] == pytest.approx(2.0)
        assert stats["mean_tortuosity"] >= 1.0

    def test_from_features_empty_graph(self):
        from src.knowledge_graph.graph import CornealKnowledgeGraph
        kg = CornealKnowledgeGraph.from_features({"fiber_density": 0.1, "tortuosity_index": 1.2})
        stats = kg.stats()
        assert stats["fiber_density"] == pytest.approx(0.1, abs=1e-6)
        assert stats["tortuosity_index"] == pytest.approx(1.2, abs=1e-6)


class TestTriageEngine:
    def test_baseline_risk_mapping(self):
        from src.knowledge_graph.rules import TriageEngine
        eng = TriageEngine()
        assert eng.triage(1)["risk_level"] == "Low"
        assert eng.triage(2)["risk_level"] == "Medium"
        assert eng.triage(3)["risk_level"] == "High"
        assert eng.triage(4)["risk_level"] == "High"

    def test_structural_escalation(self):
        from src.knowledge_graph.rules import TriageEngine
        eng = TriageEngine()
        features = {
            "fiber_density": 0.01,
            "tortuosity_index": 2.0,
            "branching_density": 0.001,
        }
        fired = eng.structural_evidence(features)
        assert "severe_fiber_loss" in fired
        assert "high_tortuosity" in fired
        assert "severe_branching_sparsity" in fired

    def test_escalation_increases_risk(self):
        from src.knowledge_graph.rules import TriageEngine
        eng = TriageEngine()
        features = {"fiber_density": 0.01, "tortuosity_index": 2.0, "branching_density": 0.001}
        triage = eng.triage(severity_grade=1, kg=features)
        assert triage["risk_level"] == "High"

    def test_low_structural_keeps_baseline(self):
        from src.knowledge_graph.rules import TriageEngine
        eng = TriageEngine()
        features = {"fiber_density": 0.2, "tortuosity_index": 1.0, "branching_density": 0.01}
        triage = eng.triage(severity_grade=1, kg=features)
        assert triage["risk_level"] == "Low"
        assert triage["structural_rules_fired"] == []

    def test_output_keys(self):
        from src.knowledge_graph.rules import TriageEngine
        eng = TriageEngine()
        result = eng.triage(severity_grade=2, kg={})
        expected_keys = {
            "risk_level", "severity_grade", "structural_rules_fired",
            "fiber_density", "tortuosity_index", "branching_density",
            "n_junctions", "referral_window_days", "referral_text",
            "evidence_path", "graph_stats",
        }
        assert set(result.keys()) == expected_keys

    def test_with_knowledge_graph_object(self):
        from src.knowledge_graph.rules import TriageEngine
        from src.knowledge_graph.graph import CornealKnowledgeGraph
        kg = CornealKnowledgeGraph.from_features({
            "fiber_density": 0.01,
            "tortuosity_index": 2.0,
            "branching_density": 0.001,
            "n_junctions": 0.0,
        })
        eng = TriageEngine()
        result = eng.triage(severity_grade=1, kg=kg)
        assert result["risk_level"] == "High"
        assert isinstance(result["graph_stats"], dict)

    def test_load_triage_engine(self):
        from src.knowledge_graph.rules import TriageEngine, load_triage_engine
        from src.utils.config import load_config
        cfg = load_config()
        eng = load_triage_engine(cfg)
        assert isinstance(eng, TriageEngine)
        assert eng.triage(1)["risk_level"] == "Low"