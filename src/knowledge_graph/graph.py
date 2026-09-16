from __future__ import annotations

import numpy as np

import networkx as nx

from .features import compute_clinical_features, fiber_segments, _path_length


def build_fiber_graph(
    skeleton: np.ndarray,
    features: dict[str, float] | None = None,
) -> nx.Graph:
    """Build a Corneal Knowledge Graph (fiber-connectivity graph) from a
    binary nerve skeleton.

    - Nodes: junctions and endpoints (keypoints) with ``node_type`` and ``pos``.
    - Edges: fiber segments between keypoints with ``length_px`` and
      ``tortuosity`` attributes.
    - Graph-level attributes: aggregated morphological features.
    """
    skeleton = (skeleton > 0).astype(np.uint8)
    graph = nx.Graph()
    graph.graph["features"] = dict(features or compute_clinical_features(skeleton))

    segments = fiber_segments(skeleton)
    keypoints: set[tuple[int, int]] = set()
    for seg in segments:
        if len(seg) >= 2:
            keypoints.add(seg[0])
            keypoints.add(seg[-1])

    for node in sorted(keypoints):
        graph.add_node(node, node_type="junction_or_endpoint", pos=node)

    for seg in segments:
        if len(seg) < 2:
            continue
        u, v = seg[0], seg[-1]
        path_len = _path_length(seg)
        x1, y1 = u
        x2, y2 = v
        straight = float(np.hypot(x2 - x1, y2 - y1))
        tort = path_len / straight if straight > 1e-6 else 1.0
        if graph.has_edge(u, v):
            continue
        graph.add_edge(u, v, length_px=path_len, tortuosity=float(tort))

    return graph


class CornealKnowledgeGraph:
    """Wrapper around a networkx fiber-connectivity graph with convenience
    analytics for diagnostic reasoning.

    Usage:
        kg = CornealKnowledgeGraph.from_mask(mask)
        kg.stats()
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        self.features: dict[str, float] = dict(graph.graph.get("features", {}))

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "CornealKnowledgeGraph":
        mask = (mask > 0).astype(np.uint8)
        return cls(build_fiber_graph(mask))

    @classmethod
    def from_features(
        cls, features: dict[str, float], graph: nx.Graph | None = None
    ) -> "CornealKnowledgeGraph":
        g = graph if graph is not None else nx.Graph()
        g.graph["features"] = dict(features)
        return cls(g)

    @property
    def n_nodes(self) -> int:
        return int(self.graph.number_of_nodes())

    @property
    def n_edges(self) -> int:
        return int(self.graph.number_of_edges())

    def mean_tortuosity(self) -> float:
        """Area-density-weighted mean tortuosity across fiber edges."""
        if self.n_edges == 0:
            return float(self.features.get("tortuosity_index", 1.0))
        lengths = np.asarray(
            [d["length_px"] for _, _, d in self.graph.edges(data=True)]
        )
        torts = np.asarray(
            [d["tortuosity"] for _, _, d in self.graph.edges(data=True)]
        )
        total = lengths.sum()
        if total <= 0:
            return 1.0
        return float((lengths * torts).sum() / total)

    def stats(self) -> dict[str, float]:
        """Aggregate reasoning-oriented graph statistics."""
        return {
            "n_nodes": float(self.n_nodes),
            "n_edges": float(self.n_edges),
            "n_connected_components": float(nx.number_connected_components(self.graph))
            if self.n_nodes
            else 0.0,
            "mean_tortuosity": self.mean_tortuosity(),
            **{k: float(v) for k, v in self.features.items()},
        }