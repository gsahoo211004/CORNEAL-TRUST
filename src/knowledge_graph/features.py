from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..utils.skeleton import (
    skeletonize_binary,
    topological_features,
    connected_components,
)
from ..utils.skeleton import branching_density as _br_density

NEIGHBORHOOD = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def fiber_segments(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Extract individual nerve fiber segments from a binary skeleton.

    Segments are traced between keypoints (junctions/endpoints) along the
    skeleton. Each returned path has >= 2 coordinate points; a path may span
    multiple length-1 edges when the branch is a single dead-end spur.
    """
    binary = (skeleton > 0).astype(np.uint8)
    pts = set(map(tuple, np.argwhere(binary)))
    if not pts:
        return []

    neighbour = _count_neighbours(binary)
    keypoints = {
        (r, c) for (r, c) in pts
        if neighbour[r, c] >= 3 or neighbour[r, c] == 1
    }
    if not keypoints:
        return [list(pts)] if binary.sum() else []

    raw_paths: list[list[tuple[int, int]]] = []
    for kp in sorted(keypoints):
        for dr, dc in NEIGHBORHOOD:
            nxt = (kp[0] + dr, kp[1] + dc)
            if nxt not in pts:
                continue
            raw_paths.append(_trace(pts, keypoints, kp, nxt))

    best: dict[frozenset[tuple[int, int]], list[tuple[int, int]]] = defaultdict(list)
    for path in raw_paths:
        if len(path) < 2:
            continue
        key = frozenset((path[0], path[-1]))
        if len(path) > len(best[key]):
            best[key] = path
    return [p for p in best.values() if len(p) >= 2]


def _trace(
    pts: set[tuple[int, int]],
    keypoints: set[tuple[int, int]],
    start: tuple[int, int],
    first: tuple[int, int],
) -> list[tuple[int, int]]:
    """Walk along the skeleton from ``start`` through ``first`` until the next
    keypoint or dead-end. Returns the full coordinate path."""
    path = [start]
    prev, cur = start, first
    while True:
        path.append(cur)
        if cur in keypoints:
            break
        nxt = _next_neighbor(pts, cur, prev)
        if nxt is None:
            break
        prev, cur = cur, nxt
    return path


def _next_neighbor(
    pts: set[tuple[int, int]], cur: tuple[int, int], prev: tuple[int, int]
) -> tuple[int, int] | None:
    for dr, dc in NEIGHBORHOOD:
        cand = (cur[0] + dr, cur[1] + dc)
        if cand in pts and cand != prev:
            return cand
    return None


def _count_neighbours(binary: np.ndarray, radius: int = 1) -> np.ndarray:
    from scipy.ndimage import convolve

    size = radius * 2 + 1
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[radius, radius] = 0.0
    return convolve(binary.astype(np.float32), kernel, mode="constant", cval=0.0)


def _path_length(path: list[tuple[int, int]]) -> float:
    """Length of a coordinate path in Euclidean units (diagonal steps weigh
    sqrt(2), so a straight line has length equal to endpoint distance)."""
    return float(
        sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
    )


def tortuosity_index(skeleton: np.ndarray) -> float:
    """Mean tortuosity across fiber segments.

    Each segment: tortuosity = path_length / straight_line_distance.
    A straight fiber has tortuosity ~1.0; higher values mean more winding.
    """
    segments = fiber_segments(skeleton)
    if not segments:
        return 1.0
    values: list[float] = []
    for seg in segments:
        path_len = _path_length(seg)
        start, end = seg[0], seg[-1]
        straight = math.hypot(end[0] - start[0], end[1] - start[1])
        if straight < 1e-6:
            continue
        values.append(path_len / straight)
    if not values:
        return 1.0
    return float(np.mean(values))


def compute_clinical_features(
    mask: np.ndarray, image_area: float | None = None
) -> dict[str, float]:
    """Compute clinical-grade morphological features from a nerve mask.

    Args:
        mask: binary nerve mask (HxW, 0/1 or 0/255) or a probability map.
        image_area: total tissue area to normalise against (pixels). Defaults
            to the mask area.

    Returns:
        Dict with n_junctions, endpoints, total_length, connected_components,
        fiber_density (nerve pixels / total pixels), branching_density
        (junctions / nerve pixel), tortuosity_index, n_segments.
    """
    mask = (mask > 0).astype(np.uint8)
    if image_area is None or image_area <= 0:
        image_area = float(mask.size)

    skel = skeletonize_binary(mask)
    topo = topological_features(skel)
    n_components = connected_components(skel)

    segs = fiber_segments(skel)
    total_length = topo["total_length"]
    nerve_px = max(float(total_length), 1.0)
    bd = _br_density(skel, total_area=nerve_px)  # junctions / nerve pixel

    return {
        "n_junctions": float(topo["n_junctions"]),
        "endpoints": float(topo["endpoints"]),
        "total_length": total_length,
        "connected_components": float(n_components),
        "fiber_density": float(total_length) / image_area,
        "branching_density": bd,
        "tortuosity_index": tortuosity_index(skel),
        "n_segments": float(len(segs)),
    }