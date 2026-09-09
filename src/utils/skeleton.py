from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize as _skel
from skimage.measure import label as cc_label


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary nerve mask (HxW, 0/1 or 0/255).

    Returns a binary skeleton array of the same shape (bool or same dtype).
    """
    binary = (mask > 0).astype(np.uint8)
    skel = _skel(binary.astype(bool))
    return skel.astype(np.uint8)


def topological_features(skeleton: np.ndarray) -> dict[str, float]:
    """Compute basic topological statistics from a binary skeleton.

    Args:
        skeleton: (H, W) boolean/uint8 array of thinned nerve fibers.

    Returns:
        Dict with:
            - n_junctions: branching points (>=3 neighbours)
            - endpoints:   terminal fiber points (1 neighbour)
            - total_length: sum of skeleton pixel counts
    """
    binary = (skeleton > 0).astype(np.uint8)
    if binary.sum() == 0:
        return {"n_junctions": 0, "endpoints": 0, "total_length": 0.0}

    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    neighbour_count = _count_neighbours(binary, k)

    nerve = binary > 0
    junctions = int(np.sum((neighbour_count >= 3) & nerve))
    endpoints = int(np.sum((neighbour_count == 1) & nerve))
    total_length = float(binary.sum())
    return {
        "n_junctions": junctions,
        "endpoints": endpoints,
        "total_length": total_length,
    }


def _count_neighbours(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Return per-pixel 8-neighbour count excluding center."""
    from scipy.ndimage import convolve

    return convolve(binary.astype(np.float32), kernel.astype(np.float32), mode="constant")


def branching_density(skeleton: np.ndarray, total_area: float = 1.0) -> float:
    """Branching junction density per unit area (junctions / area)."""
    features = topological_features(skeleton)
    if total_area <= 0:
        return 0.0
    return features["n_junctions"] / total_area


def connected_components(skeleton: np.ndarray) -> int:
    """Number of connected fiber components in the skeleton."""
    binary = (skeleton > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0
    return int(cc_label(binary).max())
