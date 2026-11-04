"""Global collinearity-aware polygon simplification (GCP, TGRS 2025).

Reference: F. Zhang, Y. Shi, X. X. Zhu, "Global Collinearity-Aware Polygonizer
for Polygonal Building Mapping in Remote Sensing", TGRS 63, 2025.

The idea: instead of Douglas-Peucker's local greedy corner picking, choose the
subsequence of polyline points that minimises

    L(p, p_hat) + lambda * |p_hat|

where ``L`` is the sum of point-to-segment distances of the dropped points and
``|p_hat|`` is the number of kept vertices.  This is solved exactly by dynamic
programming, which gives a globally optimal trade-off between fidelity and
compactness -- typically fewer vertices than DP for the same error.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gcp_simplify"]


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def gcp_simplify(points, lam: float = 1.2, closed: bool = True) -> np.ndarray:
    """Simplify a polyline with the collinearity-aware DP objective.

    ``lam`` is in pixels: larger values allow bigger deviations but fewer
    vertices.  A good starting range for 0.3 m aerial imagery is 0.8-2.0.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return pts.astype(np.float32)
    if closed:
        # rotate so that we start at the point farthest from the centroid:
        # a corner is a much better fixed endpoint than an arbitrary edge point
        centroid = pts.mean(axis=0)
        start = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))
        pts = np.roll(pts, -start, axis=0)
        pts = np.vstack([pts, pts[:1]])

    n = len(pts)
    # cost[i, j] = sum of distances of points (i, j) to the segment p_i p_j
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        for j in range(i + 2, n):
            inner = pts[i + 1:j]
            if len(inner):
                cost[i, j] = _point_segment_distance(inner, pts[i], pts[j]).sum()

    dp = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=int)
    dp[0] = 0.0
    for j in range(1, n):
        best, arg = np.inf, -1
        for i in range(j):
            if not np.isfinite(dp[i]):
                continue
            value = dp[i] + cost[i, j] + lam
            if value < best:
                best, arg = value, i
        dp[j] = best
        prev[j] = arg

    keep = []
    cur = n - 1
    while cur >= 0:
        keep.append(cur)
        cur = prev[cur]
    keep = sorted(set(keep))
    out = pts[keep]
    if closed and len(out) > 1 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return out.astype(np.float32)
