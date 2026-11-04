"""Geometry-based occlusion completion: cut notches caused by trees/shadow.

When a tree or its shadow covers part of a roof, the extracted contour bends
*inwards* around the occluder, producing a concave notch.  If the wall on both
sides of that notch runs in the same direction, the notch is not a real feature
of the building: the wall simply continues behind the tree.

This module detects such notches and replaces them with a straight chord, but
only when

* the edges on both sides are (nearly) collinear, and
* the removed area is actually covered by vegetation / shadow.

That double condition keeps genuine courtyards and L-shaped corners intact.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["cut_occlusion_notches", "occluder_mask"]


def occluder_mask(image: np.ndarray, shadow_ratio: float = 0.62) -> np.ndarray:
    """Boolean map of vegetation and cast shadow (uint8, 1 = occluder)."""
    rgb = np.asarray(image, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    veg = (exg > 30) & (g > r + 5)
    shadow = gray < shadow_ratio * float(np.median(gray))
    return (veg | shadow).astype(np.uint8)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(abs(cos))))


def cut_occlusion_notches(points, occluder: np.ndarray, tol_deg: float = 25.0,
                          min_occluded: float = 0.55, max_area_gain: float = 0.35) -> np.ndarray:
    """Replace occlusion notches by a straight chord.

    ``points``: (N,2) polygon; ``occluder``: (H,W) uint8 mask of trees/shadow.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 5 or occluder is None:
        return pts
    if _signed_area(pts) < 0:                      # enforce CCW
        pts = pts[::-1].copy()
    n = len(pts)
    h, w = occluder.shape[:2]

    # concave vertices: right turn for a CCW ring
    concave = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < 0:
            concave.append(i)
    if not concave:
        return pts

    # group cyclic runs of concave vertices
    runs = []
    start = prev = concave[0]
    for idx in concave[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n - 1:
        runs[0] = (runs[-1][0] - n, runs[0][1])
        runs.pop()

    area0 = abs(_signed_area(pts))
    drop = np.zeros(n, dtype=bool)
    for s, e in runs:
        idx_run = [k % n for k in range(s, e + 1)]
        i_start = (s - 1) % n
        i_end = (e + 1) % n
        v_before = pts[i_start] - pts[(s - 2) % n]
        v_after = pts[(e + 2) % n] - pts[i_end]
        if _angle_between(v_before, v_after) > tol_deg:
            continue
        chain = [pts[i_start]] + [pts[k] for k in idx_run] + [pts[i_end]]
        notch = np.round(np.asarray(chain)).astype(np.int32)
        x0, y0 = notch.min(axis=0).clip(0)
        x1, y1 = notch.max(axis=0) + 1
        x1, y1 = min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(local, [(notch - np.array([x0, y0])).reshape(-1, 1, 2)], 1)
        occ_frac = float((local & occluder[y0:y1, x0:x1]).sum()) / max(int(local.sum()), 1)
        if occ_frac < min_occluded:
            continue
        for k in idx_run:
            drop[k] = True

    if not drop.any():
        return pts
    out = pts[~drop]
    if len(out) < 3:
        return pts
    gain = (abs(_signed_area(out)) - area0) / max(area0, 1e-6)
    if gain > max_area_gain or gain < -0.05:
        return pts
    # drop collinear points created by the chord
    cleaned = []
    m = len(out)
    for i in range(m):
        a, b, c = out[i - 1], out[i], out[(i + 1) % m]
        v1 = b - a
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6:
            cos = abs(float(np.dot(v1, v2) / (n1 * n2)))
            if cos > 0.999:
                continue
        cleaned.append(b)
    if len(cleaned) < 3:
        return pts
    return np.asarray(cleaned, dtype=np.float32)
