"""Extend a building outward along a bright, thin ridge.

Motivation (0153 in the WHU test split): part of a footprint is a narrow bright
structure (a low wall / canopy) that disappears under trees.  The classifier
almost never fires on such a 1-2 px wide line, but the line itself is still
visible between the trees: along it the grey level stays high (>170) for dozens
of pixels.

So instead of *connecting two pieces* (``bridge.py``) or *filling a notch*
(``occlusion_geometry.py``), we **trace the ridge** starting from a boundary
endpoint that touches an occluder and grow the polygon along it.

Acceptance ("assume it is the same building, then verify"):
  * the traced path must be mostly bright (mean > 160),
  * it must start within a few pixels of the polygon boundary,
  * its direction must stay within a few degrees of the edge direction,
  * the rest of the path has to be covered by vegetation / shadow.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["trace_ridge", "extend_polygon_along_ridge"]


def _bright_map(image: np.ndarray, thr: int = 165) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    return (gray >= thr).astype(np.uint8)


def trace_ridge(bright: np.ndarray, start: np.ndarray, direction: np.ndarray,
                max_len: int = 45, max_drift: float = 5.0) -> np.ndarray:
    """Walk from ``start`` along ``direction``, hugging the bright ridge."""
    h, w = bright.shape
    d = direction / (np.linalg.norm(direction) + 1e-6)
    n = np.array([-d[1], d[0]])
    path = []
    cur = np.asarray(start, dtype=np.float32).copy()
    miss = 0
    for step in range(1, max_len + 1):
        base = np.asarray(start, dtype=np.float32) + d * step
        found = None
        for off in np.arange(-max_drift, max_drift + 0.5, 1.0):
            p = base + n * off
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h and bright[y, x]:
                found = p
                break
        if found is None:
            miss += 1
            if miss >= 3:
                break
            continue
        miss = 0
        cur = found
        path.append(cur.copy())
    return np.asarray(path, dtype=np.float32) if path else np.zeros((0, 2), np.float32)


def extend_polygon_along_ridge(points, image: np.ndarray, occluder: np.ndarray | None = None,
                               min_len: int = 8, width: float = 3.5,
                               min_bright: float = 150.0, occ_frac: float = 0.0,
                               max_len: int = 90) -> np.ndarray:
    """Return an extra thin polygon that continues the building along a ridge."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or image is None:
        return np.zeros((0, 2), np.float32)
    bright = _bright_map(image)
    gray = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occ_near = cv2.dilate(occluder, k3, iterations=2) > 0 if occluder is not None else None

    best = None
    n = len(pts)
    for i in range(n):
        a, b = pts[i - 1], pts[i]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 3:
            continue
        d = d / length
        x, y = int(round(b[0])), int(round(b[1]))
        h, w = bright.shape
        if not (0 <= x < w and 0 <= y < h):
            continue
        if occ_near is not None and occ_frac > 0 and not occ_near[y, x]:
            continue
        # Three candidate directions: along the edge, along the outward normal, along the inward normal
        cen = pts.mean(axis=0)
        nrm = np.array([-d[1], d[0]], np.float32)
        if float(np.dot(cen - b, nrm)) > 0:
            nrm = -nrm                      # point outward
        best_path = None
        for cand in (d, nrm):
            path = trace_ridge(bright, b, cand, max_len=max_len)
            if len(path) >= min_len and (best_path is None or len(path) > len(best_path)):
                best_path = path
        if best_path is None:
            continue
        path = best_path
        d = (path[-1] - path[0])
        d = d / (np.linalg.norm(d) + 1e-6)
        vals = [gray[int(round(p[1])), int(round(p[0]))] for p in path]
        if float(np.mean(vals)) < min_bright:
            continue
        if occ_frac > 0 and occluder is not None:
            occ_share = float(np.mean([occluder[int(round(p[1])), int(round(p[0]))] for p in path]))
            if occ_share < occ_frac:
                continue
        score = len(path) * float(np.mean(vals))
        if best is None or score > best[0]:
            best = (score, d, path)

    if best is None:
        return np.zeros((0, 2), np.float32)
    _, d, path = best
    nrm = np.array([-d[1], d[0]], np.float32) * (width / 2.0)
    start = path[0] - d * 1.0
    end = path[-1] + d * 1.0
    quad = np.stack([start + nrm, end + nrm, end - nrm, start - nrm]).astype(np.float32)
    return quad
