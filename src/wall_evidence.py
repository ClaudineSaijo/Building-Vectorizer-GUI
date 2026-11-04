"""Wall-evidence completion: unmixing-driven search for occluded walls.

Two ideas are combined here.

**(A) wall evidence.**  A building boundary that disappears under a tree still
leaves a colour discontinuity, because roof and canopy differ in colour even
when the roof is only partly visible.  Measuring that discontinuity across a
candidate line tells us where the wall is.

**(B) unmixing.**  Sparse canopy is semi-transparent: the observed colour is a
linear mixture ``P = a * leaf + (1 - a) * roof``.  Projecting ``P`` on the
roof-to-leaf colour axis recovers both the mixing ratio ``a`` (how much is
hidden) and the residual (how well the mixture explains the pixel).  A small
residual with small ``a`` means "there is roof underneath".

Combining them: a notch is filled only when the area to be filled has high roof
evidence, i.e. when we can actually *see* roof through the canopy there -- not
merely because the area is covered by vegetation.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["roof_evidence", "complete_notches"]


def _leaf_color(image: np.ndarray, vegetated: np.ndarray) -> np.ndarray | None:
    if vegetated.sum() < 200:
        return None
    return image.astype(np.float32)[vegetated].mean(axis=0)


def roof_evidence(image: np.ndarray, mask: np.ndarray, roof_blur: int = 15) -> np.ndarray | None:
    """Per-pixel probability that *roof* is present (possibly under canopy).

    Returns float32 (H,W) in [0,1], or None when the tile has too few buildings
    or too little vegetation to calibrate the mixture.
    """
    f = np.asarray(image, dtype=np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    exg = 2.0 * g - r - b
    veg = (exg > 30) & (g > r + 5)
    leaf = _leaf_color(f, veg)
    if leaf is None:
        return None

    roof_sel = ((mask > 0) & (exg < 5)).astype(np.float32)
    if roof_sel.sum() < 50:
        return None
    num = cv2.blur(f * roof_sel[..., None], (roof_blur, roof_blur))
    den = cv2.blur(roof_sel, (roof_blur, roof_blur))
    roof_loc = num / np.maximum(den, 1e-3)[..., None]

    axis = leaf - roof_loc
    denom = (axis * axis).sum(axis=2) + 1e-3
    alpha = np.clip(((f - roof_loc) * axis).sum(axis=2) / denom, 0.0, 1.0)
    recon = roof_loc + alpha[..., None] * axis
    residual = np.linalg.norm(f - recon, axis=2) / np.sqrt(denom)

    evidence = np.exp(-(residual ** 2) / (2 * 0.25 ** 2)) * (1.0 - alpha) * (den > 0.05)
    evidence[~np.isfinite(evidence)] = 0.0
    return np.clip(evidence, 0.0, 1.0).astype(np.float32)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _angle_between(v1, v2) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 90.0
    cos = float(np.clip(abs(float(np.dot(v1, v2)) / (n1 * n2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def complete_notches(points, evidence: np.ndarray, tol_deg: float = 25.0,
                     min_roof: float = 0.42, max_gain: float = 0.35) -> np.ndarray:
    """Fill concave notches whose interior shows enough *roof* evidence."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 5 or evidence is None:
        return pts
    if _signed_area(pts) < 0:
        pts = pts[::-1].copy()
    n = len(pts)
    h, w = evidence.shape[:2]

    concave = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross < 0:
            concave.append(i)
    if not concave:
        return pts

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
        i_start, i_end = (s - 1) % n, (e + 1) % n
        if _angle_between(pts[i_start] - pts[(s - 2) % n], pts[(e + 2) % n] - pts[i_end]) > tol_deg:
            continue
        chain = [pts[i_start]] + [pts[k] for k in idx_run] + [pts[i_end]]
        poly = np.round(np.asarray(chain)).astype(np.int32)
        x0, y0 = np.clip(poly.min(axis=0), 0, None)
        x1, y1 = poly.max(axis=0) + 1
        x0, y0, x1, y1 = int(x0), int(y0), min(int(x1), w), min(int(y1), h)
        if x1 <= x0 or y1 <= y0:
            continue
        local = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(local, [(poly - np.array([x0, y0])).reshape(-1, 1, 2)], 1)
        sel = local > 0
        mean_ev = float(evidence[y0:y1, x0:x1][sel].mean()) if sel.any() else 0.0
        if mean_ev < min_roof:
            continue
        drop[idx_run] = True

    if not drop.any():
        return pts
    out = pts[~drop]
    if len(out) < 3:
        return pts
    gain = (abs(_signed_area(out)) - area0) / max(area0, 1e-6)
    if gain > max_gain or gain < -0.05:
        return pts

    cleaned = []
    for i in range(len(out)):
        a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
        v1, v2 = b - a, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1e-6 and n2 > 1e-6 and abs(float(np.dot(v1, v2) / (n1 * n2))) > 0.999:
            continue
        cleaned.append(b)
    if len(cleaned) < 3:
        return pts
    return np.asarray(cleaned, dtype=np.float32)
