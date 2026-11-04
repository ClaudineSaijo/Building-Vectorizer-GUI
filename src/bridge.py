"""Hypothesis-driven reconnection of buildings split by trees / shadow.

Idea (the user's): assume first that the pieces belong to *one* building, then
verify.  For every pair of boundary edges that

* run in (nearly) the same direction,
* are collinear within a few pixels,
* face each other across a gap of 3-40 px,
* and whose gap is mostly covered by vegetation / shadow,

we draw the quad that closes the gap and merge the two pieces.  The verification
step is exactly the collinearity + occlusion evidence, so a genuine courtyard or
two adjacent houses (whose walls are *not* collinear) are left alone.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["find_bridges", "apply_bridges", "Bridge"]


class Bridge:
    __slots__ = ("poly_a", "poly_b", "quad", "score")

    def __init__(self, poly_a: int, poly_b: int, quad: np.ndarray, score: float) -> None:
        self.poly_a = poly_a
        self.poly_b = poly_b
        self.quad = quad
        self.score = score

    def __repr__(self) -> str:
        return f"Bridge({self.poly_a}->{self.poly_b}, score={self.score:.2f})"


def _edges(points: np.ndarray):
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 3.0:
            continue
        yield i, a, b, d / length, length


def _line_distance(p: np.ndarray, a: np.ndarray, d: np.ndarray) -> float:
    v = p - a
    return abs(float(v[0] * d[1] - v[1] * d[0]))


def find_bridges(polygons, occluder: np.ndarray, image: np.ndarray | None = None,
                 max_gap: float = 14.0, min_gap: float = 3.0,
                 tol_deg: float = 12.0, max_offset: float = 3.0,
                 min_occluded: float = 0.70) -> list[Bridge]:
    """Return candidate bridges between polygon pairs."""
    if not polygons or occluder is None:
        return []
    cos_tol = np.cos(np.radians(tol_deg))
    h, w = occluder.shape[:2]
    bridges: list[Bridge] = []

    # Local thickness map of each building (used to estimate the bridge width)
    colors: list[np.ndarray] = []
    for poly in polygons:
        tmp = np.zeros((h, w), np.uint8)
        cv2.fillPoly(tmp, [np.round(np.asarray(poly.points)).astype(np.int32).reshape(-1, 1, 2)], 1)
        sel = tmp > 0
        colors.append(image[sel].mean(axis=0) if (image is not None and sel.any()) else np.zeros(3, np.float32))

    widths: list[float] = []
    for poly in polygons:
        tmp = np.zeros((h, w), np.uint8)
        cv2.fillPoly(tmp, [np.round(np.asarray(poly.points)).astype(np.int32).reshape(-1, 1, 2)], 1)
        dt = cv2.distanceTransform(tmp, cv2.DIST_L2, 5)
        # Inscribed-circle radius x2 ~= the short-side size of that building
        widths.append(float(np.clip(2.0 * float(dt.max()), 6.0, 60.0)))

    for ia, pa in enumerate(polygons):
        pts_a = np.asarray(pa.points, dtype=np.float32)
        for _, a0, a1, da, la in _edges(pts_a):
            for ib, pb in enumerate(polygons):
                if ib <= ia:
                    continue
                pts_b = np.asarray(pb.points, dtype=np.float32)
                for _, b0, b1, db, lb in _edges(pts_b):
                    if abs(float(np.dot(da, db))) < cos_tol:      # directions do not match
                        continue
                    # Both endpoints of B must lie close to the line through A
                    if max(_line_distance(b0, a0, da), _line_distance(b1, a0, da)) > max_offset:
                        continue
                    # Find the closest pair of endpoints; the gap between them is the one to bridge
                    pairs = [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
                    dists = [float(np.linalg.norm(x - y)) for x, y in pairs]
                    k = int(np.argmin(dists))
                    gap = dists[k]
                    if not (min_gap <= gap <= max_gap):
                        continue
                    e_a, e_b = pairs[k]
                    g = e_b - e_a
                    if float(abs(np.dot(g / gap, da))) < cos_tol:
                        continue                                   # not collinear
                    # Bridge = rectangle along the gap direction, width taken from the local
                    # thickness at that building endpoint
                    w_local = float(min(widths[ia], widths[ib]))
                    # Inward normal: remove the along-edge component from the centroid direction
                    cen_a = pts_a.mean(axis=0)
                    inward = cen_a - 0.5 * (a0 + a1)
                    inward = inward - float(np.dot(inward, da)) * da
                    n_in = float(np.linalg.norm(inward))
                    nrm = inward / n_in if n_in > 1e-6 else np.array([-da[1], da[0]], np.float32)
                    quad = np.stack([e_a, e_b,
                                     e_b + nrm * w_local, e_a + nrm * w_local]).astype(np.float32)
                    area = 0.5 * abs(float(np.dot(quad[:, 0], np.roll(quad[:, 1], -1))
                                            - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))))
                    if area < 20 or area > 4000:
                        continue
                    x0, y0 = np.clip(quad.min(axis=0), 0, None).astype(int)
                    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 1
                    x1, y1 = min(x1, w), min(y1, h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    cv2.fillPoly(local, [(quad - np.array([x0, y0])).reshape(-1, 1, 2).astype(np.int32)], 1)
                    sel = local > 0
                    occ = float(occluder[y0:y1, x0:x1][sel].mean()) if sel.any() else 0.0
                    if occ < min_occluded:
                        continue
                    score = occ * (1.0 - gap / max_gap) * min(la, lb)
                    bridges.append(Bridge(ia, ib, quad, score))
    bridges.sort(key=lambda b: -b.score)
    return bridges


def apply_bridges(polygons, bridges, shape, cfg):
    """Merge only the bridged polygon pairs with each other; keep all other polygons unchanged."""
    from .vectorize import (_contour_to_polygon, _dense_contour, _polygon_area,
                            _polygon_perimeter, BuildingPolygon)

    h, w = shape
    parent = list(range(len(polygons)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    used_pairs = []
    seen = set()
    for br in bridges:
        key = (min(br.poly_a, br.poly_b), max(br.poly_a, br.poly_b))
        if key in seen:
            continue
        seen.add(key)
        union(br.poly_a, br.poly_b)
        used_pairs.append(br)

    groups: dict[int, list[int]] = {}
    for i in range(len(polygons)):
        groups.setdefault(find(i), []).append(i)

    merged_output = []
    for root, members in groups.items():
        if len(members) == 1:
            merged_output.append(polygons[members[0]])
            continue
        mask = np.zeros((h, w), np.uint8)
        for m in members:
            cv2.fillPoly(mask, [np.round(np.asarray(polygons[m].points)).astype(np.int32).reshape(-1, 1, 2)], 1)
        for br in used_pairs:
            if br.poly_a in members and br.poly_b in members:
                cv2.fillPoly(mask, [np.round(br.quad).astype(np.int32).reshape(-1, 1, 2)], 1)
        contour = _dense_contour(mask)
        if contour is None or len(contour) < 3:
            merged_output.extend(polygons[m] for m in members)
            continue
        pts, n_init = _contour_to_polygon(contour, cfg)
        if len(pts) < 3:
            merged_output.extend(polygons[m] for m in members)
            continue
        merged_output.append(BuildingPolygon(points=pts.astype(np.float32), score=0.0,
                                             regularized=False, vertices_initial=n_init,
                                             area=_polygon_area(pts), perimeter=_polygon_perimeter(pts)))
    return merged_output
