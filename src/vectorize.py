"""Pixel probability maps -> building polygons (vector features).

The pipeline mirrors the three branches described in the proposal:

1. **semantic branch**   -> binary mask from the probability map;
2. **boundary branch**   -> cut touching buildings apart with the boundary map;
3. **vertex branch**     -> turn each raster contour into a compact polygon with
   vertices placed at line intersections instead of staircase pixels;
   optionally **geometry regularization** snaps dominant edges to 45/90 degrees.

The vertex branch here is the *deterministic* counterpart of the graph-convolution
vertex regressor that is still being trained.  It exposes exactly the same
interface (``VectorConfig.use_vertex`` / ``use_geometry``) so the GUI and the
ablation study keep working once the learned head is plugged in.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

__all__ = [
    "VectorConfig",
    "BuildingPolygon",
    "VectorResult",
    "vectorize",
    "polygons_to_mask",
    "polygon_boundary_points",
    "write_geojson",
]


# --------------------------------------------------------------------------- #
# configuration / result containers
# --------------------------------------------------------------------------- #
@dataclass
class VectorConfig:
    """All switches exposed in the GUI's algorithm panel."""

    threshold: float = 0.45          # mask probability threshold (tuned on the val split)
    use_boundary: bool = True        # branch 2: boundary-constrained separation
    use_vertex: bool = True          # branch 3: vertex regression / refinement
    use_geometry: bool = True        # geometric regularization (right angles)
    edge_threshold: float = 0.45     # boundary map threshold (floor)
    edge_quantile: float = 0.90      # keep the strongest boundary responses
    use_edge_cut: bool = True        # seed the separation with the boundary band
    split_peak_size: int = 11        # max-filter window for distance peaks
    split_min_peak: float = 1.8      # ignore peaks thinner than this radius
    split_min_part: float = 0.22     # every instance keeps >= this share of the blob
    split_max_parts: int = 4         # refuse implausible over-segmentation
    split_size_factor: float = 2.0   # only split blobs this much larger than typical
    min_area: float = 40.0           # drop instances smaller than this (px^2)
    simplify_ratio: float = 0.012    # Douglas-Peucker epsilon / perimeter
    rect_tolerance_deg: float = 14.0 # max edge deviation from 0/90 deg
    rect_min_support: float = 0.55   # share of edge length that must be axis-aligned
    use_completion: bool = True      # bridge tree/shadow occlusions with straight edges
    use_bridge: bool = False         # reconnect buildings split by trees/shadow
    open_kernel: int = 3             # 形态学开运算核（1=不做开运算）
    bridge_max_gap: float = 40.0
    use_notch_cut: bool = True       # cut concave notches that are caused by occluders
    notch_tol_deg: float = 25.0      # max angle between the two edges flanking a notch
    notch_min_occ: float = 0.55      # notch area must be this much occluded
    occl_min_run: int = 6            # min occluded contour length to bridge (px)
    occl_dilate: int = 2             # occluder mask dilation (px)
    occl_max_chord_frac: float = 0.35  # never bridge a gap longer than this share of perimeter

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VectorConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class BuildingPolygon:
    """One building footprint, in image pixel coordinates."""

    points: np.ndarray                  # (N, 2) float32, closed implicitly
    score: float = 0.0                  # mean mask probability inside
    regularized: bool = False           # geometry branch applied
    vertices_initial: int = 0           # vertices before vertex refinement
    area: float = 0.0
    perimeter: float = 0.0
    geom_id: int = 0

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)

    @property
    def num_vertices(self) -> int:
        return int(len(self.points))

    def area_perimeter(self) -> tuple[float, float]:
        return float(self.area), float(self.perimeter)


@dataclass
class VectorResult:
    polygons: list[BuildingPolygon] = field(default_factory=list)
    mask: np.ndarray | None = None
    instances: int = 0
    merged_blobs: int = 0           # blobs that the boundary branch split
    regularized: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    config: VectorConfig = field(default_factory=VectorConfig)

    @property
    def num_polygons(self) -> int:
        return len(self.polygons)

    @property
    def mean_vertices(self) -> float:
        if not self.polygons:
            return 0.0
        return float(np.mean([p.num_vertices for p in self.polygons]))

    def summary(self) -> dict:
        return {
            "num_polygons": self.num_polygons,
            "instances": self.instances,
            "merged_blobs": self.merged_blobs,
            "regularized": self.regularized,
            "mean_vertices": round(self.mean_vertices, 2),
            "timings_ms": {k: round(v * 1000.0, 2) for k, v in self.timings.items()},
            "config": self.config.to_dict(),
        }


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _polygon_perimeter(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1).sum())


def _ring_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two rings rasterised on their shared bounding box."""
    pa = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    both = np.vstack([pa, pb])
    x0, y0 = np.floor(both.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(both.max(axis=0)).astype(int) + 2
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    ma = np.zeros((h, w), dtype=np.uint8)
    mb = np.zeros((h, w), dtype=np.uint8)
    off = np.array([x0, y0], dtype=np.float32)
    cv2.fillPoly(ma, [np.round(pa - off).astype(np.int32).reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mb, [np.round(pb - off).astype(np.int32).reshape(-1, 1, 2)], 1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / max(union, 1)


def _dense_polygon(contour: np.ndarray, cfg: "VectorConfig") -> np.ndarray:
    """Contour-following polygon: the safe fallback when refinement misfires."""
    perimeter = float(cv2.arcLength(contour, True))
    eps = max(1.0, cfg.simplify_ratio * perimeter)
    dense_eps = max(0.6, 0.10 * eps)
    return cv2.approxPolyDP(contour, dense_eps, True).reshape(-1, 2).astype(np.float32)


def _dedupe(points: np.ndarray, min_dist: float = 1.5) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts.astype(np.float32)
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) >= min_dist:
            kept.append(p)
    # closed polygon: last point must not coincide with the first
    while len(kept) > 2 and np.linalg.norm(kept[-1] - kept[0]) < min_dist:
        kept.pop()
    return np.asarray(kept, dtype=np.float32)


def _drop_collinear(points: np.ndarray, angle_tol_deg: float = 2.0) -> np.ndarray:
    """Remove vertices whose interior angle is (almost) 180 degrees."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n <= 3:
        return pts.astype(np.float32)
    keep = []
    cos_tol = math.cos(math.radians(180.0 - angle_tol_deg))
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        u, v = a - b, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        # interior angle at b: straight line -> 180 deg -> cos == -1
        if float(np.dot(u, v) / (nu * nv)) > cos_tol:
            continue
        keep.append(pts[i])
    if len(keep) < 3:
        return pts.astype(np.float32)
    return np.asarray(keep, dtype=np.float32)


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else -1

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def is_simple_polygon(points: np.ndarray) -> bool:
    """True when the ring has no self-intersection (O(n^2), n is tiny here)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 4:
        return True
    for i in range(n):
        a1, a2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = pts[j], pts[(j + 1) % n]
            if _segments_cross(a1, a2, b1, b2):
                return False
    return True


def _line_intersection(a, d, b, e) -> np.ndarray | None:
    """Intersection of line a + t*d and b + s*e (2-D)."""
    denom = d[0] * e[1] - d[1] * e[0]
    if abs(denom) < 1e-8:
        return None
    diff = b - a
    t = (diff[0] * e[1] - diff[1] * e[0]) / denom
    return a + t * d


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(pts) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(np.asarray(pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.array([float(vx), float(vy)], dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return None
    d /= norm
    return np.array([float(x0), float(y0)], dtype=np.float64), d


# --------------------------------------------------------------------------- #
# instance separation (boundary branch)
# --------------------------------------------------------------------------- #
def _partition_by_nearest(blob: np.ndarray, labels: np.ndarray, ids: list[int]) -> list[np.ndarray]:
    """Assign every blob pixel to its closest marker so instances tile the blob."""
    dists = np.stack([
        cv2.distanceTransform((labels != i).astype(np.uint8), cv2.DIST_L2, 3)
        for i in ids
    ])
    nearest = np.asarray(ids, dtype=np.int32)[np.argmin(dists, axis=0)]
    return [((labels == i) | ((labels <= 0) & (nearest == i) & (blob > 0))).astype(np.uint8) & blob
            for i in ids]


def _split_blob(blob: np.ndarray, cut: np.ndarray | None, cfg: VectorConfig) -> list[np.ndarray]:
    """Separate touching buildings inside one blob.

    Markers come from the boundary-constrained cut when it succeeds, otherwise
    from distance-transform peaks (the classical morphology route).  Every blob
    pixel is then assigned to its nearest marker, which yields a gap-free
    partition instead of a shrunken mask.
    """
    blob_area = int(blob.sum())
    markers = None
    if cut is not None:
        inner = (blob.astype(np.uint8) & (~cut).astype(np.uint8))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inner, connectivity=8)
        keep = [i for i in range(1, num)
                if stats[i, cv2.CC_STAT_AREA] >= max(cfg.min_area, cfg.split_min_part * blob_area)]
        if 2 <= len(keep) <= cfg.split_max_parts:
            markers = np.zeros(blob.shape, dtype=np.int32)
            for new_id, old_id in enumerate(keep, start=1):
                markers[labels == old_id] = new_id

    if markers is None:
        dist = cv2.distanceTransform(blob.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        peaks = (dist >= maximum_filter(dist, size=cfg.split_peak_size)) & (dist > cfg.split_min_peak)
        # label the peaks directly (they are separated by the filter window)
        num, markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
        markers = markers.astype(np.int32)
        if num - 1 < 2:
            return [blob.astype(np.uint8)]

    ids = [int(i) for i in np.unique(markers) if i > 0]
    if len(ids) < 2 or len(ids) > cfg.split_max_parts:
        return [blob.astype(np.uint8)]

    parts = _partition_by_nearest(blob.astype(np.uint8), markers, ids)
    parts = [p for p in parts if int(p.sum()) >= max(cfg.min_area, cfg.split_min_part * blob_area)]
    return parts if len(parts) >= 2 else [blob.astype(np.uint8)]


def _separate_instances(mask: np.ndarray, cut: np.ndarray | None,
                        cfg: VectorConfig) -> tuple[list[tuple[np.ndarray, int, int]], int]:
    """Split the mask into instances, each returned as (local_mask, x0, y0).

    Everything is processed on per-blob bounding boxes: a 1500x1500 tile may hold
    thousands of buildings, and touching full-size arrays for every one of them
    dominates the run time otherwise.
    """
    mask = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < cfg.min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((i, x, y, w, h))

    if not cfg.use_boundary:
        return [((labels[y:y + h, x:x + w] == i).astype(np.uint8), x, y) for i, x, y, w, h in boxes], 0

    # adaptive split threshold: only blobs well above the typical building size
    # are candidates, which stops individual roofs from being cut in two
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i, *_ in boxes], dtype=np.float64)
    typical = float(np.median(areas)) if len(areas) else cfg.min_area
    size_limit = max(cfg.min_area * 2.0, cfg.split_size_factor * typical, 120.0)

    split_count = 0
    instances: list[tuple[np.ndarray, int, int]] = []
    for i, x, y, w, h in boxes:
        sub = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        if sub.sum() < size_limit:
            instances.append((sub, x, y))
            continue
        sub_cut = cut[y:y + h, x:x + w] if cut is not None else None
        parts = _split_blob(sub, sub_cut, cfg)
        if len(parts) > 1:
            split_count += 1
        for part in parts:
            instances.append((part, x, y))
    return instances, split_count


# --------------------------------------------------------------------------- #
# contour -> polygon (vertex branch)
# --------------------------------------------------------------------------- #
def _dense_contour(instance: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(instance.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _occluder_map(image: np.ndarray, mask: np.ndarray, cfg: VectorConfig) -> np.ndarray | None:
    """Pixels that are likely trees or cast shadow (they bend building outlines).

    Vegetation is detected with the excess-green index, shadow by brightness
    relative to the building pixels.  Occluders inside the predicted building
    (dark roofs) are removed so they cannot trigger completion.
    """
    if image is None:
        return None
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    inside = gray[mask > 0]
    med = float(np.median(inside)) if inside.size else float(np.median(gray))
    veg = (2.0 * g - r - b > 25.0) & (g > r + 5.0)
    shadow = gray < 0.55 * max(med, 1.0)
    occ = (veg | shadow).astype(np.uint8)
    if cfg.occl_dilate > 0:
        occ = cv2.dilate(occ, _ellipse(2 * cfg.occl_dilate + 1))
    eroded_mask = cv2.erode((mask > 0).astype(np.uint8), _ellipse(3)).astype(bool)
    occ[eroded_mask] = 0
    return occ > 0


def _complete_across_occluders(pts: np.ndarray, occ: np.ndarray, cfg: VectorConfig) -> np.ndarray:
    """Replace contour runs that hug a tree/shadow by one straight chord.

    A run of contour points that sits on occluders is assumed to be the *shadow*
    of a straight wall rather than the real outline: the two supported endpoints
    are kept and everything between them is dropped, so the polygonization stage
    connects them with a straight edge.  Runs longer than a share of the
    perimeter are left alone to avoid cutting a building in half.
    """
    n = len(pts)
    if n < 12:
        return pts
    h, w = occ.shape
    xy = np.round(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
    flags = occ[xy[:, 1], xy[:, 0]]
    if not flags.any() or flags.mean() > 0.85:
        return pts

    start = int(np.argmin(flags))          # rotate so index 0 is supported
    pts_r = np.roll(pts, -start, axis=0)
    flags_r = np.roll(flags, -start)
    perimeter = float(cv2.arcLength(pts_r.reshape(-1, 1, 2).astype(np.int32), True))
    keep = np.ones(n, dtype=bool)

    i = 0
    while i < n:
        if not flags_r[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags_r[j + 1]:
            j += 1
        run = j - i + 1
        if run >= cfg.occl_min_run and j + 1 < n and i > 0:
            chord_vec = pts_r[j + 1] - pts_r[i - 1]
            chord = float(np.linalg.norm(chord_vec))
            if 1e-6 < chord <= cfg.occl_max_chord_frac * perimeter:
                if _is_straight_continuation(pts_r, i, j, chord_vec):
                    keep[i:j + 1] = False
        i = j + 1

    if keep.sum() < 0.6 * n:
        return pts
    idx = (np.arange(n) + start) % n
    keep_orig = np.zeros(n, dtype=bool)
    keep_orig[idx] = keep
    return pts[keep_orig].astype(np.float32)


def _is_straight_continuation(pts: np.ndarray, i: int, j: int, chord_vec: np.ndarray,
                              span: int = 8, tol_deg: float = 25.0) -> bool:
    """True when the wall keeps the same direction on both sides of a gap.

    A tree/shadow occlusion leaves a detour whose two supported ends still run
    along one straight wall; a genuine corner or curved facade does not.
    """
    chord = float(np.linalg.norm(chord_vec))
    if chord < 1e-6:
        return False
    cd = chord_vec / chord
    n = len(pts)

    def _dir(a: int, b: int) -> np.ndarray | None:
        idx = np.arange(a, b)
        if len(idx) < 2:
            return None
        seg = pts[idx % n]
        v = seg[-1] - seg[0]
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-6 else None

    before = _dir(i - 1 - span, i - 1)
    after = _dir(j + 1, j + 1 + span)
    cos_tol = math.cos(math.radians(tol_deg))
    for v in (before, after):
        if v is not None and float(np.dot(v, cd)) < cos_tol:
            return False
    return True


def _contour_indices(pts: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(pts[None, :, :] - np.asarray(seeds, dtype=np.float64)[:, None, :], axis=2)
    return np.argmin(dist, axis=1)


def _polygonize(contour: np.ndarray, cfg: VectorConfig) -> np.ndarray:
    """Fit one straight line per contour segment, then intersect adjacent lines.

    Douglas-Peucker only *selects* the seed corners; the actual edges are total
    least-squares fits over the whole contour run between two corners.  Vertices
    therefore sit at the true intersection of roof edges instead of on the
    staircase pixels, which is the deterministic equivalent of the learned
    vertex-offset regression.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    perimeter = float(cv2.arcLength(contour, True))
    eps = max(1.0, cfg.simplify_ratio * perimeter)
    seeds = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float64)
    if len(seeds) < 3 or n < 8:
        return seeds.astype(np.float32)

    idx = _contour_indices(pts, seeds)
    order = np.argsort(idx)
    seeds, idx = seeds[order], idx[order]
    m = len(idx)

    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(m):
        a, b = int(idx[k]), int(idx[(k + 1) % m])
        seg = pts[a:b + 1] if b > a else np.vstack([pts[a:], pts[:b + 1]])
        fit = _fit_line(seg) if len(seg) >= 3 else None
        if fit is None:
            fit = (seeds[k], seeds[(k + 1) % m] - seeds[k])
        lines.append(fit)

    verts = []
    for k in range(m):
        p1, d1 = lines[k - 1]
        p2, d2 = lines[k]
        point = _line_intersection(p1, d1, p2, d2)
        seed = seeds[k]
        if point is None or not np.all(np.isfinite(point)) or np.linalg.norm(point - seed) > 7.0:
            point = seed
        verts.append(point)

    poly = _dedupe(np.asarray(verts, dtype=np.float32), min_dist=1.2)
    return _drop_collinear(poly, angle_tol_deg=2.0)


def _contour_to_polygon(contour: np.ndarray, cfg: VectorConfig) -> tuple[np.ndarray, int]:
    """Return (vertices, number_of_initial_seed_vertices)."""
    perimeter = float(cv2.arcLength(contour, True))
    eps = max(1.0, cfg.simplify_ratio * perimeter)
    n_initial = len(cv2.approxPolyDP(contour, eps, True))

    if not cfg.use_vertex or n_initial < 3:
        # baseline keeps the raster staircase: many vertices, jagged edges
        dense_eps = max(0.6, 0.10 * eps)
        poly = cv2.approxPolyDP(contour, dense_eps, True).reshape(-1, 2)
        return poly.astype(np.float32), n_initial

    return _polygonize(contour, cfg).astype(np.float32), n_initial


# --------------------------------------------------------------------------- #
# geometric regularization
# --------------------------------------------------------------------------- #
def dominant_orientation(points: np.ndarray) -> tuple[float, float]:
    """Return (theta_rad, support) with theta in [-pi/4, pi/4).

    ``support`` is the fraction of edge length that is axis-aligned after
    rotating by theta (mod 90 degrees), i.e. how "rectilinear" the ring is.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    good = lengths > 1e-6
    if good.sum() < 2:
        return 0.0, 0.0
    angles = np.arctan2(edges[good, 1], edges[good, 0])
    weights = lengths[good]
    # angles modulo 90 degrees -> circular mean over 4x the angle
    z = np.sum(weights * np.exp(1j * 4.0 * angles))
    if abs(z) < 1e-9:
        return 0.0, 0.0
    theta = float(np.angle(z) / 4.0)
    theta = (theta + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0

    rel = (angles - theta + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
    aligned = np.abs(rel) <= math.radians(8.0)
    support = float(weights[aligned].sum() / max(weights.sum(), 1e-9))
    return theta, support


def _rotate(points: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - center) @ rot.T + center


def regularize_polygon(points: np.ndarray, cfg: VectorConfig) -> tuple[np.ndarray, bool]:
    """Snap near-axis-aligned edges onto the dominant 0/90 degree frame."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts.astype(np.float32), False
    theta, support = dominant_orientation(pts)
    if support < cfg.rect_min_support:
        return pts.astype(np.float32), False

    center = pts.mean(axis=0)
    local = _rotate(pts, -theta, center)
    tol = math.tan(math.radians(cfg.rect_tolerance_deg))
    adjusted = local.copy()

    for _ in range(6):
        new = adjusted.copy()
        changed = False
        n = len(adjusted)
        for i in range(n):
            a, b = adjusted[i], adjusted[(i + 1) % n]
            dx, dy = b[0] - a[0], b[1] - a[1]
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            if abs(dy) <= tol * abs(dx):        # horizontal edge
                y = 0.5 * (a[1] + b[1])
                new[i, 1], new[(i + 1) % n, 1] = y, y
                changed = True
            elif abs(dx) <= tol * abs(dy):      # vertical edge
                x = 0.5 * (a[0] + b[0])
                new[i, 0], new[(i + 1) % n, 0] = x, x
                changed = True
        adjusted = new
        if not changed:
            break

    back = _rotate(adjusted, theta, center)
    back = _drop_collinear(back, angle_tol_deg=2.5)
    back = _dedupe(back, min_dist=1.5)
    if len(back) < 3 or not is_simple_polygon(back):
        return pts.astype(np.float32), False
    # reject a "regularised" outline that drifted away from the original shape
    # (thin slivers are the typical symptom on long diagonal buildings)
    if _polygon_area(back) < 0.85 * _polygon_area(pts):
        return pts.astype(np.float32), False
    if _ring_iou(back, pts) < 0.90:
        return pts.astype(np.float32), False
    return back.astype(np.float32), True


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def vectorize(
    prob: np.ndarray,
    edge_prob: np.ndarray | None = None,
    config: VectorConfig | None = None,
    image: np.ndarray | None = None,
) -> VectorResult:
    """Convert a building probability map into polygons.

    ``image`` (H,W,3 uint8) enables occlusion-aware completion: contour runs that
    follow trees or cast shadow are replaced by straight chords so the polygon
    keeps the building's true straight walls.
    """
    cfg = config or VectorConfig()
    result = VectorResult(config=cfg)
    t: dict[str, float] = {}

    t0 = time.perf_counter()
    prob = np.asarray(prob, dtype=np.float32)
    mask = (prob >= cfg.threshold).astype(np.uint8)
    if mask.sum() == 0:
        result.mask = mask
        result.timings = {"threshold": time.perf_counter() - t0}
        return result
    t["threshold"] = time.perf_counter() - t0

    occluders = _occluder_map(image, mask, cfg) if (cfg.use_completion and image is not None) else None

    t0 = time.perf_counter()
    cut = None
    edge = np.asarray(edge_prob, dtype=np.float32) if edge_prob is not None else None
    if cfg.use_boundary and cfg.use_edge_cut and edge is not None and float(edge.max()) > 1e-6:
        # the boundary head is only weakly calibrated after few epochs, so the
        # cut keeps the strongest responses instead of a fixed value
        thr = max(cfg.edge_threshold, float(np.quantile(edge, cfg.edge_quantile)))
        cut = cv2.dilate((edge >= thr).astype(np.uint8), _ellipse(3))
    first = (cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(cfg.open_kernel))
             if cfg.open_kernel > 1 else mask)
    instances, split = _separate_instances(first, cut, cfg)
    t["separation"] = time.perf_counter() - t0

    result.instances = len(instances)
    result.merged_blobs = split

    t_contour = t_vertex = t_geom = 0.0
    for instance, ox, oy in instances:
        t0 = time.perf_counter()
        contour = _dense_contour(instance)
        t_contour += time.perf_counter() - t0
        if contour is None or len(contour) < 3:
            continue

        if occluders is not None:
            pts_local = contour.reshape(-1, 2).astype(np.float32)
            pts_global = pts_local + np.array([ox, oy], dtype=np.float32)
            completed = _complete_across_occluders(pts_global, occluders, cfg)
            if len(completed) >= 3 and len(completed) != len(pts_local):
                contour = np.round(completed - np.array([ox, oy], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)

        t0 = time.perf_counter()
        points, n_initial = _contour_to_polygon(contour, cfg)
        t_vertex += time.perf_counter() - t0
        if len(points) < 3:
            continue
        points = points + np.array([ox, oy], dtype=np.float32)

        area = _polygon_area(points)
        if area < cfg.min_area:
            continue

        t0 = time.perf_counter()
        regularized = False
        final = points
        if cfg.use_geometry:
            final, regularized = regularize_polygon(points, cfg)
        t_geom += time.perf_counter() - t0
        if len(final) < 3 or not is_simple_polygon(final):
            final, regularized = points, False

        if cfg.use_notch_cut and occluders is not None and len(final) >= 5:
            from .occlusion_geometry import cut_occlusion_notches

            notched = cut_occlusion_notches(final, occluders.astype(np.uint8),
                                            tol_deg=cfg.notch_tol_deg,
                                            min_occluded=cfg.notch_min_occ)
            if len(notched) >= 3 and is_simple_polygon(notched):
                final = notched

        # safety net: a polygon must explain most of its instance, otherwise the
        # vertex/geometry stages thinned it out (fall back to the dense contour)
        inst_area = int(instance.sum())
        if inst_area > 0:
            probe = np.zeros_like(instance)
            cv2.fillPoly(probe, [np.round(final - np.array([ox, oy], np.float32)).astype(np.int32).reshape(-1, 1, 2)], 1)
            if int(probe.sum()) < 0.60 * inst_area:
                dense = _dense_polygon(contour, cfg)
                if len(dense) >= 3:
                    final = dense + np.array([ox, oy], dtype=np.float32)
                    regularized = False

        # local rasterisation of the polygon: cheap and keeps the mean
        # probability (confidence) restricted to its own footprint
        x0, y0 = max(0, int(np.floor(final[:, 0].min()))), max(0, int(np.floor(final[:, 1].min())))
        x1 = min(mask.shape[1], int(np.ceil(final[:, 0].max())) + 1)
        y1 = min(mask.shape[0], int(np.ceil(final[:, 1].max())) + 1)
        score = 0.0
        if x1 > x0 and y1 > y0:
            local = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(local, [np.round(final - np.array([x0, y0])).astype(np.int32).reshape(-1, 1, 2)], 1)
            patch = prob[y0:y1, x0:x1]
            inside = local > 0
            if inside.any():
                score = float(patch[inside].mean())
        result.polygons.append(BuildingPolygon(
            points=final,
            score=score,
            regularized=regularized,
            vertices_initial=n_initial,
            area=_polygon_area(final),
            perimeter=_polygon_perimeter(final),
        ))
        result.regularized += int(regularized)

    if cfg.use_bridge and occluders is not None and len(result.polygons) > 1:
        from .bridge import apply_bridges, find_bridges

        brs = find_bridges(result.polygons, occluders.astype(np.uint8), image,
                           max_gap=cfg.bridge_max_gap)
        if brs:
            result.polygons = apply_bridges(result.polygons, brs, mask.shape, cfg)
            result.bridges = len(brs)

    result.mask = mask
    result.timings = dict(t)
    result.timings["contour"] = t_contour
    result.timings["vertex"] = t_vertex
    result.timings["geometry"] = t_geom
    return result


# --------------------------------------------------------------------------- #
# rasterisation / export helpers
# --------------------------------------------------------------------------- #
def polygons_to_mask(polygons, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    contours = [np.round(np.asarray(p.points, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
                for p in polygons if len(p.points) >= 3]
    if contours:
        cv2.fillPoly(canvas, contours, 1)
    return canvas


def polygon_boundary_points(polygons, step: float = 1.0) -> np.ndarray:
    """Densely sample all polygon edges (for Chamfer / PoLiS distances)."""
    out = []
    for poly in polygons:
        pts = np.asarray(poly.points, dtype=np.float64)
        if len(pts) < 2:
            continue
        ring = np.vstack([pts, pts[:1]])
        for a, b in zip(ring[:-1], ring[1:]):
            length = float(np.linalg.norm(b - a))
            n = max(2, int(math.ceil(length / max(step, 1e-6))) + 1)
            ts = np.linspace(0.0, 1.0, n)
            out.append(a[None, :] + ts[:, None] * (b - a)[None, :])
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(out)


def _round_points(points: np.ndarray, ndigits: int = 2) -> list[list[float]]:
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in points]


def write_geojson(path: str | Path, polygons: list[BuildingPolygon], meta: dict | None = None) -> Path:
    """Write polygons as a WGS84-less GeoJSON FeatureCollection (pixel CRS)."""
    path = Path(path)
    features = []
    for i, poly in enumerate(polygons, start=1):
        ring = _round_points(poly.points)
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append({
            "type": "Feature",
            "id": i,
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {
                "fid": i,
                "area_px": round(float(poly.area), 2),
                "perimeter_px": round(float(poly.perimeter), 2),
                "vertices": int(poly.num_vertices),
                "score": round(float(poly.score), 4),
                "regularized": bool(poly.regularized),
            },
        })
    payload = {
        "type": "FeatureCollection",
        "name": "building_footprints",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::0"}},
        "metadata": meta or {},
        "features": features,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
