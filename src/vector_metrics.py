"""Vector-level evaluation: polygon IoU, Chamfer distance, PoLiS.

Pixel metrics (mIoU / Boundary IoU) are delegated to :mod:`src.metrics` so the
GUI numbers stay comparable with ``scripts/evaluate.py``.
"""
from __future__ import annotations

import cv2
import numpy as np

from .metrics import boundary_iou, component_report, pixel_stats, scores_from_stats
from .vectorize import dominant_orientation, is_simple_polygon, polygons_to_mask

__all__ = ["evaluate_vectors", "chamfer_distance", "polis_distance", "gt_boundary_points"]


def gt_boundary_points(mask: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Dense points along the inner boundary of a binary mask."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float64)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float64)
        if len(pts) < 2:
            continue
        n = max(2, int(np.ceil(len(pts) / max(step, 1e-6))))
        idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
        out.append(pts[idx])
    return np.vstack(out) if out else np.zeros((0, 2), dtype=np.float64)


def _nearest(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point to the closest reference point (cKDTree)."""
    if len(reference) == 0 or len(points) == 0:
        return np.full(len(points), np.nan)
    from scipy.spatial import cKDTree

    dist, _ = cKDTree(reference).query(points, k=1)
    return dist.astype(np.float64)


def chamfer_distance(pred_boundary: np.ndarray, gt_boundary: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance in pixels."""
    if len(pred_boundary) == 0 or len(gt_boundary) == 0:
        return float("nan")
    d1 = _nearest(pred_boundary, gt_boundary)
    d2 = _nearest(gt_boundary, pred_boundary)
    return float(0.5 * (np.mean(d1) + np.mean(d2)))


def polis_distance(polygons: list, gt_boundary: np.ndarray) -> float:
    """PoLiS distance between polygon vertices and the GT boundary."""
    verts = [np.asarray(p.points, dtype=np.float64) for p in polygons if len(p.points) >= 3]
    if not verts or len(gt_boundary) == 0:
        return float("nan")
    verts = np.vstack(verts)
    d1 = _nearest(verts, gt_boundary)
    d2 = _nearest(gt_boundary, verts)
    return float(0.5 * (np.mean(d1) + np.mean(d2)))


def _component_matching(pred_mask: np.ndarray, gt: np.ndarray) -> dict:
    """Match GT components to predicted components by overlap.

    A GT building is matched when a single predicted instance covers at least
    half of it.  Matching is used for the geometric (Chamfer / PoLiS) metrics so
    that one completely missed building cannot dominate a whole tile's distance.
    """
    gt = (gt > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)
    num_g, gl, gs, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
    num_p, pl, ps, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    matched_gt = np.zeros(num_g, dtype=bool)
    matched_pred = np.zeros(max(num_p, 1), dtype=bool)
    if num_g <= 1 or num_p <= 1:
        return {"matched_gt": matched_gt, "matched_pred": matched_pred,
                "num_gt": max(num_g - 1, 0), "num_pred": max(num_p - 1, 0)}

    sel = (gt > 0) & (pred > 0)
    if not sel.any():
        return {"matched_gt": matched_gt, "matched_pred": matched_pred,
                "num_gt": num_g - 1, "num_pred": num_p - 1}

    g = gl[sel].astype(np.int64)
    p = pl[sel].astype(np.int64)
    codes, counts = np.unique(g * num_p + p, return_counts=True)
    g_of = codes // num_p
    p_of = codes % num_p
    order = np.lexsort((-counts, g_of))
    seen = set()
    for k in order:
        gi = int(g_of[k])
        if gi in seen:
            continue
        seen.add(gi)
        pj = int(p_of[k])
        area = int(gs[gi, cv2.CC_STAT_AREA])
        if counts[k] / max(area, 1) >= 0.5:
            matched_gt[gi] = True
            matched_pred[pj] = True

    gt_keep = matched_gt[gl] & (gt > 0)
    pred_keep = matched_pred[pl] & (pred > 0)
    return {
        "matched_gt": matched_gt,
        "matched_pred": matched_pred,
        "gt_keep": gt_keep,
        "pred_keep": pred_keep,
        "num_gt": num_g - 1,
        "num_pred": num_p - 1,
    }


def matched_chamfer(pred_mask: np.ndarray, gt: np.ndarray, match: dict | None = None) -> float:
    """Symmetric mean boundary distance over matched instances only."""
    match = match or _component_matching(pred_mask, gt)
    if "gt_keep" not in match or not match["gt_keep"].any() or not match["pred_keep"].any():
        return float("nan")
    gt = (gt > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gb = ((cv2.dilate(gt, k) - cv2.erode(gt, k)) > 0)
    pb = ((cv2.dilate(pred, k) - cv2.erode(pred, k)) > 0)
    dist_to_gt = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 5)
    dist_to_pred = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = dist_to_gt[pb & match["pred_keep"]]
    d2 = dist_to_pred[gb & match["gt_keep"]]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float(0.5 * (d1.mean() + d2.mean()))


def right_angle_rate(polygons: list, tol_deg: float = 5.0) -> float:
    """Length-weighted share of edges that are within ``tol_deg`` of 0/90 degrees.

    The dominant orientation is estimated per ring, then every edge is measured
    against that frame.  This is the strict version of ``rectilinearity`` (which
    uses an 8-degree band and no length weighting).
    """
    lengths_all = []
    aligned_all = []
    for poly in polygons:
        pts = np.asarray(poly.points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        theta, _ = dominant_orientation(pts)
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.linalg.norm(edges, axis=1)
        good = lengths > 1e-6
        if not good.any():
            continue
        ang = np.arctan2(edges[good, 1], edges[good, 0])
        rel = (ang - theta + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        lengths_all.append(lengths[good])
        aligned_all.append(np.abs(rel) <= np.radians(tol_deg))
    if not lengths_all:
        return float("nan")
    lengths = np.concatenate(lengths_all)
    aligned = np.concatenate(aligned_all)
    return float(lengths[aligned].sum() / max(lengths.sum(), 1e-9))


def topology_report(polygons: list) -> dict:
    """Validity of the vector output: simple rings, positive area, no overlap."""
    if not polygons:
        return {"valid_rate": float("nan"), "self_intersections": 0, "degenerate": 0}
    simple = sum(1 for p in polygons if is_simple_polygon(p.points))
    positive = sum(1 for p in polygons if p.area > 0)
    valid = sum(1 for p in polygons if is_simple_polygon(p.points) and p.area > 0)
    return {
        "valid_rate": valid / len(polygons),
        "self_intersections": len(polygons) - simple,
        "degenerate": len(polygons) - positive,
    }


def evaluate_vectors(polygons: list, pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """All metrics shown in the GUI's metric panel."""
    gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
    shape = gt.shape

    tp, fp, fn, _ = pixel_stats(pred_mask, gt)
    pixel = scores_from_stats(tp, fp, fn)

    poly_mask = polygons_to_mask(polygons, shape)
    tp_p, fp_p, fn_p, _ = pixel_stats(poly_mask, gt)
    poly = scores_from_stats(tp_p, fp_p, fn_p)

    gt_pts = gt_boundary_points(gt)
    pred_pts = np.vstack([np.asarray(p.points, dtype=np.float64) for p in polygons]) if polygons else np.zeros((0, 2))
    # dense sampling of polygon edges for Chamfer
    from .vectorize import polygon_boundary_points

    dense = polygon_boundary_points(polygons, step=1.0)

    rect = [dominant_orientation(p.points)[1] for p in polygons if len(p.points) >= 3]
    comp = component_report(poly_mask, gt, small_area=100)
    match = _component_matching(poly_mask, gt)
    topo = topology_report(polygons)
    gt_area = float(gt.sum())
    poly_area = float(poly_mask.sum())
    area_bias = float((poly_area - gt_area) / gt_area) if gt_area >= 200.0 else float("nan")

    return {
        "mIoU": float(pixel["iou"]),
        "f1": float(pixel["f1"]),
        "precision": float(pixel["precision"]),
        "recall": float(pixel["recall"]),
        "boundary_iou": float(boundary_iou(pred_mask, gt, width=2)),
        "polygon_iou": float(poly["iou"]),
        "chamfer": chamfer_distance(dense, gt_pts),
        "chamfer_matched": matched_chamfer(poly_mask, gt, match),
        "polis": polis_distance(polygons, gt_pts),
        "num_polygons": int(len(polygons)),
        "num_vertices": int(len(pred_pts)),
        "mean_vertices": float(np.mean([len(p.points) for p in polygons])) if polygons else 0.0,
        "mean_score": float(np.mean([p.score for p in polygons])) if polygons else 0.0,
        "rectilinearity": float(np.mean(rect)) if rect else float("nan"),
        "right_angle_rate": right_angle_rate(polygons),
        "topology_valid": float(topo["valid_rate"]),
        "self_intersections": int(topo["self_intersections"]),
        "degenerate": int(topo["degenerate"]),
        "area_bias": area_bias,
        "merge_rate": float(comp["merge_rate"]),
        "instance_recall": float(comp["gt_recall"]),
        "gt_count": int(comp["num_gt"]),
        "instance_precision": float(match["matched_pred"].sum() / max(match["num_pred"], 1)),
        "matched_small": int(match["matched_gt"].sum()),
    }
