"""Pixel / boundary / instance-level metrics for building extraction."""
from __future__ import annotations

import cv2
import numpy as np


def _as_binary(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.uint8)


def pixel_stats(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred = _as_binary(pred)
    gt = _as_binary(gt)
    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())
    return tp, fp, fn, tn


def scores_from_stats(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def boundary_iou(pred: np.ndarray, gt: np.ndarray, width: int = 2) -> float:
    """Boundary IoU (Cheng et al., 2021) with an elliptical band.

    A band of `2*width+1` around each mask boundary is extracted, then the
    intersection-over-union of the two bands is computed.  Images without any
    foreground are skipped (returns NaN).
    """
    pred = _as_binary(pred)
    gt = _as_binary(gt)
    if gt.sum() == 0:
        return float("nan")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    gd = cv2.dilate(gt, kernel) - cv2.erode(gt, kernel)
    pd = cv2.dilate(pred, kernel) - cv2.erode(pred, kernel)
    inter = int(np.logical_and(gd > 0, pd > 0).sum())
    union = int(np.logical_or(gd > 0, pd > 0).sum())
    return inter / max(union, 1)


def component_report(pred: np.ndarray, gt: np.ndarray, small_area: int = 100) -> dict:
    """Instance-level recall, small-building recall, and merge (adhesion) rate.

    A GT component is matched when its best-overlapping predicted component
    covers at least 50% of the GT component.  A merge/adhesion event is counted
    when two or more matched GT components share the same predicted component.
    """
    pred = _as_binary(pred)
    gt = _as_binary(gt)
    num_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt, connectivity=8)
    num_pred, pred_labels, _, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
    gt_areas, gt_ids, matched, merged = [], [], [], []
    pred_to_gts: dict[int, list[int]] = {}

    for gi in range(1, num_gt):
        area = int(gt_stats[gi, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = gt_labels == gi
        overlaps = np.bincount(pred_labels[comp].ravel(), minlength=num_pred)
        overlaps[0] = 0
        pj = int(overlaps.argmax())
        best = int(overlaps[pj])
        hit = best / area >= 0.5
        gt_areas.append(area)
        gt_ids.append(gi)
        matched.append(hit)
        if hit:
            pred_to_gts.setdefault(pj, []).append(gi)
            merged.append(False)
        else:
            merged.append(False)

    merged_set = set()
    for pj, gis in pred_to_gts.items():
        if len(gis) > 1:
            merged_set.update(gis)
    for idx, gi in enumerate(gt_ids):
        merged[idx] = gi in merged_set

    gt_areas_arr = np.asarray(gt_areas, dtype=np.int64)
    matched_arr = np.asarray(matched, dtype=bool)
    merged_arr = np.asarray(merged, dtype=bool)
    small = gt_areas_arr < small_area
    large = gt_areas_arr >= small_area
    return {
        "num_gt": int(len(gt_areas_arr)),
        "num_pred": int(num_pred - 1),
        "gt_matched": int(matched_arr.sum()),
        "gt_recall": float(matched_arr.mean()) if len(matched_arr) else float("nan"),
        "small_num": int(small.sum()),
        "small_matched": int(matched_arr[small].sum()),
        "small_recall": float(matched_arr[small].mean()) if small.any() else float("nan"),
        "large_recall": float(matched_arr[large].mean()) if large.any() else float("nan"),
        "merge_count": int(merged_arr.sum()),
        "merge_rate": float(merged_arr.mean()) if len(merged_arr) else float("nan"),
        "mean_gt_area": float(gt_areas_arr.mean()) if len(gt_areas_arr) else float("nan"),
    }
