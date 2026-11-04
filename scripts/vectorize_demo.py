"""Pilot: from pixel masks to vector polygons, with geometry-aware metrics.

This is a diagnostic baseline for the "geometry + semantics" direction, not a
learned polygon predictor.  It shows how much geometry is lost by naive
contour extraction and establishes the evaluation protocol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCENES = {"shadow": "弱光照/阴影", "dense": "密集建筑", "small": "小目标", "clutter": "复杂背景"}


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> float:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pb = cv2.dilate(pred, k) - cv2.erode(pred, k)
    gb = cv2.dilate(gt, k) - cv2.erode(gt, k)
    if pb.sum() == 0 or gb.sum() == 0:
        return float("nan")
    # Distance transform: distance to the nearest boundary pixel
    dist_to_pred = cv2.distanceTransform(1 - pb.astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_gt = cv2.distanceTransform(1 - gb.astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dist_to_gt[pb > 0] <= tol).mean())
    recall = float((dist_to_pred[gb > 0] <= tol).mean())
    return 2 * precision * recall / max(precision + recall, 1e-9)


def vectorize(mask: np.ndarray, min_area: float = 20, eps_ratio: float = 0.012):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = max(1.0, eps_ratio * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) >= 3:
            polys.append(approx.reshape(-1, 2))
    canvas = np.zeros_like(mask, dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(canvas, [poly], 1)
    return polys, canvas


def main() -> int:
    out_dir = ensure_dir(ROOT / "results" / "geometry")
    probs = np.load(ROOT / "results" / "probs" / "full_test.npy").astype(np.float32)
    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    masks = (np.load(ROOT / "data" / "processed" / "test_masks.npy") > 0).astype(np.uint8)
    audit = json.loads((ROOT / "results" / "dataset_audit.json").read_text(encoding="utf-8"))
    reps = audit["test"]["representative_indices"]

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    for ax, (key, label) in zip(axes.ravel(), SCENES.items()):
        idx = int(reps[key])
        gt = masks[idx]
        pred = (probs[idx] >= 0.5).astype(np.uint8)
        polys, poly_mask = vectorize(pred)
        inter = np.logical_and(poly_mask, gt).sum()
        union = np.logical_or(poly_mask, gt).sum()
        p_iou = inter / max(union, 1)
        pix_inter = np.logical_and(pred, gt).sum()
        pix_union = np.logical_or(pred, gt).sum()
        pix_iou = pix_inter / max(pix_union, 1)
        bf1 = boundary_f1(pred, gt)
        vertices = [len(p) for p in polys]
        rows.append({
            "scene": key, "index": idx,
            "pixel_iou": round(float(pix_iou), 4),
            "polygon_iou": round(float(p_iou), 4),
            "boundary_f1": round(float(bf1), 4),
            "num_polygons": len(polys),
            "mean_vertices": round(float(np.mean(vertices)), 2) if vertices else 0,
            "simplification_loss": round(float(pix_iou - p_iou), 4),
        })

        canvas = images[idx].copy()
        # Ground-truth boundary (green)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gb = cv2.dilate(gt, k) - cv2.erode(gt, k)
        canvas[gb > 0] = [40, 200, 90]
        # Predicted polygons (yellow) + vertices (red)
        for poly in polys:
            cv2.polylines(canvas, [poly], True, (255, 220, 40), 1, cv2.LINE_AA)
            for x, y in poly:
                cv2.circle(canvas, (int(x), int(y)), 1, (230, 60, 40), -1, cv2.LINE_AA)
        ax.imshow(canvas)
        ax.set_title(f"{label}  |  Pixel IoU {pix_iou:.3f}  Poly IoU {p_iou:.3f}  "
                     f"Boundary F1 {bf1:.3f}\n{len(polys)} 个多边形，平均 {np.mean(vertices) if vertices else 0:.1f} 个顶点",
                     fontsize=10)
        ax.axis("off")

    fig.suptitle("从像素掩膜到矢量多边形：几何—语义一体化输出初探\n"
                 "绿色为真值边界，黄色为简化后多边形，红点为顶点", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "vectorization_demo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    agg = {
        "method": "threshold -> connected contour -> approxPolyDP (eps=1.2% perimeter)",
        "note": "naive vectorization baseline; no learned polygon head",
        "mean_pixel_iou": round(float(np.mean([r["pixel_iou"] for r in rows])), 4),
        "mean_polygon_iou": round(float(np.mean([r["polygon_iou"] for r in rows])), 4),
        "mean_boundary_f1": round(float(np.nanmean([r["boundary_f1"] for r in rows])), 4),
        "mean_vertices_per_polygon": round(float(np.mean([r["mean_vertices"] for r in rows])), 2),
        "per_scene": rows,
    }
    (out_dir / "vector_metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
