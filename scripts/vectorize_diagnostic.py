"""Diagnostic: render the four configurations on real tiles (edge branch included)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.infer import InferenceEngine  # noqa: E402
from src.vector_metrics import evaluate_vectors  # noqa: E402
from src.vectorize import VectorConfig, vectorize  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONFIGS = [
    ("A 基线", dict(use_boundary=False, use_vertex=False, use_geometry=False)),
    ("B +边界分支", dict(use_boundary=True, use_vertex=False, use_geometry=False)),
    ("C +边界+顶点", dict(use_boundary=True, use_vertex=True, use_geometry=False)),
    ("D +边界+顶点+几何", dict(use_boundary=True, use_vertex=True, use_geometry=True)),
]


def main() -> int:
    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    gts = (np.load(ROOT / "data" / "processed" / "test_masks.npy") > 0).astype(np.uint8)
    audit = json.loads((ROOT / "results" / "dataset_audit.json").read_text(encoding="utf-8"))
    reps = audit["test"]["representative_indices"]
    engine = InferenceEngine(ROOT)
    idx = [int(reps[k]) for k in ("dense", "shadow", "small", "clutter")]

    out = ROOT / "results" / "geometry"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(CONFIGS), 2, figsize=(9.6, 15.5))
    for r, (name, overrides) in enumerate(CONFIGS):
        for c, i in enumerate(idx[:2]):
            t0 = time.perf_counter()
            prob, edge, t_inf = engine.predict_timed(images[i], "full", want_edge=True)
            res = vectorize(prob, edge, VectorConfig(**overrides), image=images[i])
            m = evaluate_vectors(res.polygons, res.mask, gts[i])
            ax = axes[r][c]
            canvas = images[i].copy()
            gb = cv2.dilate(gts[i], np.ones((3, 3), np.uint8)) - cv2.erode(gts[i], np.ones((3, 3), np.uint8))
            canvas[gb > 0] = [40, 210, 90]
            for poly in res.polygons:
                cv2.polylines(canvas, [np.round(poly.points).astype(np.int32)], True, (255, 214, 10), 1, cv2.LINE_AA)
                for x, y in poly.points:
                    cv2.circle(canvas, (int(x), int(y)), 1, (255, 60, 50), -1, cv2.LINE_AA)
            ax.imshow(canvas)
            ax.set_title(f"{name} | tile {i}\npolyIoU {m['polygon_iou']:.3f}  chamfer {m['chamfer']:.2f}  "
                         f"BIoU {m['boundary_iou']:.3f}  n={m['num_polygons']}  v={m['mean_vertices']:.1f}",
                         fontsize=8)
            ax.axis("off")
            print(f"{name:16s} tile {i:3d} n={m['num_polygons']:2d} polyIoU={m['polygon_iou']:.4f} "
                  f"chamfer={m['chamfer']:.3f} biou={m['boundary_iou']:.4f} "
                  f"vtx={m['mean_vertices']:.2f} dist={res.merged_blobs} reg={res.regularized} "
                  f"infer={t_inf*1000:.0f}ms")
    fig.tight_layout()
    fig.savefig(out / "pipeline_diagnostic.png", dpi=150)
    print("saved", out / "pipeline_diagnostic.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
