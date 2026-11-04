"""Vector-first evaluation on the WHU test split.

Instead of reporting only pixel mIoU, this walks the geometry pipeline one
switch at a time and reports vector-level metrics, which is what the thesis is
actually about:

    A  mask only                  (no boundary / vertex / geometry / completion)
    B  + boundary separation
    C  + vertex regression
    D  + geometric regularisation
    E  + occlusion completion

Metrics: polygon IoU, matched Chamfer, PoLiS, vertices, rectilinearity,
self-intersection rate, merge rate, instance recall.

    python scripts/evaluate_vectors_whu.py --samples 80
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.infer import InferenceEngine  # noqa: E402
from src.vector_metrics import evaluate_vectors  # noqa: E402
from src.vectorize import VectorConfig, is_simple_polygon, vectorize  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONFIGS = [
    ("A 仅掩膜", dict(use_boundary=False, use_vertex=False, use_geometry=False, use_completion=False)),
    ("B +边界分离", dict(use_boundary=True, use_vertex=False, use_geometry=False, use_completion=False)),
    ("C +顶点回归", dict(use_boundary=True, use_vertex=True, use_geometry=False, use_completion=False)),
    ("D +几何规则化", dict(use_boundary=True, use_vertex=True, use_geometry=True, use_completion=False)),
    ("E +遮挡补全", dict(use_boundary=True, use_vertex=True, use_geometry=True, use_completion=True)),
]

KEYS = [("polygon_iou", "多边形IoU", 4), ("chamfer_matched", "Chamfer/px", 2),
        ("polis", "PoLiS/px", 2), ("mean_vertices", "平均顶点", 1),
        ("right_angle_rate", "直角率", 3), ("area_bias", "面积偏差", 3),
        ("topology_valid", "拓扑有效", 3), ("merge_rate", "粘连率", 3)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="whu_cpu_unet")
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--use-edge", action="store_true",
                        help="边界分支模型：把预测的边界图送入实例分离（否则用形态学回退）")
    parser.add_argument("--out", default=str(ROOT / "results" / "vector_eval"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = np.load(Path(args.data) / f"{args.split}_images.npy", mmap_mode="r")
    masks = np.load(Path(args.data) / f"{args.split}_masks.npy", mmap_mode="r")
    n = len(images)
    idx = np.unique(np.linspace(0, n - 1, min(max(args.samples, 2), n)).round().astype(int))

    engine = InferenceEngine(ROOT)
    want_edge = bool(args.use_edge and engine.has_edge(args.model))
    probs, edges, gts = [], [], []
    for k, i in enumerate(idx):
        prob, edge = engine.predict(np.asarray(images[i]), args.model, want_edge=want_edge)
        probs.append(prob)
        edges.append(edge)
        gts.append((np.asarray(masks[i]) > 0).astype(np.uint8))
        if (k + 1) % 20 == 0:
            print(f"  inference {k + 1}/{len(idx)}", flush=True)

    edge_stats = {}
    if want_edge and any(e is not None for e in edges):
        import cv2

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bands = [((cv2.dilate(g, k3) - cv2.erode(g, k3)) > 0) for g in gts]
        best = (-1.0, 0.5)
        for thr in np.arange(0.30, 0.81, 0.05):
            inter = fp = fn = 0
            for e, band in zip(edges, bands):
                p = e >= thr
                inter += int((p & band).sum())
                fp += int((p & ~band).sum())
                fn += int((~p & band).sum())
            f1 = 2 * inter / max(2 * inter + fp + fn, 1)
            if f1 > best[0]:
                best = (f1, float(thr))
        edge_stats = {"edge_f1": round(best[0], 4), "edge_threshold": best[1]}
        print(f"  边界分支：F1={best[0]:.4f} @ thr {best[1]:.2f}", flush=True)

    rows = []
    for name, over in CONFIGS:
        cfg = VectorConfig(threshold=args.threshold, **over)
        acc = {key: [] for key, _, _ in KEYS}
        for prob, edge, gt, i in zip(probs, edges, gts, idx):
            res = vectorize(prob, edge if cfg.use_boundary else None, cfg, image=np.asarray(images[i]))
            m = evaluate_vectors(res.polygons, res.mask, gt)
            for key, _, _ in KEYS:
                if key in m:
                    acc[key].append(m[key])
            acc.setdefault("self_intersect", []).append(
                float(np.mean([0.0 if is_simple_polygon(p.points) else 1.0 for p in res.polygons]))
                if res.polygons else 0.0)
        row = {"config": name}
        for key, label, digits in KEYS:
            vals = [v for v in acc[key] if v is not None and np.isfinite(v)]
            row[key] = round(float(np.mean(vals)), digits) if vals else float("nan")
        rows.append(row)
        print(f"  [{name}] " + "  ".join(f"{label}={row[key]:.{digits}f}" for key, label, digits in KEYS), flush=True)

    base_vertices = rows[0]["mean_vertices"] if rows else float("nan")
    for row in rows:
        row["vertex_reduction"] = round(1.0 - row["mean_vertices"] / base_vertices, 3) if base_vertices else float("nan")

    csv_path = out_dir / "vector_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config"] + [key for key, _, _ in KEYS] + ["vertex_reduction"])
        for row in rows:
            writer.writerow([row["config"]] + [f"{row[key]:.4f}" for key, _, _ in KEYS]
                            + [f"{row['vertex_reduction']:.3f}"])
    (out_dir / "vector_eval.json").write_text(
        json.dumps({"model": args.model, "split": args.split, "samples": len(idx),
                    "threshold": args.threshold, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    labels = [c[0].split()[0] for c in CONFIGS]
    for ax, (key, title) in zip(axes, [("polygon_iou", "多边形 IoU ↑"),
                                       ("chamfer_matched", "Chamfer 距离 / px ↓"),
                                       ("mean_vertices", "平均顶点数 ↓")]):
        vals = [row[key] for row in rows]
        ax.bar(labels, vals, color="#219ebc")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for k, v in enumerate(vals):
            ax.text(k, v, f"{v:.3f}" if v < 10 else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "vector_eval.png", dpi=170, bbox_inches="tight")

    print(f"\nwrote {csv_path}\nwrote {out_dir / 'vector_eval.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
