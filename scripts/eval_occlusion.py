"""Evaluate models on the occlusion subset (trees/shadows over building roofs).

Compares several checkpoints on the same tiles so that occlusion-oriented
modules (BIEB, direction field, ...) can be judged on the samples they target,
instead of on the overall test set.

    python scripts/eval_occlusion.py --models whu_mt512 whu_bieb --threshold 0.55
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import InferenceEngine  # noqa: E402
from src.vector_metrics import evaluate_vectors  # noqa: E402
from src.vectorize import VectorConfig, vectorize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=str(ROOT / "data" / "occlusion_subset.json"))
    parser.add_argument("--models", nargs="+", default=["whu_mt512", "whu_bieb"])
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--out", default=str(ROOT / "results" / "occlusion_eval.json"))
    parser.add_argument("--no-notch", action="store_true", help="关闭遮挡凹口裁剪（对照）")
    args = parser.parse_args()

    subset = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    indices = [t["index"] for t in subset["tiles"]]
    imgs = np.load(Path(args.data) / "test_images.npy", mmap_mode="r")
    ms = np.load(Path(args.data) / "test_masks.npy", mmap_mode="r")

    engine = InferenceEngine(ROOT)
    cfg = VectorConfig(threshold=args.threshold, use_notch_cut=not args.no_notch)
    report = {}
    for model in args.models:
        miou, poly, verts, polys = [], [], [], []
        for i in indices:
            gt = (np.asarray(ms[i]) > 0).astype(np.uint8)
            img = np.asarray(imgs[i])
            prob, edge = engine.predict(img, model, want_edge=True)
            res = vectorize(prob, edge, cfg, image=img)
            m = evaluate_vectors(res.polygons, res.mask, gt)
            miou.append(m["mIoU"])
            poly.append(m["polygon_iou"])
            verts.append(m["mean_vertices"])
            polys.append(m["num_polygons"])
        report[model] = {
            "tiles": len(indices),
            "mIoU": round(float(np.mean(miou)), 4),
            "polygon_iou": round(float(np.mean(poly)), 4),
            "mean_vertices": round(float(np.mean(verts)), 2),
            "polygons": round(float(np.mean(polys)), 1),
        }
        print(f"{model:16s} mIoU {report[model]['mIoU']:.4f} | polyIoU {report[model]['polygon_iou']:.4f} "
              f"| 顶点 {report[model]['mean_vertices']:.1f}")

    base = report.get(args.models[0], {}).get("mIoU")
    if base:
        print("\n相对第一个模型：")
        for model in args.models[1:]:
            delta = (report[model]["mIoU"] - base) * 100
            print(f"  {model:16s} {delta:+.2f} 点")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
