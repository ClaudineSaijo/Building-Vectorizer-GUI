"""Rich error diagnosis for a segmentation checkpoint.

Answers "where does the error actually come from?" instead of only reporting mIoU.

Sections
--------
1. pixel level: mIoU / F1 / P / R
2. recall by ground-cover type: visible roof / cast shadow / vegetation
3. recall by building size bucket
4. false-positive composition (what the model wrongly calls building)
5. false-negative composition (what it misses)
6. boundary quality: boundary IoU and boundary F1
7. vector level: polygon IoU, Chamfer, vertices, area bias, fragmentation
8. confidence distribution on GT-positive pixels

    python scripts/diagnose.py --model whu_mt512 --limit 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import InferenceEngine  # noqa: E402
from src.vector_metrics import evaluate_vectors  # noqa: E402
from src.vectorize import VectorConfig, vectorize  # noqa: E402

SIZE_BINS = [(0, 200), (200, 500), (500, 1500), (1500, 5000), (5000, 10**9)]
SIZE_NAMES = ["<200", "200-500", "500-1.5k", "1.5k-5k", ">5k"]


def region_masks(img: np.ndarray) -> dict[str, np.ndarray]:
    rgb = img.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    med = float(np.median(gray))
    veg = (exg > 30) & (g > r + 5)
    shadow = (gray < 0.62 * med) & ~veg
    bright = gray > 1.05 * med
    return {"veg": veg, "shadow": shadow, "bright": bright}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    imgs = np.load(Path(args.data) / f"{args.split}_images.npy", mmap_mode="r")
    ms = np.load(Path(args.data) / f"{args.split}_masks.npy", mmap_mode="r")
    n = len(imgs) if args.limit <= 0 else min(args.limit, len(imgs))
    idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)

    engine = InferenceEngine(ROOT)
    cfg = VectorConfig(threshold=args.threshold)

    tp = fp = fn = 0
    region_hit = {k: 0 for k in ("visible", "shadow", "veg")}
    region_tot = {k: 0 for k in region_hit}
    size_hit = [0] * len(SIZE_BINS)
    size_tot = [0] * len(SIZE_BINS)
    fp_comp = {"veg": 0, "shadow": 0, "bright": 0, "other": 0}
    fn_comp = {"veg": 0, "shadow": 0, "other": 0}
    conf_edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
    conf_hist = np.zeros(len(conf_edges) - 1, dtype=np.int64)
    biou, bf1 = [], []
    poly_iou, cham, verts, area_bias = [], [], [], []
    per_tile = []

    for j, i in enumerate(idx):
        gt = (np.asarray(ms[i]) > 0).astype(np.uint8)
        if gt.sum() < 100:
            continue
        img = np.asarray(imgs[i])
        prob = engine.predict(img, args.model, want_edge=False)[0]
        pred = (prob >= cfg.threshold).astype(np.uint8)

        tp += int((pred & gt).sum()); fp += int((pred & ~gt.astype(bool)).sum())
        fn += int((~pred.astype(bool) & gt.astype(bool)).sum())

        reg = region_masks(img)
        for name, sel in (("visible", ~(reg["veg"] | reg["shadow"])),
                          ("shadow", reg["shadow"]), ("veg", reg["veg"])):
            m = (gt > 0) & sel
            region_tot[name] += int(m.sum()); region_hit[name] += int((pred > 0)[m].sum())

        nb, labels, stats, _ = cv2.connectedComponentsWithStats(gt, 8)
        for b_id in range(1, nb):
            area = int(stats[b_id, 4])
            bm = labels == b_id
            for k, (lo, hi) in enumerate(SIZE_BINS):
                if lo <= area < hi:
                    size_tot[k] += area; size_hit[k] += int((pred > 0)[bm].sum())
                    break

        fp_mask = (pred > 0) & (gt == 0)
        if fp_mask.any():
            fp_comp["veg"] += int((fp_mask & reg["veg"]).sum())
            fp_comp["shadow"] += int((fp_mask & reg["shadow"]).sum())
            fp_comp["bright"] += int((fp_mask & reg["bright"]).sum())
            fp_comp["other"] += int((fp_mask & ~(reg["veg"] | reg["shadow"] | reg["bright"])).sum())
        fn_mask = (pred == 0) & (gt > 0)
        if fn_mask.any():
            fn_comp["veg"] += int((fn_mask & reg["veg"]).sum())
            fn_comp["shadow"] += int((fn_mask & reg["shadow"]).sum())
            fn_comp["other"] += int((fn_mask & ~(reg["veg"] | reg["shadow"])).sum())

        positives = prob[gt > 0]
        hist, _ = np.histogram(positives, bins=conf_edges)
        conf_hist += hist

        # boundary + vector
        gt_pts = gt.astype(np.uint8)
        band = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gb = cv2.morphologyEx(gt_pts, cv2.MORPH_GRADIENT, band) > 0
        pb = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, band) > 0
        gb_d = cv2.dilate(gb.astype(np.uint8), band) > 0
        pb_d = cv2.dilate(pb.astype(np.uint8), band) > 0
        biou.append(((gb & pb_d).sum() + (pb & gb_d).sum()) / max((gb | pb).sum(), 1))
        rec = (gb & pb_d).sum() / max(gb.sum(), 1)
        pre = (pb & gb_d).sum() / max(pb.sum(), 1)
        bf1.append(2 * rec * pre / max(rec + pre, 1e-6))

        res = vectorize(prob, None, cfg, image=img)
        m = evaluate_vectors(res.polygons, res.mask, gt)
        poly_iou.append(m["polygon_iou"]); cham.append(m["chamfer_matched"])
        verts.append(m["mean_vertices"]); area_bias.append(m["area_bias"])
        per_tile.append({"tile": int(i) + 1, "mIoU": round(m["mIoU"], 4),
                         "polyIoU": round(m["polygon_iou"], 4), "polys": m["num_polygons"]})
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(idx)}", flush=True)

    inter = tp; union = tp + fp + fn
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    report = {
        "model": args.model, "tiles": len(poly_iou), "threshold": cfg.threshold,
        "pixel": {"iou": round(inter / max(union, 1), 4), "f1": round(2 * precision * recall / max(precision + recall, 1e-6), 4),
                  "precision": round(precision, 4), "recall": round(recall, 4)},
        "recall_by_region": {k: round(region_hit[k] / max(region_tot[k], 1), 4) for k in region_hit},
        "region_pixels": region_tot,
        "recall_by_size": {SIZE_NAMES[k]: round(size_hit[k] / max(size_tot[k], 1), 4) for k in range(len(SIZE_BINS))},
        "fp_composition": {k: round(v / max(sum(fp_comp.values()), 1), 3) for k, v in fp_comp.items()},
        "fn_composition": {k: round(v / max(sum(fn_comp.values()), 1), 3) for k, v in fn_comp.items()},
        "boundary": {"biou": round(float(np.mean(biou)), 4), "bf1": round(float(np.mean(bf1)), 4)},
        "vector": {"poly_iou": round(float(np.mean(poly_iou)), 4),
                   "chamfer": round(float(np.nanmean(cham)), 3),
                   "vertices": round(float(np.mean(verts)), 2),
                   "area_bias": round(float(np.nanmean(area_bias)), 4)},
        "confidence_on_gt": {f"{conf_edges[k]:.1f}-{conf_edges[k+1]:.1f}": round(float(conf_hist[k]) / max(conf_hist.sum(), 1), 3)
                             for k in range(len(conf_hist))},
        "worst_tiles": sorted(per_tile, key=lambda r: r["mIoU"])[:8],
    }

    print(f"\n[{args.model}] {report['tiles']} 张, 阈值 {cfg.threshold}")
    print(f"  像素: mIoU {report['pixel']['iou']:.4f}  F1 {report['pixel']['f1']:.4f}  P {report['pixel']['precision']:.3f}  R {report['pixel']['recall']:.3f}")
    print(f"  分区域召回: " + "  ".join(f"{k}={v:.3f}(像素{t})" for k, v in report["recall_by_region"].items() for t in [report["region_pixels"][k]]))
    print(f"  分尺寸召回: " + "  ".join(f"{k}={v:.3f}" for k, v in report["recall_by_size"].items()))
    print(f"  FP 构成: {report['fp_composition']}")
    print(f"  FN 构成: {report['fn_composition']}")
    print(f"  边界: BIoU {report['boundary']['biou']:.4f}  BF1 {report['boundary']['bf1']:.4f}")
    print(f"  矢量: polyIoU {report['vector']['poly_iou']:.4f}  Chamfer {report['vector']['chamfer']:.2f}  顶点 {report['vector']['vertices']:.1f}  面积偏差 {report['vector']['area_bias']:+.3f}")
    print(f"  GT 正样本置信度分布: {report['confidence_on_gt']}")

    out = args.out or str(ROOT / "results" / f"diagnose_{args.model}.json")
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
