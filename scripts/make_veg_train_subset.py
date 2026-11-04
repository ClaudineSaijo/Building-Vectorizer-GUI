"""Find training tiles that contain buildings under tree canopy.

Trees over roofs are rare in WHU (about 0.9% of the building area), so the
network sees very few such examples per epoch.  This script scores every training
tile by how much of its building area is covered by vegetation, and keeps the
top ones so they can be oversampled during training.

    python scripts/make_veg_train_subset.py --top 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--top", type=int, default=1200)
    parser.add_argument("--min-ratio", type=float, default=0.01)
    parser.add_argument("--out", default=str(ROOT / "data" / "veg_train_subset.json"))
    args = parser.parse_args()

    imgs = np.load(Path(args.data) / "train_images.npy", mmap_mode="r")
    ms = np.load(Path(args.data) / "train_masks.npy", mmap_mode="r")

    rows = []
    for i in range(len(imgs)):
        gt = np.asarray(ms[i]) > 0
        if gt.sum() < 500:
            continue
        img = np.asarray(imgs[i]).astype(np.float32)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        exg = 2.0 * g - r - b
        veg = (exg > 30) & (g > r + 5)
        ratio = float((gt & veg).sum()) / float(gt.sum())
        if ratio < args.min_ratio:
            continue
        rows.append({"index": i, "veg_ratio": round(ratio, 4), "area": int(gt.sum())})

    rows.sort(key=lambda r: -r["veg_ratio"])
    picked = rows[: args.top]
    Path(args.out).write_text(json.dumps({"n": len(picked), "source": "whu_processed/train",
                                          "criteria": "vegetation pixels inside building GT",
                                          "total_candidates": len(rows), "tiles": picked},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    ratios = [t["veg_ratio"] for t in picked]
    print(f"候选 {len(rows)} 张，保留 {len(picked)} 张 -> {args.out}")
    if ratios:
        print(f"植被遮挡比例：最大 {max(ratios):.3f} 中位 {np.median(ratios):.3f} 最小 {min(ratios):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
