"""Build an occlusion subset from the WHU test split.

For every test tile we measure how much of the *building boundary* is adjacent to
vegetation or cast shadow.  Tiles whose buildings are heavily occluded are the
ones the occlusion experiments should be judged on.

Output:
  data/occlusion_subset.json   ranked list of test indices + scores
  results/occlusion_sheet.png  contact sheet for a quick visual check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def occluders(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    veg = (exg > 30) & (g > r + 5)
    shadow = gray < 0.62 * float(np.median(gray))
    return (veg | shadow).astype(np.uint8)


def main() -> int:
    n_out = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    imgs = np.load(ROOT / "data" / "whu_processed" / "test_images.npy", mmap_mode="r")
    ms = np.load(ROOT / "data" / "whu_processed" / "test_masks.npy", mmap_mode="r")

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rows = []
    for i in range(len(imgs)):
        gt = (np.asarray(ms[i]) > 0).astype(np.uint8)
        if gt.sum() < 800:
            continue
        img = np.asarray(imgs[i])
        rgb = img.astype(np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        exg = 2 * g - r - b
        veg = ((exg > 30) & (g > r + 5)).astype(np.uint8)
        # 树荫：暗且靠近植被（减少深色屋顶的误判）
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        med = float(np.median(gray))
        dark = (gray < 0.60 * med).astype(np.uint8)
        near_veg = cv2.dilate(veg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        tree_shadow = ((dark > 0) & near_veg).astype(np.uint8)
        occl = np.clip(veg + tree_shadow, 0, 1).astype(np.uint8)

        # 遮挡 = 落在建筑真值轮廓内部的植被像素（树盖在屋顶上）
        n, labels, stats, _ = cv2.connectedComponentsWithStats(gt, 8)
        ratios = []
        for j in range(1, n):
            area = int(stats[j, 4])
            if area < 800:
                continue
            m = labels == j
            occ = int((m & (occl > 0)).sum())
            ratios.append(occ / area)
        if not ratios:
            continue
        worst = max(ratios)
        mean_ratio = float(np.mean(ratios))
        if worst < 0.05:
            continue
        rows.append({"index": i, "score": round(worst, 4), "mean_building_occlusion": round(mean_ratio, 4),
                     "buildings": len(ratios), "area": int(gt.sum())})

    rows.sort(key=lambda r: -r["score"])
    # 分层抽样：轻度/中度/重度/极重各取一批，避免只挑最极端的
    bins = [(0.12, 0.25), (0.25, 0.45), (0.45, 0.70), (0.70, 1.01)]
    per_bin = max(n_out // len(bins), 4)
    picked = []
    for lo, hi in bins:
        chunk = [r for r in rows if lo <= r["score"] < hi][:per_bin]
        picked.extend(chunk)
    picked.sort(key=lambda r: -r["score"])
    # 固定加入参考样本 0153（index 152）
    if not any(r["index"] == 152 for r in picked):
        ref = next((r for r in rows if r["index"] == 152), None)
        if ref:
            picked.append(ref)
    out = ROOT / "data" / "occlusion_subset.json"
    out.write_text(json.dumps({"n": len(picked),
                               "criteria": "fraction of building footprint covered by vegetation (trees over roof)",
                               "tiles": picked}, ensure_ascii=False, indent=2), encoding="utf-8")

    cols, cell = 6, 256
    n_show = min(24, len(picked))
    sheet = np.zeros((cell * ((n_show + cols - 1) // cols), cell * cols, 3), np.uint8)
    for k, item in enumerate(picked[:n_show]):
        i = item["index"]
        img = np.asarray(imgs[i]).copy()
        gt = (np.asarray(ms[i]) > 0)
        rgb = img.astype(np.float32)
        exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
        veg = (exg > 30) & (rgb[..., 1] > rgb[..., 0] + 5)
        inside = gt & veg
        img[inside] = (255, 0, 255)
        edge = cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_GRADIENT, k3) > 0
        img[edge] = (255, 60, 60)
        small = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        cv2.putText(small, f"{i + 1:04d} occ={item['score']:.2f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, cols)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = small
    cv2.imwrite(str(ROOT / "results" / "occlusion_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    print(f"输出 {out}（{len(picked)} 张）")
    print("top10:", [(t["index"] + 1, t["score"]) for t in picked[:10]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
