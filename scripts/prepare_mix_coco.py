"""Convert WHU-Mix (Vector) COCO annotations into per-image polygon records.

COCO ``segmentation`` can be
  * a flat list ``[x1, y1, x2, y2, ...]``          -> one polygon
  * a list of such lists                           -> several polygons
  * an RLE dict                                    -> decoded with pycocotools if present

Output (one JSON per split)::

    {"images": [{"file": "...", "w": 512, "h": 512,
                 "polys": [[[x, y], ...], ...]}], "stats": {...}}

Usage::

    python scripts/prepare_mix_coco.py --src "G:/BISHE_data/whu_mix_vector/train" \
        --split train --out data/vertex_labels
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_annotation_files(src: Path) -> list[Path]:
    out = []
    for p in sorted(src.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            out.append(p)
    return out


def decode_rle(seg: dict) -> list[list[float]]:
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    counts = seg.get("counts")
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    m = mask_utils.decode({"size": seg["size"], "counts": counts})
    import cv2

    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(float).tolist())
    return polys


def parse_segmentation(seg, stats: Counter) -> list[list[list[float]]]:
    if isinstance(seg, dict):
        polys = decode_rle(seg)
        stats["rle"] += 1
        return polys
    if isinstance(seg, list) and seg:
        if all(isinstance(v, (int, float)) for v in seg):
            pts = np.asarray(seg, dtype=float).reshape(-1, 2).tolist()
            return [pts]
        out = []
        for part in seg:
            if isinstance(part, list) and len(part) >= 6 and all(isinstance(v, (int, float)) for v in part):
                out.append(np.asarray(part, dtype=float).reshape(-1, 2).tolist())
        return out
    stats["empty"] += 1
    return []


def polygon_area(poly) -> float:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="目录：含 annotations/*.json 与 images/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=str(ROOT / "data" / "vertex_labels"))
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--min-vertices", type=int, default=4)
    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_files = find_annotation_files(src)
    if not ann_files:
        print(f"在 {src} 下没找到 COCO 标注（含 annotations/images 的 json）")
        return 2
    print(f"找到 {len(ann_files)} 个标注文件：{[p.name for p in ann_files]}")

    records: list[dict] = []
    stats: Counter = Counter()
    vert_hist: Counter = Counter()
    for ann_path in ann_files:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        images = {im["id"]: im for im in data.get("images", [])}
        by_img: dict[int, list] = {i: [] for i in images}
        for ann in data.get("annotations", []):
            if ann.get("iscrowd"):
                stats["crowd"] += 1
                continue
            polys = parse_segmentation(ann.get("segmentation"), stats)
            for poly in polys:
                if len(poly) < args.min_vertices or polygon_area(poly) < args.min_area:
                    stats["small"] += 1
                    continue
                by_img.setdefault(ann["image_id"], []).append(poly)
        for img_id, polys in by_img.items():
            im = images.get(img_id)
            if im is None or not polys:
                continue
            records.append({
                "file": im.get("file_name", ""),
                "w": int(im.get("width", 0)),
                "h": int(im.get("height", 0)),
                "polys": [[[round(float(x), 2), round(float(y), 2)] for x, y in poly] for poly in polys],
            })
            for poly in polys:
                vert_hist[len(poly)] += 1
            stats["images"] += 1
            stats["buildings"] += len(polys)

    target = out_dir / f"{args.split}.json"
    payload = {"split": args.split, "source": str(src), "stats": dict(stats),
               "vertex_hist": dict(sorted(vert_hist.items())), "images": records}
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    n_img = len(records)
    n_b = sum(len(r["polys"]) for r in records)
    per_img = n_b / max(n_img, 1)
    vtx = [v for v, c in vert_hist.items() for _ in range(c)]
    print(f"输出 {target}（{target.stat().st_size / 1e6:.1f} MB）")
    print(f"图像 {n_img}，建筑 {n_b}，平均每图 {per_img:.1f} 栋")
    if vtx:
        print(f"顶点数：中位 {np.median(vtx):.0f}，90 分位 {np.percentile(vtx, 90):.0f}，最大 {max(vtx)}")
    print(f"统计：{dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
