"""Compare the EdgeConv vertex regressor with the geometric baseline.

Baseline : resampled contour -> Douglas-Peucker (epsilon = 1.2% perimeter)
Model    : resampled contour -> EdgeConv offsets -> merge/dedupe -> DP (0.4%)

Metrics against the GT polygon: rasterised IoU, symmetric vertex Chamfer,
vertex count.  Run after ``scripts/train_vertex.py``.

    python scripts/eval_vertex.py --model runs/vertex_gcn/best.pt --samples 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vertex_gcn import VertexGCN, node_features, resample_ring  # noqa: E402


def dp_simplify(pts, ratio: float, closed: bool = True) -> np.ndarray:
    contour = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, closed)
    eps = max(0.8, ratio * perimeter)
    approx = cv2.approxPolyDP(contour, eps, closed).reshape(-1, 2)
    return approx


def polygon_iou(a, b, shape) -> float:
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round(np.asarray(a)).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b)).astype(np.int32)], 1)
    inter = int((ma & mb).sum())
    union = int((ma | mb).sum())
    return inter / max(union, 1)


def chamfer_vertices(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def dedupe(pts, min_dist: float = 2.0) -> np.ndarray:
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) >= min_dist:
            kept.append(p)
    while len(kept) > 3 and np.linalg.norm(kept[-1] - kept[0]) < min_dist:
        kept.pop()
    return np.asarray(kept, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "vertex_labels"))
    parser.add_argument("--split", default="val.json")
    parser.add_argument("--model", default=str(ROOT / "runs" / "vertex_gcn" / "best.pt"))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--gcn-eps", type=float, default=0.004)
    parser.add_argument("--out", default=str(ROOT / "results" / "vertex_eval.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    ck_path = Path(args.model)
    if not ck_path.exists():
        print(f"找不到模型 {ck_path}")
        return 2
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = VertexGCN(in_dim=7, max_offset=float(ck.get("args", {}).get("max_offset", 64.0)))
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"加载 {ck_path}（epoch {ck.get('epoch')}, val {ck.get('val')}）")

    split_path = Path(args.data) / args.split
    if not split_path.exists():
        print(f"找不到 {split_path}")
        return 2
    records = json.loads(split_path.read_text(encoding="utf-8"))["images"]
    rng = np.random.default_rng(1)
    if len(records) > args.samples:
        records = [records[i] for i in rng.choice(len(records), size=args.samples, replace=False)]

    rows = []
    for rec in records:
        for poly in rec["polys"]:
            gt = np.asarray(poly, dtype=np.float32)
            if len(gt) < 4:
                continue
            pts = gt
            x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 3
            local = pts - np.array([x0, y0], np.float32)
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32) + np.array([x0, y0], np.float32)
            if len(contour) < 8:
                continue
            nodes = resample_ring(contour, args.nodes).astype(np.float32)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            feat = node_features(nodes, (rec["h"], rec["w"]), dist, dist_offset=(x0, y0))

            base = dp_simplify(nodes, 0.012)
            with torch.no_grad():
                ft = torch.from_numpy(feat)[None]
                co = torch.from_numpy(nodes)[None]
                off = model(ft.reshape(-1, feat.shape[1]), co.reshape(-1, 2)).reshape(co.shape)
                moved = (co + off)[0].numpy()
            moved = dedupe(moved)
            gcn_poly = dp_simplify(moved, args.gcn_eps) if len(moved) >= 3 else moved

            shape = mask.shape
            rows.append({
                "base_iou": polygon_iou(np.round(base - [x0, y0]), local, shape),
                "gcn_iou": polygon_iou(np.round(gcn_poly - [x0, y0]), local, shape),
                "base_chamfer": chamfer_vertices(base, gt),
                "gcn_chamfer": chamfer_vertices(gcn_poly, gt),
                "base_vtx": int(len(base)),
                "gcn_vtx": int(len(gcn_poly)),
            })

    if not rows:
        print("没有可评估的样本")
        return 2
    a = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    a["n"] = len(rows)
    print(f"样本 {a['n']}")
    print(f"  几何基线：IoU {a['base_iou']:.4f}  Chamfer {a['base_chamfer']:.2f}  顶点 {a['base_vtx']:.1f}")
    print(f"  GCN    ：IoU {a['gcn_iou']:.4f}  Chamfer {a['gcn_chamfer']:.2f}  顶点 {a['gcn_vtx']:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
