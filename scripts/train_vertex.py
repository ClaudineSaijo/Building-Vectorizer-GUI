"""Train the EdgeConv vertex regressor on WHU-Mix polygons.

Each sample is one building:
  * rasterise its GT polygon, take the outer contour and resample it to N nodes
    (this mimics the inference-time input, staircase and all),
  * features per node (normalised coordinates, tangent, radial direction,
    distance to boundary),
  * target = the GT polygon vertices.

Loss = Chamfer(prediction, GT vertices) + 0.5 * one-way pred->GT + 0.1 * smoothness.

    python scripts/train_vertex.py --data data/vertex_labels --epochs 80
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.vertex_gcn import VertexGCN, node_features, resample_ring  # noqa: E402


def build_samples(records, n_nodes: int, max_samples: int | None, seed: int = 0,
                  target: str = "vertices"):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    records = [records[i] for i in order]
    samples = []
    for rec in records:
        for poly in rec["polys"]:
            pts = np.asarray(poly, dtype=np.float32)
            if len(pts) < 4:
                continue
            x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 3
            local = pts - np.array([x0, y0], dtype=np.float32)
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(local).astype(np.int32)], 1)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            if len(contour) < 8:
                continue
            nodes = resample_ring(contour + np.array([x0, y0], np.float32), n_nodes).astype(np.float32)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
            feats = node_features(nodes, (rec["h"], rec["w"]), dist, dist_offset=(x0, y0))
            if target == "dense":
                tgt = align_ring(nodes, resample_ring(pts, n_nodes).astype(np.float32))
            else:
                tgt = pts
            samples.append({"feat": feats, "nodes": nodes, "target": tgt,
                            "shape": (rec["h"], rec["w"])})
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples



def align_ring(nodes: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Cyclically shift (and optionally reverse) `ring` to match `nodes`.

    The contour produced by OpenCV starts at an arbitrary point, so a dense
    target sampled from the GT polygon needs a phase/direction alignment before
    a per-node L1 loss makes sense.
    """
    n = len(ring)
    nodes = np.asarray(nodes, dtype=np.float32)
    best_cost, best = None, ring
    for direction in (1, -1):
        cand = ring[::direction] if direction == 1 else ring[::-1]
        for shift in range(n):
            rolled = np.roll(cand, -shift, axis=0)
            cost = float(np.linalg.norm(nodes - rolled, axis=1).mean())
            if best_cost is None or cost < best_cost:
                best_cost, best = cost, rolled
    return best.astype(np.float32)


def collate(batch):
    feats = torch.from_numpy(np.stack([b["feat"] for b in batch]))
    coords = torch.from_numpy(np.stack([b["nodes"] for b in batch]))
    m = max(len(b["target"]) for b in batch)
    target = torch.zeros(len(batch), m, 2)
    mask = torch.zeros(len(batch), m, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = torch.from_numpy(b["target"])
        target[i, : len(t)] = t
        mask[i, : len(t)] = True
    return feats, coords, target, mask


def chamfer(pred, gt, mask):
    d = torch.cdist(pred, gt)
    d = d.masked_fill(~mask.unsqueeze(1), 1e6)
    d_pred_gt = d.min(dim=2).values.mean(dim=1)
    d_gt_pred = d.min(dim=1).values.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return d_pred_gt, d_gt_pred, d


def smoothness(pred):
    edges = torch.roll(pred, -1, dims=1) - pred
    lengths = edges.norm(dim=-1)
    return ((lengths - lengths.mean(dim=1, keepdim=True)) ** 2).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "vertex_labels"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--max-offset", type=float, default=64.0)
    parser.add_argument("--target", choices=["vertices", "dense"], default="dense")
    parser.add_argument("--max-samples", type=int, default=30000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(ROOT / "runs" / "vertex_gcn"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_dir = Path(args.data)
    train_json = data_dir / "train.json"
    val_json = data_dir / "val.json"
    if not train_json.exists():
        print(f"缺少 {train_json}，先运行 scripts/prepare_mix_coco.py")
        return 2

    t0 = time.time()
    cache = data_dir / f"cache_{args.target}_{args.nodes}_{args.max_samples}.npz"
    samples = None
    if cache.exists():
        try:
            blob = np.load(cache, allow_pickle=False)
            feats, nodes, targets = blob["feats"], blob["nodes"], blob["targets"]
            hw = blob["hw"]
            samples = [{"feat": feats[i], "nodes": nodes[i], "target": targets[i],
                        "shape": (int(hw[i][0]), int(hw[i][1]))} for i in range(len(feats))]
            print(f"从缓存加载 {len(samples)} 个样本（{time.time() - t0:.0f}s）", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"缓存不可用（{exc}），重新构建", flush=True)
            samples = None
    if samples is None:
        train_recs = json.loads(train_json.read_text(encoding="utf-8"))["images"]
        samples = build_samples(train_recs, args.nodes, args.max_samples, target=args.target)
        print(f"训练样本 {len(samples)}（{time.time() - t0:.0f}s）", flush=True)
        if args.target == "dense" and samples:
            np.savez(cache,
                     feats=np.stack([s["feat"] for s in samples]),
                     nodes=np.stack([s["nodes"] for s in samples]),
                     targets=np.stack([s["target"] for s in samples]),
                     hw=np.array([s["shape"] for s in samples], dtype=np.int32))
            print(f"样本缓存写入 {cache}", flush=True)
    if len(samples) < 50:
        print("样本太少")
        return 2

    rng = np.random.default_rng(0)
    order = rng.permutation(len(samples))
    n_val = max(50, int(len(samples) * args.val_ratio))
    val_idx, train_idx = order[:n_val], order[n_val:]
    train_set = [samples[i] for i in train_idx]
    val_set = [samples[i] for i in val_idx]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = VertexGCN(in_dim=samples[0]["feat"].shape[1], max_offset=args.max_offset).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    start_epoch, best = 0, float("inf")
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = int(ck.get("epoch", 0))
        best = float(ck.get("best", best))
        print(f"resume from epoch {start_epoch}", flush=True)

    def run_epoch(set_, train: bool):
        if train:
            model.train()
            rng.shuffle(set_)
        else:
            model.eval()
        total, n = 0.0, 0
        for i in range(0, len(set_), args.batch_size):
            batch = set_[i:i + args.batch_size]
            feats, coords, target, mask = collate(batch)
            feats, coords = feats.to(args.device), coords.to(args.device)
            target, mask = target.to(args.device), mask.to(args.device)
            with torch.set_grad_enabled(train):
                # batched forward builds a block-diagonal k-NN graph, so edges
                # never cross between buildings
                off = model.forward_batch(feats, coords)
                pred = coords + off
                if args.target == "dense":
                    # fixed correspondence: node i must land on resampled point i
                    loss = (pred - target).norm(dim=-1).mean() + 0.1 * smoothness(pred)
                else:
                    d_pre, d_post, _ = chamfer(pred, target, mask)
                    loss = (d_pre + d_post).mean() + 0.5 * d_pre.mean() + 0.1 * smoothness(pred)
                if train:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
            total += float(loss.detach()) * len(batch)
            n += len(batch)
        return total / max(n, 1)

    for epoch in range(start_epoch, args.epochs):
        t = time.time()
        tr = run_epoch(train_set, True)
        va = run_epoch(val_set, False)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        print(f"  epoch {epoch + 1:3d}/{args.epochs} train {tr:.4f} val {va:.4f} lr {lr:.2e} "
              f"({time.time() - t:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch + 1, "best": min(best, va), "args": vars(args)}, last_path)
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": va,
                        "args": vars(args)}, best_path)
            print(f"    new best val {va:.4f}", flush=True)
    print(f"完成，best val {best:.4f}，权重 {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
