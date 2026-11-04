"""Evaluate trained models on the official test split and save per-image diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import BuildingPatchDataset, DEFAULT_MEAN, DEFAULT_STD  # noqa: E402
from src.metrics import boundary_iou, component_report, pixel_stats, scores_from_stats  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402


def infer_probs(model, images: np.ndarray, batch_size: int = 8) -> np.ndarray:
    model.eval()
    out = []
    mean = torch.tensor(DEFAULT_MEAN.reshape(1, 3, 1, 1))
    std = torch.tensor(DEFAULT_STD.reshape(1, 3, 1, 1))
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size].astype(np.float32) / 255.0
            x = torch.from_numpy(chunk.transpose(0, 3, 1, 2))
            x = (x - mean) / std
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            out.append(torch.sigmoid(logits).numpy()[:, 0].astype(np.float16))
    return np.concatenate(out, axis=0)


def image_diagnostics(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    med = float(np.median(gray))
    num_gt, labels, stats, _ = cv2.connectedComponentsWithStats((gt > 0).astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_gt > 1 else np.array([0])
    return {
        "brightness": float(gray.mean() / 255.0),
        "shadow_frac": float((gray < 0.60 * med).mean()),
        "edge_density": float((edges > 0).mean()),
        "gt_ratio": float((gt > 0).mean()),
        "gt_count": int(num_gt - 1),
        "mean_gt_area": float(areas.mean()),
    }


def evaluate(probs: np.ndarray, gts: np.ndarray, images: np.ndarray, threshold: float) -> tuple[dict, list[dict]]:
    preds = (probs >= threshold).astype(np.uint8)
    tp = fp = fn = 0
    ious, bious = [], []
    comp_total = {"num_gt": 0, "gt_matched": 0, "small_num": 0, "small_matched": 0,
                  "merge_count": 0, "num_pred": 0}
    rows = []
    for i, (p, g) in enumerate(zip(preds, gts)):
        g = (g > 0).astype(np.uint8)
        a, b, c, _ = pixel_stats(p, g)
        tp += a
        fp += b
        fn += c
        diag = image_diagnostics(images[i], g, p)
        row = {"index": i, **diag}
        if g.sum() > 0:
            iou = a / max(a + b + c, 1)
            biou = boundary_iou(p, g, width=2)
            ious.append(iou)
            if not np.isnan(biou):
                bious.append(biou)
            row["iou"] = iou
            row["biou"] = biou
        else:
            row["iou"] = float("nan")
            row["biou"] = float("nan")
        comp = component_report(p, g, small_area=100)
        for key in comp_total:
            comp_total[key] += comp[key]
        row.update({
            "tp": a, "fp": b, "fn": c,
            "small_recall": comp["small_recall"],
            "merge_rate": comp["merge_rate"],
            "num_pred": comp["num_pred"],
            "gt_matched": comp["gt_matched"],
            "num_gt": comp["num_gt"],
        })
        rows.append(row)

    overall = scores_from_stats(tp, fp, fn)
    overall.update({
        "mean_image_iou": float(np.mean(ious)) if ious else float("nan"),
        "mean_boundary_iou": float(np.mean(bious)) if bious else float("nan"),
        "gt_recall": comp_total["gt_matched"] / max(comp_total["num_gt"], 1),
        "small_recall": comp_total["small_matched"] / max(comp_total["small_num"], 1),
        "merge_rate": comp_total["merge_count"] / max(comp_total["num_gt"], 1),
        "num_gt": comp_total["num_gt"],
        "num_pred": comp_total["num_pred"],
        "small_num": comp_total["small_num"],
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn,
    })
    return overall, rows


def load_model(model_name: str, checkpoint: Path):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = int(ckpt.get("args", {}).get("base", 16))
    model = build_model(model_name, base=base)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["unet", "msf", "full"])
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--tune", action="store_true", help="select threshold on valid split")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = ensure_dir(args.out)
    ensure_dir(out_dir / "probs")
    proc = ROOT / "data" / "processed"
    split = args.split
    images = np.load(proc / f"{split}_images.npy")
    gts = np.load(proc / f"{split}_masks.npy")
    val_images = np.load(proc / "valid_images.npy")
    val_gts = np.load(proc / "valid_masks.npy")

    for name in args.models:
        ckpt_path = ROOT / "runs" / name / "best.pt"
        if not ckpt_path.exists():
            print(f"!! missing {ckpt_path}")
            continue
        model, ckpt = load_model(name, ckpt_path)
        print(f"[{name}] checkpoint epoch {ckpt.get('epoch')} val={ckpt.get('val')}")
        threshold = args.threshold
        if args.tune:
            val_probs = infer_probs(model, val_images)
            best = (-1.0, threshold)
            for t in np.arange(0.30, 0.751, 0.05):
                overall, _ = evaluate(val_probs, val_gts, val_images, float(t))
                if overall["iou"] > best[0]:
                    best = (overall["iou"], float(t))
            threshold = best[1]
            print(f"  tuned threshold={threshold:.2f} (val IoU={best[0]:.4f})")
        probs = infer_probs(model, images)
        np.save(out_dir / "probs" / f"{name}_{split}.npy", probs)
        overall, rows = evaluate(probs, gts, images, threshold)
        summary = {
            "model": name,
            "split": split,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val": ckpt.get("val"),
            "threshold": threshold,
            "overall": overall,
            "num_images": len(images),
        }
        save_json(summary, out_dir / f"metrics_{name}_{split}.json")
        with (out_dir / f"per_image_{name}_{split}.csv").open("w", newline="", encoding="utf-8") as fh:
            fields = list(rows[0].keys()) if rows else []
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps({"model": name, "threshold": threshold, "overall": overall},
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
