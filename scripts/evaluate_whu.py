"""Evaluate a trained WHU checkpoint (threshold sweep + optional 8x TTA).

Example::

    python scripts/evaluate_whu.py --model whu_upernet_r34 --split test --tta

Add ``--verbose --examples 4`` to watch the per-image loop and dump a
prediction/GT comparison figure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer import InferenceEngine  # noqa: E402

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass


def save_examples(path: Path, images, masks, probs, threshold: float, count: int = 4) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = min(count, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n), squeeze=False)
    for k in range(n):
        img = images[k]
        gt = masks[k] > 0
        pred = probs[k] >= threshold
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = inter / max(union, 1)

        canvas = img.copy()
        gt_edge = (cv2.dilate(gt.astype(np.uint8), np.ones((3, 3), np.uint8))
                   - cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8))) > 0
        pred_edge = (cv2.dilate(pred.astype(np.uint8), np.ones((3, 3), np.uint8))
                     - cv2.erode(pred.astype(np.uint8), np.ones((3, 3), np.uint8))) > 0
        canvas[gt_edge] = [40, 210, 90]
        canvas[pred_edge] = [255, 60, 40]
        canvas[pred & ~gt] = (0.6 * canvas[pred & ~gt] + 0.4 * np.array([255, 60, 40])).astype(np.uint8)
        canvas[gt & ~pred] = (0.6 * canvas[gt & ~pred] + 0.4 * np.array([40, 120, 255])).astype(np.uint8)

        axes[k][0].imshow(img)
        axes[k][0].set_title(f"影像 #{k}", fontsize=9)
        axes[k][1].imshow(gt, cmap="gray")
        axes[k][1].set_title("真值", fontsize=9)
        axes[k][2].imshow(canvas)
        axes[k][2].set_title(f"预测(红) vs 真值(绿)  IoU={iou:.3f}", fontsize=9)
        for ax in axes[k]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="runs 下的模型目录名")
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="逐图打印 IoU / 精度 / 召回")
    parser.add_argument("--examples", type=int, default=0, help="保存 N 张预测对比图")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data)
    images = np.load(data_dir / f"{args.split}_images.npy", mmap_mode="r")
    masks = np.load(data_dir / f"{args.split}_masks.npy", mmap_mode="r")
    n = len(images) if not args.limit else min(int(args.limit), len(images))

    engine = InferenceEngine(ROOT)
    _, ckpt = engine.load(args.model)
    size = int(ckpt.get("args", {}).get("size", 512))
    thresholds = [float(t) for t in args.thresholds.split(",")]
    tp = {t: 0 for t in thresholds}
    fp = {t: 0 for t in thresholds}
    fn = {t: 0 for t in thresholds}

    print(f"模型 {args.model} | checkpoint epoch {ckpt.get('epoch')} | 验证 IoU "
          f"{ckpt.get('val', {}).get('iou', float('nan')):.4f} | 切块 {size} | TTA {args.tta}")
    print(f"{args.split} 集共 {len(images)} 张，本次评测 {n} 张\n")

    t0 = time.time()
    examples = []
    for i in range(n):
        img = np.asarray(images[i])
        prob, _ = engine.predict(img, args.model, want_edge=False, tta=args.tta)
        gt = (np.asarray(masks[i]) > 0)
        if args.verbose:
            pred = prob >= 0.5
            inter = int((pred & gt).sum())
            if gt.sum() == 0:
                print(f"  [{i + 1:4d}/{n}] 空图（真值无建筑），预测 {int(pred.sum())} px  "
                      f"({time.time() - t0:.1f}s)", flush=True)
            else:
                iou = inter / max(int((pred | gt).sum()), 1)
                prec = inter / max(int(pred.sum()), 1)
                rec = inter / max(int(gt.sum()), 1)
                print(f"  [{i + 1:4d}/{n}] IoU {iou:.4f}  P {prec:.4f}  R {rec:.4f}  "
                      f"建筑像素 {gt.mean() * 100:5.2f}%  ({time.time() - t0:.1f}s)", flush=True)
        elif (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        if len(examples) < args.examples:
            examples.append((img, gt, prob))
        for t in thresholds:
            pred = prob >= t
            tp[t] += int((pred & gt).sum())
            fp[t] += int((pred & ~gt).sum())
            fn[t] += int((~pred & gt).sum())

    rows = []
    for t in thresholds:
        iou = tp[t] / max(tp[t] + fp[t] + fn[t], 1)
        prec = tp[t] / max(tp[t] + fp[t], 1)
        rec = tp[t] / max(tp[t] + fn[t], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        rows.append({"threshold": t, "iou": iou, "f1": f1, "precision": prec, "recall": rec})
    best = max(rows, key=lambda r: r["iou"])

    print(f"\n{'阈值':>6} {'IoU':>8} {'F1':>8} {'精度':>8} {'召回':>8}")
    print("-" * 42)
    for r in rows:
        flag = "  <-- 最优" if r is best else ""
        print(f"{r['threshold']:6.2f} {r['iou']:8.4f} {r['f1']:8.4f} "
              f"{r['precision']:8.4f} {r['recall']:8.4f}{flag}")

    payload = {
        "model": args.model,
        "split": args.split,
        "num_images": n,
        "tta": bool(args.tta),
        "tile": size,
        "thresholds": rows,
        "best": best,
        "checkpoint_val_iou": ckpt.get("val", {}).get("iou"),
        "checkpoint_epoch": ckpt.get("epoch"),
    }
    out = Path(args.out) if args.out else ROOT / "runs" / args.model / f"eval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最优阈值 {best['threshold']:.2f}: IoU {best['iou']:.4f}  F1 {best['f1']:.4f}")
    print(f"结果写入 {out}")

    if examples:
        fig_path = out.parent / f"eval_examples_{args.split}.png"
        save_examples(fig_path, [e[0] for e in examples], [e[1] for e in examples],
                      [e[2] for e in examples], best["threshold"], len(examples))
        print(f"对比图写入 {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
