"""Pilot stress test: how much does the current model degrade under non-ideal imaging?

Synthetic corruptions are proxies for real adverse conditions.  This is a
diagnostic experiment, not a new training method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import boundary_iou, component_report, pixel_stats  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import ensure_dir, seed_everything  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CONDITIONS = ["clean", "low_light", "shadow", "haze", "noise", "motion_blur", "jpeg"]


def low_light(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.power(x, 2.1) * 0.55
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def shadow(img: np.ndarray, idx: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + idx)
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    overlay = np.ones((h, w), np.float32)
    for _ in range(5):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        ax, ay = rng.integers(30, 90), rng.integers(20, 60)
        ang = rng.uniform(0, 180)
        cv2.ellipse(overlay, (int(cx), int(cy)), (int(ax), int(ay)), float(ang), 0, 360,
                    0.38, -1)
    out *= overlay[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def haze(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return np.clip(0.65 * x + 0.35 * 205, 0, 255).astype(np.uint8)


def noise(img: np.ndarray, idx: int) -> np.ndarray:
    rng = np.random.default_rng(2000 + idx)
    x = img.astype(np.float32) + rng.normal(0, 24, img.shape).astype(np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray) -> np.ndarray:
    kernel = np.zeros((11, 11), np.float32)
    kernel[5, :] = 1.0 / 11
    m = cv2.getRotationMatrix2D((5, 5), 28, 1.0)
    kernel = cv2.warpAffine(kernel, m, (11, 11))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def jpeg(img: np.ndarray) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 15])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB) if ok else img


def corrupt(name: str, img: np.ndarray, idx: int) -> np.ndarray:
    if name == "clean":
        return img
    if name == "low_light":
        return low_light(img)
    if name == "shadow":
        return shadow(img, idx)
    if name == "haze":
        return haze(img)
    if name == "noise":
        return noise(img, idx)
    if name == "motion_blur":
        return motion_blur(img)
    if name == "jpeg":
        return jpeg(img)
    raise ValueError(name)


def infer(model, images: np.ndarray, batch: int = 8) -> np.ndarray:
    mean = torch.tensor([0.2570, 0.2586, 0.2346]).view(1, 3, 1, 1)
    std = torch.tensor([0.1866, 0.1819, 0.1851]).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            p = torch.sigmoid(model((x - mean) / std)).numpy()[:, 0]
            out.append(p)
    return np.concatenate(out, axis=0)


def main() -> int:
    seed_everything(42)
    torch.set_num_threads(8)
    out_dir = ensure_dir(ROOT / "results" / "robustness")
    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    masks = (np.load(ROOT / "data" / "processed" / "test_masks.npy") > 0).astype(np.uint8)
    # Fixed subsample of 120 patches containing buildings to bound CPU time
    rng = np.random.default_rng(7)
    candidates = np.where(masks.reshape(len(masks), -1).sum(1) > 0)[0]
    idx = np.sort(rng.choice(candidates, size=min(120, len(candidates)), replace=False))
    images, masks = images[idx], masks[idx]

    ckpt = torch.load(ROOT / "runs" / "full" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model("full", base=int(ckpt["args"].get("base", 16)))
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {}
    for cond in CONDITIONS:
        imgs = np.stack([corrupt(cond, im, int(i)) for im, i in zip(images, idx)])
        probs = infer(model, imgs)
        preds = (probs >= 0.5).astype(np.uint8)
        tp = fp = fn = 0
        biou, small_rec, merge, gt_num, small_num = [], 0, 0, 0, 0
        for p, g in zip(preds, masks):
            a, b, c, _ = pixel_stats(p, g)
            tp += a; fp += b; fn += c
            if g.sum() > 0:
                biou.append(boundary_iou(p, g, width=2))
            cr = component_report(p, g, small_area=100)
            gt_num += cr["num_gt"]; small_num += cr["small_num"]
            small_rec += cr["small_matched"]; merge += cr["merge_count"]
        iou = tp / max(tp + fp + fn, 1)
        results[cond] = {
            "iou": round(float(iou), 4),
            "f1": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4),
            "biou": round(float(np.nanmean(biou)), 4),
            "small_recall": round(small_rec / max(small_num, 1), 4),
            "merge_rate": round(merge / max(gt_num, 1), 4),
            "relative_iou": round(float(iou / max(results["clean"]["iou"] if "clean" in results else iou, 1e-9)), 4),
        }
        print(cond, results[cond], flush=True)

    with (out_dir / "robustness_stress.json").open("w", encoding="utf-8") as fh:
        json.dump({"num_patches": len(images), "checkpoint": str(ROOT / "runs" / "full" / "best.pt"),
                   "results": results}, fh, ensure_ascii=False, indent=2)

    labels = {"clean": "清晰", "low_light": "低照度", "shadow": "阴影", "haze": "雾霾",
              "noise": "噪声", "motion_blur": "运动模糊", "jpeg": "JPEG 压缩"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    xs = np.arange(len(CONDITIONS))
    ious = [results[c]["iou"] for c in CONDITIONS]
    bars = axes[0].bar(xs, ious, color=["#5b6573"] + ["#e07a5f"] * 6)
    for b, v in zip(bars, ious):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(xs, [labels[c] for c in CONDITIONS], rotation=20)
    axes[0].set_ylim(0, 0.7)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("非理想成像条件下的 IoU", fontsize=12)
    axes[0].grid(axis="y", alpha=0.25)

    rel = [results[c]["relative_iou"] for c in CONDITIONS]
    bars = axes[1].bar(xs, rel, color=["#5b6573"] + ["#3b7ea1"] * 6)
    for b, v in zip(bars, rel):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    axes[1].set_xticks(xs, [labels[c] for c in CONDITIONS], rotation=20)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("相对清晰影像的 IoU")
    axes[1].set_title("鲁棒性指数（1.0 = 无退化）", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("现有完整模型在合成退化下的鲁棒性诊断（120 个测试图块）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "robustness_stress.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
