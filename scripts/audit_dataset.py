"""Quantify the dataset difficulties: illumination, scale, density, background clutter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, save_json  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def shadow_fraction(gray: np.ndarray) -> float:
    """Fraction of pixels much darker than the image's own median brightness.

    A relative threshold avoids conflating an overall dark exposure with local
    shadows, which is important for this dataset (many tiles are underexposed).
    """
    med = float(np.median(gray))
    return float((gray < 0.60 * med).mean())


def patch_features(image: np.ndarray, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    num, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if num > 1 else np.array([], dtype=np.int64)
    feats = {
        "brightness": float(gray.mean() / 255.0),
        "shadow_frac": shadow_fraction(gray),
        "edge_density": float((edges > 0).mean()),
        "gt_ratio": float((mask > 0).mean()),
        "gt_count": int(len(areas)),
        "mean_area": float(areas.mean()) if len(areas) else 0.0,
        "small_count": int((areas < 100).sum()),
    }
    return feats, areas


def collect_split(split: str):
    images = np.load(ROOT / "data" / "processed" / f"{split}_images.npy")
    masks = np.load(ROOT / "data" / "processed" / f"{split}_masks.npy")
    feats, all_areas = [], []
    for i in range(len(images)):
        f, a = patch_features(images[i], masks[i])
        f["index"] = i
        feats.append(f)
        all_areas.append(a)
    areas = np.concatenate(all_areas) if all_areas else np.array([], dtype=np.int64)
    return feats, areas


def summarize(feats: list[dict], areas: np.ndarray) -> dict:
    arr = lambda k: np.array([f[k] for f in feats], dtype=np.float64)  # noqa: E731
    brightness = arr("brightness")
    shadow = arr("shadow_frac")
    edge = arr("edge_density")
    ratio = arr("gt_ratio")
    count = arr("gt_count")
    mean_area = arr("mean_area")
    return {
        "num_patches": len(feats),
        "num_buildings": int(len(areas)),
        "building_area_px": {
            "mean": float(areas.mean()),
            "median": float(np.median(areas)),
            "p10": float(np.percentile(areas, 10)),
            "p25": float(np.percentile(areas, 25)),
            "p75": float(np.percentile(areas, 75)),
            "p90": float(np.percentile(areas, 90)),
            "small_lt_100_ratio": float((areas < 100).mean()),
            "small_lt_300_ratio": float((areas < 300).mean()),
            "large_gt_2000_ratio": float((areas > 2000).mean()),
        },
        "brightness": {"mean": float(brightness.mean()), "p10": float(np.percentile(brightness, 10)),
                       "p90": float(np.percentile(brightness, 90))},
        "shadow_frac": {"mean": float(shadow.mean()), "p90": float(np.percentile(shadow, 90))},
        "edge_density": {"mean": float(edge.mean()), "p90": float(np.percentile(edge, 90))},
        "gt_ratio": {"mean": float(ratio.mean()), "p90": float(np.percentile(ratio, 90))},
        "gt_count": {"mean": float(count.mean()), "p90": float(np.percentile(count, 90))},
        "mean_area": {"mean": float(mean_area.mean()), "p10": float(np.percentile(mean_area, 10))},
        "high_shadow_patch_ratio": float((shadow > 0.25).mean()),
        "dense_patch_ratio_gt20": float((count > 20).mean()),
        "small_patch_ratio": float(((mean_area > 0) & (mean_area < 400)).mean()),
        "clutter_patch_ratio": float((edge > np.percentile(edge, 75)).mean()),
    }


def pick_representatives(feats: list[dict], used: set[int]) -> dict[str, int]:
    def pick(score, extra=None):
        ranked = sorted(feats, key=score, reverse=True)
        for f in ranked:
            if f["index"] in used:
                continue
            if extra is not None and not extra(f):
                continue
            used.add(f["index"])
            return f["index"]
        return ranked[0]["index"]

    out = {}
    out["shadow"] = pick(lambda f: f["shadow_frac"] + 0.5 * (1 - f["brightness"]),
                         lambda f: f["gt_ratio"] > 0.10 and f["gt_count"] >= 25 and 0.18 < f["brightness"] < 0.32)
    out["dense"] = pick(lambda f: f["gt_count"], lambda f: f["gt_ratio"] > 0.12 and f["gt_count"] >= 80)
    out["small"] = pick(lambda f: f["small_count"] / max(f["gt_count"], 1) + 0.2 * f["gt_count"],
                        lambda f: f["gt_count"] >= 30 and f["mean_area"] < 280)
    out["clutter"] = pick(lambda f: f["edge_density"] * (1 - min(f["gt_ratio"], 0.5)),
                         lambda f: f["gt_count"] >= 12 and f["gt_ratio"] < 0.15)
    return out


def plot_overview(test_images, test_masks, feats, areas, reps, stats, out_path: Path) -> None:
    fig = plt.figure(figsize=(13.5, 7.2))
    titles = {
        "shadow": "弱光照/阴影",
        "dense": "密集建筑",
        "small": "小目标为主",
        "clutter": "复杂背景",
    }
    for j, key in enumerate(["shadow", "dense", "small", "clutter"]):
        idx = reps[key]
        ax = fig.add_subplot(2, 4, j + 1)
        ax.imshow(test_images[idx])
        mask = test_masks[idx] > 0
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask] = [1.0, 0.15, 0.1, 0.45]
        ax.imshow(overlay)
        f = feats[idx]
        ax.set_title(f"{titles[key]}\n亮度{f['brightness']:.2f} 建筑{f['gt_count']}栋", fontsize=10)
        ax.axis("off")

    ax = fig.add_subplot(2, 4, 5)
    bins = np.logspace(1, 5, 40)
    ax.hist(np.clip(areas, 10, None), bins=bins, color="#3b6ea5")
    ax.set_xscale("log")
    ax.axvline(100, color="#d1495b", ls="--", lw=1.3, label="100 px²（小目标阈值）")
    ax.set_xlabel("单栋建筑面积（px²，对数轴）")
    ax.set_ylabel("建筑数量")
    ax.legend(fontsize=8)
    ax.set_title(f"尺度分布：中位数 {stats['building_area_px']['median']:.0f} px²", fontsize=10)

    ax = fig.add_subplot(2, 4, 6)
    ax.hist([f["brightness"] for f in feats], bins=25, color="#e09f3e")
    ax.set_xlabel("图像平均亮度")
    ax.set_ylabel("图块数量")
    ax.set_title("光照分布", fontsize=10)

    ax = fig.add_subplot(2, 4, 7)
    ax.hist([f["gt_count"] for f in feats], bins=30, color="#6a994e")
    ax.set_xlabel("单图块建筑数量")
    ax.set_ylabel("图块数量")
    ax.set_title("建筑密度分布", fontsize=10)

    ax = fig.add_subplot(2, 4, 8)
    ax.scatter([f["edge_density"] for f in feats], [f["shadow_frac"] for f in feats],
               s=10, alpha=0.5, c="#5e548e")
    ax.set_xlabel("Canny 边缘密度（背景复杂度代理）")
    ax.set_ylabel("深阴影像素比例")
    ax.set_title("背景复杂度与阴影", fontsize=10)

    fig.suptitle("Massachusetts Buildings 测试子集：四类典型困难与数据分布", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(out_dir / "figures")

    train_feats, train_areas = collect_split("train")
    test_feats, test_areas = collect_split("test")
    report = {
        "train": summarize(train_feats, train_areas),
        "test": summarize(test_feats, test_areas),
    }
    reps = pick_representatives(test_feats, set())
    report["test"]["representative_indices"] = reps
    save_json(report, out_dir / "dataset_audit.json")

    test_images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    test_masks = np.load(ROOT / "data" / "processed" / "test_masks.npy")
    plot_overview(test_images, test_masks, test_feats, test_areas, reps, report["test"],
                  fig_dir / "dataset_overview.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
