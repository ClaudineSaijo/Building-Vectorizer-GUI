"""Compare models, break down errors by difficulty, and draw qualitative panels."""
from __future__ import annotations

import argparse
import csv
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

MODELS = ["unet", "msf", "full"]
LABELS = {"unet": "U-Net 基线", "msf": "+多尺度融合", "full": "+边界监督（完整）"}
COLORS = {"unet": "#7f8c8d", "msf": "#3b7ea1", "full": "#c8553d"}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in {"index", "gt_count", "num_pred", "num_gt", "gt_matched", "tp", "fp", "fn"}:
                r[k] = int(float(v))
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def mean_metric(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def bin_rows(rows, key, edges, labels):
    out = []
    values = np.array([r[key] for r in rows], dtype=float)
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        if i == len(labels) - 1:
            subset = [r for r, v in zip(rows, values) if v >= lo]
        else:
            subset = [r for r, v in zip(rows, values) if lo <= v < hi]
        out.append((label, subset))
    return out


def plot_overall(summaries, out_path):
    metrics = [("iou", "IoU"), ("f1", "F1"), ("mean_boundary_iou", "边界 IoU"),
               ("small_recall", "小目标召回"), ("merge_rate", "粘连率（越低越好）")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(metrics[:4]))
    width = 0.26
    for j, model in enumerate(MODELS):
        vals = [summaries[model]["overall"][k] for k, _ in metrics[:4]]
        bars = axes[0].bar(x + (j - 1) * width, vals, width, label=LABELS[model], color=COLORS[model])
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x, [label for _, label in metrics[:4]])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("得分")
    axes[0].set_title("分割精度与边界完整性", fontsize=12)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    vals = [summaries[m]["overall"]["merge_rate"] for m in MODELS]
    bars = axes[1].bar(MODELS, vals, color=[COLORS[m] for m in MODELS], width=0.55)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(MODELS, ["基线", "+多尺度", "完整"], fontsize=9)
    axes[1].set_ylabel("粘连率")
    axes[1].set_title("密集区域粘连（越低越好）", fontsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def tertiles(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    q1, q2 = np.percentile(vals, [33.333, 66.667])
    return [-1e9, q1, q2, 1e9], [f"低\n<{q1:.2f}", f"中\n{q1:.2f}–{q2:.2f}", f"高\n>{q2:.2f}"]


def panel_specs(ref):
    bright_e, bright_l = tertiles(ref, "brightness")
    shadow_e, shadow_l = tertiles(ref, "shadow_frac")
    edge_e, edge_l = tertiles(ref, "edge_density")
    return [
        ("brightness", bright_e, bright_l, "光照条件（平均亮度）"),
        ("shadow_frac", shadow_e, shadow_l, "阴影/暗区占比"),
        ("gt_count", [0, 21, 51, 10**6], ["稀疏\n≤20 栋", "中等\n21–50 栋", "密集\n>50 栋"], "建筑密度"),
        ("mean_gt_area", [0, 300, 800, 10**9], ["小建筑\n<300 px²", "中建筑\n300–800", "大建筑\n>800"], "目标尺度"),
        ("edge_density", edge_e, edge_l, "背景复杂度（Canny 边缘密度）"),
        ("gt_ratio", [0, 0.10, 0.25, 1.0], ["低覆盖\n<10%", "中覆盖\n10–25%", "高覆盖\n>25%"], "建筑像素覆盖率"),
    ]


def breakdown_table(all_rows) -> dict:
    panels = panel_specs(all_rows[MODELS[0]])
    out = {}
    for key, edges, labels, _ in panels:
        out[key] = {}
        for model in MODELS:
            out[key][model] = {}
            for label, sub in bin_rows(all_rows[model], key, edges, labels):
                clean = label.replace("\n", " ")
                out[key][model][clean] = {
                    "n": len(sub),
                    "iou": mean_metric(sub, "iou"),
                    "biou": mean_metric(sub, "biou"),
                    "small_recall": mean_metric(sub, "small_recall"),
                    "merge_rate": mean_metric(sub, "merge_rate"),
                }
    return out


def plot_breakdown(all_rows, out_path):
    panels = panel_specs(all_rows[MODELS[0]])
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    for ax, (key, edges, labels, title) in zip(axes.ravel(), panels):
        width = 0.26
        x = np.arange(len(labels))
        for j, model in enumerate(MODELS):
            rows = all_rows[model]
            binned = bin_rows(rows, key, edges, labels)
            vals = [mean_metric(sub, "iou") for _, sub in binned]
            bars = ax.bar(x + (j - 1) * width, vals, width, label=LABELS[model], color=COLORS[model])
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("平均单图 IoU")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("不同困难条件下的建筑物提取精度（测试集）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def overlay_mask(image, mask, color):
    out = image.astype(np.float32) / 255.0
    overlay = np.zeros_like(out)
    overlay[:] = color
    alpha = 0.42
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * overlay[m]
    return np.clip(out, 0, 1)


def error_map(pred, gt):
    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    tp = (pred > 0) & (gt > 0)
    fp = (pred > 0) & (gt == 0)
    fn = (pred == 0) & (gt > 0)
    out[tp] = [0.25, 0.55, 0.30]
    out[fp] = [0.85, 0.20, 0.16]
    out[fn] = [0.15, 0.38, 0.80]
    return out


def plot_qualitative(test_images, test_masks, probs, reps, thresholds, out_path):
    keys = ["shadow", "dense", "small", "clutter"]
    titles = {"shadow": "弱光照 / 阴影", "dense": "密集建筑", "small": "小目标", "clutter": "复杂背景"}
    fig, axes = plt.subplots(len(keys), 6, figsize=(14.2, 9.6))
    cols = ["原图", "真值", "U-Net 基线", "+多尺度融合", "完整模型", "误差图"]
    for i, key in enumerate(keys):
        idx = reps[key]
        img = test_images[idx]
        gt = (test_masks[idx] > 0).astype(np.uint8)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(overlay_mask(img, gt, (0.15, 0.75, 0.35)))
        for j, model in enumerate(MODELS):
            pred = (probs[model][idx] >= thresholds.get(model, 0.5)).astype(np.uint8)
            iou = (pred & gt).sum() / max((pred | gt).sum(), 1)
            axes[i, 2 + j].imshow(overlay_mask(img, pred, (0.10, 0.45, 0.85)))
            axes[i, 2 + j].set_title(f"IoU {iou:.3f}", fontsize=8)
            if model == "full":
                err = error_map(pred, gt)
        axes[i, 5].imshow(err)
        axes[i, 0].set_ylabel(titles[key], fontsize=11)
        for j in range(6):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=10)
        axes[i, 5].text(0.02, -0.08, "绿 TP / 红 FP / 蓝 FN", transform=axes[i, 5].transAxes,
                        fontsize=7, va="top")
    fig.suptitle("典型困难场景下的定性对比（Massachusetts Buildings 测试集）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_failure_cases(test_images, test_masks, probs, rows, thresholds, out_path):
    """Pick the worst case in four difficult strata and show its error map."""
    valid = [r for r in rows if not np.isnan(r["iou"])]
    high_shadow = float(np.percentile([r["shadow_frac"] for r in valid], 75))
    high_edge = float(np.percentile([r["edge_density"] for r in valid], 75))
    strata = [
        ("密集粘连", lambda r: r["gt_count"] > 80 and r["gt_ratio"] > 0.15),
        ("重阴影", lambda r: r["shadow_frac"] >= high_shadow and r["gt_count"] >= 25),
        ("小目标", lambda r: r["mean_gt_area"] < 250 and r["gt_count"] >= 40),
        ("背景误检", lambda r: r["edge_density"] >= high_edge and 5 <= r["gt_count"] <= 30),
    ]
    picked, used = [], set()
    for name, cond in strata:
        candidates = sorted([r for r in valid if cond(r) and r["index"] not in used],
                            key=lambda r: r["iou"])
        if candidates:
            picked.append((name, candidates[0]))
            used.add(candidates[0]["index"])
    fig, axes = plt.subplots(len(picked), 4, figsize=(10.2, 2.7 * len(picked)))
    if len(picked) == 1:
        axes = axes[None, :]
    for i, (name, row) in enumerate(picked):
        idx = row["index"]
        img = test_images[idx]
        gt = (test_masks[idx] > 0).astype(np.uint8)
        pred = (probs["full"][idx] >= thresholds.get("full", 0.5)).astype(np.uint8)
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(f"{name}\nIoU {row['iou']:.2f}", fontsize=10)
        axes[i, 1].imshow(overlay_mask(img, gt, (0.15, 0.75, 0.35)))
        axes[i, 2].imshow(overlay_mask(img, pred, (0.10, 0.45, 0.85)))
        axes[i, 3].imshow(error_map(pred, gt))
        for j, title in enumerate(["原图", "真值", "完整模型预测", "误差（绿TP/红FP/蓝FN）"]):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(title, fontsize=9)
    fig.suptitle("完整模型的典型失败案例", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    fig_dir = ensure_dir(out_dir / "figures")

    summaries, all_rows = {}, {}
    for model in MODELS:
        path = out_dir / f"metrics_{model}_test.json"
        if not path.exists():
            print(f"skip {model}: no metrics")
            continue
        summaries[model] = json.loads(path.read_text(encoding="utf-8"))
        all_rows[model] = read_rows(out_dir / f"per_image_{model}_test.csv")
    models = [m for m in MODELS if m in summaries]
    if not models:
        raise SystemExit("no results found")

    plot_overall(summaries, fig_dir / "model_comparison.png")
    plot_breakdown(all_rows, fig_dir / "difficulty_breakdown.png")

    audit = json.loads((out_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    reps = audit["test"]["representative_indices"]
    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    masks = np.load(ROOT / "data" / "processed" / "test_masks.npy")
    probs = {m: np.load(out_dir / "probs" / f"{m}_test.npy").astype(np.float32) for m in models}
    if all(m in probs for m in MODELS):
        thresholds = {m: float(summaries[m]["threshold"]) for m in models}
        plot_qualitative(images, masks, probs, reps, thresholds, fig_dir / "qualitative_comparison.png")
        plot_failure_cases(images, masks, probs, all_rows["full"], thresholds,
                           fig_dir / "failure_cases.png")

    analysis = {"models": models, "overall": {m: summaries[m]["overall"] for m in models},
                "breakdown": breakdown_table(all_rows)}
    for key in ["brightness", "shadow_frac", "edge_density", "gt_count", "mean_gt_area"]:
        analysis[f"corr_{key}"] = {}
        for m in models:
            vals = np.array([r[key] for r in all_rows[m]], dtype=float)
            ious = np.array([r["iou"] for r in all_rows[m]], dtype=float)
            ok = ~np.isnan(ious)
            analysis[f"corr_{key}"][m] = float(np.corrcoef(vals[ok], ious[ok])[0, 1]) if ok.sum() > 2 else float("nan")
    if "unet" in all_rows and "full" in all_rows:
        base = {r["index"]: r["iou"] for r in all_rows["unet"]}
        full = {r["index"]: r["iou"] for r in all_rows["full"]}
        shared = [i for i in base if i in full and not np.isnan(base[i]) and not np.isnan(full[i])]
        delta = np.array([full[i] - base[i] for i in shared])
        analysis["full_minus_unet"] = {
            "mean_delta": float(delta.mean()),
            "improved": int((delta > 0.01).sum()),
            "degraded": int((delta < -0.01).sum()),
            "num": len(shared),
        }
        if len(delta) > 5:
            rng = np.random.default_rng(0)
            boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            analysis["full_minus_unet"]["ci95"] = [float(np.percentile(boots, 2.5)),
                                                   float(np.percentile(boots, 97.5))]
    if "unet" in all_rows and "msf" in all_rows:
        base = {r["index"]: r["iou"] for r in all_rows["unet"]}
        msf = {r["index"]: r["iou"] for r in all_rows["msf"]}
        shared = [i for i in base if i in msf and not np.isnan(base[i]) and not np.isnan(msf[i])]
        delta = np.array([msf[i] - base[i] for i in shared])
        analysis["msf_minus_unet"] = {
            "mean_delta": float(delta.mean()),
            "improved": int((delta > 0.01).sum()),
            "degraded": int((delta < -0.01).sum()),
            "num": len(shared),
        }
        if len(delta) > 5:
            rng = np.random.default_rng(0)
            boots = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
            analysis["msf_minus_unet"]["ci95"] = [float(np.percentile(boots, 2.5)),
                                                  float(np.percentile(boots, 97.5))]
    save_json(analysis, out_dir / "analysis_summary.json")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
