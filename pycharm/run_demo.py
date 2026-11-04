"""PyCharm 演示脚本：加载已训练模型，对代表性测试图块做推理并导出给 MATLAB 分析。

在 PyCharm 中以项目根目录为工作目录运行本文件。运行成功后会生成：
  matlab_exchange/patch_*.png       原图 / 真值 / 预测 / 误差图
  matlab_exchange/demo_manifest.csv 逐图块指标
  matlab_exchange/demo_summary.json 汇总指标
  pycharm_output/pycharm_run.log    运行日志
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import boundary_iou, component_report, pixel_stats  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import ensure_dir, seed_everything  # noqa: E402

SCENES = {
    "shadow": "弱光照/阴影",
    "dense": "密集建筑",
    "small": "小目标",
    "clutter": "复杂背景",
}


def select_indices() -> dict[str, int]:
    audit = json.loads((ROOT / "results" / "dataset_audit.json").read_text(encoding="utf-8"))
    reps = {k: int(v) for k, v in audit["test"]["representative_indices"].items()}
    with (ROOT / "results" / "per_image_full_test.csv").open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r["iou"] not in ("", "nan") and int(float(r["gt_count"])) >= 20]
    out = dict(reps)
    used = set(reps.values())
    # 两个高 IoU 的成功案例
    for r in sorted(rows, key=lambda r: float(r["iou"]), reverse=True):
        idx = int(r["index"])
        if idx in used or float(r["iou"]) > 0.85:
            continue
        out[f"success_{len([k for k in out if k.startswith('success')]) + 1}"] = idx
        used.add(idx)
        if len([k for k in out if k.startswith("success")]) == 2:
            break
    # 两个失败案例
    for r in sorted(rows, key=lambda r: float(r["iou"])):
        idx = int(r["index"])
        if idx in used:
            continue
        out[f"failure_{len([k for k in out if k.startswith('failure')]) + 1}"] = idx
        used.add(idx)
        if len([k for k in out if k.startswith("failure")]) == 2:
            break
    return out


def overlay(image: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    out = image.astype(np.float32) / 255.0
    m = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[m] = 0.58 * out[m] + 0.42 * c
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def error_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[(pred > 0) & (gt > 0)] = [64, 140, 77]
    out[(pred > 0) & (gt == 0)] = [217, 51, 41]
    out[(pred == 0) & (gt > 0)] = [38, 97, 204]
    return out


def main() -> int:
    started = time.time()
    seed_everything(42)
    torch.set_num_threads(8)
    ex_dir = ensure_dir(ROOT / "matlab_exchange")
    patch_dir = ensure_dir(ex_dir / "patches")
    log_dir = ensure_dir(ROOT / "pycharm_output")
    log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    device = torch.device("cpu")
    ckpt_path = ROOT / "runs" / "full" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model("full", base=int(ckpt["args"].get("base", 16)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    log(f"PyCharm demo | torch={torch.__version__} | device={device}")
    log(f"checkpoint: {ckpt_path} (epoch={ckpt.get('epoch')}, val IoU={ckpt['val']['iou']:.4f})")

    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    masks = np.load(ROOT / "data" / "processed" / "test_masks.npy")
    indices = select_indices()

    # 四方向翻转 TTA，提升演示效果的稳定性
    def predict(img: np.ndarray) -> np.ndarray:
        x0 = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        mean = torch.tensor([0.2570, 0.2586, 0.2346]).view(1, 3, 1, 1)
        std = torch.tensor([0.1866, 0.1819, 0.1851]).view(1, 3, 1, 1)
        x0 = (x0 - mean) / std
        probs = []
        with torch.no_grad():
            for k in range(4):
                x = torch.rot90(x0, k, dims=(2, 3))
                p = torch.sigmoid(model(x))
                p = torch.rot90(p, -k, dims=(2, 3))
                probs.append(p)
        return torch.stack(probs).mean(0)[0, 0].numpy()

    rows = []
    for key, idx in indices.items():
        img = images[idx]
        gt = (masks[idx] > 0).astype(np.uint8)
        prob = predict(img)
        pred = (prob >= 0.5).astype(np.uint8)
        tp, fp, fn, _ = pixel_stats(pred, gt)
        iou = tp / max(tp + fp + fn, 1)
        biou = boundary_iou(pred, gt, width=2)
        comp = component_report(pred, gt, small_area=100)
        scene = SCENES.get(key, "成功案例" if key.startswith("success") else f"失败案例 {key.split('_')[-1]}")
        group = "representative" if key in SCENES else ("success" if key.startswith("success") else "failure")
        name = f"{key}_{idx:03d}"
        cv2.imwrite(str(patch_dir / f"{name}_image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(patch_dir / f"{name}_gt.png"), gt * 255)
        cv2.imwrite(str(patch_dir / f"{name}_pred.png"), pred * 255)
        cv2.imwrite(str(patch_dir / f"{name}_pred_overlay.png"),
                    cv2.cvtColor(overlay(img, pred, (0.10, 0.45, 0.85)), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(patch_dir / f"{name}_error.png"),
                    cv2.cvtColor(error_rgb(pred, gt), cv2.COLOR_RGB2BGR))
        rows.append({
            "name": name, "index": idx, "scene": scene, "group": group,
            "iou": round(float(iou), 4), "biou": round(float(biou), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "gt_count": comp["num_gt"], "pred_count": comp["num_pred"],
            "small_recall": round(float(comp["small_recall"]), 4)
            if not np.isnan(comp["small_recall"]) else 0.0,
            "merge_rate": round(float(comp["merge_rate"]), 4)
            if not np.isnan(comp["merge_rate"]) else 0.0,
        })
        log(f"  [{scene}] index={idx} IoU={iou:.4f} BIoU={biou:.4f} "
            f"GT={comp['num_gt']} Pred={comp['num_pred']}")

    with (ex_dir / "demo_manifest.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ious = [r["iou"] for r in rows]
    ok_ious = [r["iou"] for r in rows if r["group"] != "failure"]
    bad_ious = [r["iou"] for r in rows if r["group"] == "failure"]
    summary = {
        "script": "pycharm/run_demo.py",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "device": "CPU",
        "torch": torch.__version__,
        "num_patches": len(rows),
        "mean_iou": round(float(np.mean(ious)), 4),
        "non_failure_mean_iou": round(float(np.mean(ok_ious)), 4),
        "failure_mean_iou": round(float(np.mean(bad_ious)), 4) if bad_ious else 0.0,
        "min_iou": round(float(np.min(ious)), 4),
        "max_iou": round(float(np.max(ious)), 4),
        "mean_biou": round(float(np.mean([r["biou"] for r in rows])), 4),
        "mean_small_recall": round(float(np.mean([r["small_recall"] for r in rows])), 4),
        "mean_merge_rate": round(float(np.mean([r["merge_rate"] for r in rows])), 4),
        "elapsed_sec": round(time.time() - started, 2),
    }
    (ex_dir / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log("summary: " + json.dumps(summary, ensure_ascii=False))
    log("PYCHARM_DEMO_OK")
    (log_dir / "pycharm_run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
