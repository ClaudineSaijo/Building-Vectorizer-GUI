"""Reproducible configuration ablation for the vectorization prototype.

Runs the four configurations described in the proposal (baseline, +boundary,
+boundary+vertex, +boundary+vertex+geometry) with one or more trained models on
a sample of the test split, and writes CSV / JSON / a comparison figure.

Example
-------
    .\\.venv\\Scripts\\python.exe scripts\\ablation_benchmark.py --samples 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app.compare import CONFIGS, BatchJob, run_batch, save_csv  # noqa: E402
from src.infer import InferenceEngine  # noqa: E402
from src.utils import ensure_dir  # noqa: E402
from src.vectorize import VectorConfig  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot(rows: list[dict], path: Path) -> None:
    models = sorted({r["model"] for r in rows})
    labels = ["A\n基线", "B\n+边界", "C\n+B+顶点", "D\n+B+V+几何"]
    colors = {"unet": "#8ecae6", "msf": "#219ebc", "full": "#023047"}
    x = np.arange(len(CONFIGS))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    specs = [
        ("polygon_iou", "多边形 IoU ↑", axes[0], (0, 1)),
        ("chamfer_matched", "Chamfer 距离 / px ↓", axes[1], None),
        ("mean_vertices", "平均顶点数 ↓", axes[2], None),
        ("rectilinearity", "几何规整度 ↑", axes[3], (0, 1)),
    ]
    for key, title, ax, ylim in specs:
        for k, model in enumerate(models):
            vals = [next((r[key] for r in rows if r["model"] == model and r["config"] == c[0]), np.nan)
                    for c in CONFIGS]
            ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width, label=model,
                   color=colors.get(model))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="default: all available")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=str(ROOT / "results" / "ablation"))
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    proc = ROOT / "data" / "processed"
    images = np.load(proc / f"{args.split}_images.npy")
    gts = (np.load(proc / f"{args.split}_masks.npy") > 0).astype(np.uint8)
    n = max(2, min(args.samples, len(images)))
    idx = np.unique(np.linspace(0, len(images) - 1, n).round().astype(int))

    engine = InferenceEngine(ROOT)
    models = args.models or engine.available_models()
    if not models:
        print("no checkpoints found under runs/*/best.pt")
        return 1

    job = BatchJob(
        images=[np.asarray(images[i]) for i in idx],
        gts=[gts[i] for i in idx],
        model_names=list(models),
        configs=CONFIGS,
        base_cfg=VectorConfig(threshold=args.threshold),
    )
    rows = run_batch(engine, job, progress=lambda done, total: print(f"\r  inference {done}/{total}", end=""))
    print()

    csv_path = save_csv(out_dir / "ablation_compare.csv", rows)
    (out_dir / "ablation_compare.json").write_text(
        json.dumps({"samples": [int(i) for i in idx], "models": list(models),
                    "configs": [c[0] for c in CONFIGS], "rows": rows},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    plot(rows, out_dir / "ablation_compare.png")

    header = f"{'config':30s} {'model':6s} {'polyIoU':>8s} {'chamfer':>8s} {'vtx':>6s} {'merge':>6s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['config'][:30]:30s} {r['model']:6s} {r['polygon_iou']:8.4f} "
              f"{r['chamfer_matched']:8.3f} {r['mean_vertices']:6.1f} {r['merge_rate']:6.3f}")
    print(f"\nwrote {csv_path}\nwrote {out_dir / 'ablation_compare.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
