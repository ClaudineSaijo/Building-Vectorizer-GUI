"""Plot training curves for the three ablation models."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
LABELS = {"unet": "U-Net 基线", "msf": "+多尺度融合", "full": "+边界监督（完整）"}
COLORS = {"unet": "#7f8c8d", "msf": "#3b7ea1", "full": "#c8553d"}


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for model in ("unet", "msf", "full"):
        path = ROOT / "runs" / model / "history.json"
        if not path.exists():
            continue
        h = json.loads(path.read_text(encoding="utf-8"))["history"]
        epochs = [r["epoch"] for r in h]
        axes[0].plot(epochs, [r["train_loss"] for r in h], label=LABELS[model], color=COLORS[model])
        axes[1].plot(epochs, [r["val_iou"] for r in h], label=LABELS[model], color=COLORS[model])
        axes[2].plot(epochs, [r["val_f1"] for r in h], label=LABELS[model], color=COLORS[model])
    axes[0].set_title("训练损失")
    axes[1].set_title("验证集 IoU")
    axes[2].set_title("验证集 F1")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("IoU")
    axes[2].set_ylabel("F1")
    axes[0].legend(fontsize=8)
    fig.suptitle("三组模型的训练收敛过程（验证集仅 4 景影像）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "results" / "figures" / "training_curves.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
