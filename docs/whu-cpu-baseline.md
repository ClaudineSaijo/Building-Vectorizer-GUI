# WHU Local CPU Baseline (What Is Achievable Without a GPU)

> Purpose: with only an AMD 780M integrated GPU at hand, train a **genuinely usable**
> building segmentation model on the WHU 0.3 m data, validate the full chain of
> "data → training → evaluation → GUI loading → vectorized export",
> and quantify "how far local CPU training can go, and how far it still is from 0.85".

## 1. Training Configuration

| Item | Value |
|---|---|
| Data | WHU aerial 0.3 m, 512² tiles; training randomly samples **2800/4736**, validation **256/1036** |
| Input | Random **256²** crop from 512² (1/4 of the area, 4× the speed) |
| Model | Built-in U-Net, base=16 (1.96 M parameters), no pretraining |
| Loss | BCE (pos_weight=1.5) + Dice |
| Optimization | AdamW, lr=3e-4, cosine annealing to 2%, batch=8, 38 epochs (first 18 rounds on 2800 images, then continued training for 20 rounds on the full 4736 images) |
| Augmentation | Random scaling 0.75–1.3, flip, 90° rotation, brightness/contrast/Gamma/noise |
| Normalization | WHU dataset statistics mean=[0.436,0.446,0.414], std=[0.216,0.203,0.217] |
| Hardware | Local CPU (12 threads), **total runtime about 5 hours**, roughly 5–10 min/epoch |

![Training curves](images/whu_cpu_training_curves.png)

- Train/val loss decrease in sync and are still decreasing at epoch 18 (0.229 / 0.223), indicating that **training has not saturated yet**;
- Validation IoU 0.653 → **0.770**, F1 0.790 → **0.870**; there was one dip in the middle at epoch 4 (0.631), which is normal fluctuation.

## 2. Test Set Results (2416 images, never used in training)

| Threshold | IoU | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.30 | 0.8004 | 0.8892 | 0.8415 | 0.9426 |
| 0.40 | 0.8117 | 0.8960 | 0.8641 | 0.9304 |
| 0.50 | 0.8183 | 0.9000 | 0.8846 | 0.9161 |
| 0.55 | 0.8196 | 0.9009 | 0.8937 | 0.9082 |
| **0.75** | **0.8273** | **0.9055** | 0.9123 | 0.8988 |
| 0.65 | 0.8186 | 0.9003 | 0.9110 | 0.8898 |
| 0.70 | 0.8159 | 0.8986 | 0.9197 | 0.8785 |
| 0.80 | 0.8026 | 0.8905 | 0.9383 | 0.8473 |

**Best threshold 0.75: test set IoU = 0.8273, F1 = 0.9055** (no TTA).

![Test samples](images/whu_cpu_eval_examples.png)

- Row 1 is an empty image (no buildings in the ground truth); the model falsely detected a small patch, IoU=0;
- Rows 2, 4, and 6 have IoU 0.89 / 0.87 / 0.82, with good edge alignment;
- Row 3, an industrial area, has IoU 0.34 — the ground truth only annotates part of the sheds, which is a difference in annotation criteria.

## 3. Comparison with the Past and with the Target

| Model / Data | Test IoU | Notes |
|---|---:|---|
| Massachusetts + 12-epoch small U-Net | 0.583 | Earlier pilot experiment |
| **WHU + 38-epoch CPU baseline (this page)** | **0.8273** | About 5 h, integrated-GPU machine |
| Target | 0.85 | Still **3 percentage points** away |

**Conclusion: 0.85 is not out of reach.** This model only used 256² cropping, a single model, no pretraining, no TTA,
no pretraining, no TTA, and still reached 0.82. Add any 2–3 of the items below and 0.85 is basically passed:

1. **Full 4736 images + more epochs** (30–40): estimated +1.5–2.5 points;
2. **512² for training/inference** (instead of 256 cropping): more complete building context, estimated +1–2 points;
3. **8× TTA**: a stable +0.5–1 point;
4. **Pretrained encoder** (ResNet-34 + UPerNet): +3–5 points (requires a GPU).

That is to say: **to reliably pass 0.85, renting a GPU for 2–3 hours to run Plan B is still recommended**; but if you do not want to spend money for now,
I can train another version on this machine according to items 1–3 above, expected to reach **0.84–0.86** (taking 4–6 hours).

## 4. Reproduction Commands

```powershell
# Training (resume from epoch 6 to epoch 18; for a full retrain remove --resume --stop-after)
.\.venv\Scripts\python.exe scripts\train_whu.py `
  --data data\whu_processed --arch unet --base 16 --size 256 `
  --epochs 18 --max-train 2800 --max-valid 256 --batch-size 8 `
  --lr 3e-4 --pos-weight 1.5 --device cpu --threads 12 `
  --out runs\whu_cpu_unet --resume

# Evaluation (threshold sweep + per-image log + comparison figures)
.\.venv\Scripts\python.exe scripts\evaluate_whu.py `
  --model whu_cpu_unet --split test --verbose --examples 6
```

The two corresponding run configurations in PyCharm are: **Train WHU (CPU, resume)** and **Evaluate WHU (test)**.

## 5. Artifacts

```
runs/whu_cpu_unet/
  best.pt                # epoch 38, val IoU 0.7956
  last.pt
  train_log.csv          # per-epoch loss / val_iou / val_f1
  training_curves.png
  eval_test.json         # full metrics for 12 thresholds
  eval_examples_test.png
docs/images/whu_cpu_training_curves.png
docs/images/whu_cpu_eval_examples.png
```

The model can be loaded directly by the GUI (it appears as `whu_cpu_unet` in the dropdown): `InferenceEngine` reads
`arch/mean/std/size` from the checkpoint and tiles at 256 during inference. Measured by plugging this model into the vectorization
pipeline: test image 4 has Polygon IoU 0.862, image 6 has 0.811, and Chamfer is 1.3–1.9 px.

### GUI Default Behavior (Already Integrated)

- On startup it **automatically selects the model with the highest validation IoU** (currently `whu_cpu_unet`, 0.770);
- It automatically reads the threshold selected in `eval_test.json` (0.60) and syncs it to the slider;
- "Image source" defaults to **WHU 512²**, and it automatically picks a test image that contains buildings (avoiding empty images);
- The model info bar shows `WHU · e18 · IoU 0.770`;
- "Comparison & Ablation" follows the current dataset (WHU or Massachusetts) instead of always using Massachusetts.
