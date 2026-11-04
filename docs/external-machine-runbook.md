# External Machine Runbook (RUNBOOK)

> This package can reproduce training and evaluation on another machine
> (Windows / Linux, CPU or GPU).
> Goal: after running the specified experiments, pack `runs/` and `results/` and send them back.

---

## 0. One-Minute Overview

```bash
# 1) Unpack and enter the directory
cd BISHE

# 2) Install dependencies (for GPU, install the CUDA build of torch from the official site first)
pip install -r requirements.txt

# 3) Generate data (if the package does not contain data/whu_processed)
python scripts/prepare_whu.py --root data/raw --out data/whu_processed --size 512

# 4) Train (on GPU use --device cuda --amp; on CPU drop these two flags)
python scripts/train_whu.py --data data/whu_processed --arch full --base 16 \
  --size 384 --shadow-aug 0.5 --shadow-weighted-loss --epochs 6 --batch-size 4 \
  --lr 2e-5 --device cuda --amp --init-from runs/whu_mt512/best.pt \
  --out runs/whu_shadow --resume

# 5) Evaluate
python scripts/evaluate_whu.py --model whu_shadow --split test --limit 300
python scripts/diagnose.py --model whu_shadow --limit 300

# 6) Pack and send back
zip -r results_back.zip runs results
```

---

## 1. Environment Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 – 3.12 (3.13 also works, but some wheels may be missing) |
| RAM | ≥ 8 GB (16 GB recommended for 512 input training) |
| VRAM | Optional; 6 GB runs 512/batch 8, 12 GB runs 1024/batch 4 |
| Disk | Code 2 MB + raw data 700 MB + processed data 8 GB |

Install:

```bash
pip install -r requirements.txt
# GPU machine: install the CUDA build of torch first, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Data Preparation

### Case A: the package already contains `data/whu_processed/`

Skip directly to step 3. The directory should contain:

```
data/whu_processed/
  train_images.npy  train_masks.npy
  valid_images.npy  valid_masks.npy
  test_images.npy   test_masks.npy
  meta.json
```

### Case B: the package contains only `data/raw/` (108 tif files)

```bash
python scripts/prepare_whu.py --root data/raw --out data/whu_processed --size 512
```

Takes about 10–20 minutes and produces 8 GB of memmap files.

> If you have the original WHU archive (0.3 m cropped tiles), just unpack it into
> the structure `data/raw/{train,valid,test}/{sat,map}/xxx.tif`.

---

## 3. Experiment List (by priority)

### E1 Shadow-aware training (**highest priority**)

```bash
python scripts/train_whu.py --data data/whu_processed --arch full --base 16 \
  --size 384 --shadow-aug 0.5 --shadow-weighted-loss \
  --epochs 6 --batch-size 4 --lr 2e-5 --pos-weight 1.5 --edge-weight 0.5 \
  --device cuda --amp --num-workers 4 \
  --init-from runs/whu_mt512/best.pt --out runs/whu_shadow --resume
```

| Platform | Estimate |
|---|---|
| CPU 12 threads | 3 hours |
| RTX 4090 | 8–12 minutes |

> Requires `runs/whu_mt512/best.pt`. If the package does not include it, contact us
> as described in E4 to obtain it, or drop `--init-from` (training from scratch,
> which will perform worse).

### E2 1024 resolution (**largest unknown gain**, GPU only)

```bash
python scripts/train_whu.py --data data/whu_processed --arch full --base 16 \
  --size 1024 --epochs 8 --batch-size 4 --lr 1.5e-5 \
  --pos-weight 1.5 --edge-weight 0.5 \
  --device cuda --amp --num-workers 4 \
  --init-from runs/whu_mt512/best.pt --out runs/whu_1024 --resume
```

| Platform | Estimate |
|---|---|
| RTX 4090 | 40–60 minutes |
| CPU | Not recommended (about 30 hours) |

### E3 BIEB completeness branch 512 (next priority)

```bash
python scripts/train_whu.py --data data/whu_processed --arch full --base 16 \
  --size 512 --bieb --bieb-weight 0.2 --epochs 8 --batch-size 4 --lr 1.5e-5 \
  --pos-weight 1.5 --edge-weight 0.5 --device cuda --amp --num-workers 4 \
  --init-from runs/whu_bieb/best.pt --out runs/whu_bieb512 --resume
```

### E4 Evaluation and diagnosis (after training finishes)

```bash
# Full test set threshold sweep
python scripts/evaluate_whu.py --model whu_shadow --split test

# Shadow subset comparison
python scripts/eval_occlusion.py --subset data/shadow_subset.json \
  --models whu_mt512 whu_shadow --threshold 0.55

# Error source diagnosis (by region / by size / FP-FN composition / confidence)
python scripts/diagnose.py --model whu_shadow --limit 300
```

---

## 4. What to Send Back

```
runs/<experiment name>/best.pt          ← weights (most important)
runs/<experiment name>/train_log.csv    ← per-epoch metrics
runs/<experiment name>/eval_test*.json  ← evaluation results
results/*.json                          ← diagnosis and comparisons
```

Packing:

```bash
# Linux / macOS
zip -r results_back.zip runs results
# Windows
Compress-Archive -Path runs,results -DestinationPath results_back.zip
```

If the results are large, sending back only `runs/*/best.pt` plus all `*.csv` / `*.json` is enough.

---

## 5. Notes

1. **Do not modify the files in `data/`**, otherwise the results are not comparable;
2. Training uses `seed=42` by default; do not change the random seed in the `.py` files;
3. If VRAM is insufficient, lower `--batch-size` to 2 or 1, **keeping all other parameters unchanged**;
4. After an interruption, rerun the same command (with `--resume`) to continue training;
5. Each epoch's log is written to `runs/<experiment name>/train_log.csv`; if NaN appears or val IoU stops improving for a long time, take a screenshot and let us know.

---

## 6. Expected Values for Each Experiment (to judge whether it ran correctly)

| Experiment | Starting point | Expected val IoU | Notes |
|---|---|---:|---|
| E1 shadow-aware (384, 6 epochs) | whu_mt512 | 0.86–0.88 | Shadow pixel recall should rise from 0.84 to 0.87+ |
| E2 1024 (8 epochs) | whu_mt512 | 0.88–0.91 | Resolution increase, unknown upper bound |
| E3 BIEB (512, 8 epochs) | whu_bieb | 0.87–0.89 | Compare with a control under the same conditions |

If the values deviate noticeably (e.g. < 0.80), the data or the starting weights are probably wrong; please verify them first.
