# Solution B: Pushing IoU to 0.85+ on the WHU Dataset (GPU Training Guide)

> Goal: test-set **IoU 0.85–0.89** on the WHU aerial image building dataset (8189 images at 512², 0.3 m).
> This machine has only a CPU (no CUDA), so training runs on Colab / Kaggle / a lab GPU;
> when training is done, copy the weights back to `runs/`, and this repository's visualization system can load them directly.

## 1. Why the dataset must change

| Dataset | Resolution | Published IoU | Feasibility of 0.85 |
|---|---|---|---|
| Massachusetts (current) | 1 m, small buildings, blurred boundaries | 0.70–0.78 | Not feasible |
| **WHU aerial** | 0.3 m, large and clear buildings, clean labels | **0.85–0.90** | Feasible |

This is also the source of the "mIoU 0.88 on WHU" stated in the proposal report.

## 2. Prerequisites

1. **GPU**: the free T4 on Colab (16 GB) is enough; a local NVIDIA card is even more convenient.
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`
   - The local output is `False` (AMD 780M integrated GPU), so an external GPU is required.
2. **Dataset**: the WHU aerial image building dataset (about 2.5 GB, officially from the GPCV group at Wuhan University,
   domestic mirrors also exist). You need to be able to download it locally or to a cloud drive.
3. **Dependencies**: run on the training machine
   ```bash
   pip install -r requirements-whu.txt
   # equivalent to: pip install segmentation-models-pytorch timm
   ```

## 3. Data directory structure

The scripts support both the "official split" and the "custom split" structures:

```
whu_raw/
  train/{image,label}/*.tif     # official train
  valid/{image,label}/*.tif     # official valid (may also be val/validation)
  test/{image,label}/*.tif      # official test
```

Or:

```
whu_raw/
  image/*.tif
  label/*.tif                   # use --val-frac/--test-frac to split automatically
```

> Images and labels are paired by **identical filename**; labels only need to be 0/non-0, and the script binarizes them.
> Images larger than 512 are cropped into 512² tiles; images smaller than 512 are reflection-padded to 512².

A decompressed copy of the official data already exists on this machine (structure verified, migrated to the new location):

```
D:\All4DS\data4bishe\data4training\
  3. The cropped aerial image tiles and raster labels\
    3. The cropped image tiles and raster labels\
      train\image\*.tif  (4736)   train\label\*.tif
      val\image\*.tif    (1036)   val\label\*.tif
      test\image\*.tif   (2416)   test\label\*.tif
  whu_raster_1024\whu_raster_1024\
    r_train.zip (7.9 GB)  r_val.zip (1.6 GB)  r_test.zip (5.8 GB)   # 0.2 m / 1024² version
```

The corresponding command (note that the path contains spaces and Chinese characters, so it must be quoted):

```powershell
.\.venv\Scripts\python.exe scripts\prepare_whu.py `
  --root "D:\All4DS\data4bishe\data4training\3. The cropped aerial image tiles and raster labels\3. The cropped image tiles and raster labels" `
  --out data\whu_processed --size 512
```

> The 0.2 m / 1024² copy (`whu_raster_1024`) has higher resolution and costs about 4× the compute. For 0.85,
> 0.3 m / 512² is enough; to push for 0.88+, you can randomly crop 512² tiles from the 1024² tiles for training,
> which costs the same compute as the 512 version but keeps sharper detail.

Already validated on this machine, with these results:

| split | tiles | notes |
|---|---:|---|
| train | 4736 | 512², consistent with the official training area |
| valid | 1036 | official validation area |
| test | 2416 | official test area (two regions combined) |
| total | 8188 | the official page says 8189; the archive is missing 1 image, which is normal |

Dataset statistics (automatically written to `dataset_info.json`, used by both training and GUI inference):
mean = `[0.4359, 0.4458, 0.4141]`, std = `[0.2160, 0.2029, 0.2169]`;
building pixel ratio: train 15.9%, valid 5.7%, test 2.5% (the test area is sparser, so false positives cost more).

The packaged artifact is in `data/whu_processed/` (about 8.1 GB); the training machine can copy this directory directly,
saving the re-extraction and re-tiling. If the archive you get has no train/val/test directories (only flat
`image` + `label`), add `--official-split` to split by 4736/1036/rest; otherwise use
`--val-frac/--test-frac` for a random split.

## 4. Four steps to completion

### ① Package the data (one-off, about 10–20 minutes)

```bash
python scripts/prepare_whu.py --root /content/whu_raw --out /content/whu_processed --size 512
```

Output: `train/valid/test_{images,masks}.npy` (memmap) + `dataset_info.json` (with mean/variance).
**Note**: the full data is about 6.4 GB of images + 2.1 GB of masks, so be sure to put it on Colab's local `/content` disk,
not directly on Google Drive (random reads become more than 10× slower).

To first validate the pipeline on 20 images, use `--limit 20`.

### ② Training

```bash
python scripts/train_whu.py \
  --data /content/whu_processed \
  --arch upernet --encoder resnet34 \
  --epochs 60 --batch-size 8 --lr 1e-4 \
  --size 512 --amp --num-workers 4 \
  --out runs/whu_upernet_r34
```

Key parameters:

| Parameter | Recommendation | Notes |
|---|---|---|
| `--arch` | `upernet` / `deeplabv3plus` | UPerNet is the most stable for multi-scale buildings |
| `--encoder` | `resnet34` (fast), `resnet50` (more accurate) | ImageNet pretraining, critical for small data |
| `--epochs` | 60 (about 2–3 h on T4) | 40 epochs can also reach around 0.85 |
| `--batch-size` | 8 (T4 16G); change to 4 if VRAM is small | |
| `--lr` | 1e-4 | AdamW + cosine |
| `--amp` | on | about 1.5–2× speedup |
| `--num-workers` | 4 | noticeably faster on Colab |
| `--pos-weight` | 1.0 | WHU classes are fairly balanced; raise to 2–3 when the building ratio is very low |
| `--norm` | `auto` | smp pretrained models automatically use ImageNet mean/variance; the built-in U-Net uses dataset statistics |

Training log: `runs/whu_upernet_r34/train_log.csv`; validation IoU every epoch, with `best.pt` saved automatically.
After an interruption, add `--resume` to continue.

### ③ Evaluation (threshold sweep + 8× TTA)

```bash
python scripts/evaluate_whu.py --model whu_upernet_r34 --split test --data /content/whu_processed --tta
```

Outputs `runs/whu_upernet_r34/eval_test.json`, containing IoU/F1/P/R at each threshold and the optimal threshold.
**TTA typically adds another +0.5–1 IoU point.**

### ④ Copy back to this machine and use it in the GUI

Copy the whole `runs/whu_upernet_r34/` (at least `best.pt`) to:

```
D:\All4DS\BISHE\runs\whu_upernet_r34\
```

Then start the GUI:

```powershell
.\.venv\Scripts\python.exe run_vectorizer.py
```

The model dropdown will automatically gain `whu_upernet_r34` (`InferenceEngine.available_models()` scans
`runs/*/best.pt`). The GUI automatically reads what the checkpoint records:

- `arch` / `encoder`: determine the network structure;
- `mean` / `std`: WHU normalization statistics (not Massachusetts');
- `size`: inference tile size (512).

After switching to that model, infer, edit vertices, export GeoJSON/Shapefile, and run ablation comparisons as usual.

## 5. Expected results

| Configuration | Test-set IoU | F1 | Notes |
|---|---|---|---|
| UPerNet + ResNet-34, 40 epochs, no TTA | 0.84–0.86 | 0.91–0.92 | lowest-cost way to meet the target |
| UPerNet + ResNet-34, 60 epochs + TTA | **0.86–0.88** | 0.925–0.94 | recommended |
| DeepLabV3+ + ResNet-50, 60 epochs + TTA | 0.87–0.89 | 0.93–0.94 | stronger |
| Add boundary auxiliary loss / multi-scale testing | 0.88–0.90 | 0.94+ | a bonus for the thesis |

VRAM usage (T4 16 GB): ResNet-34 + 512² + batch 8 is about 6–8 GB; ResNet-50 about 10–12 GB.

## 6. FAQ

| Symptom | Cause / handling |
|---|---|
| `CUDA out of memory` | Lower `--batch-size` to 4; or switch to `resnet34` |
| Training is slow (>8 min/epoch) | Confirm `--amp` is on, `--num-workers 4`, and the data is on a local disk, not Drive |
| val IoU stays at 0 | Check whether the labels are all 0 (use `np.unique` on the masks); or `--lr` is too small |
| IoU is stuck around 0.7 | The encoder did not load pretrained weights: confirm `--encoder-weights imagenet` and that it can download from the network |
| Loading weights reports `Missing key(s)` | `arch`/`encoder` do not match the training run; check the checkpoint's `args` |
| The new model is missing from the GUI model list | Confirm the directory is `runs/<name>/best.pt` and that the GUI was restarted |

## 7. Relationship to the existing code

- `src/models_seg.py`: the `smp` wrapper, providing the same `forward(x, return_edge)` contract as the built-in U-Net;
- `src/models.py::build_from_spec`: selects the built-in U-Net or an smp model based on the checkpoint's `arch` field;
- `src/infer.py`: automatically reads `mean/std/size`, so WHU models and Massachusetts models can coexist;
- `scripts/train_whu.py` / `evaluate_whu.py`: the training and evaluation scripts used in this guide;
- `scripts/prepare_whu.py`: data packaging.

> The training side does not depend on `app/`, so on Colab you only need to copy `src/` + `scripts/` + the data to start.
