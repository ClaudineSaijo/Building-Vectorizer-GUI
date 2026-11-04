# Daily Report 2026-11-02

## 1. Today's Output

### 1.1 Three New Models (Compared on the Same Batch of 300 Test Images)

| Model | Configuration | Best IoU | Relative to Baseline |
|---|---|---:|---:|
| `whu_mt512` (baseline) | 512, base16, multi-task | **0.8307** | — |
| `whu_dir` | 384, +direction-field head, 4 epochs | 0.8265 | −0.42 |
| `whu_prior` | 384, +4-channel prior, 3 epochs | 0.8202 | −1.05 |

> Note: The absolute values on the 300-image subset are lower than the 0.8626 on the full 2416 images because the proportion of hard samples is higher; the three models are comparable on the same subset.

**Conclusion**: Neither of the two "input prompt / auxiliary task" schemes improved mask accuracy, so both are kept as negative ablation results.

### 1.2 Code Implementation

| File | Contents |
|---|---|
| `src/direction_field.py` | Boundary direction-field target generation (contour tangential direction) + cosine loss |
| `src/enhanced_image.py` | Label-derived "occlusion-free enhanced image" (ESBI) |
| `src/models.py` | Added `dir_head` (2ch) and `bieb_head` (3ch), each +0.06 M parameters |
| `scripts/train_whu.py` | `--direction-head` / `--bieb` and the corresponding loss weights |
| `scripts/day_runner.py` | Failsafe warning for a missing `--resume` |

### 1.3 base=24 Capacity Experiment (In Progress)

- Epoch **11/20**, best val IoU **0.7988** (epoch 9);
- The window closed automatically at 22:55 and paused; it will resume at 06:30 tomorrow morning and run to epoch 20;
- Expected to finish **around 11:00**.

## 2. Today's Problems

### Missing `--resume` caused 4.5 hours of wasted compute

When the window resumed at 16:00, the base=24 job restarted from epoch 1 because the command did not include `--resume`; everything up to epoch 9 reached in the morning (val 0.799) was lost, and `best.pt` was overwritten as well.

**Fixed**:
1. Added `--resume` to the base=24 job;
2. `day_runner.py` now checks: if `last.pt` already exists under `--out` but the command has no `--resume`, it logs a warning directly.

## 3. Conclusions on Data and Methods

1. **Adding an input channel (4ch) does not work**: val −1 point, consistent with the BIENet paper's conclusion that "prompts are worse than supervision";
2. **The direction-field head does learn correctly** (the predicted arrows lie strictly along roof edges), but 4 epochs of training are not enough to improve the mask;
3. **GCP's collinearity simplification is not better than our DP under the same vertex budget** (yesterday's conclusion, unchanged today);
4. The real next step is **BIEB completeness reconstruction** — using the "occlusion-free enhanced image" as the reconstruction target to force the network to complete the occluded parts.

## 4. Tomorrow's Plan

| Time | Task |
|---|---|
| 06:30–11:00 | base=24 resumed run (epochs 11→20) |
| From 11:00 | **Completeness reconstruction branch training** (384, 4 epochs, about 2 hours) |
| From 13:00 | BIEB evaluation (pixel + vector + occlusion subset) |
| Afternoon | If BIEB works → formal 512 training; if not → wrap up and switch to paper writing |

## 5. To-Do List

- [ ] "Capacity vs resolution" conclusion once base=24 finishes
- [ ] BIEB occlusion-subset evaluation (including 0153)
- [ ] Update the vector evaluation protocol (add two rows for direction field / BIEB)
- [ ] Organize the figures and tables for Chapters 3 and 4
