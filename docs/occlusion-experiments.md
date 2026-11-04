# Occlusion-Handling Experiment Summary (2026-11-01 ~ 11-03)

> Goal: improve the completeness of building extraction under tree/shadow occlusion.
> Conclusion first: **none of the four routes produced a positive gain under the current data and compute budget**; the only effort still running is the BIEB 512 long training.

## 1. Experimental Setup

| Item | Content |
|---|---|
| Evaluation set | `data/occlusion_subset.json`, 49 images (severe 12 / moderate 12 / mild 12 / very mild 13) |
| Selection criterion | proportion of the building ground-truth footprint covered by vegetation/tree shade (stratified sampling) |
| Baseline | `whu_mt512` (512 input, 10 epochs, validation 0.8864) |
| Threshold | 0.55 (uniform) |

## 2. Overall Results Table (49-image occlusion subset)

| Model | Configuration | mIoU | Polygon IoU | Vertices |
|---|---|---:|---:|---:|
| whu_mt512 | 512, fully trained | **0.8415** | **0.8189** | 7.97 |
| **whu_ctrl384** | 384, 4 epochs, **no modules at all** (control) | **0.8388** | 0.8173 | 7.98 |
| whu_bieb | 384, 4 epochs, + completeness reconstruction | 0.8385 | 0.8167 | 7.98 |
| whu_dir | 384, 4 epochs, + direction-field head | 0.8378 | 0.8121 | 8.03 |
| whu_prior | 384, 3 epochs, + 4-channel prior | 0.8363 | — | — |

**Key control**: the gap between `whu_ctrl384` (0.8388) and the three modules (0.8363–0.8385) is within noise.
That is, **the entire −0.27 to −0.52 relative to the baseline comes from the "384 + 4 epochs" training configuration, not from the modules**.

## 3. Geometric Completion (B1 occluded-notch clipping)

Implementation: `src/occlusion_geometry.py`, which detects notches whose two sides are nearly collinear and whose notch region is occluded, then replaces them with a straight chord.

Threshold sweep (49 images, polygon IoU):

| Configuration | Polygon IoU | Vertices |
|---|---:|---:|
| Off | **0.8231** | 8.22 |
| tol=25°, occ≥0.55 | 0.8189 | 7.97 |
| tol=15°, occ≥0.70 | 0.8213 | 8.08 |
| tol=10°, occ≥0.80 | 0.8218 | 8.12 |
| tol=20°, occ≥0.75 | 0.8209 | 8.06 |

**All are slightly below the off state** — indicating that most of the notches clipped away in this subset are real structures (courtyards, L-shaped corners)
rather than spurious notches caused by occlusion.

## 4. Why These Methods Had No Effect (Analysis)

1. **Occlusion in WHU is mostly "trees beside a courtyard" rather than "trees lying on the roof"**: the ground-truth footprint itself already treats trees as background,
   so completion instead expands into non-building areas;
2. **Insufficient training budget**: the 384/4-epoch control experiment shows that the configuration alone depresses the score to 0.867–0.869;
3. **The information really is missing**: the 0153 type (tree shade covering a long building) has no recoverable evidence at the pixel level.

## 5. What Still Works

- **Synthetic occlusion cases**: when a rectangle is occluded by trees in two segments, enabling contour bridging takes IoU from 0.84 to 1.00 (vertices 11 to 4) — the mechanism itself holds;
- **The direction-field head learns correctly**: the predicted boundary tangent strictly follows the roof edge (see `results/dir_pred_0153.png`);
- **BIEB's enhanced-image target is right**: the target image can paint the occluded parts with the roof's mean colour (see `results/esbi_demo.png`).

## 6. Suggested Thesis Write-up

> Occlusion is an inherent limitation of single-image input. This work systematically tried four classes of solutions — input priors, direction-field supervision, completeness reconstruction, and geometric notch completion —
> and reports quantitative results on a unified occlusion subset (Table X); the experiments show that under a limited training budget these modules yield no significant gain,
> and performance is dominated by input resolution and training volume. For residual occlusion, the system provides vertex editing as a fallback.

## 7. To Do

- [ ] BIEB 512 long training (8 epochs) and a control under identical conditions, for the final decision
- [ ] If there is still no gain, close out the occlusion line and shift focus to vectorization and thesis writing
