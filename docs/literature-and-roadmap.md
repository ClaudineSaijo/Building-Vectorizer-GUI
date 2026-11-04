# Literature Review and Roadmap Adjustments (Night of 2026-11-01)

> 8 PDFs have been read (full texts are in `papers/*.txt`). This document records: the usable part of
> each paper, the measured conclusions I obtained after reproducing them myself, and the impact on the
> route for Chapters 3/4.

## 1. Occlusion-related (3 papers)

### BIENet (TGRS 2024) — Borrowable, moderate cost

- **Approach**: use the **labels** to synthesize an "enhanced image" from the original image — take the
  interior mean color of each building → blend it with the original image by a ratio →
  brighten the foreground and darken the background, yielding an image that is "occlusion-free, high-contrast,
  and has complete contours";
  then use an **extra decoder (BIEB) to reconstruct this enhanced image** as auxiliary supervision, with the
  segmentation branch using both the encoder features and the BIEB features;
  plus residual learning (BRL).
- **Results**: on WHU, **+1.49 IoU / +0.82 F1** over HD-Net (with gains on Massachusetts, Inria, and AIRS as well).
- **Code**: github.com/tqwhdx19/bienet.
- **Significance to us**: it proves that "**giving a target**" (reconstructing the enhanced image) is more
  effective than "**giving a hint**" (the 4-channel prior we added).
  Our 4-channel experiment dropped val from 0.8864 to 0.8676→0.8504, which corroborates exactly this point.

### BOMSC-Net (TGRS 2022) — The closest fit to our occlusion problem

- **Problem statement**: explicitly points out that "occlusion by trees and shadows makes extraction results
  **discontinuous**".
- **Module**: Direction Field Optimization Module (DFOM) — predicts a **direction field** and uses the
  directional information to correct boundaries.
- **Significance to us**: the direction field is precisely "which way the boundary should go", and across
  occluded regions it can be continued along the field;
  this is more systematic than the hand-written "directional-consistency straight-line bridging" we have, and it is learnable.
  It has the **lowest cost** (adding a 2-channel head) and is **the most worthwhile thing to do first**.

### PCL–PTD Net (JSTARS 2023) — Low reference value

Parallel cross learning plus pixel transfer deconvolution, an encoder structural trick; it does not fit the
"vector output" goal, so it is cited only as related work.

## 2. Vectorization/Polygonization (4 papers)

### GCP: Global Collinearity-Aware Polygonizer (TGRS 2025) — Reproduced; the conclusion is "no clear advantage"

- **Approach**: Mask2Former mask → sample polylines → Transformer regression refinement → **collinearity-aware
  DP simplification**
  (objective `L(p, p̂) + λ|p̂|`, where L is the sum of distances from the removed points to the corresponding
  line segments), globally optimal.
- **Paper's claim**: when applied to arbitrary polylines, this simplification module outperforms Douglas-Peucker.
- **I reproduced the simplification module** (`src/polygonize_gcp.py`) and measured it on our 97 predicted polygons:

| Method | Mean vertices | Chamfer to the original polygon |
|---|---:|---:|
| DP (ε=1.2%) | **6.64** | **0.938** |
| GCP λ=1.6 | 8.55 | 0.768 |
| GCP λ=3.0 | 7.52 | 0.904 |
| GCP λ=5.0 | 6.79 | 1.052 |
| GCP λ=8.0 | 6.03 | 1.261 |

**Conclusion**: under the **same vertex budget** (≈6.6), DP's Chamfer is in fact better (0.938 vs 1.05).
The value of GCP is that it provides a **smooth, adjustable "fidelity–compression" curve** (monotonically
controlled by λ),
rather than "fewer vertices".

→ **Significance to us**: this shows that **the bottleneck is not the "simplification algorithm" but the
jaggedness of the mask itself**.
This also explains why GCN cannot push the vertex count down: the input contour itself is jittery.

### Machine-learned Regularization and Polygonization (ICPR 2021)

- **Approach**: FCN segmentation → **GAN-regularized boundaries** (closer to right angles/regular outlines) →
  then train a CNN to predict **sparse corners** from the regularized result.
- **Significance to us**: the order "regularize first, then take corners" is clever; we currently "take corners
  first, then regularize".
  This can serve as a comparison scheme for Chapter 4 (if time permits).

### CornerDYO (IEEE Access 2026, open access)

- **Approach**: multi-scale features + bidirectional fusion + **corner-aware refinement** → joint prediction of
  corners and edges → planar graph construction.
- **Significance to us**: the most recent work closest to "detect feature points in pixels and then connect
  edges"; it is worth reading its corner annotation and connection rules closely.

### Vertex Aided Building Polygonization (JURSE 2023)

- **Approach**: RGB directly to polygon, **no preset vertex count, no thresholds**; AP 48.1%, IoU 84.1%.
- **Significance to us**: it proves that "end-to-end vertex output" is feasible; however, its metric definitions
  differ from ours, so only a qualitative comparison is possible.

## 3. Not applicable (2 papers)

- **Amodal dataset (TGRS 2024)**: MVB-Zurich / MSB-Dortmund are both **oblique multi-view** data, not
  orthophoto aerial images,
  and cannot be used directly for WHU; its "remote sensing boundary loss" can be borrowed.
- **PCL–PTD Net**: see above.

## 4. Roadmap adjustments

| Original plan | Adjusted | Reason |
|---|---|---|
| 4-channel prior as the main thrust | **Downgraded to an ablation** (wrap up as soon as round 3 finishes tomorrow) | Measured val −1~3 points; the literature also shows "a hint is not as good as a target" |
| None | **New: direction field head (2 channels)** | BOMSC proves it works, lowest cost, directly serves vector continuity |
| None | **New: completeness reconstruction branch (BIEB)** | BIENet proves it works; can be the second selling point of Chapter 3 |
| GCN to reduce vertices | **Reposition GCN as "fidelity refinement"**, leaving compression to DP/λ curves | Measured GCN vertices 7.3 > DP 6.2; the literature shows the bottleneck is in the mask |
| None | **New: GCP simplification with tunable λ**, used to plot the "vertex count vs Chamfer" curve | It makes a very nice figure in the paper |

## 5. Plan for tomorrow (11/2)

1. At 06:30, automatically resume the 4-channel run to round 3 (as an ablation; stop once results are out);
2. Implement the **direction field head**: GT boundary tangents (obtained from the gradient of the distance
   transform or from contour tangents),
   loss = cosine similarity on the boundary band; at inference time, bridge occlusions along the direction field;
3. If the direction field works, add BIEB (if time permits).
