# Project status — living snapshot

> Keep this file up to date. Any new session (or any context compaction) can resume
> from here without re-reading the transcript.

Last updated: 2026-11-04 18:00

---

## 1. What the project is

Building extraction from aerial imagery, end to end:

```
U-Net probability map -> threshold -> instance separation -> contour
-> vertex fit (DP seeds + line fit + intersection) -> rectilinear regularization
-> occlusion completion -> GeoJSON / Shapefile
```

GUI in `app/`, CLI entry `run_vectorizer.py`, batch comparison in `app/compare.py`.

---

## 2. Confirmed numbers (do not re-derive)

### Segmentation (WHU 0.3 m, 512 px input, test split 2416 tiles)

| Model | val IoU | test IoU | notes |
|---|---:|---:|---|
| 256 px baseline | 0.7956 | — | early model |
| multi-task 256 | 0.8239 | — | + edge head |
| 384 px | 0.8681 | — | |
| **`whu_mt512`** | **0.8864** | **0.8626** | current main model, F1 0.9263 @0.55 |
| `whu_bieb512` (4 epochs) | 0.8864 | — | BIEB, parity with baseline |
| `whu_shadow` (6 epochs, 384) | 0.8703 | — | +1.4 dark-region recall |

### Vector level (512 model, 150 test tiles, configs A–E)

| Config | PolyIoU | Vertices | Right-angle rate | Chamfer |
|---|---:|---:|---:|---:|
| mask contour only | 0.582 | 20.2 | 0.390 | 1.49 |
| + vertex fitting | 0.568 | 6.2 | 0.546 | 1.46 |
| + regularization | 0.567 | 6.1 | **0.642** | 1.48 |
| + occlusion completion | 0.567 | 6.1 | 0.641 | **1.43** |

Trade: **−1.5 PolyIoU for −70 % vertices and +25 pt right-angle rate.**

### Diagnostic (300 test tiles) — the key finding

| Slice | share of GT | recall |
|---|---:|---:|
| visible roof | 93.0 % | 0.933 |
| dark (shadow + dark canopy) | 6.2 % | 0.701 |
| bright green canopy | 0.9 % | 0.740 |

| Distance to GT boundary | recall |
|---|---:|
| 0–1.5 px | **0.700** |
| 1.5–3 px | 0.890 |
| 3–5 px | 0.934 |
| >10 px | 0.975 |

**Conclusion: the "occlusion loss" is mostly a boundary-localisation loss.**

| Occluded blob size | recall |
|---|---:|
| <50 px (thin slivers) | 0.507 |
| 50–150 px | 0.618 |
| 150–400 px | 0.817 |
| >1000 px | 0.940 |

---

## 3. What has been tried on occlusion (and failed)

| Idea | Result |
|---|---|
| 4th input channel (ExG / dark prior) | −1.05 IoU |
| Boundary direction field head | −0.42 |
| BIEB, 4 epochs @384 | −0.30 |
| Amodal double mask, 10 epochs @384 | −0.01 (noise) |
| Geometric notch cutting | −0.42 PolyIoU |
| Bridge reconnection (`src/bridge.py`) | −5.1 at gap ≤14; +0.29 only at gap ≤4 |
| Bright-ridge extension (`src/ridge_extend.py`) | −0.67 |
| Gradient edge snapping | −2.85 |
| Retinex shadow lifting | −8.4 |
| Multi-scale TTA (512+448) | −0.25 |
| Mask dilation ±1–2 px | −2.6 / −9.3 |
| **Shadow-weighted loss** | **+1.4 dark-region recall** (only positive) |
| **BIEB @512, 4 epochs** | **+2.0 IoU on tile 0153**, parity overall |

---

## 4. Reference tile 0153 (hard case)

- GT building with a long thin wing under canopy.
- 3,783 missed px: roof-coloured 45.9 %, dark 17.4 %, green 12.4 %.
- Dark green canopy and shadowed roof are **not separable** at pixel level
  (AUC 0.52–0.57 for brightness, texture and Laplacian variance).
- The 79.5 % of dark canopy already falls inside the "dark" rule, so the combined
  (dark ∪ green) mask is the correct supervision set.

---

## 5. Current run

**HR-Amodal** — the most promising occlusion plan (running 2026-11-04):

```
data    : G:/BISHE_data/whu02  (WHU 0.2 m, 2943 train / 627 valid / 2220 test, 512 px native crops)
model   : full + BIEB head
loss    : BCE/Dice + edge + BIEB + boundary-weighted (×4 within 2 px) + dark-weighted + green-weighted
epochs  : 8  (≈40 min/epoch on CPU)
init    : runs/whu_mt512/best.pt
out     : runs/whu02_hr
```

Acceptance criteria: boundary-band recall ≥0.85, dark-region recall ≥0.88,
tile 0153 mIoU ≥0.80.

---

## 6. Data locations

| Path | Content |
|---|---|
| `data/whu_processed` (D:) | 0.3 m, 512 px, 4736 / 1036 / 2416 |
| `G:/BISHE_data/whu02` | 0.2 m, 512 px native crops, 2943 / 627 / 2220 |
| `G:/BISHE_data/processed` etc. | junctions for older datasets |
| `data/vertex_labels` | WHU-Mix COCO polygons (43,585 train tiles) |

---

## 7. Open decisions

1. If HR-Amodal hits the acceptance numbers → adopt 0.2 m as the main pipeline.
2. If it fails → next candidate is distance-field (TSDF) supervision.
3. Cloud GPU is unavailable; 1024 px training is not feasible locally.
