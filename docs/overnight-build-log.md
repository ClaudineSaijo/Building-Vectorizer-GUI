# Overnight Automated Build Log (2026-11-01 01:30 – 02:05)

> The user has gone to rest and authorized automatic execution. This file records what was done during this period, how far it was verified, and where to continue tomorrow morning.

## 1. Completed

### 1.1 Vertex Regression Pipeline (true-label route, corresponding to the selected option ②)

| File | Purpose | Status |
|---|---|---|
| [scripts/prepare_mix_coco.py](scripts/prepare_mix_coco.py) | WHU-Mix COCO annotations → per-image polygon JSON (`data/vertex_labels/<split>.json`), plus vertex distribution statistics | ✅ Compiles, pending verification on real data |
| [src/vertex_gcn.py](src/vertex_gcn.py) | EdgeConv vertex-offset network (3 layers, k=8, hidden 64, ~40k parameters) + equal-arc-length resampling + node features | ✅ Trainable on synthetic data |
| [scripts/train_vertex.py](scripts/train_vertex.py) | Training script: Chamfer + one-way snapping + edge-length smoothing, resumable from checkpoints | ✅ Runs |
| [scripts/eval_vertex.py](scripts/eval_vertex.py) | GCN vs. geometric baseline (DP simplification): polygon IoU / vertex Chamfer / vertex count | ✅ Runs |

### 1.2 Synthetic-Data Verification Results (160 rectangular buildings)

| Scheme | Polygon IoU | Vertex Chamfer | Vertex count |
|---|---:|---:|---:|
| Geometric baseline (DP) | **0.977** | **1.26** | **4.0** |
| GCN (first version, offset cap 12 px) | 0.891 | 9.55 | 23.5 |
| GCN (offset cap 64 px) | 0.933 | 5.00 | 25.1 |
| GCN (cap 96 px + faster learning rate) | 0.958 | 3.91 | 20.9 |

**Conclusion: the Chamfer-style "nearest-vertex snapping" loses to the geometric baseline.** The problem is not whether the code runs, but the **supervision form**: many nodes stop on edges, the loss is still hovering around ~23.8 by epoch 150, and there is no direct constraint on "fewer vertices".

### 1.3 Key Fix: Dense Regression Target + Ring Alignment (verified successfully at 02:25)

After switching to an L1 target that "resamples the GT polygon by arc length into 64 points, in one-to-one correspondence with the input nodes", a result **beating the geometric baseline** appeared for the first time:

| Scheme | Polygon IoU | Vertex Chamfer | Vertex count |
|---|---:|---:|---:|
| Geometric baseline (DP) | 0.9769 | **1.26** | **4.0** |
| GCN dense regression (unaligned) | 0.509 | 31.4 | 32.3 |
| **GCN dense regression (after ring alignment)** | **0.9822** | 2.41 | 5.7 |

That middle 0.509 was a real bug: the starting point of an OpenCV contour is arbitrary, whereas GT polygon resampling starts from vertex 0, so there is a phase difference and a direction difference between the two; per-node L1 then pulls nodes to the wrong positions. The fix is `align_ring()`: iterate over the 64 cyclic shifts × 2 directions, take the alignment with the smallest mean distance, and then feed it into the loss.

The remaining gap is in Chamfer (2.41 vs 1.26) and vertex count (5.7 vs 4.0), which belongs to post-processing tuning (DP threshold, corner snapping); continue tuning with real data tomorrow morning.

## 2. Tomorrow Morning's Priorities (already updated per the new conclusion)

1. **Retrain the dense-regression GCN on real data** (WHU-Mix, `--target dense`) to confirm that the conclusion drawn on synthetic data transfers;
2. Tune post-processing: DP threshold / direction snapping, to push Chamfer and vertex count below the baseline;
3. If it still fails to meet the bar on real data → honestly downgrade it to an ablation, and write the main line as "multi-task + boundary + geometric regularization + occlusion completion + vector evaluation".

## 3. Other Status

- Multi-task segmentation: `whu_mt_edge` at epoch 10/12, val IoU 0.8202 (peak 0.8239 at epoch 6), expected to finish at 02:20;
- The overnight plan `overnight_plan.py` is still waiting, after which it automatically enters the 384→512 fine-tuning;
- Downloads: `val/test1/test2` are complete (G drive); `train.zip` was deleted and is being re-downloaded due to an abnormal resume offset, currently about 400 MB / 22 GB;
- The daily queue `runs/dayplan/tasks.json` has had the vertex task's dependency changed to `data/vertex_labels/train.json`;
- `data/whu_processed` (8 GB) is still on the D drive (in use for training); migrate it to the G drive after 22:30.

## 4. Automatic Check at 06:35 Tomorrow Morning

Reminder set: check the overnight results → confirm downloads → unzip → run the COCO conversion → retrain the GCN on real data → update the queue → remind to launch `start_day_runner.bat`.
