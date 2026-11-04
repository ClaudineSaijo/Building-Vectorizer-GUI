# Thesis Architecture and Implementation Plan

> Title: **Research on Building Vectorization Extraction Methods for High-Resolution Remote Sensing Imagery Based on Multi-Task Deep Learning**
> Positioning: not "pushing segmentation IoU to 0.85", but an integrated method **from pixels to database-ready vectors** + vertex regression under real vector supervision + a vector-level evaluation system.
> This document serves simultaneously as the thesis writing outline and the execution plan for the remaining work, and is updated as experiments progress.

---

## 1. Core Contributions (the three points the thesis must establish)

1. **Multi-task semantic–boundary integration**: a shared backbone outputs both building masks and boundary maps, and the boundary map drives instance separation, replacing the empirical "morphology + watershed" post-processing;
2. **Graph-convolution vertex regression under real vector supervision**: EdgeConv vertex offset regression is trained on the true polygon labels of WHU-Mix (Vector), pulling jagged contours into regular polygons, combined with geometric regularization and **occlusion-aware completion**;
3. **A vector-first evaluation protocol and prototype system**: polygon IoU / Chamfer / PoLiS / vertex compression rate / right-angle rate / topology validity rate as the primary measures, together with an interactive visualization prototype (import → inference → editing → export → ablation).

> Writing principle: segmentation metrics (mIoU/F1) go into the appendix tables as "base-map quality"; all main tables contain vector metrics only.

---

## 2. Thesis Chapter Architecture

### Chapter 1 Introduction

| Section | Content | Figures/Tables |
|---|---|---|
| 1.1 Research background and significance | Applications of building vector data (planning/registration/disaster/database updating); the pain point that raster masks are hard to put into a database | Figure 1-1 Mask vs. vector comparison |
| 1.2 Research status at home and abroad | 1.2.1 Semantic segmentation (FCN/U-Net/DeepLab/HRNet/SegFormer)<br>1.2.2 Instance separation and boundary refinement (Mask R-CNN, boundary-aware)<br>1.2.3 Polygon generation and vectorization (Polygon-RNN/Curve-GCN/PolyGNN/BoundaryFormer/heatmap-based)<br>1.2.4 Existing shortcomings (error accumulation, variable vertex ordering, occlusion and regularity, evaluation criteria) | Table 1-1 Comparison of representative methods |
| 1.3 Research content and technical route | Three contributions + technical route diagram | **Figure 1-2 Technical route** |
| 1.4 Thesis organization | One sentence per chapter | — |

### Chapter 2 Related Theory and Technical Foundations

| Section | Content |
|---|---|
| 2.1 Convolutional neural networks and semantic segmentation | Encoder–decoder, skip connections, receptive field |
| 2.2 Multi-scale context and attention | ASPP, FPN, attention gates (formulas + roles) |
| 2.3 Graph neural networks and EdgeConv | Graph construction, message passing, EdgeConv formula |
| 2.4 Polygon geometric processing | Douglas-Peucker, piecewise line fitting, dominant direction estimation, right-angle/parallel constraints, self-intersection detection |
| 2.5 Evaluation metrics | Pixel (IoU/F1), boundary (BIoU), vector (PolyIoU/Chamfer/PoLiS/vertex count/right-angle rate/topology validity rate), **with formulas given** |
| 2.6 Datasets | WHU aerial, Massachusetts, **WHU-Mix (Vector)**; annotation scale and splits | Table 2-1 Dataset comparison |

### Chapter 3 Multi-Task Building Extraction Network

| Section | Content | Figures/Tables |
|---|---|---|
| 3.1 Overall architecture | Shared backbone + semantic head + boundary head | **Figure 3-1 Network structure** |
| 3.2 Shared backbone and multi-scale context | U-Net encoder–decoder + ASPP bottleneck | — |
| 3.3 Semantic segmentation branch | 1×1 output, BCE(pos_weight)+Dice | — |
| 3.4 Boundary branch | Morphological gradient to generate boundary bands, weighted BCE, boundary F1 calibration | Figure 3-2 Boundary map visualization |
| 3.5 Training strategy | Warm start, staged training (semantics first, then joint), cosine annealing, augmentation, AMP | Table 3-1 Hyperparameters |
| 3.6 Experiments in this chapter | Resolution comparison (256/384/512), with/without boundary head, boundary F1 | Table 3-2, Table 3-3 |

### Chapter 4 Building Contour Vectorization Based on Graph Convolution

| Section | Content | Figures/Tables |
|---|---|---|
| 4.1 From mask to initial contour | Thresholding, morphological opening, **boundary-constrained instance separation** (boundary-band cutting + distance transform + nearest-seed partitioning) | **Figure 4-1 Flowchart** |
| 4.2 Graph construction and node features | Equal-arc-length resampling (N=64), k nearest neighbors, features (coordinates/tangent/boundary probability/distance transform/curvature) | Figure 4-2 Feature illustration |
| 4.3 EdgeConv vertex regression | 3-layer EdgeConv, (dx,dy) per node; loss = Chamfer + nearest-vertex L1 + edge-length smoothing | **Figure 4-3 Network and loss** |
| 4.4 Polygon simplification and geometric regularization | DP simplification, dominant direction + right-angle snapping, shape-preserving validation (area/IoU fallback) | Figure 4-4 Before and after regularization |
| 4.5 Occlusion-aware completion | Tree/shadow detection (excess green + relative brightness), direction consistency criterion, straight-line bridging, length constraints | **Figure 4-5 Decision illustration** |
| 4.6 Experiments in this chapter | Comparison with geometric baseline; pseudo labels vs. true labels; cross-domain test II | Table 4-1/4-2/4-3 |

### Chapter 5 Experiments and Result Analysis

| Section | Content |
|---|---|
| 5.1 Experimental environment and data | CPU/GPU configuration, three datasets, splits |
| 5.2 Evaluation protocol | Cites `docs/vector-evaluation-protocol.md`, explaining "why not report only mIoU" |
| 5.3 Comparison with mainstream methods | U-Net / DeepLabv3+ / Mask R-CNN (run what can be run; cite literature values with notes for what cannot) |
| 5.4 Ablation experiments | A mask only / B +boundary / C +vertex / D +geometry / E +completion; plus "pseudo labels vs. true labels" |
| 5.5 Cross-dataset generalization | WHU→Massachusetts, WHU-Mix test II (no geographic overlap) |
| 5.6 Visualization prototype system | Feature list + screenshots + operation flow (corresponding to "expected outcome item 4" of the proposal) |
| 5.7 Failure cases and error analysis | Dense adhesion, shadow false detection, occlusion completion overshoot, topology repair failure |

### Chapter 6 Summary and Outlook

- 6.1 Work summary (corresponding to the three contributions, with quantitative conclusions)
- 6.2 Shortcomings and outlook: differentiable polygon rendering, end-to-end topological consistency, 3D roofs/multi-view, large models and remote sensing foundation models

### Appendix

- A Code structure and reproduction commands
- B Metric definitions and implementation details
- C Dataset statistics

---

## 3. Figure and Table List (planned in advance, saved as you go)

| Number | Type | Content | Status |
|---|---|---|---|
| Figure 1-1 | Illustration | Raster mask vs. vector polygon | To do |
| Figure 1-2 | Flow | Overall technical route diagram | To do (can be adapted from the existing pipeline diagram) |
| Figure 3-1 | Structure | Multi-task network | To draw |
| Figure 3-2 | Result | Boundary map vs. ground-truth boundary comparison | Material available |
| Figure 4-1 | Flow | Mask→instance→contour→graph | To draw |
| Figure 4-3 | Structure | EdgeConv vertex regression | To draw |
| Figure 4-5 | Illustration | Occlusion decision and bridging | Synthetic figure available |
| Figure 5-1 | Screenshot | GUI main interface | Available |
| Figure 5-2 | Chart | Bar chart of vector metrics for five configurations | Available |
| Table 3-1 | Table | Training hyperparameters | Available |
| Table 4-1 | Table | Vertex regression vs. geometric baseline | To run |
| Table 5-1 | Table | Comparison experiments | To run |
| Table 5-2 | Table | Ablation | A–E available, vertex to be added |

---

## 4. Experiment Matrix (tick when finished)

| Number | Experiment | Variable | Metrics | Status |
|---|---|---|---|---|
| E1 | Backbone comparison | unet / msf / full | mIoU, F1, BIoU | Partially complete (Massachusetts) |
| E2 | Input resolution | 256 / 384 / 512 | mIoU, boundary F1 | **Overnight training in progress** |
| E3 | Boundary head | With / without (unet_edge ablation) | Boundary F1, adhesion rate | Training |
| E4 | Instance separation | Morphological / boundary-driven | Adhesion rate, instance recall | To run (requires E3 weights) |
| E5 | Vertex regression | Geometric baseline / pseudo-label GCN / true-label GCN | Chamfer, vertex count, PolyIoU | **To do (main line)** |
| E6 | Geometric regularization | On / off | Right-angle rate, PolyIoU | Available |
| E7 | Occlusion completion | On / off | Chamfer, synthetic case IoU | Available (real subset to be added) |
| E8 | Cross-dataset | WHU→MA, WHU-Mix→test II | mIoU, PolyIoU | To run |
| E9 | TTA / ensemble | None / 8×TTA / two-model average | mIoU, Chamfer | To run |

---

## 5. Timeline (taking a November proposal and April–May defense the following year as an example)

### Stage 0: Base maps and data (11/1 – 11/3)
- Wrap up overnight training (256/384/512 multi-task) → summarize
- WHU-Mix extraction, COCO conversion, vertex statistics, spot checks
- Deliverables: `data/vertex_labels/`, data health-check table, base-map model weights

### Stage 1: First version of vertex regression (11/4 – 11/10)
- `vertex_dataset.py` / `vertex_gcn.py` / `train_vertex.py`
- Overfit validation on 500 instances → formal training on 5k instances
- Compare with `_polygonize` and decide whether the target is met (vertex count −20%, Chamfer not increased, PolyIoU within −0.5)
- Deliverables: GCN weights, comparison table, failure analysis

### Stage 2: Geometry and occlusion (11/11 – 11/17)
- Shape-preserving validation for regularization, dedicated occlusion completion work (select 50 samples of buildings pressed by trees)
- Deliverables: ablation E6/E7 results, representative figures

### Stage 3: Comparison and generalization (11/18 – 11/24)
- Complete E1/E3/E4/E8/E9; cross-domain experiments
- Deliverables: main table 5-1, ablation 5-2, generalization 5-3

### Stage 4: System and manuscript (11/25 – 12/10)
- GUI refinement (vertex model switching, model management, report export)
- First draft of the thesis (following the chapters of this document)
- Deliverables: first draft + all figures and tables + demo video

### Stage 5: Revision and defense (mid-December – April of the following year)
- Plagiarism check, supervisor revision, blind review, defense slides
- Deliverables: final version, defense materials

---

## 6. Risks and Countermeasures

| Risk | Probability | Impact | Countermeasure |
|---|---|---|---|
| GCN cannot beat the geometric baseline | Medium | Chapter 4 weakened | Degrade to "ablation/negative result", change the main line to multi-task + boundary + regularization + evaluation; the thesis still stands |
| Insufficient pseudo-label quality | Medium | Poor supervision signal | Already switched to WHU-Mix true labels; pseudo labels used only as a control |
| WHU-Mix annotation offset | Medium | Metrics unreliable | After extraction, first do an overlay spot check on 20 images; if necessary, use only the manually corrected subset |
| Insufficient CPU computing power | High | Experiment scale limited | Tiered resolutions (512 for key experiments, 256 for the rest); run TTA only on the final model |
| Fragmenting into "stacking modules" | Medium | Novelty questioned | The whole thesis revolves around one line "pixels→structure", with ablations proving the necessity of each step |
| Insufficient time | Medium | Rushed thesis | Stages 0–3 are the minimum deliverable; Stage 4 system features can be trimmed |

---

## 7. Current Progress (2026-11-01 01:30)

| Item | Status |
|---|---|
| Data | WHU 0.3 m packaged (4736/1036/2416); WHU-Mix vector val complete, test2 complete, train downloading; shapefile queued |
| Base-map model | `whu_cpu_unet`: val 0.7956 / test IoU 0.8273 @0.75 |
| Multi-task | `whu_mt_edge` training (12 epochs, currently epoch 8 val 0.8112), early boundary F1 0.70 |
| Overnight plan | 384→512 fine-tuning + second seed + no-ASPP ablation, expected to finish at 22:30 |
| Vector evaluation | Five-configuration protocol implemented (A–E), main table produced |
| Occlusion completion | Implemented (direction consistency criterion), synthetic case IoU 0.84→1.00 |
| GUI | WHU data integrated, automatic model/threshold selection, five-configuration ablation, vertex editing, export |

## 8. The Next 72 Hours

1. Download complete → extract → COCO conversion → vertex statistics and spot checks (daytime 11/1)
2. Wrap up overnight training → summarize metrics of the four models → update the protocol table and GUI defaults (evening 11/1)
3. Vertex regression data pipeline + model + first training run (11/2)
4. Compare with the geometric baseline → decide whether to enter Stage 2 (11/3)
