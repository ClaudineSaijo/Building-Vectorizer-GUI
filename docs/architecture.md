# Engineering Logical Architecture

> A visualization prototype for "multi-task deep learning + building vectorization". About 3200 lines of Python,
> split into three layers: the **presentation layer (app/)**, the **algorithm layer (src/)**, and the
> **task layer (scripts/)**. The dependency direction is one-way: `scripts/app → src`, and `src` never
> depends on the interface in reverse.

## 1. Layered View

```
Presentation / interaction layer ──────────────────────────────── app/
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ gui.py          │ → │ viewer.py       │   │ compare.py      │   │ sysinfo.py      │
   │ main window /   │   │ canvas / vertex │   │ batch compare   │   │ resource probe  │
   │ dispatch        │   │ editing         │   │ window          │   │                 │
   └────────┬────────┘   └─────────────────┘   └────────┬────────┘   └─────────────────┘
            │                                           │
Algorithm / domain layer ──────────────────────────────────── src/
            ▼                                           ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ infer.py        │ → │ vectorize.py    │ → │ vector_metrics  │
   │ load / sliding- │   │ mask → vector   │   │ vector metrics  │
   │ window inference│   │                 │   │                 │
   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
            │                     │                     │
            ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ models.py       │   │ esri_shapefile  │   │ metrics.py      │
   │ network         │   │ .shp writer     │   │ pixel / boundary│
   │ structures      │   │                 │   │ metrics         │
   └────────┬────────┘   └─────────────────┘   └─────────────────┘
            │
   ┌─────────────────┐   ┌─────────────────┐
   │ dataset.py      │   │ losses.py       │   (used during training only; inference does not depend on them)
   └─────────────────┘   └─────────────────┘
Task layer / entry points ────────────────────────────────────── scripts/ + run_vectorizer.py
   train.py  evaluate.py  ablation_benchmark.py  selfcheck_*.py ...
```

Dependencies form a **directed acyclic graph** (measured import edges):

```
run_vectorizer.py ──► app.gui
app.gui ──► app.compare, app.viewer, app.sysinfo
        └─► src.infer, src.vectorize, src.vector_metrics, src.esri_shapefile
app.compare ──► src.vectorize, src.vector_metrics
src.infer ──► src.models, src.dataset
src.vector_metrics ──► src.vectorize, src.metrics
scripts/* ──► src.* (ablation_benchmark additionally reuses the batch functions of app.compare)
```

There is no `import app` anywhere in `src/`, so the algorithm layer can be tested separately from the
interface — which is also why `scripts/selfcheck_vectorize.py` can validate the whole pipeline in a
display-less environment.

## 2. Module Responsibilities

| Module | Lines | Responsibility | Key interface |
|---|---:|---|---|
| `app/gui.py` | 753 | Main window, widgets, config snapshot, thread dispatch, metric refresh, export | `BuildingVectorizerApp` |
| `app/viewer.py` | 319 | Zoom/pan, layer compositing, polygon drawing, vertex add/delete/drag | `ImageViewer.set_image/set_overlays/set_polygons` |
| `app/compare.py` | 303 | Batch evaluation of 4 config groups + comparison window + bar chart | `BatchJob`, `run_batch`, `CompareWindow` |
| `app/sysinfo.py` | 112 | CPU/memory/process usage (ctypes, no third-party dependencies) | `cpu_percent`, `memory_usage` |
| `src/infer.py` | 168 | Model cache, sliding-window inference, image loading and channel/bit-depth normalization | `InferenceEngine.predict` |
| `src/models.py` | 147 | U-Net + ASPP + attention gate + boundary head | `building_unet` / `build_model` |
| `src/vectorize.py` | 548 | Mask → instances → polygons → regularization, GeoJSON writing | `vectorize(prob, edge, cfg)` |
| `src/vector_metrics.py` | 149 | Polygon IoU, matched Chamfer, PoLiS, rectilinearity, merge rate | `evaluate_vectors` |
| `src/esri_shapefile.py` | 124 | Minimal Shapefile writer (.shp/.shx/.dbf) | `write_polygon_shapefile` |
| `src/metrics.py` | 90 | Pixel/boundary metrics, instance statistics | `pixel_stats`, `boundary_iou`, `component_report` |
| `src/dataset.py` | 82 | Patch dataset, augmentation, normalization constants | `BuildingPatchDataset` |
| `src/losses.py` | 39 | BCE+Dice, boundary-weighted loss | `SegEdgeLoss` |
| `src/utils.py` | 43 | Seeding, JSON, parameter counting | — |

## 3. Core Data Contracts

The whole chain passes only four kinds of objects, with stable types, which makes it easy to replace
any single link:

```python
image:  np.ndarray  # (H, W, 3) uint8, RGB
prob:   np.ndarray  # (H, W) float32, [0,1] building probability
edge:   np.ndarray | None  # (H, W) float32, boundary probability (full model only)

VectorConfig:  # all switches and parameters of the algorithm (dataclass, used immutably)
  threshold=0.50, use_boundary=True, use_vertex=True, use_geometry=True,
  edge_threshold=0.45, edge_quantile=0.90, use_edge_cut=True,
  min_area=40, simplify_ratio=0.012, rect_tolerance_deg=14,
  rect_min_support=0.55, split_peak_size=11, split_min_peak=1.8,
  split_min_part=0.22, split_max_parts=4, split_size_factor=2.0

VectorResult:
  polygons: list[BuildingPolygon]   # points(N,2) float32 + area/perimeter/score/regularized
  mask: (H,W) uint8
  instances, merged_blobs, regularized: int
  timings: dict[str, float]         # threshold/separation/contour/vertex/geometry
  config: VectorConfig

metrics: dict[str, float]  # mIoU, boundary_iou, polygon_iou, chamfer_matched,
                           # polis, f1, rectilinearity, merge_rate,
                           # instance_recall, instance_precision, ...
```

## 4. Runtime Sequences

### 4.1 Single-image inference (F5)

```
User clicks "Run inference"
  └─ gui._on_run()
       ├─ cfg = gui._config()                 # Tk variables → VectorConfig snapshot
       ├─ disable buttons, update status bar text
       └─ threading.Thread(work)
            ├─ engine.predict_timed(image, model, want_edge)
            │     ├─ reflect-pad to an integer multiple of tile
            │     ├─ split into tiles with 0.5·tile stride, batch forward
            │     ├─ sigmoid → probabilities, weighted fusion with ramp weights
            │     └─ crop back to the original size
            ├─ vectorize(prob, edge, cfg)     # see 4.4
            └─ queue.put(("run", payload))
  ← gui._poll_queue() (every 60 ms, main thread)
       └─ _apply_run(payload)
            ├─ view.set_overlays(mask, boundary, gt)
            ├─ view.set_polygons(result.polygons)
            ├─ _refresh_metrics()             # evaluate_vectors
            ├─ _show_timing()                 # per-stage timings
            └─ re-enable buttons
```

**Threading rule**: Tk widgets are only touched on the main thread; worker threads only perform
numpy/torch computation and hand results back through `queue.Queue`; the main thread consumes them by
polling with `after()`. The comparison window uses the same pattern.

### 4.2 Vertex editing

```
Drag a vertex (viewer._on_left_drag)
  └─ directly modify BuildingPolygon.points[j] (in-place numpy write)
       └─ _draw_polygons()                # redraw only the vector layer, do not resample the base image
Mouse release → on_vertices_changed callback
  └─ gui._on_vertices_changed()
       ├─ result.polygons = polygons      # make export use the edited geometry
       └─ _refresh_metrics(polygons)      # recompute polygon IoU/Chamfer
```

"Reset vectors" takes another path: it reuses the cached `_prob/_edge` and only re-runs `vectorize()`,
without re-running inference, so it completes in milliseconds.

### 4.3 Batch comparison

```
CompareWindow._start()
  └─ sample N test tiles + the current VectorConfig as the baseline
       └─ run_batch(engine, job)
            for each model:
                for each image: engine.predict → cache prob/edge   # inference runs only once
                for each config: vectorize(cached prob, cfg) → evaluate_vectors
            → rows[{config, model, mIoU, polyIoU, chamfer, ...}]
       └─ main thread fills the Treeview + draws the bar chart + gives the "best option" conclusion
```

The caching of model × tile results in `run_batch` is the key: the 4 config groups share a single
inference pass, so "4 configs × 3 models × 16 images" needs only 48 forward passes.

### 4.4 Internal vectorization pipeline (`vectorize`)

```
prob ──threshold──► mask ──morphological opening──► blobs (by connected component + bbox crop)
                                  │
edge ──quantile threshold + dilation──► cut ───┤
                                  ▼
                    _separate_instances: only for blobs "significantly larger than the median area"
                      ├─ with cut: use the connected components after cutting as seeds
                      ├─ no seed: use local maxima of the distance transform as seeds
                      └─ nearest-seed assignment (no gaps, no area shrinkage)
                                  │
                        each instance → _dense_contour (CHAIN_APPROX_NONE)
                                  │
                 _contour_to_polygon: DP corner selection → piecewise line fitting → vertices from intersections
                                  │
                 regularize_polygon: dominant direction estimation → snap to 0/90° → self-intersection/area checks
                                  │
                         BuildingPolygon (area/perimeter/confidence/regularized flag)
```

## 5. State and Ownership

| State | Location | Notes |
|---|---|---|
| Algorithm config | Tk `*_var` | The interface is the single source of truth; `_config()` produces a snapshot on each run, so interface changes during a run do not affect it |
| Current image / ground truth | `gui.current` dict | `image`, `gt`, `name`, `label` |
| Latest result | `gui.result` (`VectorResult`) | Export, reset, and metric refresh all read from it |
| Editable polygons | `viewer.polygons` | The same object batch as `result.polygons`; editing modifies the objects themselves |
| Probability cache | `gui._prob/_edge` | Reused by "Reset vectors" and config switching to avoid repeated inference |
| Model | `InferenceEngine._cache` | Cached by name within the process, so switching models does not reload |

## 6. Extension Points

| What you want to replace | Where to change | Impact |
|---|---|---|
| Segmentation network/backbone | `src/models.py::build_model`, keep the `forward(x, return_edge)` contract | Training scripts + inference engine adapt automatically |
| Vertex regression (a real GCN) | Call the model inside `src/vectorize.py::_contour_to_polygon` | GUI/metrics/export unchanged |
| Instance separation strategy | `src/vectorize.py::_separate_instances` | Same as above |
| Add an evaluation metric | Return value of `src/vector_metrics.py::evaluate_vectors` + `app/compare.py::METRIC_COLUMNS` | Tables and charts gain one column automatically |
| Add an export format | Mirror `write_geojson` / `write_polygon_shapefile`, add an item to the export menu in `gui.py` | Local |
| GeoTIFF geographic coordinates | Write `.prj` in `src/esri_shapefile.py`, change the CRS in `write_geojson` | Export layer |

## 7. Design Tradeoffs and Known Issues

1. **`app/compare.py` mixes "batch logic" and the "Tk window" in one file**. `run_batch`
   is a pure function with no UI dependency and is reused directly by the CLI; if it keeps growing,
   it should be split into `app/batch.py` (pure computation) + `app/compare_window.py` (interface).
2. **`app/gui.py` at 753 lines is on the large side**, carrying four concerns: layout, state, threading,
   and export. It could be further split into `gui_export.py` and `gui_sections.py`, but the current size is
   still maintainable.
3. **Vertex regression is a deterministic geometric algorithm, not a learning module**. The interface
   (`use_vertex`) is already designed for a learned version, so replacing it does not affect the upper layers.
4. **The inference thread has no progress callback**; a 1500×1500 image only shows the "inference running…"
   text. For fine-grained progress, `InferenceEngine.predict` would need a `progress_cb`.
5. **No GPU abstraction**: `InferenceEngine` is fixed to CPU (`map_location="cpu"`); switching to a GPU
   requires device management in `load()` and `predict()`.
6. **`src/metrics.py` and `src/vector_metrics.py` coexist**: the former is an early pixel-level evaluation
   that is reused later, the latter is vector-level evaluation; both are kept to stay aligned with the
   historical results of `scripts/evaluate.py`, and merging them would break the comparability of the
   existing metrics.
