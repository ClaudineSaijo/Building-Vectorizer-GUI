"""Desktop demo system for multi-task building vectorization.

Launch with ``python run_vectorizer.py`` (or ``python -m app.gui``).

The window follows the deliverable described in the proposal: load an image,
toggle the boundary / vertex / geometry branches, run joint inference, inspect
the raster mask and the vector polygons overlaid on the image, drag vertices,
export GeoJSON / Shapefile, and read run time plus CPU / memory usage.  A second
window runs the configuration ablation and recommends the best setting.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

if __package__ in (None, ""):  # allow `python app/gui.py` as well as `python -m app.gui`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.compare import CompareWindow
from app.sysinfo import FPSMeter, gpu_info, memory_usage, process_memory_mb
from app.viewer import ImageViewer
from src.esri_shapefile import write_polygon_shapefile
from src.infer import MODEL_LABELS, InferenceEngine, load_rgb_image
from src.vector_metrics import evaluate_vectors
from src.vectorize import VectorConfig, dominant_orientation, vectorize, write_geojson

BG = "#eef1f5"
PANEL = "#ffffff"
TEXT = "#1c2632"
MUTED = "#5b6675"
ACCENT = "#1f6feb"
OK = "#1a7f37"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class ScrollableFrame(ttk.Frame):
    """Vertically scrollable container for the control panel."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=PANEL)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Panel.TFrame")
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_wheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)

    def _unbind_wheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class BuildingVectorizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.project_root = _project_root()
        self.title("建筑物矢量化提取与可视化系统  ·  多任务深度学习原型")
        self.geometry("1480x960")
        self.minsize(1180, 720)
        self.configure(bg=BG)

        self.engine = InferenceEngine(self.project_root)
        self.current: dict | None = None
        self.result = None
        self._prob: np.ndarray | None = None
        self._edge: np.ndarray | None = None
        self._running = False
        self._edited = False
        self._queue: queue.Queue = queue.Queue()
        self.current_model = "full"

        self.tiles = None
        self.tile_masks = None
        self.tile_meta: list = []
        self.whu_tiles = None
        self.whu_masks = None
        self.samples: list[dict] = []
        self.sample_index = 0

        self._metric_vars: dict[str, tk.StringVar] = {}
        self._init_vars()
        self._init_style()
        self._build_toolbar()
        self._build_body()
        self._build_status_bar()
        self._bind_keys()

        self._fps = FPSMeter()
        self.after(80, self._poll_queue)
        self.after(160, self._bootstrap)

    # ------------------------------------------------------------- variables
    def _init_vars(self) -> None:
        self.var_source = tk.StringVar(value="tile")
        self.var_sample = tk.StringVar()
        self.var_image_info = tk.StringVar(value="未载入影像")
        self.model_var = tk.StringVar(value="")
        self.var_model_info = tk.StringVar(value="")
        self.edge_var = tk.BooleanVar(value=True)
        self.vertex_var = tk.BooleanVar(value=True)
        self.geom_var = tk.BooleanVar(value=True)
        self.completion_var = tk.BooleanVar(value=True)
        self.tta_var = tk.BooleanVar(value=False)
        self.threshold_var = tk.DoubleVar(value=0.45)
        self.edge_thr_var = tk.DoubleVar(value=0.45)
        self.min_area_var = tk.DoubleVar(value=40.0)
        self.simplify_var = tk.DoubleVar(value=0.012)
        self.show_mask = tk.BooleanVar(value=True)
        self.show_boundary = tk.BooleanVar(value=True)
        self.show_gt = tk.BooleanVar(value=True)
        self.show_poly = tk.BooleanVar(value=True)
        self.show_vertex = tk.BooleanVar(value=True)
        self.alpha_var = tk.DoubleVar(value=0.45)
        self.var_hint = tk.StringVar(value="滚轮缩放 · 拖动平移 · 拖拽顶点编辑 · 双击边插入顶点 · 右键删除顶点")
        self.var_coord = tk.StringVar(value="")
        self.var_res = tk.StringVar(value="")
        self.var_stage = tk.StringVar(value="就绪")
        self.var_branch = tk.StringVar(value="")
        for key in ("mIoU", "boundary_iou", "polygon_iou", "chamfer", "polis", "f1",
                    "num_polygons", "mean_vertices", "area_ratio", "rectilinearity",
                    "merge_rate", "instance_recall"):
            self._metric_vars[key] = tk.StringVar(value="—")

    def _init_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base_font = ("Microsoft YaHei UI", 9)
        self.option_add("*Font", base_font)
        style.configure(".", font=base_font, background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Title.TLabel", background=PANEL, foreground=TEXT,
                        font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Metric.TLabel", background=PANEL, foreground=ACCENT,
                        font=("Consolas", 11, "bold"))
        style.configure("TLabelframe", background=PANEL, bordercolor="#d6dbe3")
        style.configure("TLabelframe.Label", background=PANEL, foreground="#334155",
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.configure("TRadiobutton", background=PANEL, foreground=TEXT)
        style.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                        padding=(14, 7), borderwidth=0, focusthickness=0)
        style.map("Accent.TButton", background=[("active", "#1558c0"), ("disabled", "#9db8e5")])
        style.configure("Tool.TButton", padding=(10, 5))
        style.configure("TButton", padding=(8, 5))
        style.configure("TSpinbox", padding=3)
        style.configure("Treeview", rowheight=22, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Status.TLabel", background="#e3e8ef", foreground=MUTED, padding=(8, 3))

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg="#ffffff", height=52, highlightthickness=1,
                       highlightbackground="#dbe1e8")
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="  建筑物矢量化提取与可视化系统", bg="#ffffff", fg=TEXT,
                 font=("Microsoft YaHei UI", 13, "bold")).pack(side="left", padx=(14, 6))
        tk.Label(bar, text="多任务分割 + 边界分离 + 顶点回归", bg="#ffffff", fg=MUTED,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(0, 20))

        self.btn_run = ttk.Button(bar, text="运行推理  F5", style="Accent.TButton",
                                  command=self._on_run)
        self.btn_run.pack(side="left", padx=4, pady=9)
        ttk.Button(bar, text="打开影像", style="Tool.TButton",
                   command=self._open_local).pack(side="left", padx=4)
        ttk.Button(bar, text="导出 GeoJSON", style="Tool.TButton",
                   command=self._export_geojson).pack(side="left", padx=4)
        ttk.Button(bar, text="导出 Shapefile", style="Tool.TButton",
                   command=self._export_shapefile).pack(side="left", padx=4)
        ttk.Button(bar, text="对比与消融", style="Tool.TButton",
                   command=self._open_compare).pack(side="left", padx=4)
        ttk.Button(bar, text="使用说明", style="Tool.TButton",
                   command=self._show_help).pack(side="right", padx=12)

    def _build_body(self) -> None:
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ---------------- left control column ----------------
        left = tk.Frame(body, bg=PANEL, width=352)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        actions = ttk.Frame(left, style="Panel.TFrame", padding=(10, 8))
        actions.pack(side="bottom", fill="x")
        ttk.Button(actions, text="运行联合推理", style="Accent.TButton",
                   command=self._on_run).pack(fill="x", pady=(0, 6))
        row = ttk.Frame(actions, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="重置矢量", command=self._reset_polygons).pack(
            side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(row, text="导出结果", command=self._export_menu).pack(
            side="left", expand=True, fill="x", padx=(4, 0))

        scroll = ScrollableFrame(left)
        scroll.pack(fill="both", expand=True)
        panel = scroll.inner
        panel.configure(padding=(12, 10))

        self._section_source(panel)
        self._section_model(panel)
        self._section_algorithm(panel)
        self._section_metrics(left)

        # ---------------- right viewer ----------------
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        card = tk.Frame(right, bg="#ffffff", highlightthickness=1, highlightbackground="#dbe1e8")
        card.pack(fill="both", expand=True, padx=(8, 10), pady=(8, 4))
        self.viewer = ImageViewer(card, on_vertices_changed=self._on_vertices_changed,
                                  on_status=lambda s: self.var_coord.set(s))
        self.viewer.pack(fill="both", expand=True)

        self._build_layer_bar(right)

        legend = tk.Frame(right, bg=BG)
        legend.pack(fill="x", padx=14, pady=(0, 6))
        for color, text in (("#ffd60a", "矢量多边形"), ("#ffffff", "顶点"), ("#00c8a0", "分割掩膜"),
                            ("#ff463c", "边界带"), ("#3cdc5a", "真值边界")):
            sw = tk.Label(legend, bg=color, width=2, relief="solid", bd=1,
                          highlightbackground="#9aa4b2")
            sw.pack(side="left", padx=(0, 3))
            tk.Label(legend, text=text, bg=BG, fg=MUTED, font=("Microsoft YaHei UI", 8)).pack(
                side="left", padx=(0, 12))
        self.var_hint_label = tk.Label(legend, textvariable=self.var_hint, bg=BG, fg=MUTED,
                                       font=("Microsoft YaHei UI", 8))
        self.var_hint_label.pack(side="right")

    def _section_source(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="1  影像来源", padding=(10, 8))
        box.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Radiobutton(row, text="内置 256²", value="tile", variable=self.var_source,
                        command=self._on_source_change).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(row, text="WHU 512²", value="whu", variable=self.var_source,
                        command=self._on_source_change).pack(side="left", padx=(0, 8))
        ttk.Radiobutton(row, text="高分辨率", value="hires", variable=self.var_source,
                        command=self._on_source_change).pack(side="left")

        row2 = ttk.Frame(box, style="Panel.TFrame")
        row2.pack(fill="x", pady=(6, 0))
        self.cmb_sample = ttk.Combobox(row2, textvariable=self.var_sample, state="readonly", width=24)
        self.cmb_sample.pack(side="left", fill="x", expand=True)
        self.cmb_sample.bind("<<ComboboxSelected>>", lambda e: self._load_sample(self.cmb_sample.current()))
        ttk.Button(row2, text="◀", width=3, command=lambda: self._step_sample(-1)).pack(side="left", padx=(4, 2))
        ttk.Button(row2, text="▶", width=3, command=lambda: self._step_sample(1)).pack(side="left")

        row3 = ttk.Frame(box, style="Panel.TFrame")
        row3.pack(fill="x", pady=(6, 0))
        ttk.Button(row3, text="打开本地影像…", command=self._open_local).pack(side="left")
        ttk.Button(row3, text="载入真值…", command=self._load_gt).pack(side="left", padx=6)

        ttk.Label(box, textvariable=self.var_image_info, style="Muted.TLabel",
                  wraplength=300, justify="left").pack(fill="x", pady=(6, 0))

    def _section_model(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="2  语义分割模型", padding=(10, 8))
        box.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(box, style="Panel.TFrame")
        row.pack(fill="x")
        self.cmb_model = ttk.Combobox(row, textvariable=self.model_var, state="readonly", width=16,
                                      values=list(MODEL_LABELS))
        self.cmb_model.pack(side="left")
        self.cmb_model.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())
        ttk.Label(row, textvariable=self.var_model_info, style="Muted.TLabel").pack(
            side="left", padx=(8, 0))

    def _section_algorithm(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="3  算法配置（鼠标点击切换）", padding=(10, 8))
        box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(box, text="边界分支：用边界图切断粘连建筑",
                        variable=self.edge_var, command=self._on_flag_change).pack(anchor="w")
        ttk.Checkbutton(box, text="顶点回归：轮廓简化为规整多边形",
                        variable=self.vertex_var, command=self._on_flag_change).pack(anchor="w")
        ttk.Checkbutton(box, text="几何约束：直角 / 平行规则化",
                        variable=self.geom_var, command=self._on_flag_change).pack(anchor="w")
        ttk.Checkbutton(box, text="遮挡补全：树木/阴影处用直线跨接",
                        variable=self.completion_var, command=self._on_flag_change).pack(anchor="w")
        ttk.Checkbutton(box, text="TTA 增强：8× 翻转/旋转平均（更准、慢 8 倍）",
                        variable=self.tta_var, command=self._on_flag_change).pack(anchor="w")

        self._slider(box, "分割阈值", self.threshold_var, 0.10, 0.90, 0.01)
        self._slider(box, "边界阈值", self.edge_thr_var, 0.20, 0.90, 0.01)
        self._slider(box, "最小面积 / px²", self.min_area_var, 0, 400, 10)
        self._slider(box, "简化系数 ε", self.simplify_var, 0.002, 0.05, 0.002)
        self.threshold_var.trace_add("write", lambda *_: self._update_branch_hint())

        ttk.Label(box, textvariable=self.var_branch, style="Muted.TLabel",
                  wraplength=300, justify="left").pack(fill="x", pady=(4, 0))

    def _slider(self, parent, label: str, var: tk.Variable, lo: float, hi: float, step: float) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=(6, 0))
        name = ttk.Label(row, text=label, style="Panel.TLabel", width=13)
        name.pack(side="left")
        value = ttk.Label(row, text="", style="Muted.TLabel", width=6)
        value.pack(side="right")

        def fmt(v) -> str:
            v = float(v)
            return f"{v:.3f}" if step < 0.01 else (f"{v:.2f}" if step < 1 else f"{v:.0f}")

        value.configure(text=fmt(var.get()))
        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=lambda v: value.configure(text=fmt(v)))
        scale.pack(side="left", fill="x", expand=True, padx=(2, 4))
        # keep the number in sync when the value is set programmatically
        # (e.g. the threshold applied from a model's evaluation result)
        var.trace_add("write", lambda *_: value.configure(text=fmt(var.get())))

    def _build_layer_bar(self, parent) -> None:
        """Layer switches live next to the canvas, where they are always visible."""
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", padx=14, pady=(2, 0))
        opts = [
            ("分割掩膜", self.show_mask),
            ("边界带", self.show_boundary),
            ("真值边界", self.show_gt),
            ("矢量多边形", self.show_poly),
            ("顶点", self.show_vertex),
        ]
        for text, var in opts:
            ttk.Checkbutton(bar, text=text, variable=var, command=self._sync_layers).pack(
                side="left", padx=(0, 10))
        ttk.Label(bar, text="掩膜透明度", background=BG, foreground=MUTED).pack(side="left", padx=(16, 4))
        ttk.Scale(bar, from_=0.0, to=0.9, variable=self.alpha_var, orient="horizontal", length=110,
                  command=lambda v: self.viewer.set_layers(mask_alpha=float(v))).pack(side="left")

    def _section_metrics(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="4  精度指标与运行信息", padding=(10, 6))
        box.pack(side="bottom", fill="x", padx=10, pady=(0, 8))
        rows = [
            ("mIoU", "mIoU"), ("boundary_iou", "边界 IoU"), ("polygon_iou", "多边形 IoU"),
            ("chamfer", "Chamfer / px"), ("polis", "PoLiS / px"), ("f1", "F1"),
            ("rectilinearity", "几何规整度"), ("merge_rate", "粘连率"),
            ("instance_recall", "实例召回"), ("num_polygons", "多边形数量"),
            ("mean_vertices", "平均顶点数"), ("area_ratio", "提取面积占比"),
        ]
        for i, (key, label) in enumerate(rows):
            r, c = divmod(i, 2)
            ttk.Label(box, text=label, style="Panel.TLabel").grid(
                row=r, column=c * 2, sticky="w", pady=1, padx=(0, 4))
            ttk.Label(box, textvariable=self._metric_vars[key], style="Metric.TLabel").grid(
                row=r, column=c * 2 + 1, sticky="e", pady=1, padx=(0, 14))
        box.columnconfigure(1, weight=1)
        box.columnconfigure(3, weight=1)

        ttk.Separator(box, orient="horizontal").grid(
            row=6, column=0, columnspan=4, sticky="ew", pady=(6, 4))

        self.var_time = tk.StringVar(value="—")
        self.var_stage_detail = tk.StringVar(value="—")
        self.var_hw = tk.StringVar(value="—")
        for k, (label, var) in enumerate((("耗时", self.var_time), ("分阶段", self.var_stage_detail),
                                          ("资源", self.var_hw))):
            ttk.Label(box, text=label, style="Panel.TLabel").grid(
                row=7 + k, column=0, sticky="nw", pady=1)
            ttk.Label(box, textvariable=var, style="Muted.TLabel", wraplength=250,
                      justify="left").grid(row=7 + k, column=1, columnspan=3, sticky="w", pady=1)

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self, bg="#e3e8ef", height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.var_stage, bg="#e3e8ef", fg="#1c2632",
                 font=("Microsoft YaHei UI", 8)).pack(side="left", padx=8)
        tk.Label(bar, textvariable=self.var_coord, bg="#e3e8ef", fg=MUTED,
                 font=("Consolas", 8)).pack(side="right", padx=8)
        self.var_status_right = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.var_status_right, bg="#e3e8ef", fg=MUTED,
                 font=("Consolas", 8)).pack(side="right", padx=12)

    def _bind_keys(self) -> None:
        self.bind("<F5>", lambda e: self._on_run())
        self.bind("<Control-Return>", lambda e: self._on_run())
        self.bind("<Control-o>", lambda e: self._open_local())
        self.bind("<Control-s>", lambda e: self._export_geojson())

    # --------------------------------------------------------------- startup
    def _bootstrap(self) -> None:
        try:
            self.tiles = np.load(self.project_root / "data" / "processed" / "test_images.npy",
                                 mmap_mode="r")
            self.tile_masks = np.load(self.project_root / "data" / "processed" / "test_masks.npy",
                                      mmap_mode="r")
            self.tile_meta = json.loads(
                (self.project_root / "data" / "processed" / "test_meta.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            self.tiles = self.tile_masks = None
            self.tile_meta = []

        # WHU aerial tiles (0.3 m, 512²) packed by scripts/prepare_whu.py
        try:
            whu = self.project_root / "data" / "whu_processed"
            self.whu_tiles = np.load(whu / "test_images.npy", mmap_mode="r")
            self.whu_masks = np.load(whu / "test_masks.npy", mmap_mode="r")
        except Exception:  # noqa: BLE001
            self.whu_tiles = self.whu_masks = None

        # if WHU data is available, start there: the best model was trained on it
        if self.whu_tiles is not None:
            self.var_source.set("whu")

        self._refresh_sample_list()
        models = self.engine.available_models()
        if models:
            self.cmb_model.configure(values=models or list(MODEL_LABELS))
            # highest validation IoU wins, unless the launcher pinned a model
            default = self.engine.best_model() or models[0]
            pinned = getattr(self, "_model_locked", False) and self.model_var.get() in models
            if not pinned:
                self.model_var.set(default)
            self.current_model = self.model_var.get()
            try:
                info = self.engine.model_info(self.current_model)
                self.var_model_info.set(self._model_desc(info))
            except Exception:  # noqa: BLE001
                self.var_model_info.set("")
        else:
            self.var_stage.set("未找到 runs/*/best.pt，请先训练模型")

        self._update_hw_label()
        self.after(1000, self._tick_status)
        if self.samples:
            self._load_sample(self._pick_start_sample())
        self._apply_model_threshold()
        self._auto_pick_model_for_source()

    def _pick_start_sample(self) -> int:
        """Start on a tile that actually contains buildings (empty fields are bad demos)."""
        if self.var_source.get() != "whu" or self.whu_masks is None:
            return 0
        n = min(len(self.whu_masks), 300)
        coverage = [float(np.asarray(self.whu_masks[i]).mean()) for i in range(n)]
        candidates = [i for i, c in enumerate(coverage) if 0.02 <= c <= 0.30]
        if candidates:
            return candidates[len(candidates) // 2]
        return int(np.argmax(coverage)) if coverage else 0

    def _refresh_sample_list(self) -> None:
        """Fill the sample combobox for the selected source."""
        self.samples = []
        if self.var_source.get() == "tile" and self.tiles is not None:
            for i in range(min(len(self.tiles), 400)):
                meta = self.tile_meta[i] if i < len(self.tile_meta) else {}
                name = f"{i + 1:03d} · {meta.get('tile', 'tile')} (y{meta.get('y', 0)},x{meta.get('x', 0)})"
                self.samples.append({"kind": "tile", "index": i, "label": name})
        elif self.var_source.get() == "whu" and getattr(self, "whu_tiles", None) is not None:
            for i in range(min(len(self.whu_tiles), 500)):
                self.samples.append({"kind": "whu", "index": i,
                                     "label": f"WHU 测试 {i + 1:04d} · 512²"})
        elif self.var_source.get() == "hires":
            sat_dir = self.project_root / "data" / "raw" / "test" / "sat"
            for p in sorted(sat_dir.glob("*.tiff")) + sorted(sat_dir.glob("*.tif")):
                self.samples.append({"kind": "hires", "path": str(p), "label": f"1500² · {p.stem}"})
        values = [s["label"] for s in self.samples]
        self.cmb_sample.configure(values=values)
        if values:
            self.sample_index = 0
            self.var_sample.set(values[0])
        else:
            self.var_sample.set("")

    def _source_dataset(self) -> str:
        return "whu" if self.var_source.get() == "whu" else "massachusetts"

    def _auto_pick_model_for_source(self) -> None:
        """Pick the best checkpoint trained on the same domain as the loaded imagery."""
        if getattr(self, "_model_locked", False):
            return
        names = self.engine.models_for_dataset(self._source_dataset())
        if not names:
            return
        if names[0] != self.model_var.get():
            self.model_var.set(names[0])
            self._on_model_change()
            self.var_stage.set(f"已切换到与当前数据同域的模型：{names[0]}")

    def _on_source_change(self) -> None:
        self._refresh_sample_list()
        if self.samples:
            self._load_sample(self._pick_start_sample())
        self._auto_pick_model_for_source()

    @staticmethod
    def _model_desc(info: dict) -> str:
        iou = info.get("val_iou", float("nan"))
        desc = f"e{info.get('epoch')} · IoU {iou:.3f}"
        arch = info.get("arch")
        encoder = info.get("encoder")
        dataset = str(info.get("dataset") or "")
        prefix = "WHU · " if dataset == "whu" else ""
        if arch and arch not in {"unet", "msf", "full"}:
            prefix += f"{arch}/{encoder} · "
        return prefix + desc

    def _apply_model_threshold(self) -> None:
        """Use the threshold that evaluation picked for this checkpoint, if recorded."""
        try:
            thr = self.engine.best_threshold(self.current_model)
        except Exception:  # noqa: BLE001
            thr = None
        if thr is None:
            try:
                same_domain = self.engine.model_dataset(self.current_model) == "whu"
            except Exception:  # noqa: BLE001
                same_domain = False
            thr = 0.60 if same_domain else 0.45
            self.threshold_var.set(float(thr))
            return
        self.threshold_var.set(float(thr))
        self.var_stage.set(f"已按 {self.current_model} 的评估结果把分割阈值设为 {thr:.2f}")

    def _on_model_change(self) -> None:
        self.current_model = self.model_var.get()
        mismatch = False
        try:
            info = self.engine.model_info(self.current_model)
            mismatch = info.get("dataset") not in (None, "", self._source_dataset())
            self.var_model_info.set(self._model_desc(info) + ("  ⚠ 数据域不匹配" if mismatch else ""))
        except Exception:  # noqa: BLE001
            self.var_model_info.set("")
        self._apply_model_threshold()
        if mismatch:
            self.var_stage.set(
                f"注意：{self.current_model} 与当前影像数据域不同（模型训练于 "
                f"{info.get('dataset')}），指标仅供参考；建议切到同域模型。")
        self._update_branch_hint()

    def _on_flag_change(self) -> None:
        self._update_branch_hint()

    def _update_branch_hint(self) -> None:
        model = self.model_var.get()
        parts = []
        if self.edge_var.get():
            has_edge = False
            try:
                has_edge = self.engine.has_edge(model)
            except Exception:  # noqa: BLE001 - model may not be loaded yet
                has_edge = False
            if has_edge:
                parts.append("边界分支：使用模型边界头输出")
            else:
                parts.append("边界分支：该模型无边界头，回退距离变换实例分离")
        if self.vertex_var.get():
            parts.append("顶点回归：轮廓→顶点优化")
        if self.geom_var.get():
            parts.append("几何约束：主方向 + 直角")
        try:
            thr = float(self.threshold_var.get())
            if thr < 0.30:
                parts.append("⚠ 阈值过低（<0.30），虚警会明显增多")
            elif thr > 0.80:
                parts.append("⚠ 阈值过高（>0.80），会漏掉较多建筑")
        except Exception:  # noqa: BLE001
            pass
        self.var_branch.set("；".join(parts) if parts else "三个分支均已关闭（纯阈值栅格结果）")

    def _step_sample(self, delta: int) -> None:
        if not self.samples:
            return
        self.sample_index = (self.sample_index + delta) % len(self.samples)
        self._load_sample(self.sample_index)

    def _load_sample(self, index: int) -> None:
        if not self.samples:
            return
        index = max(0, min(index, len(self.samples) - 1))
        self.sample_index = index
        item = self.samples[index]
        self.var_sample.set(item["label"])
        if item["kind"] == "tile":
            image = np.asarray(self.tiles[item["index"]])
            gt = (np.asarray(self.tile_masks[item["index"]]) > 0).astype(np.uint8)
            name = f"tile_{item['index']:03d}"
        elif item["kind"] == "whu":
            image = np.asarray(self.whu_tiles[item["index"]])
            gt = (np.asarray(self.whu_masks[item["index"]]) > 0).astype(np.uint8)
            name = f"whu_test_{item['index']:04d}"
        else:
            image = load_rgb_image(item["path"])
            gt = self._load_hires_gt(Path(item["path"]))
            name = Path(item["path"]).stem
        self._set_current(image, gt, name, item["label"])

    def _load_hires_gt(self, sat_path: Path) -> np.ndarray | None:
        map_path = sat_path.parent.parent / "map" / f"{sat_path.stem}.tif"
        if not map_path.exists():
            return None
        label = load_rgb_image(map_path)
        return (label.max(axis=2) > 127).astype(np.uint8)

    def _set_current(self, image: np.ndarray, gt: np.ndarray | None, name: str, label: str) -> None:
        self.current = {"image": image, "gt": gt, "name": name, "label": label}
        h, w = image.shape[:2]
        has_gt = "有真值" if gt is not None else "无真值"
        self.var_image_info.set(f"{w}×{h} px · {has_gt} · {label}")
        self.viewer.set_image(image)
        self.viewer.set_overlays(None, None, gt)
        self.viewer.set_polygons([])
        self.result = None
        self._prob = None
        self._edge = None
        self._edited = False
        for var in self._metric_vars.values():
            var.set("—")
        self.var_stage.set(f"已载入 {name}，按 F5 运行联合推理")
        self._update_branch_hint()

    def _open_local(self) -> None:
        path = filedialog.askopenfilename(
            title="选择遥感影像",
            filetypes=[("遥感影像", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            image = load_rgb_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("读取失败", f"无法读取影像：\n{exc}")
            return
        self.var_source.set("local")
        self.samples = [{"kind": "local", "path": path, "label": Path(path).name}]
        self.sample_index = 0
        self.cmb_sample.configure(values=[Path(path).name])
        self.var_sample.set(Path(path).name)
        self._set_current(image, None, Path(path).stem, Path(path).name)

    def _load_gt(self) -> None:
        if self.current is None:
            return
        path = filedialog.askopenfilename(title="选择真值掩膜",
                                          filetypes=[("影像/掩膜", "*.tif *.tiff *.png *.jpg"), ("所有文件", "*.*")])
        if not path:
            return
        label = load_rgb_image(path)
        gt = (label.max(axis=2) > 127).astype(np.uint8) if label.ndim == 3 else (label > 127).astype(np.uint8)
        if gt.shape != self.current["image"].shape[:2]:
            messagebox.showerror("尺寸不匹配", "真值掩膜与影像尺寸不一致。")
            return
        self.current["gt"] = gt
        self.viewer.set_overlays(self.result.mask if self.result else None, None, gt)
        self.var_image_info.set(self.var_image_info.get().split(" · ")[0] + " · 有真值 · " +
                                self.current["label"])
        if self.result is not None:
            self._refresh_metrics(self.viewer.polygons)

    # ------------------------------------------------------------------ run
    def _config(self) -> VectorConfig:
        return VectorConfig(
            threshold=float(self.threshold_var.get()),
            use_boundary=bool(self.edge_var.get()),
            use_vertex=bool(self.vertex_var.get()),
            use_geometry=bool(self.geom_var.get()),
            use_completion=bool(self.completion_var.get()),
            edge_threshold=float(self.edge_thr_var.get()),
            min_area=float(self.min_area_var.get()),
            simplify_ratio=float(self.simplify_var.get()),
        )

    def _on_run(self) -> None:
        if self.current is None:
            messagebox.showinfo("提示", "请先载入影像。")
            return
        if self._running:
            return
        if not self.engine.available_models():
            messagebox.showerror("缺少权重", "未找到 runs/*/best.pt，请先运行训练脚本。")
            return
        cfg = self._config()
        model = self.model_var.get()
        self.current_model = model
        want_edge = cfg.use_boundary and self.engine.has_edge(model)
        tta = bool(self.tta_var.get())
        image = self.current["image"]
        self._running = True
        self.btn_run.configure(state="disabled", text="推理中…")
        self.var_stage.set("正在执行语义分割" + ("、边界分离" if want_edge else "")
                           + "与顶点回归…" + ("（TTA 8×）" if tta else ""))
        self.update_idletasks()

        def work() -> None:
            try:
                prob, edge, t_inf = self.engine.predict_timed(image, model, want_edge, tta=tta)
                t0 = time.perf_counter()
                result = vectorize(prob, edge if want_edge else None, cfg, image=image)
                t_vec = time.perf_counter() - t0
                self._queue.put(("run", {
                    "prob": prob, "edge": edge, "result": result,
                    "t_inf": t_inf, "t_vec": t_vec, "cfg": cfg, "model": model,
                    "tta": tta,
                }))
            except Exception:  # noqa: BLE001
                self._queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "run":
                    self._apply_run(payload)
                elif kind == "error":
                    self._running = False
                    self.btn_run.configure(state="normal", text="运行推理  F5")
                    self.var_stage.set("运行失败")
                    messagebox.showerror("运行失败", payload)
        except queue.Empty:
            pass
        self.after(60, self._poll_queue)

    def _apply_run(self, payload: dict) -> None:
        self._running = False
        self.btn_run.configure(state="normal", text="运行推理  F5")
        self._prob = payload["prob"]
        self._edge = payload["edge"]
        self.result = payload["result"]
        self._edited = False
        cfg: VectorConfig = payload["cfg"]

        boundary = None
        if cfg.use_boundary:
            if payload["edge"] is not None:
                boundary = (payload["edge"] >= cfg.edge_threshold).astype(np.uint8)
            else:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                m = self.result.mask
                boundary = ((cv2.dilate(m, k) - cv2.erode(m, k)) > 0).astype(np.uint8)
        gt = self.current["gt"] if self.current else None
        self.viewer.set_overlays(self.result.mask, boundary, gt)
        self.viewer.set_polygons(self.result.polygons)

        self._refresh_metrics(self.result.polygons)
        self._show_timing(payload["t_inf"], payload["t_vec"], self.result)

        summary = self.result.summary()
        self.var_stage.set(
            f"完成：{summary['num_polygons']} 个多边形 · {summary['instances']} 个实例 · "
            f"边界分离 {summary['merged_blobs']} 组 · 规则化 {summary['regularized']} 个"
            + ("（顶点已手工编辑）" if self._edited else ""))
        self._edited = False

    def _show_timing(self, t_inf: float, t_vec: float, result) -> None:
        total = t_inf + t_vec
        self.var_time.set(f"总 {total:.2f} s（推理 {t_inf:.2f} + 矢量 {t_vec:.2f}）")
        stages = result.timings
        order = [("threshold", "阈值"), ("separation", "分离"), ("contour", "轮廓"),
                 ("vertex", "顶点"), ("geometry", "几何")]
        self.var_stage_detail.set(" · ".join(
            f"{label} {stages.get(key, 0) * 1000:.0f}ms" for key, label in order))
        self._update_hw_label()

    def _update_hw_label(self) -> None:
        mem = memory_usage()
        pmem = process_memory_mb()
        cpu = self._fps.cpu
        bits = [gpu_info()]
        if cpu is not None:
            bits.append(f"CPU {cpu:.0f}%")
        if mem is not None:
            bits.append(f"内存 {mem[0]:.1f}/{mem[1]:.1f} GB")
        if pmem is not None:
            bits.append(f"进程 {pmem:.0f} MB")
        self.var_hw.set(" · ".join(bits))

    def _tick_status(self) -> None:
        self._fps.poll()
        self._update_hw_label()
        self.var_status_right.set(self.var_hw.get())
        self.after(1200, self._tick_status)

    # -------------------------------------------------------------- metrics
    def _refresh_metrics(self, polygons: list) -> None:
        if self.result is None:
            return
        gt = self.current["gt"] if self.current else None
        h, w = (self.current["image"].shape[:2] if self.current else (1, 1))
        if gt is not None:
            metrics = evaluate_vectors(polygons, self.result.mask, gt)
            for key in ("mIoU", "boundary_iou", "polygon_iou", "f1", "merge_rate", "instance_recall"):
                value = metrics[key]
                self._metric_vars[key].set("—" if not np.isfinite(value) else f"{value:.4f}")
            for key, digits in (("chamfer", 2), ("polis", 2)):
                value = metrics[key]
                self._metric_vars[key].set("—" if not np.isfinite(value) else f"{value:.{digits}f}")
        else:
            for key in ("mIoU", "boundary_iou", "polygon_iou", "f1", "chamfer", "polis",
                        "merge_rate", "instance_recall"):
                self._metric_vars[key].set("—")

        rect = [dominant_orientation(p.points)[1] for p in polygons if len(p.points) >= 3]
        self._metric_vars["rectilinearity"].set(f"{np.mean(rect):.3f}" if rect else "—")
        area = sum(float(p.area) for p in polygons)
        self._metric_vars["num_polygons"].set(f"{len(polygons)}")
        self._metric_vars["mean_vertices"].set(
            f"{np.mean([len(p.points) for p in polygons]):.1f}" if polygons else "0")
        self._metric_vars["area_ratio"].set(f"{100.0 * area / max(h * w, 1):.2f}%")

    def _on_vertices_changed(self, polygons: list) -> None:
        self._edited = True
        # keep the result object in sync so exports use the edited vertices
        if self.result is not None:
            self.result.polygons = list(polygons)
        self._refresh_metrics(polygons)
        self.var_stage.set(f"顶点已编辑：{len(polygons)} 个多边形，指标已实时更新")

    def _reset_polygons(self) -> None:
        if self._prob is None:
            return
        cfg = self._config()
        image = self.current["image"] if self.current else None
        self.result = vectorize(self._prob, self._edge if cfg.use_boundary else None, cfg, image=image)
        self._edited = False
        gt = self.current["gt"] if self.current else None
        boundary = None
        if cfg.use_boundary and self._edge is not None:
            boundary = (self._edge >= cfg.edge_threshold).astype(np.uint8)
        self.viewer.set_overlays(self.result.mask, boundary, gt)
        self.viewer.set_polygons(self.result.polygons)
        self._refresh_metrics(self.result.polygons)
        self.var_stage.set(f"已按当前配置重新生成 {self.result.num_polygons} 个多边形")

    # ----------------------------------------------------------------- export
    def _export_menu(self) -> None:
        if self.result is None:
            messagebox.showinfo("提示", "请先运行推理。")
            return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="导出 GeoJSON", command=self._export_geojson)
        menu.add_command(label="导出 Shapefile", command=self._export_shapefile)
        menu.add_separator()
        menu.add_command(label="导出指标 JSON", command=self._export_metrics)
        menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def _default_stem(self) -> str:
        name = self.current["name"] if self.current else "result"
        cfg = self._config()
        tag = ("b" if cfg.use_boundary else "n") + ("v" if cfg.use_vertex else "n") + ("g" if cfg.use_geometry else "n")
        return f"{name}_{self.model_var.get()}_{tag}"

    def _export_geojson(self) -> None:
        if self.result is None or not self.viewer.polygons:
            messagebox.showinfo("提示", "当前没有可导出的矢量结果。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".geojson",
                                            initialfile=f"{self._default_stem()}.geojson",
                                            filetypes=[("GeoJSON", "*.geojson"), ("JSON", "*.json")])
        if not path:
            return
        meta = {
            "source": self.current["name"] if self.current else "",
            "model": self.model_var.get(),
            "config": self._config().to_dict(),
            "vertex_edited": bool(self._edited),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_geojson(path, self.viewer.polygons, meta)
        self.var_stage.set(f"已导出 GeoJSON：{Path(path).name}（{len(self.viewer.polygons)} 个要素）")

    def _export_shapefile(self) -> None:
        if self.result is None or not self.viewer.polygons:
            messagebox.showinfo("提示", "当前没有可导出的矢量结果。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".shp",
                                            initialfile=f"{self._default_stem()}.shp",
                                            filetypes=[("Shapefile", "*.shp")])
        if not path:
            return
        shp, shx, dbf = write_polygon_shapefile(path, self.viewer.polygons)
        self.var_stage.set(f"已导出 Shapefile：{shp.name} / {shx.name} / {dbf.name}")

    def _export_metrics(self) -> None:
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            initialfile=f"{self._default_stem()}_metrics.json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        gt = self.current["gt"] if self.current else None
        payload = {
            "source": self.current["name"] if self.current else "",
            "model": self.model_var.get(),
            "vector_config": self._config().to_dict(),
            "vector_summary": self.result.summary(),
            "metrics": evaluate_vectors(self.viewer.polygons, self.result.mask, gt) if gt is not None else None,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.var_stage.set(f"已导出指标 JSON：{Path(path).name}")

    # ------------------------------------------------------------------ misc
    def _sync_layers(self) -> None:
        self.viewer.set_layers(
            show_mask=self.show_mask.get(),
            show_boundary=self.show_boundary.get(),
            show_gt=self.show_gt.get(),
            show_polygons=self.show_poly.get(),
            show_vertices=self.show_vertex.get(),
            mask_alpha=float(self.alpha_var.get()),
        )

    def _open_compare(self) -> None:
        if self.var_source.get() == "whu" and self.whu_tiles is not None:
            tiles, masks = self.whu_tiles, self.whu_masks
            names = [f"WHU {i:04d}" for i in range(len(tiles))]
        elif self.tiles is not None:
            tiles, masks = self.tiles, self.tile_masks
            names = [self.tile_meta[i].get("tile", str(i)) if i < len(self.tile_meta) else str(i)
                     for i in range(len(tiles))]
        else:
            messagebox.showinfo("提示", "未找到测试数据（data/processed 或 data/whu_processed）。")
            return
        CompareWindow(self, self.engine, tiles, masks, names, self._config())

    def _show_help(self) -> None:
        messagebox.showinfo(
            "使用说明",
            "1. 选择影像：内置 256²（Massachusetts）、WHU 512²（航空 0.3 m）、1500² 高分辨率样例，"
            "或打开本地 GeoTIFF/PNG/JPG。\n"
            "2. 在“算法配置”中点击勾选/取消边界分支、顶点回归、几何约束，即可对比不同配置。\n"
            "3. 按 F5（或“运行推理”）执行语义分割、边界分离与顶点回归的联合推理。\n"
            "4. 画布中：滚轮缩放、拖动平移、拖拽白色顶点编辑、双击边插入顶点、右键删除顶点。\n"
            "5. 指标面板实时显示 mIoU、边界 IoU、多边形 IoU、Chamfer、PoLiS 及耗时/资源占用。\n"
            "6. 导出 GeoJSON 或 Shapefile；点击“对比与消融”在当前数据集上批量比较 5 组配置。",
        )


def main() -> int:
    app = BuildingVectorizerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
