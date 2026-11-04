"""Zoomable image canvas with raster overlays and editable polygons.

The base image plus the mask/boundary overlays are composited once in NumPy and
only the visible crop is resampled on each redraw, so zooming into a 1500x1500
tile stays fast.  Polygons are vector canvas items, which keeps vertex dragging
smooth and lets the outline stay crisp at any zoom level.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np
from PIL import Image, ImageTk

__all__ = ["ImageViewer"]

COL_POLY = "#ffd60a"
COL_POLY_SEL = "#ff7b00"
COL_VERTEX = "#ffffff"
COL_VERTEX_SEL = "#ff3b30"


class ImageViewer(tk.Frame):
    """A pan/zoom view with mask, boundary, GT and polygon layers."""

    def __init__(self, master, on_vertices_changed: Callable | None = None,
                 on_status: Callable[[str], None] | None = None, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, bg="#1b1d22", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        self.on_vertices_changed = on_vertices_changed
        self.on_status = on_status

        self._base: np.ndarray | None = None       # uint8 RGB
        self._mask: np.ndarray | None = None
        self._boundary: np.ndarray | None = None
        self._gt: np.ndarray | None = None
        self._composite: np.ndarray | None = None

        self.mask_alpha = 0.45
        self.show_mask = True
        self.show_boundary = True
        self.show_polygons = True
        self.show_vertices = True
        self.show_gt = True

        self.polygons: list = []
        self._poly_items: list[int] = []
        self._vertex_items: list[int] = []

        self.scale = 1.0
        self.off_x = 0.0
        self.off_y = 0.0
        self._photo: ImageTk.PhotoImage | None = None
        self._image_item: int | None = None
        self._drag_vertex: tuple[int, int] | None = None
        self._pan_anchor: tuple[float, float] | None = None
        self._selected: int = -1

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.bind("<Configure>", lambda e: self.redraw())

    # ------------------------------------------------------------------ data
    def set_image(self, image_rgb: np.ndarray) -> None:
        self._base = np.ascontiguousarray(image_rgb[:, :, :3].astype(np.uint8))
        # a new image invalidates every overlay from the previous one
        self._mask = self._boundary = self._gt = None
        self._composite = None
        self.polygons = []
        self.fit()

    def set_overlays(self, mask: np.ndarray | None = None, boundary: np.ndarray | None = None,
                     gt: np.ndarray | None = None) -> None:
        shape = self.image_shape

        def _clean(layer):
            if layer is None:
                return None
            arr = np.asarray(layer)
            return (arr > 0) if arr.shape[:2] == shape else None

        self._mask = _clean(mask)
        self._boundary = _clean(boundary)
        self._gt = _clean(gt)
        self._composite = None
        self.redraw()

    def set_polygons(self, polygons: list) -> None:
        self.polygons = list(polygons)
        self._selected = -1
        self.redraw()

    @property
    def image_shape(self) -> tuple[int, int]:
        if self._base is None:
            return (0, 0)
        return self._base.shape[:2]

    # -------------------------------------------------------------- composite
    def _build_composite(self) -> np.ndarray | None:
        if self._base is None:
            return None
        canvas = self._base.astype(np.float32)
        if self.show_mask and self._mask is not None and self._mask.any():
            color = np.array([0, 200, 160], dtype=np.float32)
            a = self.mask_alpha
            m = self._mask
            canvas[m] = (1 - a) * canvas[m] + a * color
        if self.show_boundary and self._boundary is not None and self._boundary.any():
            b = self._boundary
            canvas[b] = 0.25 * canvas[b] + 0.75 * np.array([255, 70, 60], dtype=np.float32)
        if self.show_gt and self._gt is not None and self._gt.any():
            import cv2

            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gb = (cv2.dilate(self._gt.astype(np.uint8), k) - cv2.erode(self._gt.astype(np.uint8), k)) > 0
            canvas[gb] = 0.35 * canvas[gb] + 0.65 * np.array([60, 220, 90], dtype=np.float32)
        return canvas.clip(0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ view
    def fit(self) -> None:
        self.update_idletasks()
        h, w = self.image_shape
        if h == 0 or w == 0:
            return
        cw = max(self.canvas.winfo_width(), 64)
        ch = max(self.canvas.winfo_height(), 64)
        margin = 12
        self.scale = min((cw - 2 * margin) / w, (ch - 2 * margin) / h)
        self.scale = max(self.scale, 0.02)
        self.off_x = (cw - w * self.scale) / 2
        self.off_y = (ch - h * self.scale) / 2
        self.redraw()

    def zoom_to(self, factor: float, cx: float | None = None, cy: float | None = None) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        cx = cw / 2 if cx is None else cx
        cy = ch / 2 if cy is None else cy
        new_scale = float(np.clip(self.scale * factor, 0.02, 40.0))
        k = new_scale / self.scale
        self.off_x = cx - (cx - self.off_x) * k
        self.off_y = cy - (cy - self.off_y) * k
        self.scale = new_scale
        self.redraw()

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale + self.off_x, y * self.scale + self.off_y

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.off_x) / self.scale, (y - self.off_y) / self.scale

    # --------------------------------------------------------------- drawing
    def _draw_base(self) -> None:
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if self._base is None:
            return
        if self._composite is None:
            self._composite = self._build_composite()
        h, w = self._composite.shape[:2]
        x0 = max(0, int(np.floor((0 - self.off_x) / self.scale)))
        x1 = min(w, int(np.ceil((cw - self.off_x) / self.scale)))
        y0 = max(0, int(np.floor((0 - self.off_y) / self.scale)))
        y1 = min(h, int(np.ceil((ch - self.off_y) / self.scale)))
        if x1 <= x0 or y1 <= y0:
            return
        crop = self._composite[y0:y1, x0:x1]
        tw = max(1, int(round((x1 - x0) * self.scale)))
        th = max(1, int(round((y1 - y0) * self.scale)))
        img = Image.fromarray(crop)
        resample = Image.BILINEAR if self.scale < 1.0 else Image.NEAREST
        img = img.resize((tw, th), resample)
        self._photo = ImageTk.PhotoImage(img)
        cx = self.off_x + x0 * self.scale
        cy = self.off_y + y0 * self.scale
        if self._image_item is None:
            self._image_item = self.canvas.create_image(cx, cy, anchor="nw", image=self._photo)
            self.canvas.tag_lower(self._image_item)
        else:
            self.canvas.coords(self._image_item, cx, cy)
            self.canvas.itemconfigure(self._image_item, image=self._photo)

    def _draw_polygons(self) -> None:
        for item in self._poly_items + self._vertex_items:
            self.canvas.delete(item)
        self._poly_items.clear()
        self._vertex_items.clear()
        if not self.show_polygons:
            return
        width = 2 if self.scale < 2 else 3
        for idx, poly in enumerate(self.polygons):
            pts = np.asarray(poly.points, dtype=np.float64).reshape(-1, 2)
            if len(pts) < 2:
                continue
            coords = []
            for x, y in pts:
                coords.extend(self.to_canvas(x, y))
            color = COL_POLY_SEL if idx == self._selected else COL_POLY
            self._poly_items.append(
                self.canvas.create_line(*coords, coords[0], coords[1],
                                        fill=color, width=width, tags="poly")
            )
            if self.show_vertices and self.scale >= 0.75:
                r = 2 if self.scale < 2 else 3
                for x, y in pts:
                    cx, cy = self.to_canvas(x, y)
                    fill = COL_VERTEX_SEL if idx == self._selected else COL_VERTEX
                    self._vertex_items.append(
                        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                                fill=fill, outline="#111111", width=1, tags="vertex")
                    )

    def redraw(self) -> None:
        self._draw_base()
        self._draw_polygons()

    # ------------------------------------------------------------- interaction
    def _hit_vertex(self, cx: float, cy: float) -> tuple[int, int] | None:
        best = None
        best_d = 9.0
        for i, poly in enumerate(self.polygons):
            for j, (x, y) in enumerate(np.asarray(poly.points, dtype=np.float64).reshape(-1, 2)):
                px, py = self.to_canvas(x, y)
                d = float(np.hypot(px - cx, py - cy))
                if d < best_d:
                    best_d = d
                    best = (i, j)
        return best

    def _on_left_press(self, event) -> None:
        self.canvas.focus_set()
        hit = self._hit_vertex(event.x, event.y)
        if hit is not None:
            self._drag_vertex = hit
            self._selected = hit[0]
            self._draw_polygons()
            self._status(f"拖动顶点 ({hit[0] + 1}, {hit[1] + 1})；双击边可插入顶点，右键删除顶点")
        else:
            self._pan_anchor = (event.x, event.y)
            self.canvas.configure(cursor="fleur")

    def _on_left_drag(self, event) -> None:
        if self._drag_vertex is not None:
            i, j = self._drag_vertex
            x, y = self.to_image(event.x, event.y)
            h, w = self.image_shape
            self.polygons[i].points[j] = [float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))]
            self._draw_polygons()
            return
        if self._pan_anchor is not None:
            dx = event.x - self._pan_anchor[0]
            dy = event.y - self._pan_anchor[1]
            self.off_x += dx
            self.off_y += dy
            self._pan_anchor = (event.x, event.y)
            self._draw_base()
            self._draw_polygons()

    def set_layers(self, show_mask: bool | None = None, show_boundary: bool | None = None,
                   show_gt: bool | None = None, show_polygons: bool | None = None,
                   show_vertices: bool | None = None, mask_alpha: float | None = None) -> None:
        if show_mask is not None:
            self.show_mask = bool(show_mask)
        if show_boundary is not None:
            self.show_boundary = bool(show_boundary)
        if show_gt is not None:
            self.show_gt = bool(show_gt)
        if show_polygons is not None:
            self.show_polygons = bool(show_polygons)
        if show_vertices is not None:
            self.show_vertices = bool(show_vertices)
        if mask_alpha is not None:
            self.mask_alpha = float(mask_alpha)
        self._composite = None
        self.redraw()

    def _on_left_release(self, event) -> None:
        moved = self._drag_vertex is not None
        self._drag_vertex = None
        self._pan_anchor = None
        self.canvas.configure(cursor="crosshair")
        if moved and self.on_vertices_changed is not None:
            self.on_vertices_changed(self.polygons)

    def _on_double_click(self, event) -> None:
        hit = self._hit_vertex(event.x, event.y)
        if hit is not None:
            return
        # insert a vertex on the closest edge
        x, y = self.to_image(event.x, event.y)
        best = None
        best_d = 12.0
        for i, poly in enumerate(self.polygons):
            pts = np.asarray(poly.points, dtype=np.float64).reshape(-1, 2)
            for j in range(len(pts)):
                a, b = pts[j], pts[(j + 1) % len(pts)]
                pa = np.array(self.to_canvas(*a))
                pb = np.array(self.to_canvas(*b))
                p = np.array([event.x, event.y], dtype=np.float64)
                ab = pb - pa
                denom = float(ab @ ab)
                t = 0.0 if denom == 0 else float(np.clip((p - pa) @ ab / denom, 0, 1))
                d = float(np.linalg.norm(p - (pa + t * ab)))
                if d < best_d:
                    best_d = d
                    best = (i, j, a + t * (b - a))
        if best is not None:
            i, j, pt = best
            pts = np.asarray(self.polygons[i].points, dtype=np.float64).reshape(-1, 2)
            new = np.vstack([pts[:j + 1], np.asarray(pt, dtype=np.float64)[None, :], pts[j + 1:]])
            self.polygons[i].points = new.astype(np.float32)
            self._selected = i
            self._draw_polygons()
            self._status(f"已在多边形 {i + 1} 的第 {j + 1} 条边插入顶点")
            if self.on_vertices_changed is not None:
                self.on_vertices_changed(self.polygons)

    def _on_right_click(self, event) -> None:
        hit = self._hit_vertex(event.x, event.y)
        if hit is None:
            return
        i, j = hit
        pts = np.asarray(self.polygons[i].points, dtype=np.float64).reshape(-1, 2)
        if len(pts) <= 3:
            self._status("多边形至少需要 3 个顶点，无法继续删除")
            return
        self.polygons[i].points = np.delete(pts, j, axis=0).astype(np.float32)
        self._selected = i
        self._draw_polygons()
        self._status(f"已删除多边形 {i + 1} 的第 {j + 1} 个顶点")
        if self.on_vertices_changed is not None:
            self.on_vertices_changed(self.polygons)

    def _on_wheel(self, event) -> None:
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.zoom_to(factor, event.x, event.y)

    def _on_motion(self, event) -> None:
        if self._base is None:
            return
        x, y = self.to_image(event.x, event.y)
        h, w = self.image_shape
        if 0 <= x < w and 0 <= y < h:
            self._status(f"x={x:7.1f}  y={y:7.1f}   zoom={self.scale * 100:5.0f}%")

    def _status(self, text: str) -> None:
        if self.on_status is not None:
            self.on_status(text)
