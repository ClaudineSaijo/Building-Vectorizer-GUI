"""Render documentation screenshots of the GUI (main window + ablation window).

    .\\.venv\\Scripts\\python.exe scripts\\capture_gui_screens.py
"""
from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path

import numpy as np
from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.compare import CONFIGS, BatchJob, CompareWindow, run_batch  # noqa: E402
from app.gui import BuildingVectorizerApp  # noqa: E402
from src.vectorize import VectorConfig  # noqa: E402

OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def _dpi_scale() -> float:
    """Tk reports virtualised coordinates on a scaled display; ImageGrab does not."""
    try:
        return max(1.0, ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0)
    except Exception:  # noqa: BLE001
        return 1.0


def grab(widget, path: Path) -> None:
    widget.update()
    time.sleep(0.6)
    widget.update()
    s = _dpi_scale()
    x, y = widget.winfo_rootx(), widget.winfo_rooty()
    w, h = widget.winfo_width(), widget.winfo_height()
    ImageGrab.grab(bbox=(int(x * s), int(y * s), int((x + w) * s), int((y + h) * s))).save(path)
    print("saved", path, f"({int(w * s)}x{int(h * s)})")


def main() -> int:
    app = BuildingVectorizerApp()
    app.geometry("1480x960+30+20")
    app.attributes("-topmost", True)
    app.update()

    # default source is the WHU aerial test tile (best model + best threshold)
    app.var_source.set("whu")
    app._refresh_sample_list()
    app._load_sample(app._pick_start_sample())
    app.update()

    cfg = app._config()
    model = app.model_var.get()
    prob, edge, t_inf = app.engine.predict_timed(app.current["image"], model,
                                                cfg.use_boundary and app.engine.has_edge(model))
    from src.vectorize import vectorize

    t0 = time.perf_counter()
    result = vectorize(prob, edge if app.engine.has_edge(model) else None, cfg)
    t_vec = time.perf_counter() - t0
    app._apply_run({"prob": prob, "edge": edge, "result": result, "t_inf": t_inf,
                    "t_vec": t_vec, "cfg": cfg, "model": model})
    app.update()
    grab(app, OUT / "main_window.png")

    # ablation window on the same dataset, filled synchronously
    if app.var_source.get() == "whu" and app.whu_tiles is not None:
        tiles, masks = app.whu_tiles, app.whu_masks
        names = [f"WHU {i:04d}" for i in range(len(tiles))]
    else:
        tiles, masks = app.tiles, app.tile_masks
        names = [app.tile_meta[i].get("tile", str(i)) for i in range(len(tiles))]
    win = CompareWindow(app, app.engine, tiles, masks, names, cfg)
    win.attributes("-topmost", True)
    win.update()
    # sample a handful of the current dataset for the ablation table
    idx = np.unique(np.linspace(0, len(tiles) - 1, 8).round().astype(int))
    job = BatchJob(images=[np.asarray(tiles[i]) for i in idx],
                   gts=[np.asarray(masks[i]) for i in idx],
                   model_names=app.engine.available_models(), configs=CONFIGS, base_cfg=cfg)
    rows = run_batch(app.engine, job)
    win._done(rows)
    win.geometry("1560x800+60+40")
    win.update()
    grab(win, OUT / "compare_window.png")
    win.destroy()
    app.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
