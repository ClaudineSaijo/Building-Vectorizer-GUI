"""Launch the building vectorization demo GUI.

Usage (from the project root)::

    .\\.venv\\Scripts\\python.exe run_vectorizer.py

In PyCharm: choose the run configuration **Building Vectorizer GUI** (or right
click this file -> Run 'run_vectorizer').  The interpreter must be the project
venv ``.venv``; if the window does not show up, the console now prints a
diagnostic banner and any startup error is written to ``logs/gui_crash.log``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "logs" / "gui_crash.log"


def _banner(text: str) -> None:
    print(f"[GUI] {text}", flush=True)


def _report_crash(exc: BaseException) -> None:
    """Make startup failures visible instead of exiting silently."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}", encoding="utf-8")
        _banner(f"启动失败，完整堆栈已写入 {LOG_PATH}")
    except Exception:  # noqa: BLE001
        pass
    print(text, file=sys.stderr, flush=True)
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "可视化系统启动失败",
            f"{type(exc).__name__}: {exc}\n\n完整信息见：\n{LOG_PATH}",
        )
        root.destroy()
    except Exception:  # noqa: BLE001
        pass


def _bring_to_front(app) -> None:
    """Raise the window above the IDE instead of letting it open behind it."""
    try:
        app.deiconify()
        app.lift()
        app.attributes("-topmost", True)
        app.after(600, lambda: app.attributes("-topmost", False))
        app.focus_force()
        app.update()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="建筑物矢量化提取与可视化系统")
    parser.add_argument("--image", help="启动后直接载入的影像路径")
    parser.add_argument("--model", default="", help="启动时锁定的分割模型（默认自动选最优）")
    parser.add_argument("--sample", type=int, default=None, help="启动后载入的 WHU 测试样本序号（0 起）")
    args = parser.parse_args()

    _banner(f"解释器: {sys.executable}")
    _banner(f"工作目录: {os.getcwd()}")
    _banner(f"项目根目录: {ROOT}")

    try:
        import tkinter  # noqa: F401

        _banner(f"Tk 版本: {tkinter.TkVersion}")
    except Exception as exc:  # noqa: BLE001
        _banner(f"当前解释器缺少 tkinter：{exc}")
        _banner("请在 PyCharm 中把解释器切换为项目 .venv（Python 3.13 (BISHE venv)）")
        return 2

    from app.gui import BuildingVectorizerApp  # noqa: E402
    from src.infer import load_rgb_image  # noqa: E402

    _banner("正在创建主窗口…")
    app = None
    try:
        app = BuildingVectorizerApp()
        if args.model in app.engine.available_models():
            app._model_locked = True
            app.model_var.set(args.model)
            app.current_model = args.model
            app._on_model_change()
        if args.sample is not None:
            def _open_sample() -> None:
                app.var_source.set("whu")
                app._refresh_sample_list()
                app._load_sample(args.sample)
                app.update_idletasks()
                try:
                    app.viewer.fit()
                except Exception:
                    pass
                app._on_run()

            app.after(800, _open_sample)
        if args.image:
            app.after(400, lambda: app._set_current(
                load_rgb_image(args.image), None, Path(args.image).stem, Path(args.image).name))
        _bring_to_front(app)
        _banner("窗口已创建。若屏幕上看不到，请检查任务栏（窗口可能被 IDE 遮挡）或按 Alt+Tab。")
        _banner(f"窗口大小: {app.winfo_width()}x{app.winfo_height()}，"
                f"位置: ({app.winfo_x()}, {app.winfo_y()})")
        app.mainloop()
    except BaseException as exc:  # noqa: BLE001 - also catch SystemExit/KeyboardInterrupt traces
        if isinstance(exc, SystemExit) or isinstance(exc, KeyboardInterrupt):
            raise
        _report_crash(exc)
        return 1
    finally:
        if app is not None:
            try:
                app.destroy()
            except Exception:  # noqa: BLE001
                pass
    _banner("窗口已关闭，程序退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
