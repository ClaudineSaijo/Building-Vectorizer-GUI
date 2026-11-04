"""Headless smoke test for the Tk GUI: boot, run inference, edit a vertex, export."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows GBK console: never crash on characters like "²"
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from app.gui import BuildingVectorizerApp  # noqa: E402

ERRORS: list[str] = []


def run_sequence(app: BuildingVectorizerApp) -> None:
    app.update()
    print("boot ok:", app.current["label"] if app.current else None,
          "samples:", len(app.samples), "models:", app.engine.available_models())
    app.edge_var.set(True)
    app.vertex_var.set(True)
    app.geom_var.set(True)
    app._on_run()

    deadline = time.time() + 90
    while app.result is None and time.time() < deadline:
        app.update()
        time.sleep(0.02)
    if app.result is None:
        ERRORS.append("no result from inference")
        return

    print("result:", app.result.num_polygons, "polygons,", len(app.viewer.polygons), "drawn")
    print("metrics:", {k: v.get() for k, v in app._metric_vars.items()})
    print("time:", app.var_time.get(), "|", app.var_stage_detail.get())
    app.update()

    poly = app.viewer.polygons[0]
    before = poly.points[0].copy()
    poly.points[0] = poly.points[0] + 2.0
    app._on_vertices_changed(app.viewer.polygons)
    app.update()
    print("edited first vertex:", before, "->", poly.points[0])
    app._reset_polygons()
    app.update()
    print("reset ok:", app.result.num_polygons)

    out = ROOT / "results" / "geometry" / "_gui_export"
    out.mkdir(parents=True, exist_ok=True)
    from src.esri_shapefile import write_polygon_shapefile
    from src.vectorize import write_geojson

    gj = write_geojson(out / "gui.geojson", app.viewer.polygons, {"gui": True})
    shp = write_polygon_shapefile(out / "gui.shp", app.viewer.polygons)
    print("export:", gj.stat().st_size, [p.stat().st_size for p in shp])

    app._open_compare()
    app.update()
    for child in app.winfo_children():
        if child.winfo_class() == "Toplevel":
            print("compare window:", child.title())
            child.destroy()
    app.update()
    print("GUI_SMOKE_OK")


def main() -> int:
    app = BuildingVectorizerApp()

    def go() -> None:
        try:
            run_sequence(app)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            ERRORS.append("exception")
        finally:
            app.quit()

    app.after(300, go)
    app.mainloop()
    app.destroy()
    if ERRORS:
        print("ERRORS:", ERRORS)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
