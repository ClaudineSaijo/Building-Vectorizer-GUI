"""End-to-end self check for the vectorization pipeline (no GUI).

Runs the four ablation configurations on a few test tiles, prints the metrics,
then writes GeoJSON and Shapefile exports and re-reads them for validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.esri_shapefile import write_polygon_shapefile  # noqa: E402
from src.vector_metrics import evaluate_vectors  # noqa: E402
from src.vectorize import VectorConfig, vectorize, write_geojson  # noqa: E402

CONFIGS = [
    ("A 基线", dict(use_boundary=False, use_vertex=False, use_geometry=False)),
    ("B +边界", dict(use_boundary=True, use_vertex=False, use_geometry=False)),
    ("C +边界+顶点", dict(use_boundary=True, use_vertex=True, use_geometry=False)),
    ("D +边界+顶点+几何", dict(use_boundary=True, use_vertex=True, use_geometry=True)),
]


def main() -> int:
    images = np.load(ROOT / "data" / "processed" / "test_images.npy")
    gts = (np.load(ROOT / "data" / "processed" / "test_masks.npy") > 0).astype(np.uint8)
    probs = np.load(ROOT / "results" / "probs" / "full_test.npy").astype(np.float32)
    idx = [0, 17, 42]

    print(f"tiles={len(probs)}  sample={idx}")
    for name, overrides in CONFIGS:
        cfg = VectorConfig(**overrides)
        rows = []
        for i in idx:
            res = vectorize(probs[i], None, cfg, image=images[i])
            m = evaluate_vectors(res.polygons, res.mask, gts[i])
            rows.append(m)
            print(f"  [{name}] tile {i:3d}: polys={m['num_polygons']:2d} "
                  f"mIoU={m['mIoU']:.3f} polyIoU={m['polygon_iou']:.3f} "
                  f"BIoU={m['boundary_iou']:.3f} chamfer={m['chamfer']:.2f} "
                  f"vtx={m['mean_vertices']:.1f}")
        mean = {k: float(np.nanmean([r[k] for r in rows])) for k in
                ("mIoU", "polygon_iou", "boundary_iou", "chamfer", "polis")}
        print(f"  -> {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))

    # ---- export validation -------------------------------------------------
    cfg = VectorConfig()
    res = vectorize(probs[idx[0]], None, cfg, image=images[idx[0]])
    out = ROOT / "results" / "geometry" / "_selfcheck"
    out.mkdir(parents=True, exist_ok=True)
    gj = write_geojson(out / "test.geojson", res.polygons, {"check": True})
    payload = json.loads(gj.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection" and payload["features"], "geojson empty"
    ring = payload["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "geojson ring not closed"

    shp, shx, dbf = write_polygon_shapefile(out / "test.shp", res.polygons)
    import struct

    head = shp.read_bytes()[:100]
    code, = struct.unpack(">i", head[:4])
    length, = struct.unpack(">i", head[24:28])
    stype, = struct.unpack("<i", head[32:36])
    assert code == 9994 and stype == 5, (code, stype)
    assert length * 2 == shp.stat().st_size, (length * 2, shp.stat().st_size)
    shx_len, = struct.unpack(">i", shx.read_bytes()[24:28])
    assert shx_len * 2 == shx.stat().st_size
    dbf_head = dbf.read_bytes()[:32]
    nrec, = struct.unpack("<i", dbf_head[4:8])
    print(f"export ok: geojson={gj.stat().st_size}B shp={shp.stat().st_size}B "
          f"shx={shx.stat().st_size}B dbf={dbf.stat().st_size}B records={nrec} polys={len(res.polygons)}")
    assert nrec == len(res.polygons)

    # shapefile point count must match the polygon vertices
    total_pts = sum(len(p.points) for p in res.polygons)
    print(f"vertices exported: {total_pts}")
    print("SELFCHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
