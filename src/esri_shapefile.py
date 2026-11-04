"""Minimal ESRI Shapefile writer for building footprint polygons.

Only what the project needs: a type-5 (Polygon) ``.shp`` plus its ``.shx``
index and a dBASE-III ``.dbf`` attribute table.  Implemented directly so the
demo system does not depend on GDAL / fiona / pyshp.
"""
from __future__ import annotations

import datetime as _dt
import struct
from pathlib import Path

import numpy as np

__all__ = ["write_polygon_shapefile"]

SHAPE_TYPE_POLYGON = 5
_FILE_CODE = 9994
_VERSION = 1000


def _ring(points: np.ndarray) -> np.ndarray:
    """Shapefile rings must be explicitly closed and clockwise (outer)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    # signed area > 0 means counter-clockwise -> flip to clockwise for outer rings
    x, y = pts[:, 0], pts[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed > 0:
        pts = pts[::-1]
    return pts


def _polygon_content(polygon, points: np.ndarray) -> bytes:
    xs, ys = points[:, 0], points[:, 1]
    parts = struct.pack("<i", 0)
    header = struct.pack("<i", SHAPE_TYPE_POLYGON)
    bbox = struct.pack("<4d", xs.min(), ys.min(), xs.max(), ys.max())
    counts = struct.pack("<2i", 1, len(points))
    coords = b"".join(struct.pack("<2d", px, py) for px, py in points)
    return header + bbox + counts + parts + coords


def _dbf_bytes(polygons: list, numeric_rows: list[dict]) -> bytes:
    fields = [
        ("FID", "N", 6, 0),
        ("AREA", "N", 16, 2),
        ("PERIM", "N", 16, 2),
        ("NVERT", "N", 6, 0),
        ("SCORE", "N", 9, 4),
        ("REG", "L", 1, 0),
    ]
    record_len = 1 + sum(f[2] for f in fields)
    header_len = 32 + 32 * len(fields) + 1
    today = _dt.date.today()
    out = bytearray()
    out += struct.pack("<BBBBIHH", 0x03, today.year - 1900, today.month, today.day,
                       len(polygons), header_len, record_len)
    out += b"\x00" * 20
    for name, ftype, size, dec in fields:
        out += name.encode("ascii").ljust(11, b"\x00")[:11]
        out += ftype.encode("ascii")
        out += b"\x00" * 4
        out += struct.pack("<BB", size, dec)
        out += b"\x00" * 14
    out += b"\r"

    for idx, row in enumerate(numeric_rows, start=1):
        record = bytearray(b" ")
        record += str(idx).rjust(6)[:6].encode("ascii")
        record += f"{row['area']:.2f}".rjust(16)[:16].encode("ascii")
        record += f"{row['perimeter']:.2f}".rjust(16)[:16].encode("ascii")
        record += str(row["vertices"]).rjust(6)[:6].encode("ascii")
        record += f"{row['score']:.4f}".rjust(9)[:9].encode("ascii")
        record += (b"T" if row["regularized"] else b"F")
        out += record
    out += b"\x1a"
    return bytes(out)


def write_polygon_shapefile(path: str | Path, polygons: list) -> tuple[Path, Path, Path]:
    """Write ``path.shp``/``.shx``/``.dbf`` for the given ``BuildingPolygon`` list.

    Coordinates are image pixel coordinates (x right, y down); the shapefile has
    no projection file because no georeferencing is attached to the tiles.
    """
    path = Path(path)
    if path.suffix.lower() in {".shp", ".shx", ".dbf"}:
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)

    shapes: list[bytes] = []
    points_list: list[np.ndarray] = []
    rows: list[dict] = []
    for poly in polygons:
        pts = _ring(poly.points)
        points_list.append(pts)
        shapes.append(_polygon_content(poly, pts))
        rows.append({
            "area": float(poly.area),
            "perimeter": float(poly.perimeter),
            "vertices": int(poly.num_vertices),
            "score": float(poly.score),
            "regularized": bool(poly.regularized),
        })

    # ---- .shp ----
    all_pts = np.vstack(points_list) if points_list else np.zeros((0, 2))
    if len(all_pts):
        xmin, ymin = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
        xmax, ymax = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    else:
        xmin = ymin = xmax = ymax = 0.0

    body = bytearray()
    for i, content in enumerate(shapes, start=1):
        body += struct.pack(">2i", i, len(content) // 2)
        body += content
    total_words = (100 + len(body)) // 2
    header = struct.pack(">i", _FILE_CODE) + b"\x00" * 20
    header += struct.pack(">i", total_words)
    header += struct.pack("<2i", _VERSION, SHAPE_TYPE_POLYGON)
    header += struct.pack("<4d", xmin, ymin, xmax, ymax)
    header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    (path.with_suffix(".shp")).write_bytes(header + bytes(body))

    # ---- .shx ----
    shx_body = bytearray()
    offset = 50  # 100 bytes = 50 words
    for i, content in enumerate(shapes, start=1):
        shx_body += struct.pack(">2i", i, offset)
        offset += (8 + len(content)) // 2
    total_shx = (100 + len(shx_body)) // 2
    shx_header = struct.pack(">i", _FILE_CODE) + b"\x00" * 20
    shx_header += struct.pack(">i", total_shx)
    shx_header += struct.pack("<2i", _VERSION, SHAPE_TYPE_POLYGON)
    shx_header += struct.pack("<4d", xmin, ymin, xmax, ymax)
    shx_header += struct.pack("<4d", 0.0, 0.0, 0.0, 0.0)
    (path.with_suffix(".shx")).write_bytes(shx_header + bytes(shx_body))

    # ---- .dbf ----
    (path.with_suffix(".dbf")).write_bytes(_dbf_bytes(polygons, rows))
    return path.with_suffix(".shp"), path.with_suffix(".shx"), path.with_suffix(".dbf")
