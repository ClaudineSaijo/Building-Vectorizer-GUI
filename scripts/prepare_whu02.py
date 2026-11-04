"""Pack the WHU 0.2 m / 1024 px tiles into 512 px memmaps.

Why this exists
---------------
The 0.3 m WHU tiles we have been training on put the boundary error at ~1.5 px:
recall in the 0-1.5 px band is only 0.70 while the interior reaches 0.975.  The
0.2 m dataset is 1.5x finer per pixel, so cropping a *native* 512 px window (no
resampling) makes every building 1.5x larger in pixels at exactly the same
compute cost.  That is the cheapest way to test whether resolution is the real
bottleneck.

Reads straight out of ``r_train.zip`` / ``r_val.zip`` / ``r_test.zip`` (imagery in
``train/``, labels in ``train_label/``), so no 15 GB intermediate extraction.

    python scripts/prepare_whu02.py --crops 1 --out G:/BISHE_data/whu02
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ZIPS = {
    "train": Path(r"D:\All4DS\data4bishe\data4training\whu_raster_1024\whu_raster_1024\r_train.zip"),
    "valid": Path(r"D:\All4DS\data4bishe\data4training\whu_raster_1024\whu_raster_1024\r_val.zip"),
    "test": Path(r"D:\All4DS\data4bishe\data4training\whu_raster_1024\whu_raster_1024\r_test.zip"),
}
PREFIX = {"train": ("train/", "train_label/"), "valid": ("val/", "val_label/"), "test": ("test/", "test_label/")}


def _pairs(zf: zipfile.ZipFile, split: str) -> list[tuple[str, str]]:
    img_prefix, lab_prefix = PREFIX[split]
    images = {n.rsplit("/", 1)[-1].lower(): n for n in zf.namelist()
              if n.startswith(img_prefix) and n.lower().endswith((".tif", ".tiff"))}
    labels = {n.rsplit("/", 1)[-1].lower(): n for n in zf.namelist()
              if n.startswith(lab_prefix) and n.lower().endswith((".tif", ".tiff"))}
    keys = sorted(set(images) & set(labels))
    return [(images[k], labels[k]) for k in keys]


def _crops(img: np.ndarray, lab: np.ndarray, size: int, n_crops: int) -> list[tuple[np.ndarray, np.ndarray]]:
    h, w = img.shape[:2]
    if n_crops <= 1:
        y0, x0 = (h - size) // 2, (w - size) // 2
        offs = [(y0, x0)]
    elif n_crops == 2:
        offs = [(0, 0), (h - size, w - size)]
    else:
        offs = [(y, x) for y in (0, h - size) for x in (0, w - size)]
    out = []
    for y0, x0 in offs:
        out.append((img[y0:y0 + size, x0:x0 + size], lab[y0:y0 + size, x0:x0 + size]))
    return out


def pack(split: str, out_dir: Path, size: int, n_crops: int) -> dict:
    zpath = ZIPS[split]
    if not zpath.exists():
        print(f"[{split}] 缺少 {zpath}")
        return {}
    zf = zipfile.ZipFile(zpath)
    pairs = _pairs(zf, split)
    print(f"[{split}] {len(pairs)} 对, 每对 {n_crops} 个裁剪", flush=True)

    imgs_path = out_dir / f"{split}_images.npy"
    msks_path = out_dir / f"{split}_masks.npy"
    total = len(pairs) * n_crops
    imgs = np.lib.format.open_memmap(imgs_path, mode="w+", dtype=np.uint8,
                                     shape=(total, size, size, 3))
    msks = np.lib.format.open_memmap(msks_path, mode="w+", dtype=np.uint8,
                                     shape=(total, size, size))
    k = 0
    for a, b in pairs:
        img = np.array(Image.open(io.BytesIO(zf.read(a))).convert("RGB"))
        lab = np.array(Image.open(io.BytesIO(zf.read(b))).convert("L"))
        for ci, cl in _crops(img, lab, size, n_crops):
            imgs[k] = ci
            msks[k] = (cl > 0).astype(np.uint8) * 255
            k += 1
        if k % 400 == 0:
            print(f"  {split} {k}/{total}", flush=True)
    imgs.flush(); msks.flush()
    # 通道统计
    sample = np.asarray(imgs[::max(1, k // 400)]).astype(np.float32) / 255.0
    mean = sample.reshape(-1, 3).mean(axis=0)
    std = sample.reshape(-1, 3).std(axis=0)
    print(f"[{split}] 完成 {k} 张, mean={np.round(mean,4).tolist()} std={np.round(std,4).tolist()}", flush=True)
    return {"count": int(k), "mean": mean.tolist(), "std": std.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=r"G:/BISHE_data/whu02")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--crops", type=int, default=1, choices=[1, 2, 4])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": "WHU 0.2 m 1024 px tiles", "gsd": 0.2, "crop": args.size, "crops_per_tile": args.crops}
    for split in ("train", "valid", "test"):
        info = pack(split, out, args.size, args.crops)
        if info:
            meta[split] = info
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入", out / "meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
