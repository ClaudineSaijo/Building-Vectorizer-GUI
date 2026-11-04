"""Crop 1500x1500 Massachusetts tiles into 256x256 patches and pack .npy arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def crop_positions(length: int, patch: int, stride: int) -> list[int]:
    coords = list(range(0, length - patch + 1, stride))
    if not coords or coords[-1] != length - patch:
        coords.append(length - patch)
    return sorted(set(coords))


def read_pair(sat_path: Path, map_path: Path):
    image = np.asarray(Image.open(sat_path).convert("RGB"), dtype=np.uint8)
    mask_rgb = np.asarray(Image.open(map_path).convert("RGB"), dtype=np.uint8)
    mask = (mask_rgb.max(axis=2) > 127).astype(np.uint8)
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"shape mismatch: {sat_path} {image.shape} vs {mask.shape}")
    return image, mask


def collect(split: str, patch: int, stride: int, min_ratio: float, empty_keep: float, rng):
    sat_dir = RAW / split / "sat"
    map_dir = RAW / split / "map"
    sat_files = sorted(sat_dir.glob("*.tiff"))
    images, masks, meta = [], [], []
    for sat_path in tqdm(sat_files, desc=f"crop {split}"):
        map_path = map_dir / sat_path.name.replace(".tiff", ".tif")
        if not map_path.exists():
            print(f"  missing mask, skip {sat_path.name}")
            continue
        image, mask = read_pair(sat_path, map_path)
        h, w = mask.shape
        for y in crop_positions(h, patch, stride):
            for x in crop_positions(w, patch, stride):
                m = mask[y:y + patch, x:x + patch]
                ratio = float(m.mean())
                if split == "train":
                    if ratio == 0:
                        if rng.random() > empty_keep:
                            continue
                    elif ratio < min_ratio:
                        continue
                images.append(image[y:y + patch, x:x + patch].copy())
                masks.append(m.copy())
                meta.append({"tile": sat_path.stem, "y": y, "x": x, "ratio": ratio})
    if not images:
        raise RuntimeError(f"no patches produced for {split}")
    return np.stack(images), np.stack(masks), meta


def stats(images: np.ndarray) -> dict:
    sample = images[:: max(len(images) // 500, 1)].astype(np.float32) / 255.0
    mean = sample.mean(axis=(0, 1, 2))
    std = sample.std(axis=(0, 1, 2))
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-ratio", type=float, default=0.002)
    parser.add_argument("--empty-keep", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    info = {"patch": args.patch, "stride": args.stride, "splits": {}}
    for split in ("train", "valid", "test"):
        images, masks, meta = collect(split, args.patch, args.stride, args.min_ratio, args.empty_keep, rng)
        np.save(OUT / f"{split}_images.npy", images)
        np.save(OUT / f"{split}_masks.npy", masks)
        with (OUT / f"{split}_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        info["splits"][split] = {
            "num": int(len(images)),
            "mean_building_ratio": float(masks.mean()),
            "positive_ratio": float((masks.reshape(len(masks), -1).mean(1) > 0).mean()),
            "stats": stats(images),
        }
        print(f"[{split}] {len(images)} patches, building ratio {masks.mean():.4f}")
    with (OUT / "dataset_info.json").open("w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
