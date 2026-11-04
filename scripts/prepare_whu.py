"""Pack the WHU aerial building dataset into ``.npy`` memmaps.

The official release is a folder tree of image/label pairs (512x512 GeoTIFF or
PNG).  This script discovers the pairs, tiles or pads them to a fixed size,
splits into train/valid/test and writes memory-mapped arrays so training does
not have to hold 8k images in RAM.

Expected layouts (both are accepted)::

    WHU/
      train/{image,label}/*.tif
      valid/{image,label}/*.tif        # or val/
      test/{image,label}/*.tif

    WHU/
      image/*.tif
      label/*.tif                      # split by --val-frac / --test-frac

Example::

    .\\.venv\\Scripts\\python.exe scripts\\prepare_whu.py ^
        --root data\\whu_raw --out data\\whu_processed --size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows GBK console would otherwise crash on characters like "²"
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from src.infer import load_rgb_image  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "valid": "valid", "val": "valid", "validation": "valid",
    "test": "test", "testing": "test",
}
IMAGE_DIR_NAMES = ("image", "images", "img", "images_rgb", "rgb")
LABEL_DIR_NAMES = ("label", "labels", "mask", "masks", "gt")


def _find_dir(parent: Path, names) -> Path | None:
    if not parent.is_dir():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() in names:
            return child
    return None


def _index_labels(label_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in label_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            index[p.stem.lower()] = p
    return index


def _pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    labels = _index_labels(label_dir)
    out = []
    for img in sorted(image_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXT:
            continue
        lab = labels.get(img.stem.lower())
        if lab is not None:
            out.append((img, lab))
    return out


def _fit_tiles(image: np.ndarray, mask: np.ndarray, size: int):
    """Yield (image, mask) tiles of exactly size x size."""
    h, w = image.shape[:2]
    if h < size or w < size:
        ph, pw = max(h, size), max(w, size)
        image = np.pad(image, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, ph - h), (0, pw - w)), mode="reflect")
        h, w = image.shape[:2]
    for y in range(0, h - size + 1, size):
        for x in range(0, w - size + 1, size):
            yield image[y:y + size, x:x + size], mask[y:y + size, x:x + size]


def _read_pair(img_path: Path, lab_path: Path, size: int):
    image = load_rgb_image(img_path)
    raw = load_rgb_image(lab_path)
    mask = (raw.max(axis=2) > 0).astype(np.uint8)
    if image.shape[:2] != mask.shape[:2]:
        import cv2

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    for tile_img, tile_mask in _fit_tiles(image, mask, size):
        yield tile_img, tile_mask


def _discover(root: Path) -> tuple[dict[str, list[tuple[Path, Path]]], bool]:
    """Return ({split: pairs}, used_official_splits)."""
    splits: dict[str, list[tuple[Path, Path]]] = {}
    official = False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        split = SPLIT_ALIASES.get(child.name.lower())
        if split is None:
            continue
        img_dir = _find_dir(child, IMAGE_DIR_NAMES) or child
        lab_dir = _find_dir(child, LABEL_DIR_NAMES)
        if lab_dir is None:
            continue
        pairs = _pairs(img_dir, lab_dir)
        if pairs:
            splits[split] = pairs
            official = True
    if official:
        return splits, True

    img_dir = _find_dir(root, IMAGE_DIR_NAMES)
    lab_dir = _find_dir(root, LABEL_DIR_NAMES)
    if img_dir is None or lab_dir is None:
        raise SystemExit(f"在 {root} 下找不到 image/label 目录，请检查数据集结构")
    return {"all": _pairs(img_dir, lab_dir)}, False


def _split_pairs(pairs: list, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_test = int(round(len(pairs) * test_frac))
    n_val = int(round(len(pairs) * val_frac))
    test = [pairs[i] for i in idx[:n_test]]
    val = [pairs[i] for i in idx[n_test:n_test + n_val]]
    train = [pairs[i] for i in idx[n_test + n_val:]]
    return {"train": train, "valid": val, "test": test}


def _official_split(pairs: list, n_train: int = 4736, n_valid: int = 1036):
    """WHU's published counts: 4736 train / 1036 valid / 2416 test.

    The archive has no split folders, so this assumes the files are ordered by
    the original cropping sequence (train area, then evaluation area, then the
    two test areas).  Verify with the shapefiles if exact comparability matters.
    """
    pairs = list(pairs)
    if len(pairs) < n_train + n_valid:
        raise SystemExit(f"文件数不足（{len(pairs)}），无法按官方划分切分")
    return {
        "train": pairs[:n_train],
        "valid": pairs[n_train:n_train + n_valid],
        "test": pairs[n_train + n_valid:],
    }


def _pack(split: str, pairs: list[tuple[Path, Path]], out_dir: Path, size: int, limit: int | None):
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        print(f"  [{split}] 空，跳过")
        return 0, None, None, None
    images_path = out_dir / f"{split}_images.npy"
    masks_path = out_dir / f"{split}_masks.npy"
    # first pass: count tiles so the memmap can be allocated up front
    from PIL import Image

    counts = []
    for img_path, _lab in pairs:
        with Image.open(img_path) as im:
            w, h = im.size
        counts.append((max(h, size) // size) * (max(w, size) // size))
    total_tiles = sum(counts)
    images = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8,
                                       shape=(total_tiles, size, size, 3))
    masks = np.lib.format.open_memmap(masks_path, mode="w+", dtype=np.uint8,
                                      shape=(total_tiles, size, size))
    # stats are accumulated while writing: reopening the memmap right after the
    # write can still see unsynced pages on Windows and produce wrong values
    step = max(1, total_tiles // 500)
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    seen = 0
    k = 0
    for n, (img_path, lab_path) in enumerate(pairs, 1):
        for tile_img, tile_mask in _read_pair(img_path, lab_path, size):
            images[k] = tile_img
            masks[k] = tile_mask
            if split == "train" and k % step == 0:
                a = tile_img.astype(np.float32) / 255.0
                s1 += a.sum(axis=(0, 1))
                s2 += (a ** 2).sum(axis=(0, 1))
                seen += a.shape[0] * a.shape[1]
            k += 1
        if n % 200 == 0:
            print(f"  [{split}] {n}/{len(pairs)} 张", flush=True)
    images.flush()
    masks.flush()
    del images, masks
    print(f"  [{split}] 写出 {k} 个 {size}x{size} 图块 -> {images_path.name}")
    stats = None
    if split == "train" and seen:
        mean = s1 / seen
        var = np.maximum(s2 / seen - mean ** 2, 0.0)
        stats = {"mean": np.round(mean, 6).tolist(), "std": np.round(np.sqrt(var), 6).tolist()}
    return k, images_path, masks_path, stats


def _stats(images_path: Path, max_samples: int = 400) -> dict:
    """Fallback: compute channel statistics from an existing memmap."""
    images = np.load(images_path, mmap_mode="r")
    n = len(images)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_samples)).round().astype(int)) if n else []
    if len(idx) == 0:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    seen = 0
    for start in range(0, len(idx), 50):
        chunk = np.stack([np.asarray(images[i], dtype=np.float32) / 255.0 for i in idx[start:start + 50]])
        s1 += chunk.sum(axis=(0, 1, 2))
        s2 += (chunk ** 2).sum(axis=(0, 1, 2))
        seen += chunk.shape[0] * chunk.shape[1] * chunk.shape[2]
    mean = s1 / max(seen, 1)
    var = np.maximum(s2 / max(seen, 1) - mean ** 2, 0.0)
    return {"mean": np.round(mean, 6).tolist(), "std": np.round(np.sqrt(var), 6).tolist()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "data" / "whu_raw"), help="WHU 原始数据根目录")
    parser.add_argument("--out", default=str(ROOT / "data" / "whu_processed"), help="输出目录")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.10, help="无官方划分时的验证集比例")
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="每个 split 最多处理多少张原图（调试用）")
    parser.add_argument("--official-split", action="store_true",
                        help="按 WHU 公开数量切分（4736/1036/其余），假定文件按裁剪顺序排列")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"数据目录不存在：{root}")
        print("请先下载 WHU 航空影像建筑物数据集并解压到该目录。")
        return 1
    out_dir = ensure_dir(args.out)

    splits, official = _discover(root)
    if not official:
        if args.official_split:
            splits = _official_split(splits["all"])
            official = True
            print("按官方数量切分（假定文件顺序 = 裁剪顺序）")
        else:
            splits = _split_pairs(splits["all"], args.val_frac, args.test_frac, args.seed)
    print(f"发现划分：{ {k: len(v) for k, v in splits.items()} } (官方划分={official})")

    info = {"size": args.size, "official_split": official, "splits": {}}
    for split in ("train", "valid", "test"):
        pairs = splits.get(split, [])
        n, images_path, masks_path, stats = _pack(split, pairs, out_dir, args.size, args.limit or None)
        if n == 0:
            continue
        entry = {"num": int(n), "images": images_path.name, "masks": masks_path.name}
        if split == "train":
            entry["stats"] = stats or _stats(images_path)
        info["splits"][split] = entry

    (out_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"\n完成：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
