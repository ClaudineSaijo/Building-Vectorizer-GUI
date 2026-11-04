"""Dataset for the WHU aerial building dataset (512x512 tiles).

Reads the packed ``.npy`` memmaps produced by ``scripts/prepare_whu.py`` and
applies scale + crop + flip + photometric augmentation.  Normalisation stats are
passed in explicitly because WHU uses its own mean/std (the ImageNet-pretrained
encoders still expect the WHU statistics at the model input; see the training
script help for details).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import augment_pair
from .occlusion_prior import PRIOR_MEAN, PRIOR_STD, occlusion_prior
from .losses import soft_dice_loss  # noqa: F401  (re-exported for convenience)
from .metrics import pixel_stats, scores_from_stats  # noqa: F401  (re-exported)

__all__ = ["WhuDataset", "random_scale_crop"]


def random_scale_crop(image: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator,
                      scale_range: tuple[float, float] = (0.75, 1.3)) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    s = float(rng.uniform(*scale_range))
    nh, nw = max(size, int(round(h * s))), max(size, int(round(w * s)))
    interp = cv2.INTER_LINEAR if s >= 1 else cv2.INTER_AREA
    image = cv2.resize(image, (nw, nh), interpolation=interp)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    y0 = int(rng.integers(0, nh - size + 1))
    x0 = int(rng.integers(0, nw - size + 1))
    return image[y0:y0 + size, x0:x0 + size], mask[y0:y0 + size, x0:x0 + size]


def occlusion_augment(image, mask, rng, p: float = 0.5, max_patches: int = 2):
    """Paste vegetation over buildings, keeping the mask intact.

    Two modes, chosen at random:

    * ``patch`` — one or two small canopy patches on a building (teaches local
      robustness, mimics a tree poking into a roof);
    * ``half``  — a large block covering ~half of one building (teaches the
      network to *complete* a missing part, which is what dense canopy does).

    In both modes the label keeps saying "building", so the network learns to
    infer the footprint from the surrounding roof and edge directions.
    """
    if p <= 0 or rng.random() > p:
        return image, mask
    rgb = image.astype(np.int16)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    veg = exg > 30
    if veg.sum() < 400:
        return image, mask

    if rng.random() < 0.5:
        return _occlude_patches(image, mask, veg, rng, max_patches)
    return _occlude_half(image, mask, veg, rng)


def _occlude_patches(image, mask, veg, rng, max_patches: int):
    bys, bxs = np.nonzero(mask)
    if len(bys) == 0:
        return image, mask
    vys, vxs = np.nonzero(veg)
    out = image.copy()
    h, w = image.shape[:2]
    for _ in range(int(rng.integers(1, max_patches + 1))):
        size = int(rng.integers(10, 34))
        if size >= h or size >= w:
            continue
        j = int(rng.integers(0, len(bys)))
        k = int(rng.integers(0, len(vys)))
        y0 = int(np.clip(vys[k] - size // 2, 0, h - size))
        x0 = int(np.clip(vxs[k] - size // 2, 0, w - size))
        py = int(np.clip(bys[j] - size // 2, 0, h - size))
        px = int(np.clip(bxs[j] - size // 2, 0, w - size))
        out[py:py + size, px:px + size] = image[y0:y0 + size, x0:x0 + size]
    return out, mask


def _occlude_half(image, mask, veg, rng):
    """Cover roughly half of one building with a canopy block."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    blobs = [i for i in range(1, n) if stats[i, 4] >= 400]
    if not blobs:
        return image, mask
    idx = blobs[int(rng.integers(0, len(blobs)))]
    x, y, w, h, area = (int(stats[idx, k]) for k in range(5))
    h_img, w_img = image.shape[:2]
    if rng.random() < 0.5:                      # cover left/right half
        size_w, size_h = max(w // 2, 8), h
        px = x if rng.random() < 0.5 else x + w - size_w
        py = y
    else:                                       # cover top/bottom half
        size_w, size_h = w, max(h // 2, 8)
        py = y if rng.random() < 0.5 else y + h - size_h
        px = x
    size_w = min(size_w, w_img)
    size_h = min(size_h, h_img)
    px = int(np.clip(px, 0, w_img - size_w))
    py = int(np.clip(py, 0, h_img - size_h))

    vys, vxs = np.nonzero(veg)
    k = int(rng.integers(0, len(vys)))
    sy = int(np.clip(vys[k] - size_h // 2, 0, h_img - size_h))
    sx = int(np.clip(vxs[k] - size_w // 2, 0, w_img - size_w))
    patch = image[sy:sy + size_h, sx:sx + size_w]
    if patch.shape[:2] != (size_h, size_w):
        return image, mask
    out = image.copy()
    out[py:py + size_h, px:px + size_w] = patch
    return out, mask


def shadow_augment(image, mask, rng, p: float = 0.5, strength: tuple[float, float] = (0.45, 0.75)):
    """Darken a rectangular region of the image, keeping the label unchanged.

    Simulates cast shadow over a roof, which is where recall drops (0.70 vs 0.93
    on plain roof in the WHU test split).
    """
    if p <= 0 or rng.random() > p:
        return image, mask
    h, w = image.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(ys) < 200:
        return image, mask
    j = int(rng.integers(0, len(ys)))
    size = int(rng.integers(60, max(61, min(h, w) // 2)))
    y0 = int(np.clip(ys[j] - size // 2, 0, max(h - size, 0)))
    x0 = int(np.clip(xs[j] - size // 2, 0, max(w - size, 0)))
    gain = float(rng.uniform(*strength))
    out = image.astype(np.float32)
    out[y0:y0 + size, x0:x0 + size] *= gain
    return np.clip(out, 0, 255).astype(np.uint8), mask


class WhuDataset(Dataset):
    def __init__(self, images, masks, mean, std, size: int = 512, train: bool = False,
                 seed: int = 42, scale_range: tuple[float, float] = (0.75, 1.3),
                 edge_width: int = 2, occlusion_p: float = 0.0, use_prior: bool = False,
                 shadow_p: float = 0.0, boost_indices: tuple[int, ...] = (),
                 boost_factor: int = 1) -> None:
        self.images = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else images
        self.masks = np.load(masks, mmap_mode="r") if isinstance(masks, (str, Path)) else masks
        if len(self.images) != len(self.masks):
            raise ValueError(f"image/mask count mismatch: {len(self.images)} vs {len(self.masks)}")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.size = int(size)
        self.train = train
        self.seed = int(seed)
        self.scale_range = scale_range
        self.edge_width = int(edge_width)
        self.occlusion_p = float(occlusion_p)
        self.use_prior = bool(use_prior)
        self.shadow_p = float(shadow_p)
        # oversample tiles that contain trees over roofs (they are rare in WHU)
        self._index = list(range(len(self.images)))
        if boost_indices and boost_factor > 1:
            extra = []
            for i in boost_indices:
                extra.extend([i] * (boost_factor - 1))
            self._index.extend(extra)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        real = self._index[idx] if hasattr(self, "_index") else idx
        image = np.asarray(self.images[real], dtype=np.uint8)
        mask = (np.asarray(self.masks[real]) > 0).astype(np.uint8)
        if self.train:
            rng = np.random.default_rng(self.seed + 100003 * self._epoch + idx)
            image, mask = random_scale_crop(image, mask, self.size, rng, self.scale_range)
            image, mask = augment_pair(image, mask, rng)
            if self.occlusion_p > 0:
                image, mask = occlusion_augment(image, mask, rng, self.occlusion_p)
            if self.shadow_p > 0:
                image, mask = shadow_augment(image, mask, rng, self.shadow_p)
        if image.shape[0] != self.size or image.shape[1] != self.size:
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = image.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        if self.use_prior:
            prior = (occlusion_prior(image) - PRIOR_MEAN) / PRIOR_STD
            x = np.concatenate([x, prior[..., None]], axis=2)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(mask.astype(np.float32))[None]
        e = torch.from_numpy(self._edge(mask).astype(np.float32))[None]
        return x, y, e, idx

    def _edge(self, mask: np.ndarray) -> np.ndarray:
        """Boundary band of the building mask (supervision for the edge head)."""
        from .dataset import build_edge_target

        return build_edge_target(mask, width=self.edge_width)
