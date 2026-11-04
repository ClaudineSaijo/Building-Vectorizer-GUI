"""Dataset and augmentation for binary building extraction."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset-level RGB statistics (filled by prepare_data.py and copied here as a
# fallback).  Values are in [0, 1].
DEFAULT_MEAN = np.array([0.2570, 0.2586, 0.2346], dtype=np.float32)
DEFAULT_STD = np.array([0.1866, 0.1819, 0.1851], dtype=np.float32)


def build_edge_target(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Binary boundary band (uint8) from a binary mask."""
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    dil = cv2.dilate(mask, kernel)
    ero = cv2.erode(mask, kernel)
    return ((dil - ero) > 0).astype(np.uint8)


def augment_pair(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Geometric + photometric augmentation.

    Photometric jitter intentionally covers brightness/contrast/gamma changes so
    that illumination robustness can be probed on the real test tiles.
    """
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    if rng.random() < 0.5:
        image, mask = np.rot90(image, 1), np.rot90(mask, 1)
    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)

    if rng.random() < 0.8:
        alpha = float(rng.uniform(0.75, 1.30))         # contrast
        beta = float(rng.uniform(-28, 28))             # brightness (0-255)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.5:
        gamma = float(rng.uniform(0.7, 1.4))
        table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        image = cv2.LUT(image, table)
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 7), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=float(rng.uniform(0.4, 0.9)))
    return image, mask


class BuildingPatchDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray | str | Path,
        masks: np.ndarray | str | Path,
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        train: bool = False,
        seed: int = 42,
        edge_width: int = 2,
    ) -> None:
        self.images = np.load(images, mmap_mode="r") if isinstance(images, (str, Path)) else images
        self.masks = np.load(masks, mmap_mode="r") if isinstance(masks, (str, Path)) else masks
        assert len(self.images) == len(self.masks)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        self.train = train
        self.seed = seed
        self.edge_width = edge_width
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = np.asarray(self.images[idx], dtype=np.uint8)
        mask = (np.asarray(self.masks[idx]) > 0).astype(np.uint8)
        if self.train:
            rng = np.random.default_rng(self.seed + 100003 * self._epoch + idx)
            image, mask = augment_pair(image, mask, rng)
        edge = build_edge_target(mask, self.edge_width)
        x = image.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        y = torch.from_numpy(mask.astype(np.float32))[None]
        e = torch.from_numpy(edge.astype(np.float32))[None]
        return x, y, e, idx
