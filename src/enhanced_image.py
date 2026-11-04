"""Label-derived "enhanced image" target (BIENet, TGRS 2024).

Take each building in the label, replace its pixels by a blend of the original
appearance and the building's own mean colour, boost the foreground contrast and
dim the background.  The result looks like a clean, unoccluded roof and is used as
the supervision of an auxiliary reconstruction branch (BIEB): to rebuild it, the
network has to encode "what the building should look like", which helps under
canopy and shadow.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["enhanced_image"]


def enhanced_image(image: np.ndarray, mask: np.ndarray, mean_ratio: float = 0.6,
                   fg_gain: float = 1.18, bg_gain: float = 0.72) -> np.ndarray:
    """Return an (H,W,3) uint8 "clean roof" image."""
    img = np.asarray(image, dtype=np.float32)
    m = (np.asarray(mask) > 0).astype(np.uint8)
    out = img.copy()

    n, labels = cv2.connectedComponents(m, connectivity=8)
    for idx in range(1, n):
        sel = labels == idx
        if not sel.any():
            continue
        mean = img[sel].mean(axis=0)
        out[sel] = img[sel] * (1.0 - mean_ratio) + mean * mean_ratio

    out = out * fg_gain
    bg = img * bg_gain
    result = np.where(m[..., None] > 0, out, bg)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
