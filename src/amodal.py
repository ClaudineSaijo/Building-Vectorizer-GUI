"""Amodal (visible + complete) supervision targets.

The core difficulty of occlusion is that the label is *amodal* (the full
footprint) while the image is *modal* (only the visible part).  Instead of asking
one head to bridge that gap implicitly, we supervise **two** heads:

* ``visible``  — what the image actually shows  (GT minus trees / cast shadow)
* ``amodal``   — the complete footprint         (the GT itself)

The gap between the two targets is exactly the occlusion, so the network is
forced to learn "the building continues under the canopy".
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["visible_target", "amodal_consistency"]


def occluder_map(image: np.ndarray, shadow_ratio: float = 0.62) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    veg = (exg > 30) & (g > r + 5)
    shadow = gray < shadow_ratio * float(np.median(gray))
    return (veg | shadow).astype(np.uint8)


def visible_target(image: np.ndarray, mask: np.ndarray, grow: int = 1) -> np.ndarray:
    """GT restricted to the part of the roof that is actually visible."""
    occ = occluder_map(image)
    if grow > 0:
        occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    vis = ((mask > 0) & (occ == 0)).astype(np.float32)
    return vis


def amodal_consistency(amodal_prob, visible_prob, band=None, weight: float = 0.2):
    """Penalise the amodal prediction falling *inside* the visible prediction."""
    import torch

    deficit = torch.relu(visible_prob - amodal_prob)
    if band is not None:
        deficit = deficit * band
    return weight * deficit.mean()
