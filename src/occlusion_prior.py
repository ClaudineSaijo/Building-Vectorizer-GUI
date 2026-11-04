"""Occlusion prior as a network input channel.

Two cues that a pixel is *not* free roof:

* **vegetation** — excess green index ``2G - R - B`` (trees, hedges, lawn);
* **shadow** — brightness well below the image median.

The value is in ``[0, 1]``: high where the ground is covered by canopy or cast
shadow.  Feeding it as a 4th channel lets the network learn how much to trust
the appearance at each pixel instead of a hand-written rule.
"""
from __future__ import annotations

import numpy as np

__all__ = ["occlusion_prior", "PRIOR_MEAN", "PRIOR_STD"]

PRIOR_MEAN = 0.5
PRIOR_STD = 0.25


def occlusion_prior(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32)[..., :3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2.0 * g - r - b
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    med = float(np.median(gray))
    veg = 1.0 / (1.0 + np.exp(-(exg - 25.0) / 10.0))
    dark = 1.0 / (1.0 + np.exp(-(0.80 * max(med, 1.0) - gray) / 12.0))
    return np.clip(veg + 0.6 * dark, 0.0, 1.0).astype(np.float32)
