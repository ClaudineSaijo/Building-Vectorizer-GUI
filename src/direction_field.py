"""Boundary direction field: target generation and loss.

Motivation (BOMSC-Net, TGRS 2022): when a building boundary is cut by trees or
cast shadow, the *direction* of the wall is still evident on both sides.  A
2-channel unit-vector field (cos, sin) along the boundary lets the network carry
that direction across the gap, instead of guessing from colour.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

__all__ = ["boundary_direction_target", "direction_loss"]


def boundary_direction_target(mask: np.ndarray, width: int = 2, smooth: int = 3):
    """Return (field, band): field is (2,H,W) unit vectors, band is (H,W) uint8.

    The target is only defined on a narrow band around the building boundary.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    h, w = m.shape
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    acc = np.zeros((h, w), np.float32)

    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 8:
            continue
        k = 2
        nxt = np.roll(pts, -k, axis=0)
        prev = np.roll(pts, k, axis=0)
        vec = nxt - prev
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.maximum(norm, 1e-6)
        xs = pts[:, 0].astype(int).clip(0, w - 1)
        ys = pts[:, 1].astype(int).clip(0, h - 1)
        np.add.at(dx, (ys, xs), vec[:, 0])
        np.add.at(dy, (ys, xs), vec[:, 1])
        np.add.at(acc, (ys, xs), 1.0)

    defined = acc > 0
    if not defined.any():
        return np.zeros((2, h, w), np.float32), np.zeros((h, w), np.uint8)

    dx = dx / np.maximum(acc, 1e-6)
    dy = dy / np.maximum(acc, 1e-6)
    norm = np.sqrt(dx * dx + dy * dy)
    dx = dx / np.maximum(norm, 1e-6)
    dy = dy / np.maximum(norm, 1e-6)
    # orient consistently: a unit vector and its negative describe the same wall
    flip = (dx < 0) | ((np.abs(dx) < 1e-6) & (dy < 0))
    dx[flip] *= -1.0
    dy[flip] *= -1.0

    if smooth > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth, smooth))
        dx = cv2.blur(dx, (smooth, smooth))
        dy = cv2.blur(dy, (smooth, smooth))

    band = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)))
            - cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))))
    band = (band > 0) & defined
    band = band.astype(np.uint8)
    return np.stack([dx, dy], axis=0), band


def direction_loss(pred: torch.Tensor, target: torch.Tensor, band: torch.Tensor,
                   weight: float = 1.0) -> torch.Tensor:
    """Cosine alignment + unit-norm penalty on the boundary band.

    pred/target: (B,2,H,W); band: (B,H,W) with 1 where the target is defined.
    """
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-6)
    target_n = target / (target.norm(dim=1, keepdim=True) + 1e-6)
    cos = (pred_n * target_n).sum(dim=1)
    mask = band > 0
    if mask.sum() < 1:
        return pred.sum() * 0.0
    align = (1.0 - cos)[mask].mean()
    norm_pen = (pred.norm(dim=1) - 1.0)[mask].pow(2).mean()
    return weight * (align + 0.1 * norm_pen)
