"""Loss functions for segmentation with optional boundary supervision."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (probs * target).sum(dim=dims)
    union = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


class SegEdgeLoss:
    def __init__(
        self,
        bce_pos_weight: float = 2.0,
        edge_pos_weight: float = 4.0,
        edge_weight: float = 0.35,
        dice_weight: float = 0.5,
    ) -> None:
        self.bce_pos_weight = bce_pos_weight
        self.edge_pos_weight = edge_pos_weight
        self.edge_weight = edge_weight
        self.dice_weight = dice_weight

    def __call__(self, logits, target, edge_logits=None, edge_target=None):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=torch.tensor(self.bce_pos_weight, device=logits.device)
        )
        dice = soft_dice_loss(logits, target)
        seg_loss = bce + self.dice_weight * dice
        parts = {"bce": float(bce.detach()), "dice": float(dice.detach()), "edge": 0.0}
        total = seg_loss
        if edge_logits is not None and edge_target is not None:
            edge_bce = F.binary_cross_entropy_with_logits(
                edge_logits, edge_target,
                pos_weight=torch.tensor(self.edge_pos_weight, device=edge_logits.device),
            )
            total = total + self.edge_weight * edge_bce
            parts["edge"] = float(edge_bce.detach())
        return total, parts
