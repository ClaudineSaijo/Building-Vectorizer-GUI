"""Optional segmentation backbones from ``segmentation-models-pytorch``.

Used by the WHU pipeline: a pretrained encoder plus UPerNet / DeepLabV3+ /
FPN decoder.  Imported lazily so the base project keeps working without the
extra dependency (``pip install segmentation-models-pytorch``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SmpSegmenter", "build_segmenter", "SMP_ARCHS", "SMP_ENCODERS", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SMP_ARCHS = {
    "upernet": "UPerNet",
    "deeplabv3plus": "DeepLabV3Plus",
    "deeplabv3+": "DeepLabV3Plus",
    "unet_smp": "Unet",
    "fpn": "FPN",
    "manet": "MAnet",
    "pspnet": "PSPNet",
}

# encoders that are known to work well for building extraction
SMP_ENCODERS = ("resnet34", "resnet50", "resnet101", "efficientnet-b4", "mit_b2", "convnext_tiny")


class SmpSegmenter(nn.Module):
    """Thin wrapper exposing the same forward contract as ``BuildingUNet``."""

    use_edge = False

    def __init__(self, arch: str, encoder: str, encoder_weights: str | None = "imagenet",
                 in_channels: int = 3) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "需要 segmentation-models-pytorch：pip install segmentation-models-pytorch"
            ) from exc

        arch_key = arch.lower()
        if arch_key not in SMP_ARCHS:
            raise ValueError(f"unknown smp arch: {arch}; choose from {sorted(SMP_ARCHS)}")
        cls = getattr(smp, SMP_ARCHS[arch_key])
        self.arch = arch_key
        self.encoder = encoder
        self.net = cls(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

    def forward(self, x: torch.Tensor, return_edge: bool = False):
        return self.net(x)


def build_segmenter(arch: str, encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                    in_channels: int = 3) -> SmpSegmenter:
    return SmpSegmenter(arch, encoder, encoder_weights, in_channels)
