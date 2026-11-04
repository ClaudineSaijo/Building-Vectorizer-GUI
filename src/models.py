"""Segmentation networks: plain U-Net baseline and a multi-scale / edge-aware U-Net."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None) -> None:
        super().__init__()
        mid_ch = out_ch if mid_ch is None else mid_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP(nn.Module):
    """Lightweight atrous spatial pyramid pooling (multi-scale context)."""

    def __init__(self, channels: int, out_channels: int | None = None, rates=(1, 2, 4, 8)) -> None:
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        self.branches = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                self.branches.append(nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                                   nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels * (len(rates) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        feats.append(F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.) for skip-connection filtering."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)
        nn.init.constant_(self.psi.bias, 2.0)  # start close to "pass-through"
        self.bn = nn.BatchNorm2d(inter_ch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        att = self.relu(self.bn(self.W_gate(gate) + self.W_skip(skip)))
        att = self.sigmoid(self.psi(att))
        return skip * att


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool) -> None:
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att = AttentionGate(in_ch, skip_ch, max(skip_ch // 2, 8))
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention:
            skip = self.att(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class BuildingUNet(nn.Module):
    """U-Net with optional ASPP bottleneck, attention-gated skips, edge head.

    Configurations used in the ablation:
      * `unet`  : use_msf=False, use_edge=False
      * `msf`   : use_msf=True,  use_edge=False
      * `full`  : use_msf=True,  use_edge=True
    """

    def __init__(
        self,
        in_channels: int = 3,
        base: int = 32,
        use_msf: bool = False,
        use_edge: bool = False,
        use_dir: bool = False,
        use_bieb: bool = False,
        use_amodal: bool = False,
        use_attention: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.use_msf = use_msf
        self.use_edge = use_edge
        self.use_dir = use_dir
        self.use_bieb = use_bieb
        self.use_amodal = use_amodal
        self.in_channels = int(in_channels)
        self.stem = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c4, c4 * 2)
        if dropout > 0:
            self.bottleneck = nn.Sequential(self.bottleneck, nn.Dropout2d(dropout))
        self.context = ASPP(c4 * 2, c4 * 2) if use_msf else nn.Identity()

        self.up4 = UpBlock(c4 * 2, c4, c4, use_attention=use_attention)
        self.up3 = UpBlock(c4, c3, c3, use_attention=use_attention)
        self.up2 = UpBlock(c3, c2, c2, use_attention=use_attention)
        self.up1 = UpBlock(c2, c1, c1, use_attention=use_attention)
        self.seg_head = nn.Conv2d(c1, 1, 1)
        if use_amodal:
            self.vis_head = nn.Conv2d(c1, 1, 1)
        if use_bieb:
            self.bieb_head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, 3, 1),
            )
        if use_dir:
            self.dir_head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, 2, 1),
            )
        if use_edge:
            self.edge_head = nn.Sequential(
                nn.Conv2d(c1, c1, 3, padding=1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, 1, 1),
            )

    def forward(self, x: torch.Tensor, return_edge: bool = False):
        """Return segmentation logits, optionally with the boundary logits.

        Training keeps the historical ``(seg, edge)`` tuple; inference returns a
        single tensor unless ``return_edge`` is requested (used by the GUI to
        drive boundary-based instance separation).
        """
        s1 = self.stem(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.context(self.bottleneck(self.pool(s4)))
        d4 = self.up4(b, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        seg = self.seg_head(d1)
        if (self.use_dir or self.use_bieb or self.use_amodal) and (self.training or return_edge):
            out = [seg]
            if self.use_edge:
                out.append(self.edge_head(d1))
            if self.use_dir:
                out.append(self.dir_head(d1))
            if self.use_bieb:
                out.append(self.bieb_head(d1))
            if self.use_amodal:
                out.append(self.vis_head(d1))
            return tuple(out)
        if self.use_edge and (self.training or return_edge):
            edge = self.edge_head(d1)
            return seg, edge
        return seg


def build_model(name: str, base: int = 32, in_channels: int = 3, use_dir: bool = False,
                use_bieb: bool = False, use_amodal: bool = False) -> BuildingUNet:
    name = name.lower()
    if name == "unet":
        return BuildingUNet(base=base, use_msf=False, use_edge=False, in_channels=in_channels, use_dir=use_dir, use_bieb=use_bieb, use_amodal=use_amodal)
    if name == "msf":
        return BuildingUNet(base=base, use_msf=True, use_edge=False, in_channels=in_channels, use_dir=use_dir, use_bieb=use_bieb, use_amodal=use_amodal)
    if name == "full":
        return BuildingUNet(base=base, use_msf=True, use_edge=True, in_channels=in_channels, use_dir=use_dir, use_bieb=use_bieb, use_amodal=use_amodal)
    if name == "unet_edge":  # multi-task without ASPP: cheaper, good CPU fallback
        return BuildingUNet(base=base, use_msf=False, use_edge=True, in_channels=in_channels, use_dir=use_dir, use_bieb=use_bieb, use_amodal=use_amodal)
    raise ValueError(f"unknown model: {name}")



def inflate_first_conv(state: dict, in_channels: int = 4) -> dict:
    """Grow a 3-channel stem to `in_channels` so warm starts still work."""
    out = dict(state)
    for key, value in list(out.items()):
        if value.ndim == 4 and value.shape[1] == 3 and in_channels > 3 and key.endswith("weight"):
            extra = value.mean(dim=1, keepdim=True) if in_channels == 4 else value.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)
            out[key] = torch.cat([value, extra], dim=1)
            break
    return out


def build_from_spec(spec: dict):
    """Build a model from checkpoint ``args``.

    ``unet`` / ``msf`` / ``full`` use the in-repo U-Net; any other architecture
    name is delegated to :mod:`src.models_seg` (segmentation-models-pytorch),
    which is how the WHU models are stored.
    """
    arch = str(spec.get("arch") or spec.get("model") or "unet").lower()
    if arch in {"unet", "msf", "full", "unet_edge"}:
        return build_model(arch, base=int(spec.get("base", 32)), in_channels=int(spec.get("in_channels", 3)),
                       use_dir=bool(spec.get("use_dir") or spec.get("direction_head")),
                       use_bieb=bool(spec.get("use_bieb") or spec.get("bieb")),
                       use_amodal=bool(spec.get("use_amodal") or spec.get("amodal")))
    from .models_seg import build_segmenter

    weights = spec.get("encoder_weights", "imagenet")
    if isinstance(weights, str) and weights.strip().lower() in {"", "none", "null", "false"}:
        weights = None
    return build_segmenter(
        arch=arch,
        encoder=str(spec.get("encoder") or "resnet34"),
        encoder_weights=weights,
        in_channels=int(spec.get("in_channels", 3)),
    )
