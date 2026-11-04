"""Model loading and tiled inference (CPU friendly).

The GUI and the batch benchmark share this engine, so a single model instance is
kept in memory and reused between runs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import DEFAULT_MEAN, DEFAULT_STD
from .occlusion_prior import PRIOR_MEAN, PRIOR_STD, occlusion_prior
from .models import build_from_spec

__all__ = ["InferenceEngine", "load_rgb_image", "MODEL_LABELS"]

MODEL_LABELS = {
    "unet": "U-Net 基线",
    "msf": "U-Net + 多尺度 ASPP",
    "full": "多尺度 + 边界分支",
}


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Read any common raster into a uint8 RGB array.

    Handles 8/16-bit TIFF, grayscale and RGBA inputs; 16-bit data is rescaled to
    the 0-255 range so the ImageNet-style normalisation stays valid.
    """
    import cv2

    path = Path(path)
    # cv2.imread cannot open non-ASCII paths on Windows (it returns None and
    # spams warnings), so such files go straight to Pillow.
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if str(path).isascii() else None
    if data is not None:
        # OpenCV decodes to BGR(A); the network was trained on RGB tiles
        if data.ndim == 3 and data.shape[2] == 4:
            b, g, r, a = cv2.split(data)
            data = np.dstack([r, g, b, a])
        elif data.ndim == 3 and data.shape[2] >= 3:
            data = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image

        with Image.open(path) as im:
            data = np.asarray(im.convert("RGB"))
    if data is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    elif data.shape[2] == 4:
        alpha = data[:, :, 3:4].astype(np.float32) / 255.0
        rgb = data[:, :, :3].astype(np.float32)
        data = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    elif data.shape[2] > 3:
        data = data[:, :, :3]
    if data.dtype == np.uint16:
        data = (data.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif data.dtype != np.uint8:
        if data.dtype.kind == "f":
            scale = 255.0 if data.max() <= 1.0 else 1.0
            data = (data * scale).clip(0, 255).astype(np.uint8)
        else:
            info = np.iinfo(data.dtype)
            data = (data.astype(np.float32) / max(info.max, 1) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(data[:, :, :3])


def _ramp(length: int, overlap: int) -> np.ndarray:
    w = np.ones(length, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


class InferenceEngine:
    """Lazy model cache + tiled forward pass."""

    def __init__(self, root: str | Path, tile: int = 256, batch: int = 4, threads: int | None = None) -> None:
        self.root = Path(root)
        self.tile = int(tile)
        self.batch = int(batch)
        self._cache: dict[str, tuple] = {}
        self._stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._dataset: dict[str, str] = {}
        if threads:
            torch.set_num_threads(int(threads))

    # ---------------------------------------------------------------- models
    def checkpoint_path(self, name: str) -> Path:
        return self.root / "runs" / name / "best.pt"

    def available_models(self) -> list[str]:
        """Every ``runs/<name>/best.pt`` that is a segmentation model.

        Vertex-regression checkpoints live under ``runs/vertex_*`` as well but are
        not full extractors, so they are skipped here.
        """
        runs = self.root / "runs"
        found = {p.parent.name for p in runs.glob("*/best.pt")} if runs.exists() else set()
        found = {n for n in found if not n.lower().startswith("vertex")}
        preferred = [n for n in ("full", "msf", "unet") if n in found]
        return preferred + sorted(found - set(preferred))

    def best_model(self) -> str | None:
        """Model with the highest validation IoU (used as the GUI default)."""
        names = self.available_models()
        if not names:
            return None
        scored = []
        for name in names:
            try:
                scored.append((self.model_info(name).get("val_iou", float("nan")), name))
            except Exception:  # noqa: BLE001
                continue
        scored = [s for s in scored if s[0] == s[0]]  # drop NaN
        if not scored:
            return names[0]
        return max(scored)[1]

    def model_dataset(self, name: str) -> str:
        """Training dataset recorded in the checkpoint ('whu' or 'massachusetts')."""
        if name not in self._dataset:
            self._dataset[name] = str(self.model_info(name).get("dataset") or "")
        return self._dataset[name]

    def models_for_dataset(self, dataset: str) -> list[str]:
        """Names trained on ``dataset``, best validation IoU first."""
        scored = []
        for name in self.available_models():
            try:
                if self.model_dataset(name) != dataset:
                    continue
                scored.append((self.model_info(name).get("val_iou", float("nan")), name))
            except Exception:  # noqa: BLE001
                continue
        scored = [s for s in scored if s[0] == s[0]]
        return [name for _, name in sorted(scored, reverse=True)]

    def best_threshold(self, name: str) -> float | None:
        """Threshold recorded by ``scripts/evaluate_whu.py`` for this model, if any."""
        for fname in ("eval_test.json", "eval_valid.json"):
            path = self.root / "runs" / name / fname
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                thr = float(data.get("best", {}).get("threshold", "nan"))
                if thr == thr:
                    return thr
            except Exception:  # noqa: BLE001
                continue
        return None

    def load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        ckpt_path = self.checkpoint_path(name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = dict(ckpt.get("args", {}))
        if "arch" not in args:
            args["arch"] = name if name in {"unet", "msf", "full"} else "unet"
        model = build_from_spec(args)
        model.load_state_dict(ckpt["model"])
        model.eval()

        mean = np.asarray(args.get("mean", DEFAULT_MEAN), dtype=np.float32).reshape(3)
        std = np.asarray(args.get("std", DEFAULT_STD), dtype=np.float32).reshape(3)
        self._stats[name] = (mean, std)
        self._cache[name] = (model, ckpt)
        return model, ckpt

    def model_info(self, name: str) -> dict:
        _, ckpt = self.load(name)
        args = ckpt.get("args", {})
        return {
            "model": name,
            "label": MODEL_LABELS.get(name, args.get("arch", name)),
            "epoch": ckpt.get("epoch"),
            "val_iou": float(ckpt.get("val", {}).get("iou", float("nan"))),
            "base": int(args.get("base", 0) or 0),
            "arch": args.get("arch", name),
            "encoder": args.get("encoder"),
            "dataset": args.get("dataset", "massachusetts" if name in {"unet", "msf", "full"} else ""),
        }

    def has_edge(self, name: str) -> bool:
        """True when the checkpoint has the auxiliary boundary head."""
        model, _ = self.load(name)
        return bool(getattr(model, "use_edge", False))

    def stats_for(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._stats:
            self.load(name)
        return self._stats[name]

    # ------------------------------------------------------------ inference
    def _normalise(self, batch: np.ndarray, stats: tuple[np.ndarray, np.ndarray] | None = None,
                   in_channels: int = 3) -> torch.Tensor:
        mean, std = stats or (DEFAULT_MEAN, DEFAULT_STD)
        x = torch.from_numpy(batch.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        mean_t = torch.tensor(np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1))
        std_t = torch.tensor(np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1))
        x = (x - mean_t) / std_t
        if in_channels == 4:
            priors = np.stack([occlusion_prior(im) for im in batch])
            priors = torch.from_numpy((priors - PRIOR_MEAN) / PRIOR_STD).unsqueeze(1)
            x = torch.cat([x, priors], dim=1)
        return x

    def _positions(self, length: int, tile: int) -> list[int]:
        if length <= tile:
            return [0]
        step = max(tile // 2, 1)
        pos = list(range(0, length - tile + 1, step))
        if pos[-1] != length - tile:
            pos.append(length - tile)
        return pos

    def _forward(self, model, batch: np.ndarray, use_edge: bool,
                 stats: tuple[np.ndarray, np.ndarray] | None = None):
        in_ch = int(getattr(model, "in_channels", 0) or 0)
        if in_ch == 0:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    in_ch = int(module.in_channels)
                    break
        logits = model(self._normalise(batch, stats, in_ch), return_edge=use_edge)
        if isinstance(logits, tuple):
            seg_logits = logits[0]
            edge_logits = logits[1] if len(logits) > 1 else None
        else:
            seg_logits, edge_logits = logits, None
        probs = torch.sigmoid(seg_logits).numpy()[:, 0]
        edges = torch.sigmoid(edge_logits).numpy()[:, 0] if edge_logits is not None else None
        return probs, edges

    @staticmethod
    def _tta_forward(batch: np.ndarray, k: int) -> np.ndarray:
        """Dihedral transform k of a (N,H,W,C) batch."""
        out = batch
        if k & 1:
            out = out[:, :, ::-1, :]
        if k & 2:
            out = out[:, ::-1, :, :]
        if k & 4:
            out = np.rot90(out, 1, axes=(1, 2))
        return out

    @staticmethod
    def _tta_inverse(arr: np.ndarray, k: int) -> np.ndarray:
        out = arr
        if k & 4:
            out = np.rot90(out, -1, axes=(1, 2))
        if k & 2:
            out = out[:, ::-1, :]
        if k & 1:
            out = out[:, :, ::-1]
        return out

    def predict(self, image: np.ndarray, model_name: str = "full",
                want_edge: bool = False, tta: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (probability map, boundary probability map or None).

        ``tta=True`` averages the eight dihedral transforms (4 flips x 2 rotations),
        which costs 8x the inference time but recovers roughly 1 IoU point.
        """
        model, ckpt = self.load(model_name)
        use_edge = want_edge and bool(getattr(model, "use_edge", False))
        stats = self.stats_for(model_name)
        # models trained on 512 tiles (WHU) should be run at their own tile size
        tile = int(ckpt.get("args", {}).get("size") or self.tile)
        img = np.asarray(image, dtype=np.uint8)
        h, w = img.shape[:2]
        ph, pw = max(tile, int(np.ceil(h / tile) * tile)), max(tile, int(np.ceil(w / tile) * tile))
        if ph > h or pw > w:
            pad = ((0, ph - h), (0, pw - w), (0, 0))
            canvas = np.pad(img, pad, mode="reflect")
        else:
            canvas = img

        ys, xs = self._positions(ph, tile), self._positions(pw, tile)
        overlap = max(tile - (ys[1] - ys[0]), 0) if len(ys) > 1 else 0
        wy = _ramp(tile, overlap)
        wx = _ramp(tile, overlap)
        weight = np.outer(wy, wx).astype(np.float32)
        wacc = np.zeros((ph, pw), dtype=np.float32)
        for y in ys:
            for x in xs:
                wacc[y:y + tile, x:x + tile] += weight

        patches, coords = [], []
        for y in ys:
            for x in xs:
                patches.append(canvas[y:y + tile, x:x + tile])
                coords.append((y, x))

        n_aug = 8 if tta else 1
        acc = np.zeros((ph, pw), dtype=np.float32)
        edge_acc = np.zeros((ph, pw), dtype=np.float32) if use_edge else None
        with torch.no_grad():
            for k in range(n_aug):
                for start in range(0, len(patches), self.batch):
                    chunk = np.stack(patches[start:start + self.batch])
                    if tta:
                        chunk = np.ascontiguousarray(self._tta_forward(chunk, k))
                    probs, edges = self._forward(model, chunk, use_edge, stats)
                    if tta:
                        probs = self._tta_inverse(probs, k)
                        if edges is not None:
                            edges = self._tta_inverse(edges, k)
                    for j, prob in enumerate(probs):
                        y, x = coords[start + j]
                        acc[y:y + tile, x:x + tile] += prob * weight
                        if edge_acc is not None and edges is not None:
                            edge_acc[y:y + tile, x:x + tile] += edges[j] * weight
        denom = np.maximum(wacc * n_aug, 1e-6)
        prob_map = acc / denom
        edge_map = edge_acc / denom if edge_acc is not None else None
        return prob_map[:h, :w].astype(np.float32), (edge_map[:h, :w].astype(np.float32) if edge_map is not None else None)

    def predict_timed(self, image: np.ndarray, model_name: str = "full",
                      want_edge: bool = False, tta: bool = False) -> tuple[np.ndarray, np.ndarray | None, float]:
        t0 = time.perf_counter()
        prob, edge = self.predict(image, model_name, want_edge, tta=tta)
        return prob, edge, time.perf_counter() - t0
