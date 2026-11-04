"""Graph-convolution vertex regression (EdgeConv).

Takes the initial contour nodes of a building and predicts a 2-D offset per node,
turning the staircase contour into a compact polygon.  Deliberately small
(~40k parameters) so it trains on a CPU in minutes per epoch.

Design
------
* graph: k-NN graph on the (x, y) node coordinates, edges in both directions
* block: EdgeConv = MLP([x_i, x_j - x_i]) -> max-aggregate over neighbours
* head : per-node 2-D displacement, scaled by a factor so it can move a few pixels
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_ch, out_ch),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch, out_ch),
        )
        self.skip = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, C); edge_index: (2, E) with rows [src, dst]."""
        src, dst = edge_index[0], edge_index[1]
        msg = self.mlp(torch.cat([x[src], x[dst] - x[src]], dim=-1))
        agg = torch.full((x.size(0), msg.size(1)), -1e9, dtype=msg.dtype, device=msg.device)
        agg = agg.scatter_reduce(0, dst.unsqueeze(-1).expand_as(msg), msg, reduce="amax")
        return F.relu(self.skip(x) + agg)


class VertexGCN(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 64, layers: int = 3, k: int = 8,
                 max_offset: float = 12.0) -> None:
        super().__init__()
        self.k = int(k)
        self.max_offset = float(max_offset)
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([EdgeConv(hidden, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 2)

    # ------------------------------------------------------------------ #
    def knn_edges(self, coords: torch.Tensor) -> torch.Tensor:
        """Bidirectional k-NN edges for one graph, coords (N, 2)."""
        n = coords.size(0)
        k = min(self.k, max(n - 1, 1))
        dist = torch.cdist(coords, coords)
        dist.fill_diagonal_(float("inf"))
        idx = dist.topk(k, dim=1, largest=False).indices            # (N, k)
        src = torch.arange(n, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = idx.reshape(-1)
        return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """feats (N, F), coords (N, 2) -> offsets (N, 2) in pixels."""
        h = F.relu(self.inp(feats))
        edges = self.knn_edges(coords)
        for block in self.blocks:
            h = block(h, edges)
        return torch.tanh(self.head(h)) * self.max_offset

    def forward_batch(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Batched version: feats (B,N,F), coords (B,N,2) -> offsets (B,N,2).

        The batch is processed as one disconnected graph (block-diagonal edges),
        which removes the per-sample Python loop.
        """
        b, n, _ = coords.shape
        k = min(self.k, max(n - 1, 1))
        dist = torch.cdist(coords, coords)                       # (B, N, N)
        eye = torch.eye(n, dtype=torch.bool, device=coords.device).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        idx = dist.topk(k, dim=2, largest=False).indices         # (B, N, k)
        offs = (torch.arange(b, device=coords.device) * n).view(b, 1, 1)
        dst = (idx + offs).reshape(-1)
        src_local = torch.arange(n, device=coords.device).view(1, n, 1).expand(b, n, k)
        src = (src_local + offs).reshape(-1)
        edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

        h = F.relu(self.inp(feats.reshape(b * n, -1)))
        for block in self.blocks:
            h = block(h, edges)
        out = torch.tanh(self.head(h)) * self.max_offset
        return out.reshape(b, n, 2)


# ---------------------------------------------------------------------- #
# feature construction (shared by training and inference)
# ---------------------------------------------------------------------- #
def resample_ring(pts, n: int):
    """Resample a closed ring to ``n`` points at equal arc length (numpy)."""
    import numpy as np

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return np.repeat(pts[:1], n, axis=0)
    ring = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < len(seg) - 1 and cum[j + 1] < t:
            j += 1
        denom = max(cum[j + 1] - cum[j], 1e-9)
        w = (t - cum[j]) / denom
        out[i] = ring[j] * (1.0 - w) + ring[j + 1] * w
    return out


def node_features(nodes, image_shape, dist_map=None, dist_offset=(0.0, 0.0)):
    """Per-node features: normalised xy, tangent, local step, distance to boundary.

    ``dist_map`` is an optional distance transform of the building mask; its value
    at the node tells the network how deep the initial contour sits inside the
    shape (a staircase contour gives near-zero values everywhere).  ``dist_offset``
    maps global node coordinates into the local ``dist_map`` frame.
    """
    import numpy as np

    pts = np.asarray(nodes, dtype=np.float32)
    n = len(pts)
    h, w = image_shape[:2]
    nxt = np.roll(pts, -1, axis=0)
    prev = np.roll(pts, 1, axis=0)
    tangent = nxt - prev
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-6)
    center = pts.mean(axis=0, keepdims=True)
    rel = (pts - center) / np.maximum(np.linalg.norm(pts - center, axis=1, keepdims=True), 1e-6)
    feats = [
        pts[:, 0:1] / max(w, 1),
        pts[:, 1:2] / max(h, 1),
        tangent,
        rel,
    ]
    if dist_map is not None:
        local = pts - np.asarray(dist_offset, dtype=np.float32)
        yy = np.clip(local[:, 1].astype(int), 0, dist_map.shape[0] - 1)
        xx = np.clip(local[:, 0].astype(int), 0, dist_map.shape[1] - 1)
        d = dist_map[yy, xx].astype(np.float32)[:, None]
        d = np.clip(d / 12.0, 0.0, 1.0)
    else:
        d = np.zeros((n, 1), dtype=np.float32)
    feats.append(d)
    return np.concatenate(feats, axis=1).astype(np.float32)
