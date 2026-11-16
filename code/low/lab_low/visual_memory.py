"""MEM-inspired space/time attention on clean VAE patches, before target pooling.

This is a trainable VAE-latent adapter, not the parameter-free ViT modification
from MEM (arXiv:2603.03596). The persistent reference is encoded independently;
only the rolling observation window participates in causal temporal attention.
"""
import math

import torch
from torch import nn


def temporal_position(seconds, width, dtype):
    """Fixed sinusoidal relative-time code, exactly zero at the current frame."""
    frequency = torch.exp(-math.log(10000.) * torch.arange(
        0, width, 2, device=seconds.device, dtype=torch.float32) / width)
    phase = seconds.float()[..., None] * frequency
    code = torch.stack((phase.sin(), phase.cos() - 1), dim=-1).flatten(-2)
    return code[..., :width].to(dtype)


class SpaceTimeBlock(nn.Module):
    def __init__(self, width, heads, temporal):
        super().__init__()
        self.heads = heads
        self.spatial_norm = nn.LayerNorm(width)
        self.spatial = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.temporal_norm = nn.LayerNorm(width) if temporal else None
        self.temporal = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.) if temporal else None
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x, valid, seconds):
        b, k, n, d = x.shape
        # Invalid slots may contain padding copies. Remove them before any mixing.
        x = x.masked_fill(~valid[:, :, None, None], 0)
        spatial = x.reshape(b*k, n, d)
        query = self.spatial_norm(spatial)
        spatial = spatial + self.spatial(query, query, query, need_weights=False)[0]
        x = spatial.reshape(b, k, n, d)
        if self.temporal is not None:
            time_code = temporal_position(seconds, d, x.dtype)
            query = self.temporal_norm(x + time_code[:, :, None])
            query = query.permute(0, 2, 1, 3).reshape(b*n, k, d)
            causal = torch.ones(k, k, dtype=torch.bool, device=x.device).tril()
            allowed = causal[None] & valid[:, None, :]
            # An invalid query before the first valid frame otherwise has no key.
            # Give it a dummy self edge, then discard its entire output below.
            allowed = allowed | (torch.eye(k, dtype=torch.bool, device=x.device)[None]
                                 & ~valid[:, :, None])
            blocked = (~allowed[:, None, None]).expand(b, n, self.heads, k, k)
            recalled = self.temporal(query, query, query,
                attn_mask=blocked.reshape(b*n*self.heads, k, k), need_weights=False)[0]
            x = x + recalled.reshape(b, n, k, d).permute(0, 2, 1, 3)
        x = x + self.ff(self.ff_norm(x))
        return x.masked_fill(~valid[:, :, None, None], 0)


class SpatialTemporalMemory(nn.Module):
    def __init__(self, width=256, heads=8, depth=4, temporal_every=4):
        super().__init__()
        if depth < 1 or temporal_every < 1 or temporal_every > depth or width % heads:
            raise ValueError("Require depth >= temporal_every >= 1 and width divisible by heads")
        self.blocks = nn.ModuleList(SpaceTimeBlock(width, heads, (i+1) % temporal_every == 0)
                                    for i in range(depth))

    def forward(self, x, valid, seconds):
        if x.ndim != 4 or x.shape[:2] != valid.shape or valid.shape != seconds.shape:
            raise ValueError("Expected patches[B,K,N,D], valid[B,K], seconds[B,K]")
        if not valid[:, -1].all() or not torch.isfinite(seconds).all() or (seconds > 0).any():
            raise ValueError("Memory must have a valid present and finite nonfuture times")
        # Ignore padded times when verifying order, but never allow valid reordering.
        for times, keep in zip(seconds, valid):
            selected = times[keep]
            if (selected[1:] < selected[:-1]).any():
                raise ValueError("Valid rolling observations must be in temporal order")
        for block in self.blocks:
            x = block(x, valid, seconds)
        return x
