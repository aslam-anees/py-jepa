"""Sinusoidal positional embeddings (1D / 2D / 3D) and interpolation helpers.

Why sinusoidal? They're parameter-free, deterministic, and resolution-agnostic —
which means you can fine-tune at a different image size than you pre-trained at
and still get useful position information by interpolating in 2D/3D pixel space.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_1d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid = np.arange(grid_size, dtype=float)
    pos_embed = _get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """Return ``[grid_size*grid_size, embed_dim]`` 2D positional embeddings."""
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    grid_depth: int,
    cls_token: bool = False,
    uniform_power: bool = False,
) -> np.ndarray:
    """Return ``[D*H*W, embed_dim]`` 3D positional embeddings for video.

    When ``uniform_power=True`` all three axes get equal embedding dimension
    (the V-JEPA default for ``vit_giant`` style configs); otherwise the temporal
    axis gets twice the spatial axes (the V-JEPA ``vit_base`` default).
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(grid_h, grid_d, grid_w)

    if not uniform_power:
        h_dim = w_dim = embed_dim // 4
        d_dim = embed_dim // 2
    else:
        h_dim = w_dim = d_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = _get_1d_sincos_pos_embed_from_grid(h_dim, grid_h)
    emb_w = _get_1d_sincos_pos_embed_from_grid(w_dim, grid_w)
    emb_d = _get_1d_sincos_pos_embed_from_grid(d_dim, grid_d)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    new_grid: Tuple[int, ...],
    old_grid: Tuple[int, ...],
) -> torch.Tensor:
    """Resize a 2D or 3D positional embedding to a new spatial/temporal grid.

    Inputs:
        pos_embed: ``[1, N_old, D]``
        new_grid:  ``(H, W)`` or ``(T, H, W)``
        old_grid:  ``(H_old, W_old)`` or ``(T_old, H_old, W_old)``

    Returns ``[1, prod(new_grid), D]``.
    """
    B, N, D = pos_embed.shape
    if len(new_grid) == 2:
        H_old, W_old = old_grid
        H, W = new_grid
        if (H, W) == (H_old, W_old):
            return pos_embed
        pe = pos_embed.reshape(1, H_old, W_old, D).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(H, W), mode="bicubic", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, H * W, D)
    if len(new_grid) == 3:
        T_old, H_old, W_old = old_grid
        T, H, W = new_grid
        if (T, H, W) == (T_old, H_old, W_old):
            return pos_embed
        pe = pos_embed.reshape(1, T_old, H_old, W_old, D).permute(0, 4, 1, 2, 3)
        pe = F.interpolate(pe, size=(T, H, W), mode="trilinear", align_corners=False)
        return pe.permute(0, 2, 3, 4, 1).reshape(1, T * H * W, D)
    raise ValueError(f"new_grid must have 2 or 3 dims, got {new_grid}")
