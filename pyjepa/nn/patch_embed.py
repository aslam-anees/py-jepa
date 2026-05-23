"""Patch embedding via strided convolution (the standard ViT tokenizer)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """2D image → token sequence via a single conv with kernel/stride = patch_size."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchEmbed3D(nn.Module):
    """3D video → token sequence via Conv3d (tubelet) embedding.

    Following V-JEPA, we group ``tubelet_size`` consecutive frames into one tube
    and tokenize each tube independently.
    """

    def __init__(
        self,
        patch_size: int = 16,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) → (B, N, D) where N = (T/ts) * (H/ps) * (W/ps)
        return self.proj(x).flatten(2).transpose(1, 2)
