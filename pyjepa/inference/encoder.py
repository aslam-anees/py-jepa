"""Friendly inference wrappers around a JEPA encoder."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from ..device.backend import get_device, to_device


class Encoder:
    """Thin inference wrapper around any JEPA encoder.

    Sets ``eval()`` mode, disables grad, optionally pools tokens into a single vector.
    """

    def __init__(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        pooling: str = "mean",
        normalize: bool = False,
    ) -> None:
        self.device = device or get_device()
        self.module = module.to(self.device).eval()
        self.pooling = pooling
        self.normalize = normalize

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, pool: Optional[str] = None) -> torch.Tensor:
        x = x.to(self.device)
        tokens = self.module(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if hasattr(tokens, "last_hidden_state"):
            tokens = tokens.last_hidden_state
        pool = pool or self.pooling
        if tokens.ndim == 3:
            if pool == "mean":
                feat = tokens.mean(dim=1)
            elif pool == "cls":
                feat = tokens[:, 0]
            elif pool == "max":
                feat = tokens.amax(dim=1)
            elif pool == "none":
                feat = tokens
            else:
                raise ValueError(f"unknown pooling {pool}")
        else:
            feat = tokens
        if self.normalize and feat.ndim == 2:
            feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat


@torch.no_grad()
def embed_image(
    encoder: nn.Module,
    images: torch.Tensor,
    pooling: str = "mean",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """One-shot image embedding helper. ``images``: ``(B, C, H, W)``."""
    enc = Encoder(encoder, device=device, pooling=pooling)
    return enc(images)


@torch.no_grad()
def embed_video(
    encoder: nn.Module,
    clips: torch.Tensor,
    pooling: str = "mean",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """One-shot video embedding helper. ``clips``: ``(B, C, T, H, W)``."""
    enc = Encoder(encoder, device=device, pooling=pooling)
    return enc(clips)
