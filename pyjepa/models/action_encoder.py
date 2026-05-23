"""Small modules used by LeWM / world-model JEPAs to embed actions and bridge
between encoder/predictor dimensions."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ActionEmbedder(nn.Module):
    """Smooth then MLP-embed an action sequence.

    Input: ``(B, T, action_dim)``  →  ``(B, T, emb_dim)``.

    The 1D conv with kernel=1 is just a learned per-feature smoothing/projection
    — it's there so the action stream can have a different intrinsic dimension
    from the rest of the network without forcing one big projection.
    """

    def __init__(
        self,
        input_dim: int = 10,
        smoothed_dim: Optional[int] = None,
        emb_dim: int = 64,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        smoothed_dim = smoothed_dim or input_dim
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)


class MLPProjector(nn.Module):
    """Lightweight projector head: Linear → Norm → Act → Linear.

    Used as the projector / pred_proj heads of the LeWM model (and as a generic
    drop-in for any time you need a small MLP).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        norm_fn: Optional[type[nn.Module]] = nn.LayerNorm,
        act_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        norm_layer = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
