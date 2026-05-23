"""Plain MLP / FeedForward used inside transformer blocks and projectors."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with configurable normalization and activation.

    The ViT MLP (no internal norm), the LeWM projector (with norm), and the
    finetune head all fit through this same class — just change ``norm_layer``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Optional[type[nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeedForward(nn.Module):
    """Pre-norm Transformer FFN (LayerNorm → Linear → GELU → Linear).

    Used by the LeWM transformer blocks. Kept separate from :class:`MLP` because
    the order of norm/linear differs from the ViT MLP and we don't want to
    accidentally break a checkpoint that depends on the specific layout.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
