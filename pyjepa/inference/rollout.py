"""Latent rollout helpers for world models.

Two flavours:
- :func:`rollout_latent`: pure functional rollout given encoder + predictor + actions.
- :class:`Rollout`: stateful version that keeps a per-call history buffer (useful
  inside an interactive policy loop).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange


@torch.no_grad()
def rollout_latent(
    model: nn.Module,
    initial: dict,
    actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    """Run a latent rollout for ``actions``.

    Args:
        model: a LeWM-style model implementing ``.encode()`` and ``.predict()``.
        initial: dict with ``pixels`` of shape ``(B, T_ctx, C, H, W)``.
        actions: ``(B, T_total, A)`` where ``T_total >= T_ctx``.
        history_size: how many past frames the predictor conditions on.

    Returns ``(B, T_total + 1, D)`` predicted embeddings.
    """
    initial = dict(initial)
    initial["action"] = actions[:, : initial["pixels"].size(1)]
    out = model.encode(initial)
    emb = out["emb"]  # (B, T_ctx, D)

    act_full = model.action_encoder(actions)  # (B, T_total, A_emb)
    T_total = actions.size(1)
    for t in range(initial["pixels"].size(1), T_total):
        ctx_emb = emb[:, -history_size:]
        ctx_act = act_full[:, t - history_size + 1 : t + 1]
        if ctx_act.size(1) < history_size:
            pad = act_full[:, : history_size - ctx_act.size(1)]
            ctx_act = torch.cat([pad, ctx_act], dim=1)
        pred = model.predict(ctx_emb, ctx_act)[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
    return emb


class Rollout:
    """Stateful wrapper used inside a policy loop.

    Maintains a window of the last ``history_size`` embeddings/actions and steps
    one frame at a time as the agent acts.
    """

    def __init__(self, model: nn.Module, history_size: int = 3) -> None:
        self.model = model
        self.history_size = history_size
        self.emb_history: Optional[torch.Tensor] = None
        self.act_history: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset(self, info: dict) -> torch.Tensor:
        """Encode the initial observation window and return the embedding."""
        out = self.model.encode(info)
        self.emb_history = out["emb"]
        self.act_history = out.get("act_emb")
        return self.emb_history

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """Predict the next-state embedding given one new action."""
        if self.emb_history is None:
            raise RuntimeError("call reset(info) before step()")
        act_emb = self.model.action_encoder(action.unsqueeze(1))
        self.act_history = torch.cat([self.act_history, act_emb], dim=1) if self.act_history is not None else act_emb

        ctx_emb = self.emb_history[:, -self.history_size :]
        ctx_act = self.act_history[:, -self.history_size :]
        pred = self.model.predict(ctx_emb, ctx_act)[:, -1:]
        self.emb_history = torch.cat([self.emb_history, pred], dim=1)
        return pred.squeeze(1)
