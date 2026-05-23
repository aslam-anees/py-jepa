"""Physical-surprise scoring.

For a learned world model, the prediction error on an observed trajectory acts
as a measure of how *physically implausible* the trajectory is — LeWM uses this
for "surprise" evaluation (Sec. 5.3 of the paper). High prediction error means
the model was surprised by what it saw.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


@torch.no_grad()
def surprise_score(
    model: nn.Module,
    info: dict,
    history_size: int = 3,
    reduction: str = "per_step",
) -> torch.Tensor:
    """Compute prediction error along an observed trajectory.

    Args:
        model: any LeWM-style model.
        info: dict with ``pixels`` ``(B, T, C, H, W)`` and ``action`` ``(B, T, A)``.
        history_size: predictor conditioning window.
        reduction: ``"per_step"`` returns ``(B, T - history_size)``;
            ``"mean"`` returns ``(B,)`` mean per trajectory;
            ``"trajectory"`` returns one scalar per batch.

    Returns prediction error in the chosen reduction.
    """
    info = dict(info)
    if "action" in info:
        info["action"] = torch.nan_to_num(info["action"], 0.0)

    out = model.encode(info)
    emb = out["emb"]                  # (B, T, D)
    act_emb = out["act_emb"]          # (B, T, A_emb)

    B, T, _ = emb.shape
    errors: list[torch.Tensor] = []
    for t in range(history_size, T):
        ctx_emb = emb[:, t - history_size : t]
        ctx_act = act_emb[:, t - history_size : t]
        pred = model.predict(ctx_emb, ctx_act)[:, -1]
        target = emb[:, t]
        errors.append((pred - target).pow(2).mean(dim=-1))   # (B,)

    err = torch.stack(errors, dim=1)  # (B, T - history_size)
    if reduction == "per_step":
        return err
    if reduction == "mean":
        return err.mean(dim=1)
    if reduction == "trajectory":
        return err.sum(dim=1)
    raise ValueError(f"unknown reduction {reduction}")
