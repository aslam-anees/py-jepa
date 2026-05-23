"""Bare prediction loss used inside higher-level JEPA losses."""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import torch
import torch.nn.functional as F


def prediction_loss(
    preds: Union[torch.Tensor, Sequence[torch.Tensor]],
    targets: Union[torch.Tensor, Sequence[torch.Tensor]],
    loss_exp: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """``mean( |pred - target|^p ) / p`` — the V-JEPA prediction loss.

    With ``loss_exp=1`` this is L1; with ``loss_exp=2`` it's MSE-like. Lists of
    tensors are averaged element-wise (one term per mask).
    """
    if isinstance(preds, torch.Tensor):
        preds = [preds]
    if isinstance(targets, torch.Tensor):
        targets = [targets]

    total = 0.0
    for z, h in zip(preds, targets):
        diff = torch.abs(z - h)
        if loss_exp != 1.0:
            diff = diff ** loss_exp
        total = total + diff.mean() / loss_exp
    n = len(preds) if isinstance(preds, Iterable) else 1
    return total / max(1, n) if reduction == "mean" else total
