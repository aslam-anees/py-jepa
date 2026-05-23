"""The canonical V-JEPA / I-JEPA loss: prediction term + variance regularizer.

We split the variance regularizer into its own module so users can swap it for
SIGReg (LeWM) or VICReg without changing the prediction code.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prediction import prediction_loss


class VarianceRegularizer(nn.Module):
    """Hinge variance regularizer: ``mean( relu(1 - sqrt(var(z) + eps)) )``.

    Encourages each feature dimension to maintain unit variance across the
    patch axis, preventing the JEPA encoder from collapsing to a constant.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: Sequence[torch.Tensor]) -> torch.Tensor:
        if isinstance(preds, torch.Tensor):
            preds = [preds]
        std = sum(torch.sqrt(z.var(dim=1) + self.eps) for z in preds) / len(preds)
        return torch.mean(F.relu(1.0 - std))


class JEPALoss(nn.Module):
    """L = prediction_loss + reg_coeff * variance_regularizer.

    Mirrors the V-JEPA loss exactly. Pass ``regularizer=SIGReg(...)`` to use the
    LeWM regularizer instead.
    """

    def __init__(
        self,
        loss_exp: float = 1.0,
        reg_coeff: float = 0.0,
        regularizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.loss_exp = loss_exp
        self.reg_coeff = reg_coeff
        self.regularizer = regularizer if regularizer is not None else VarianceRegularizer()

    def forward(
        self,
        preds: Sequence[torch.Tensor] | torch.Tensor,
        targets: Sequence[torch.Tensor] | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_pred = prediction_loss(preds, targets, loss_exp=self.loss_exp)
        if self.reg_coeff > 0:
            loss_reg = self.regularizer(preds)
            total = loss_pred + self.reg_coeff * loss_reg
            return {"loss": total, "loss_pred": loss_pred, "loss_reg": loss_reg}
        return {"loss": loss_pred, "loss_pred": loss_pred}
