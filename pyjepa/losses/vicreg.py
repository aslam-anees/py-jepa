"""VICReg — Variance / Invariance / Covariance regularizer.

A useful baseline regularizer for JEPAs that do *not* use an EMA target
encoder. Often combined with the prediction loss as ``L = L_pred + λ_v V + λ_c C``.

Reference: Bardes et al. 2022, "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning."
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICRegLoss(nn.Module):
    """VICReg loss: ``λ_s * L_inv  +  λ_v * L_var  +  λ_c * L_cov``."""

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """``x`` and ``y`` are ``[N, D]``. If ``y`` is None, only var+cov terms are used."""
        if x.ndim == 3:
            x = x.reshape(-1, x.size(-1))
        if y is not None and y.ndim == 3:
            y = y.reshape(-1, y.size(-1))

        N, D = x.shape
        loss_inv = torch.tensor(0.0, device=x.device)
        if y is not None:
            loss_inv = F.mse_loss(x, y)

        x_c = x - x.mean(dim=0)
        std_x = torch.sqrt(x_c.var(dim=0) + self.eps)
        loss_var = torch.mean(F.relu(1.0 - std_x))
        cov_x = (x_c.T @ x_c) / (N - 1)
        loss_cov = _off_diagonal(cov_x).pow_(2).sum().div(D)

        if y is not None:
            y_c = y - y.mean(dim=0)
            std_y = torch.sqrt(y_c.var(dim=0) + self.eps)
            loss_var = (loss_var + torch.mean(F.relu(1.0 - std_y))) / 2
            cov_y = (y_c.T @ y_c) / (N - 1)
            loss_cov = (loss_cov + _off_diagonal(cov_y).pow_(2).sum().div(D)) / 2

        total = self.sim_coeff * loss_inv + self.std_coeff * loss_var + self.cov_coeff * loss_cov
        return {"loss": total, "loss_inv": loss_inv, "loss_var": loss_var, "loss_cov": loss_cov}
