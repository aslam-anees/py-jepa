"""SIGReg — Sketched Isotropic Gaussian Regularizer (LeWM).

Reference: Maes & Le Lidec et al. (2026), Eq. (3-5).

Pushes the distribution of embeddings toward a standard isotropic Gaussian by
matching its empirical characteristic function to that of N(0, I) along many
random 1D projections. This is what lets LeWM train end-to-end without an EMA
target encoder — the regularizer alone prevents collapse.

Implementation notes:
- Single-GPU version (the paper's single-GPU regime). For multi-GPU we'd want
  to all-gather projections across ranks, but the math is identical.
- The Epps-Pulley statistic is a weighted L2 distance between empirical and
  target characteristic functions sampled at ``knots`` points in ``[0, 3]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer.

    Args:
        knots: number of integration points for the Epps-Pulley statistic.
        num_proj: number of random 1D projections drawn per forward pass.

    Forward input: ``proj`` of shape ``(T, B, D)`` — typically ``emb.transpose(0, 1)``
    so the time axis is first.

    Forward output: scalar tensor, the average statistic across projections and time.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # Random projection matrix (D, num_proj), columns L2-normalized
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0).clamp_min_(1e-8))

        # x_t: (T, B, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # Epps-Pulley err: mean over batch axis (B = dim=-3 after unsqueeze)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)  # scale by batch
        return statistic.mean()
