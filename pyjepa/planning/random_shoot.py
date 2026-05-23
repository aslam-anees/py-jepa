"""Random shooting — sample N trajectories, pick the best. Baseline / warm-start."""

from __future__ import annotations

import torch
import torch.nn as nn

from .policy import PlanConfig


class RandomShootingPlanner:
    """Single-shot random sampling. Cheap, useful as a CEM warm-start."""

    def __init__(self, config: PlanConfig | None = None) -> None:
        self.cfg = config or PlanConfig()

    def __call__(self, model: nn.Module, info: dict, action_init: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        device = next(model.parameters()).device
        H, A = cfg.horizon, cfg.action_dim
        S = cfg.num_samples

        batched: dict = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                v = v.to(device)
                batched[k] = v.unsqueeze(1).expand(v.shape[0], S, *v.shape[1:])
            else:
                batched[k] = v

        B = next(v.size(0) for v in batched.values() if torch.is_tensor(v)) if any(torch.is_tensor(v) for v in batched.values()) else 1
        samples = (
            torch.rand(B, S, H, A, device=device) * (cfg.action_max - cfg.action_min) + cfg.action_min
        )
        cost = model.get_cost(batched, samples)  # (B, S)
        best_idx = cost.argmin(dim=-1)
        best = samples[torch.arange(B), best_idx]  # (B, H, A)
        return best.mean(dim=0)
