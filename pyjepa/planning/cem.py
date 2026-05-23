"""Cross-Entropy Method planner — refits a diagonal Gaussian to elite samples.

Why CEM? It's the planner used in PLDM / LeWM evaluation. Simple, effective for
short horizons, embarrassingly parallel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .policy import PlanConfig


@dataclass
class CEMConfig(PlanConfig):
    momentum: float = 0.0          # interpolate (1-m)*new + m*old between iters


class CEMPlanner:
    """CEM optimizer over flat action sequences ``(H, A)``.

    Optimizes:
        action_seq* = argmin_a  E[ model.get_cost(info, a) ]

    by repeatedly sampling from a Gaussian, scoring with the model, and
    refitting the Gaussian to the top-``K`` candidates.
    """

    def __init__(self, config: CEMConfig | PlanConfig | None = None) -> None:
        if config is None:
            config = CEMConfig()
        if not isinstance(config, CEMConfig):
            cfg = CEMConfig()
            cfg.__dict__.update(config.__dict__)
            config = cfg
        self.cfg = config

    def __call__(self, model: nn.Module, info: dict, action_init: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        device = next(model.parameters()).device
        H, A = cfg.horizon, cfg.action_dim
        S = cfg.num_samples

        # Prepare a batched info dict with sample dim S
        batched: dict = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                v = v.to(device)
                batched[k] = v.unsqueeze(1).expand(v.shape[0], S, *v.shape[1:])
            else:
                batched[k] = v

        # Initial mean / std
        if action_init is not None:
            mean = action_init.to(device).clone()
        elif cfg.init_mean is not None:
            mean = cfg.init_mean.to(device).clone()
        else:
            mean = torch.zeros(H, A, device=device)
        std = torch.full_like(mean, cfg.init_std)

        for _ in range(cfg.num_iters):
            # Sample (B, S, H, A) — B inferred from info batch
            B = next(v.size(0) for v in batched.values() if torch.is_tensor(v)) if any(torch.is_tensor(v) for v in batched.values()) else 1
            noise = torch.randn(B, S, H, A, device=device)
            samples = mean[None, None] + std[None, None] * noise
            samples = samples.clamp(cfg.action_min, cfg.action_max)

            cost = model.get_cost(batched, samples)  # (B, S)
            elite_idx = torch.topk(cost, k=cfg.num_elites, largest=False, dim=-1).indices  # (B, K)

            # Gather elites then refit mean/std across batch & elite axes
            elite = torch.gather(
                samples,
                dim=1,
                index=elite_idx[..., None, None].expand(-1, -1, H, A),
            )  # (B, K, H, A)
            new_mean = elite.mean(dim=(0, 1))
            new_std = elite.std(dim=(0, 1)).clamp_min(1e-4)
            mean = cfg.momentum * mean + (1 - cfg.momentum) * new_mean
            std = cfg.momentum * std + (1 - cfg.momentum) * new_std

        return mean.clamp(cfg.action_min, cfg.action_max)
