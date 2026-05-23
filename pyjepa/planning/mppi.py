"""Model-Predictive Path Integral planner.

MPPI is a soft-min update rule that weights every sampled trajectory by
``exp(-cost / temperature)`` and takes the (weighted) mean. It's smoother than
CEM (no hard elite cutoff) and often works better for continuous control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .policy import PlanConfig


@dataclass
class MPPIConfig(PlanConfig):
    noise_std: float = 1.0
    update_momentum: float = 0.5


class MPPIPlanner:
    """Simple MPPI — single iteration of softmin reweighting per call."""

    def __init__(self, config: MPPIConfig | PlanConfig | None = None) -> None:
        if config is None:
            config = MPPIConfig()
        if not isinstance(config, MPPIConfig):
            cfg = MPPIConfig()
            cfg.__dict__.update(config.__dict__)
            config = cfg
        self.cfg = config

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

        mean = action_init.to(device).clone() if action_init is not None else torch.zeros(H, A, device=device)

        for _ in range(cfg.num_iters):
            B = next(v.size(0) for v in batched.values() if torch.is_tensor(v)) if any(torch.is_tensor(v) for v in batched.values()) else 1
            noise = torch.randn(B, S, H, A, device=device) * cfg.noise_std
            samples = mean[None, None] + noise
            samples = samples.clamp(cfg.action_min, cfg.action_max)

            cost = model.get_cost(batched, samples)  # (B, S)
            cost = cost - cost.min(dim=-1, keepdim=True).values
            w = torch.softmax(-cost / max(1e-6, cfg.temperature), dim=-1)  # (B, S)
            new_mean = (w[..., None, None] * samples).sum(dim=1).mean(dim=0)  # (H, A)
            mean = cfg.update_momentum * mean + (1 - cfg.update_momentum) * new_mean

        return mean.clamp(cfg.action_min, cfg.action_max)
