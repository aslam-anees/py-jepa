"""Policy wrappers — combine a planner + a world model into a callable that picks
actions for a given observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn


@dataclass
class PlanConfig:
    """Common knobs shared by all sampling-based planners."""

    horizon: int = 8                # plan length (number of action steps)
    action_block: int = 1           # actions executed before re-planning
    num_samples: int = 256          # candidate sequences per iteration
    num_elites: int = 32            # top-K kept (CEM)
    num_iters: int = 4              # refinement iterations
    action_dim: int = 2
    action_min: float = -1.0
    action_max: float = 1.0
    temperature: float = 1.0        # MPPI softmin temperature
    init_mean: Optional[torch.Tensor] = None
    init_std: float = 1.0
    seed: int = 0


class RandomPolicy:
    """Pure-noise baseline — useful for sanity checks and as a planner warm-start."""

    def __init__(self, config: PlanConfig) -> None:
        self.cfg = config
        self.gen = torch.Generator().manual_seed(config.seed)

    def __call__(self, info: dict) -> torch.Tensor:
        H, A = self.cfg.horizon, self.cfg.action_dim
        a = torch.rand(H, A, generator=self.gen) * (self.cfg.action_max - self.cfg.action_min) + self.cfg.action_min
        return a


class WorldModelPolicy:
    """Plan with a learned world model. Calls ``solver`` to optimize a cost.

    Solver signature: ``solver(model, info, action_init=None) -> best_actions``.
    """

    def __init__(
        self,
        model: nn.Module,
        solver: Any,
        config: PlanConfig,
        process: Optional[dict[str, Callable]] = None,
        transform: Optional[dict[str, Callable]] = None,
    ) -> None:
        self.model = model
        self.solver = solver
        self.cfg = config
        self.process = process or {}
        self.transform = transform or {}
        self._last_actions: Optional[torch.Tensor] = None

    def _apply_transform(self, info: dict) -> dict:
        out = {}
        for k, v in info.items():
            if k in self.transform:
                out[k] = self.transform[k](v)
            else:
                out[k] = v
        return out

    def _apply_process(self, info: dict) -> dict:
        out = {}
        for k, v in info.items():
            if k in self.process:
                out[k] = self.process[k](v)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def __call__(self, info: dict) -> torch.Tensor:
        info = self._apply_transform(info)
        info = self._apply_process(info)
        init = self._last_actions
        actions = self.solver(self.model, info, action_init=init)
        # Cache last plan, shifted by one step (warm-start)
        self._last_actions = torch.roll(actions, shifts=-self.cfg.action_block, dims=0)
        return actions[: self.cfg.action_block]
