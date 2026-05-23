"""EMA target encoder updates and momentum schedule.

The I-JEPA / V-JEPA recipe maintains a target network whose parameters are an
exponential moving average of the online encoder. The momentum ``m`` typically
warms from ``0.996`` to ``1.0`` over training so the target moves quickly at the
start (more signal) and stabilizes near the end (less noise).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .schedulers import LinearSchedule


class EMA:
    """In-place EMA update for a pair of online/target modules.

    Typical usage::

        ema = EMA(online_encoder, target_encoder, momentum_schedule)
        for step in training:
            train_step()
            ema.update()  # advances the momentum schedule and copies params
    """

    def __init__(
        self,
        online: nn.Module,
        target: nn.Module,
        momentum_schedule: Iterable[float] | None = None,
        constant_momentum: float = 0.996,
    ) -> None:
        self.online = online
        self.target = target
        self.schedule = iter(momentum_schedule) if momentum_schedule is not None else None
        self.constant = constant_momentum
        # Always disable grad on the target
        for p in self.target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self) -> float:
        m = next(self.schedule) if self.schedule is not None else self.constant
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.mul_(m).add_(p_o.detach().data, alpha=1.0 - m)
        return m


class MomentumScheduler(LinearSchedule):
    """Convenience alias — linear schedule clipped to ``[0, 1]``."""

    def step(self) -> float:
        m = super().step()
        return float(min(1.0, max(0.0, m)))
