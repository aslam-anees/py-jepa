"""Step-level LR / WD schedulers used by JEPA training.

These are *not* ``torch.optim.lr_scheduler`` subclasses — we step them once per
training iteration (not per epoch) and they return the new LR/WD on call,
which is more ergonomic for the JEPA training loop.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class WarmupCosineSchedule:
    """Linear warmup → cosine decay from ``ref_lr`` to ``final_lr``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        start_lr: float,
        ref_lr: float,
        T_max: int,
        final_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = max(1, warmup_steps)
        self.T_max = max(1, T_max - warmup_steps)
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.warmup_steps:
            progress = self._step / self.warmup_steps
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self._step - self.warmup_steps) / self.T_max
            new_lr = self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
            new_lr = max(self.final_lr, new_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd.get("step", 0)


class CosineWDSchedule:
    """Cosine schedule for weight decay (matches the I-JEPA / V-JEPA recipe)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        ref_wd: float,
        T_max: int,
        final_wd: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = max(1, T_max)
        self._step = 0

    def step(self) -> float:
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)
        for group in self.optimizer.param_groups:
            if not group.get("WD_exclude", False):
                group["weight_decay"] = new_wd
        return new_wd

    def state_dict(self) -> dict:
        return {"step": self._step}

    def load_state_dict(self, sd: dict) -> None:
        self._step = sd.get("step", 0)


class LinearSchedule:
    """Generic linear schedule from ``start`` to ``end`` over ``T`` steps.

    Use it for any monotone parameter — EMA momentum, dropout, anneal coefficients."""

    def __init__(self, start: float, end: float, total_steps: int) -> None:
        self.start = start
        self.end = end
        self.total = max(1, total_steps)
        self._step = 0

    def step(self) -> float:
        progress = min(1.0, self._step / self.total)
        self._step += 1
        return self.start + progress * (self.end - self.start)

    def __iter__(self):
        return self

    def __next__(self) -> float:
        return self.step()
