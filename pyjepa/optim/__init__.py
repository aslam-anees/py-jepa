"""Optimization utilities — schedulers, EMA target encoder, optimizer factories."""

from .ema import EMA, MomentumScheduler
from .factory import build_optimizer, build_param_groups
from .schedulers import CosineWDSchedule, LinearSchedule, WarmupCosineSchedule

__all__ = [
    "CosineWDSchedule",
    "EMA",
    "LinearSchedule",
    "MomentumScheduler",
    "WarmupCosineSchedule",
    "build_optimizer",
    "build_param_groups",
]
