"""Training engines.

We provide one engine per model family. Each engine takes a fully-built
model + loss + data loader and exposes a simple ``fit()`` entry point. Researchers
who want a custom loop can ignore engines entirely and call the underlying
``train_step`` functions directly.
"""

from .base import BaseTrainer, TrainConfig, TrainState
from .finetune import FinetuneConfig, LinearProbeTrainer, finetune_classifier
from .ijepa import IJEPAConfig, IJEPATrainer, train_ijepa
from .lewm import LeWMConfig, LeWMTrainer, train_lewm
from .vjepa import VJEPAConfig, VJEPATrainer, train_vjepa

__all__ = [
    "BaseTrainer",
    "FinetuneConfig",
    "IJEPAConfig",
    "IJEPATrainer",
    "LeWMConfig",
    "LeWMTrainer",
    "LinearProbeTrainer",
    "TrainConfig",
    "TrainState",
    "VJEPAConfig",
    "VJEPATrainer",
    "finetune_classifier",
    "train_ijepa",
    "train_lewm",
    "train_vjepa",
]
