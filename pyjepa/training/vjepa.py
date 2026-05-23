"""V-JEPA training engine.

The training loop is mathematically identical to I-JEPA — the only differences
are the data shape (5D video tensor) and the mask collator (random tube /
multi-block 3D). So we just re-use :class:`IJEPATrainer` and re-export with a
clearer name.
"""

from __future__ import annotations

from .ijepa import IJEPAConfig as VJEPAConfig
from .ijepa import IJEPATrainer as VJEPATrainer
from .ijepa import train_ijepa as train_vjepa

__all__ = ["VJEPAConfig", "VJEPATrainer", "train_vjepa"]
