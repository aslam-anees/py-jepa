"""Mask sampling strategies for JEPA training.

Each strategy is a *collator*: a callable that takes a list of raw samples and
returns ``(batch, masks_enc, masks_pred)`` — drop it into ``DataLoader(collate_fn=...)``.

Available strategies:
    MultiBlock3DMaskCollator   — V-JEPA style spatiotemporal block masking
    MultiBlockMaskCollator     — I-JEPA style image block masking
    RandomTubeMaskCollator     — V-JEPA random spatiotemporal tubes
    RandomPatchMaskCollator    — simple per-patch Bernoulli mask (MAE-like)
"""

from .multiblock import MultiBlock3DMaskCollator, MultiBlockMaskCollator
from .random_patch import RandomPatchMaskCollator
from .tube import RandomTubeMaskCollator
from .utils import apply_masks

__all__ = [
    "MultiBlock3DMaskCollator",
    "MultiBlockMaskCollator",
    "RandomPatchMaskCollator",
    "RandomTubeMaskCollator",
    "apply_masks",
]
