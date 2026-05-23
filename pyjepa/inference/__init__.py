"""Inference helpers — encode, rollout, and physical-surprise scoring.

These wrap raw model calls in friendlier APIs for downstream use:
- :class:`Encoder` — turn a JEPA encoder into a ``features = encoder(images)`` callable.
- :class:`Rollout` — autoregressive latent rollout over a learned world model.
- :func:`surprise_score` — per-step "implausibility" score for an observed trajectory.
"""

from .encoder import Encoder, embed_image, embed_video
from .rollout import Rollout, rollout_latent
from .surprise import surprise_score

__all__ = [
    "Encoder",
    "Rollout",
    "embed_image",
    "embed_video",
    "rollout_latent",
    "surprise_score",
]
