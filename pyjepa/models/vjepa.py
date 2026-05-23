"""V-JEPA — Video Joint Embedding Predictive Architecture.

Same overall structure as I-JEPA but the encoder is a video ViT (3D patches) and
the mask collator usually produces multiple context masks per clip. We reuse
:class:`IJEPA` here because the math is identical — the only difference is the
underlying patch embedding and pos-embed.
"""

from __future__ import annotations

from .ijepa import IJEPA


class VJEPA(IJEPA):
    """Alias of IJEPA for clarity in user code. The encoder must be a video ViT."""

    pass
