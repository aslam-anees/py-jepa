"""I-JEPA — Image Joint Embedding Predictive Architecture.

This is the top-level model that bundles encoder, target encoder (EMA), and
predictor. The standard I-JEPA training loop calls:

    h = target_encoder(x)              # target rep (no_grad, normalized, masked)
    z = encoder(x, ctxt_masks)         # context rep
    z = predictor(z, h, ctxt_masks, tgt_masks)
    loss = ||z - h||

We expose ``forward(x, masks_enc, masks_pred)`` that returns ``(z, h)``.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.tensors import apply_masks


class IJEPA(nn.Module):
    """End-to-end I-JEPA model: encoder + target-encoder + predictor."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        target_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder if target_encoder is not None else copy.deepcopy(encoder)
        # The target encoder is updated via EMA, never via grad.
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """In-place EMA update of target encoder params: m*target + (1-m)*online."""
        for p_o, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_o.detach().data, alpha=1.0 - momentum)

    def encode_target(self, x: torch.Tensor, masks_pred) -> list[torch.Tensor]:
        """No-grad target encoding + layer-norm + apply target masks."""
        with torch.no_grad():
            h = self.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))
            return apply_masks(h, masks_pred, concat=False)

    def encode_context(self, x: torch.Tensor, masks_enc) -> torch.Tensor:
        return self.encoder(x, masks=masks_enc)

    def forward(
        self,
        x: torch.Tensor,
        masks_enc,
        masks_pred,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns ``(predictions, targets)``, each a list of ``[B, K, D]``."""
        h = self.encode_target(x, masks_pred)
        z = self.encode_context(x, masks_enc)
        z = self.predictor(z, h, masks_enc, masks_pred)
        return z, h

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper — encode without masks, return token features ``[B, N, D]``."""
        return self.encoder(x)
