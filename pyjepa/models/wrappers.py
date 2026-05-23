"""Multi-mask wrappers — run the encoder once per context mask and stack outputs.

These exist because the reference V-JEPA / I-JEPA code expects to sample
multiple context masks per image and re-run the encoder on each. We isolate that
behavior so the underlying ViT/predictor stay mask-agnostic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiMaskWrapper(nn.Module):
    """Run ``backbone`` once per (encoder) mask and concat outputs along batch."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def embed_dim(self) -> int:
        return self.backbone.embed_dim

    @property
    def num_heads(self) -> int:
        return self.backbone.num_heads

    def forward(self, x: torch.Tensor, masks: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        if masks is None:
            return self.backbone(x)
        if not isinstance(masks, list):
            masks = [masks]
        outs = [self.backbone(x, masks=[m]) for m in masks]
        return torch.cat(outs, dim=0)


class PredictorMultiMaskWrapper(nn.Module):
    """Same idea but for the predictor — runs once per (ctxt, tgt) pair."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        ctxt: torch.Tensor,
        tgt: torch.Tensor,
        masks_ctxt,
        masks_tgt,
    ) -> list[torch.Tensor]:
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        outs: list[torch.Tensor] = []
        N = len(ctxt) // len(masks_ctxt)
        for i, (mc, mt) in enumerate(zip(masks_ctxt, masks_tgt)):
            # slice the slab corresponding to this context-mask variant
            ci = ctxt[i * N : (i + 1) * N]
            outs.append(self.backbone(ci, tgt, [mc], [mt], mask_index=i))
        return outs
