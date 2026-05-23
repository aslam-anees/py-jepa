"""Simple per-patch Bernoulli mask — MAE-style. Useful as a baseline."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


class RandomPatchMaskCollator:
    """Bernoulli per-patch mask. Useful for ablations / MAE-style baselines.

    cfgs_mask: list of dicts each with a ``ratio`` key.
    """

    def __init__(
        self,
        cfgs_mask: Sequence[dict],
        crop_size: int | tuple[int, int] = 224,
        num_frames: int = 1,
        patch_size: int = 16,
        tubelet_size: int = 1,
    ) -> None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.height = crop_size[0] // patch_size
        self.width = crop_size[1] // patch_size
        self.duration = max(1, num_frames // tubelet_size)
        self.num_patches = self.duration * self.height * self.width
        self.ratios = [m.get("ratio", 0.75) for m in cfgs_mask]

    def step(self) -> None:
        return

    def __call__(self, batch: Iterable):
        batch_list = list(batch)
        batch_size = len(batch_list)
        collated = torch.utils.data.default_collate(batch_list)

        masks_enc, masks_pred = [], []
        for ratio in self.ratios:
            n_keep = max(1, int(self.num_patches * (1 - ratio)))
            n_pred = self.num_patches - n_keep
            e_list, p_list = [], []
            for _ in range(batch_size):
                idx = torch.randperm(self.num_patches)
                e_list.append(idx[:n_keep].sort().values)
                p_list.append(idx[n_keep:].sort().values)
            masks_enc.append(torch.stack(e_list))
            masks_pred.append(torch.stack(p_list))
        return collated, masks_enc, masks_pred
