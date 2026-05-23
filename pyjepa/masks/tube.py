"""Random tube masking — same spatial mask applied across all frames in a clip.

Used by V-JEPA when ``mask_type='random_tube'``. The "tube" name comes from the
fact that every frame within a clip masks out the same spatial patches, so the
masked region forms a temporal tube through the video.
"""

from __future__ import annotations

from multiprocessing import Value
from typing import Iterable, Sequence

import numpy as np
import torch


class RandomTubeMaskCollator:
    """Returns ``(batch, masks_enc, masks_pred)`` with random tube masking.

    cfgs_mask: list of dicts each with a ``ratio`` key (fraction to mask out).
    """

    def __init__(
        self,
        cfgs_mask: Sequence[dict],
        crop_size: int | tuple[int, int] = 224,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
    ) -> None:
        self.mask_generators = [
            _TubeGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size if isinstance(patch_size, int) else patch_size[0],
                temporal_patch_size=tubelet_size,
                ratio=m.get("ratio", 0.9),
            )
            for m in cfgs_mask
        ]

    def step(self) -> None:
        for g in self.mask_generators:
            g.step()

    def __call__(self, batch: Iterable):
        batch_list = list(batch)
        batch_size = len(batch_list)
        collated = torch.utils.data.default_collate(batch_list)
        masks_enc, masks_pred = [], []
        for g in self.mask_generators:
            e, p = g(batch_size)
            masks_enc.append(e)
            masks_pred.append(p)
        return collated, masks_enc, masks_pred


class _TubeGenerator:
    def __init__(
        self,
        *,
        crop_size: int | tuple[int, int],
        num_frames: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        ratio: float,
    ) -> None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.height = crop_size[0] // spatial_patch_size
        self.width = crop_size[1] // spatial_patch_size
        self.duration = max(1, num_frames // temporal_patch_size)
        self.ratio = ratio
        self.num_patches_spatial = self.height * self.width
        self.num_keep_spatial = int(self.num_patches_spatial * (1.0 - self.ratio))
        self.num_keep = self.num_keep_spatial * self.duration
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def __call__(self, batch_size: int):
        masks_enc, masks_pred = [], []
        for _ in range(batch_size):
            mask = np.concatenate(
                [
                    np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                    np.ones(self.num_keep_spatial),
                ]
            )
            np.random.shuffle(mask)
            mask = torch.from_numpy(np.tile(mask, (self.duration, 1))).flatten()
            mask_p = torch.argwhere(mask == 0).squeeze(-1)
            mask_e = torch.nonzero(mask).squeeze(-1)
            masks_enc.append(mask_e)
            masks_pred.append(mask_p)
        return (
            torch.utils.data.default_collate(masks_enc),
            torch.utils.data.default_collate(masks_pred),
        )
