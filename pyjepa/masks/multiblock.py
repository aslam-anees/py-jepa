"""Multi-block masking — the V-JEPA / I-JEPA "predict large blocks" strategy.

Each sample's mask is the union of ``npred`` randomly placed rectangular blocks.
The context mask is the *complement* of the union; the predictor mask is the
union itself. Sampling shares the same block *shape* across the batch (set at
each iteration via :meth:`step`) which makes batched gather/scatter shape-stable.

3D (video) and 2D (image) variants share most of the code path.
"""

from __future__ import annotations

import math
from multiprocessing import Value
from typing import Iterable, Sequence

import torch


class MultiBlock3DMaskCollator:
    """V-JEPA style mask collator. Returns ``(batch, masks_enc, masks_pred)``.

    Args:
        cfgs_mask: list of dicts with per-mask config:
            ``spatial_scale``: (min, max) of the predict block area fraction
            ``temporal_scale``: (min, max) of temporal block length fraction
            ``aspect_ratio``: (min, max) of spatial block aspect ratio
            ``num_blocks``: how many blocks to union per sample
            ``max_temporal_keep`` (optional): cap context to first X*duration frames
            ``max_keep`` (optional): cap kept context tokens
        crop_size: spatial size of input crops in pixels.
        num_frames: temporal length in frames.
        patch_size: spatial patch size.
        tubelet_size: temporal patch size.
    """

    def __init__(
        self,
        cfgs_mask: Sequence[dict],
        crop_size: int | tuple[int, int] = 224,
        num_frames: int = 16,
        patch_size: int | tuple[int, int] = 16,
        tubelet_size: int = 2,
    ) -> None:
        self.mask_generators = [
            _Block3DGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size if isinstance(patch_size, int) else patch_size[0],
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get("spatial_scale", (0.15, 0.15)),
                temporal_pred_mask_scale=m.get("temporal_scale", (1.0, 1.0)),
                aspect_ratio=m.get("aspect_ratio", (0.75, 1.5)),
                npred=m.get("num_blocks", 1),
                max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                max_keep=m.get("max_keep"),
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


class MultiBlockMaskCollator(MultiBlock3DMaskCollator):
    """2D image variant — just calls the 3D collator with ``num_frames=1``."""

    def __init__(
        self,
        cfgs_mask: Sequence[dict],
        crop_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__(
            cfgs_mask=cfgs_mask,
            crop_size=crop_size,
            num_frames=1,
            patch_size=patch_size,
            tubelet_size=1,
        )


class _Block3DGenerator:
    """Single-mask generator. Sampling is *block-size shared* across a batch step
    (via :meth:`step`) so that all samples in one iteration use the same
    rectangle size and the gather/scatter kernels stay shape-stable."""

    def __init__(
        self,
        *,
        crop_size: int | tuple[int, int],
        num_frames: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        spatial_pred_mask_scale: tuple[float, float],
        temporal_pred_mask_scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
        npred: int,
        max_context_frames_ratio: float,
        max_keep: int | None,
    ) -> None:
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.height = crop_size[0] // spatial_patch_size
        self.width = crop_size[1] // spatial_patch_size
        self.duration = max(1, num_frames // temporal_patch_size)
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        # Shared across worker processes so all workers sample the same block size
        self._itr_counter = Value("i", -1)

    def step(self) -> int:
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def _sample_block_size(self, generator: torch.Generator) -> tuple[int, int, int]:
        min_t, max_t = self.temporal_pred_mask_scale
        t = max(1, int(self.duration * (min_t + torch.rand(1, generator=generator).item() * (max_t - min_t))))

        min_s, max_s = self.spatial_pred_mask_scale
        s = min_s + torch.rand(1, generator=generator).item() * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * s)

        min_ar, max_ar = self.aspect_ratio
        ar = min_ar + torch.rand(1, generator=generator).item() * (max_ar - min_ar)

        h = int(round(math.sqrt(spatial_num_keep * ar)))
        w = int(round(math.sqrt(spatial_num_keep / ar)))
        h = min(h, self.height)
        w = min(w, self.width)
        return t, h, w

    def _sample_block_mask(self, b_size: tuple[int, int, int]) -> torch.Tensor:
        t, h, w = b_size
        top = torch.randint(0, max(1, self.height - h + 1), (1,)).item()
        left = torch.randint(0, max(1, self.width - w + 1), (1,)).item()
        start = torch.randint(0, max(1, self.duration - t + 1), (1,)).item()

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0
        return mask

    def __call__(self, batch_size: int):
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(g)

        total = self.duration * self.height * self.width
        min_keep_enc = min_keep_pred = total

        masks_pred_list, masks_enc_list = [], []
        for _ in range(batch_size):
            for _try in range(50):
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()
                mask_p = torch.argwhere(mask_e == 0).squeeze(-1)
                mask_e = torch.nonzero(mask_e).squeeze(-1)
                if mask_e.numel() > 0:
                    break
            min_keep_pred = min(min_keep_pred, mask_p.numel())
            min_keep_enc = min(min_keep_enc, mask_e.numel())
            masks_pred_list.append(mask_p)
            masks_enc_list.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        masks_pred_list = [m[:min_keep_pred] for m in masks_pred_list]
        masks_enc_list = [m[:min_keep_enc] for m in masks_enc_list]
        return (
            torch.utils.data.default_collate(masks_enc_list),
            torch.utils.data.default_collate(masks_pred_list),
        )
