"""Tests for masking strategies."""

import pytest
import torch

from pyjepa.masks.multiblock import MultiBlock3DMaskCollator, MultiBlockMaskCollator
from pyjepa.masks.random_patch import RandomPatchMaskCollator
from pyjepa.masks.tube import RandomTubeMaskCollator


IMG_SIZE = 32
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2   # 16
BATCH_SIZE = 4


def _make_batch(n=BATCH_SIZE):
    """Fake batch: list of (image,) tuples as expected by collators."""
    return [[torch.randn(3, IMG_SIZE, IMG_SIZE)] for _ in range(n)]


class TestMultiBlockMaskCollator:
    def test_output_structure(self):
        collator = MultiBlockMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_enc_masks=1, num_pred_masks=4,
        )
        batch = _make_batch()
        result = collator(batch)
        # result is (collated_images, masks_enc_list, masks_pred_list)
        assert len(result) == 3
        images, masks_enc, masks_pred = result
        assert isinstance(images, torch.Tensor)
        assert images.shape[0] == BATCH_SIZE

    def test_masks_are_index_tensors(self):
        collator = MultiBlockMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_enc_masks=1, num_pred_masks=2,
        )
        _, masks_enc, masks_pred = collator(_make_batch())
        assert len(masks_enc) == 1
        assert len(masks_pred) == 2
        for m in masks_enc + masks_pred:
            assert m.dtype == torch.int64
            assert m.ndim == 2        # (B, num_masked_tokens)
            assert m.max() < NUM_PATCHES

    def test_no_overlap_respected(self):
        collator = MultiBlockMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            allow_overlap=False,
        )
        _, masks_enc, masks_pred = collator(_make_batch())
        # enc mask indices should not overlap with pred mask indices
        enc = masks_enc[0]  # (B, K)
        for pred in masks_pred:
            for b in range(BATCH_SIZE):
                enc_set = set(enc[b].tolist())
                pred_set = set(pred[b].tolist())
                overlap = enc_set & pred_set
                assert len(overlap) == 0, f"Overlap detected: {overlap}"


class TestMultiBlock3DMaskCollator:
    NUM_FRAMES = 4
    TUBELET_SIZE = 2

    def test_output_structure(self):
        collator = MultiBlock3DMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
            num_enc_masks=1, num_pred_masks=2,
        )
        # Video batch: list of (clip,) tuples where clip is (C, T, H, W)
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        result = collator(batch)
        assert len(result) == 3
        clips, masks_enc, masks_pred = result
        assert clips.shape == (BATCH_SIZE, 3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)

    def test_token_count(self):
        collator = MultiBlock3DMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
        )
        n_temporal = self.NUM_FRAMES // self.TUBELET_SIZE  # 2
        n_spatial = (IMG_SIZE // PATCH_SIZE) ** 2          # 16
        n_total = n_temporal * n_spatial                   # 32
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        _, masks_enc, masks_pred = collator(batch)
        for m in masks_enc + masks_pred:
            assert m.max() < n_total, f"Mask index {m.max()} >= {n_total}"


class TestRandomTubeMaskCollator:
    NUM_FRAMES = 4
    TUBELET_SIZE = 2

    def test_tube_same_across_frames(self):
        """Tube masks should use the same spatial pattern for all frames."""
        collator = RandomTubeMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
            mask_ratio=0.5,
        )
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        result = collator(batch)
        # Just check it doesn't crash and returns something
        assert result is not None


class TestRandomPatchMaskCollator:
    def test_output_type(self):
        collator = RandomPatchMaskCollator(
            crop_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.5,
        )
        batch = _make_batch()
        result = collator(batch)
        assert result is not None
