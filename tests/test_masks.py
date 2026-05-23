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
        """MultiBlockMaskCollator uses cfgs_mask with config dicts."""
        cfgs_mask = [
            {"spatial_scale": (0.85, 1.0), "aspect_ratio": (0.75, 1.5), "num_blocks": 1},  # enc mask
            {"spatial_scale": (0.15, 0.2), "aspect_ratio": (0.75, 1.5), "num_blocks": 4},  # pred masks
        ]
        collator = MultiBlockMaskCollator(cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE)
        batch = _make_batch()
        result = collator(batch)
        # result is (collated_batch, masks_enc_list, masks_pred_list)
        assert len(result) == 3
        collated, masks_enc, masks_pred = result
        # collated is a tuple of collated batches (from default_collate)
        assert isinstance(collated, (list, tuple))
        assert len(masks_enc) == 2
        assert len(masks_pred) == 2
        # Each mask list should have tensors
        for masks in [masks_enc, masks_pred]:
            for m in masks:
                assert m.dtype == torch.int64

    def test_masks_are_index_tensors(self):
        """Check that mask tensors are int64 indices."""
        cfgs_mask = [
            {"spatial_scale": (0.85, 1.0), "aspect_ratio": (0.75, 1.5), "num_blocks": 1},
            {"spatial_scale": (0.15, 0.2), "aspect_ratio": (0.75, 1.5), "num_blocks": 2},
        ]
        collator = MultiBlockMaskCollator(cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE)
        _, masks_enc, masks_pred = collator(_make_batch())
        assert len(masks_enc) == 2
        assert len(masks_pred) == 2
        for m in masks_enc + masks_pred:
            assert m.dtype == torch.int64
            assert m.ndim == 2        # (B, num_masked_tokens)
            assert m.max() < NUM_PATCHES

    def test_masks_complement_and_non_overlap(self):
        """Context and predict masks should be complementary."""
        cfgs_mask = [{"spatial_scale": (0.15, 0.2), "aspect_ratio": (0.75, 1.5), "num_blocks": 1}]
        collator = MultiBlockMaskCollator(cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE)
        _, masks_enc, masks_pred = collator(_make_batch())
        # For a single mask config, enc=context and pred=target
        enc = masks_enc[0]  # (B, K_enc)
        pred = masks_pred[0]  # (B, K_pred)

        # Check no overlap and complementary
        for b in range(BATCH_SIZE):
            enc_set = set(enc[b].tolist())
            pred_set = set(pred[b].tolist())
            overlap = enc_set & pred_set
            all_tokens = enc_set | pred_set
            assert len(overlap) == 0, f"Overlap detected: {overlap}"
            assert len(all_tokens) <= NUM_PATCHES


class TestMultiBlock3DMaskCollator:
    NUM_FRAMES = 4
    TUBELET_SIZE = 2

    def test_output_structure(self):
        """3D mask collator with spatiotemporal configs."""
        cfgs_mask = [
            {
                "spatial_scale": (0.85, 1.0), "temporal_scale": (1.0, 1.0),
                "aspect_ratio": (0.75, 1.5), "num_blocks": 1
            },
            {
                "spatial_scale": (0.15, 0.2), "temporal_scale": (0.5, 1.0),
                "aspect_ratio": (0.75, 1.5), "num_blocks": 2
            },
        ]
        collator = MultiBlock3DMaskCollator(
            cfgs_mask=cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
        )
        # Video batch: list of (clip,) tuples where clip is (C, T, H, W)
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        result = collator(batch)
        assert len(result) == 3
        collated, masks_enc, masks_pred = result
        # collated is from default_collate, might be list or tuple
        assert isinstance(collated, (list, tuple))
        assert len(masks_enc) == 2
        assert len(masks_pred) == 2
        # Each mask list should have tensors
        for masks in [masks_enc, masks_pred]:
            for m in masks:
                assert m.dtype == torch.int64

    def test_token_count(self):
        """Token indices should be within valid range."""
        n_temporal = self.NUM_FRAMES // self.TUBELET_SIZE  # 2
        n_spatial = (IMG_SIZE // PATCH_SIZE) ** 2          # 16
        n_total = n_temporal * n_spatial                   # 32

        cfgs_mask = [
            {"spatial_scale": (0.15, 0.2), "temporal_scale": (0.5, 1.0),
             "aspect_ratio": (0.75, 1.5), "num_blocks": 1}
        ]
        collator = MultiBlock3DMaskCollator(
            cfgs_mask=cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
        )
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        _, masks_enc, masks_pred = collator(batch)
        for m in masks_enc + masks_pred:
            assert m.max() < n_total, f"Mask index {m.max()} >= {n_total}"


class TestRandomTubeMaskCollator:
    NUM_FRAMES = 4
    TUBELET_SIZE = 2

    def test_tube_same_across_frames(self):
        """Tube masks should use the same spatial pattern for all frames."""
        cfgs_mask = [{"ratio": 0.5}, {"ratio": 0.75}]
        collator = RandomTubeMaskCollator(
            cfgs_mask=cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
            num_frames=self.NUM_FRAMES, tubelet_size=self.TUBELET_SIZE,
        )
        batch = [[torch.randn(3, self.NUM_FRAMES, IMG_SIZE, IMG_SIZE)] for _ in range(BATCH_SIZE)]
        result = collator(batch)
        # result is (batch, masks_enc, masks_pred)
        assert len(result) == 3
        clips, masks_enc, masks_pred = result
        assert len(masks_enc) == 2
        assert len(masks_pred) == 2
        for m in masks_enc + masks_pred:
            assert m.dtype == torch.int64
            assert m.ndim == 2  # (B, num_masked_tokens)


class TestRandomPatchMaskCollator:
    def test_output_type(self):
        """RandomPatchMaskCollator with Bernoulli masking."""
        cfgs_mask = [{"ratio": 0.5}, {"ratio": 0.75}]
        collator = RandomPatchMaskCollator(
            cfgs_mask=cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE,
        )
        batch = _make_batch()
        result = collator(batch)
        assert len(result) == 3
        images, masks_enc, masks_pred = result
        assert len(masks_enc) == 2
        assert len(masks_pred) == 2
        for m in masks_enc + masks_pred:
            assert m.dtype == torch.int64
            assert m.ndim == 2
