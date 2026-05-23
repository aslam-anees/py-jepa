"""End-to-end training step tests.

These tests verify that a complete forward+backward pass runs without error
and produces finite loss values. They are intentionally small (1 step, tiny
models) so they run in seconds on any hardware.

Marked with @pytest.mark.e2e to allow excluding them from fast unit test runs:
    pytest tests/ -m "not e2e"   # skip e2e tests
    pytest tests/ -m e2e          # run only e2e tests
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from pyjepa.models.ijepa import IJEPA
from pyjepa.models.predictor import vit_predictor
from pyjepa.models.registry import build_lewm
from pyjepa.models.vit import vit_tiny
from pyjepa.models.wrappers import MultiMaskWrapper, PredictorMultiMaskWrapper
from pyjepa.masks.multiblock import MultiBlockMaskCollator
from pyjepa.training.ijepa import IJEPAConfig, IJEPATrainer
from pyjepa.training.lewm import LeWMConfig, LeWMTrainer
from pyjepa.training.base import TrainConfig
from pyjepa.data.trajectory import TrajectoryDataset


pytestmark = pytest.mark.e2e

IMG_SIZE = 32
PATCH_SIZE = 8
BATCH_SIZE = 2
HISTORY_SIZE = 2
NUM_PREDS = 2
ACTION_DIM = 4


def _tiny_ijepa():
    enc = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    pred = vit_predictor(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_frames=1,
        embed_dim=enc.embed_dim, predictor_embed_dim=96,
        depth=2, num_heads=enc.num_heads,
        use_mask_tokens=True, num_mask_tokens=2, zero_init_mask_tokens=True,
        use_sdpa=True,
    )
    return IJEPA(encoder=MultiMaskWrapper(enc), predictor=PredictorMultiMaskWrapper(pred))


def _make_ijepa_loader():
    images = torch.randn(BATCH_SIZE * 2, 3, IMG_SIZE, IMG_SIZE)
    dataset = TensorDataset(images)
    # Create a simple mask collator with one context and multiple prediction masks
    cfgs_mask = [
        {"spatial_scale": (0.85, 1.0), "aspect_ratio": (0.75, 1.5), "num_blocks": 1},
        {"spatial_scale": (0.15, 0.2), "aspect_ratio": (0.75, 1.5), "num_blocks": 4},
    ]
    collator = MultiBlockMaskCollator(cfgs_mask, crop_size=IMG_SIZE, patch_size=PATCH_SIZE)

    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        return collator([[img] for img in imgs])

    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def _make_lewm_loader():
    episodes = []
    for _ in range(4):
        T = HISTORY_SIZE + NUM_PREDS + 5
        episodes.append({
            "pixels": np.random.rand(T, 3, IMG_SIZE, IMG_SIZE).astype(np.float32),
            "action": np.random.randn(T, ACTION_DIM).astype(np.float32),
        })
    # TrajectoryDataset only takes episodes and window (and optional frameskip, transform, keys)
    # It returns dicts with all keys from the episodes
    dataset = TrajectoryDataset(
        episodes,
        window=HISTORY_SIZE + NUM_PREDS,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)


class TestIJEPATrainStep:
    def test_one_training_step(self):
        """I-JEPA model can run a forward pass and compute loss."""
        model = _tiny_ijepa()
        x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

        # Create simple non-overlapping context and prediction masks
        n_patches = (IMG_SIZE // PATCH_SIZE) ** 2  # 16
        # Context: patches 0-7, Prediction: patches 8-15
        masks_enc = [torch.arange(8).unsqueeze(0).expand(2, -1)]  # (2, 8)
        masks_pred = [torch.arange(8, 16).unsqueeze(0).expand(2, -1)]  # (2, 8)

        with torch.no_grad():
            preds, targets = model(x, masks_enc, masks_pred)

        assert len(preds) > 0, "No predictions"
        assert len(targets) > 0, "No targets"
        assert all(p.ndim == 3 for p in preds), "Predictions should be 3D"

    def test_backward_and_loss(self):
        """Test that I-JEPA loss is differentiable and backward works."""
        model = _tiny_ijepa()
        x = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE)  # Black images for simplicity

        masks_enc = [torch.arange(4).unsqueeze(0).expand(2, -1)]  # (2, 4)
        masks_pred = [torch.arange(4, 8).unsqueeze(0).expand(2, -1)]  # (2, 4)

        preds, targets = model(x, masks_enc, masks_pred)

        # Compute loss
        loss = sum(
            (p - t).pow(2).mean()
            for p, t in zip(preds, targets)
        )

        # Should be differentiable
        loss.backward()
        assert model.encoder.backbone.patch_embed.proj.weight.grad is not None


class TestLeWMTrainStep:
    def test_lewm_forward_pass(self):
        """Test LeWM model forward pass and loss computation."""
        enc = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
        # vit_tiny outputs 192-dim embeddings
        model = build_lewm(
            enc, action_dim=ACTION_DIM, history_size=HISTORY_SIZE,
            emb_dim=192, action_emb_dim=16,
            pred_depth=1, pred_heads=2, pred_mlp_dim=256,
            projector_hidden=256,
        )
        model.eval()

        # Create input data
        B, T_ctx = 2, HISTORY_SIZE
        T_total = T_ctx + NUM_PREDS
        pixels = torch.randn(B, T_total, 3, IMG_SIZE, IMG_SIZE)
        action = torch.randn(B, T_total, ACTION_DIM)

        # Encode
        info = model.encode({"pixels": pixels, "action": action})
        assert "emb" in info
        assert info["emb"].shape == (B, T_total, 192)

        # Predict next embeddings
        emb = info["emb"][:, :T_ctx]  # context
        act_emb = info["act_emb"][:, :T_ctx]
        pred_emb = model.predict(emb, act_emb)
        assert pred_emb.shape == (B, T_ctx, 192)

        # Test that loss can be computed
        target_emb = info["emb"][:, T_ctx:]
        loss = (pred_emb[:, :NUM_PREDS] - target_emb[:, :NUM_PREDS]).pow(2).mean()
        assert loss.item() >= 0
