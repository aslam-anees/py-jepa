"""Tests for model forward passes, EMA update, and LeWM encode/predict/rollout."""

import pytest
import torch

from pyjepa.models.ijepa import IJEPA
from pyjepa.models.lewm import ARPredictor
from pyjepa.models.predictor import vit_predictor
from pyjepa.models.registry import build_lewm
from pyjepa.models.vit import vit_tiny
from pyjepa.models.wrappers import MultiMaskWrapper, PredictorMultiMaskWrapper


# ---- Helpers ----

def _tiny_ijepa(img_size=32, patch_size=8):
    enc = vit_tiny(img_size=img_size, patch_size=patch_size)
    pred = vit_predictor(
        img_size=img_size, patch_size=patch_size, num_frames=1,
        embed_dim=enc.embed_dim, predictor_embed_dim=96,
        depth=2, num_heads=enc.num_heads,
        use_mask_tokens=True, num_mask_tokens=2, zero_init_mask_tokens=True,
        use_sdpa=True,
    )
    return IJEPA(encoder=MultiMaskWrapper(enc), predictor=PredictorMultiMaskWrapper(pred))


def _tiny_lewm(history_size=2):
    enc = vit_tiny(img_size=32, patch_size=8)  # outputs 192-dim embeddings
    return build_lewm(
        enc, action_dim=4, history_size=history_size,
        emb_dim=192,  # match encoder output (vit_tiny.embed_dim = 192)
        action_emb_dim=16,
        pred_depth=1, pred_heads=2, pred_mlp_dim=256,  # scale with emb_dim
        projector_hidden=256,
    )


# ---- ViT forward ----

class TestVisionTransformer:
    def test_image_forward(self, tiny_vit, small_batch):
        out = tiny_vit(small_batch)
        # ViT returns (B, N_patches, D) not (B, D)
        assert out.ndim == 3
        assert out.shape[0] == 2  # batch
        assert out.shape[-1] == tiny_vit.embed_dim  # feature dim

    def test_video_forward(self, tiny_video_vit, small_video_batch):
        # (B, C, T, H, W) -> (B, N_patches, D)
        out = tiny_video_vit(small_video_batch)
        assert out.ndim == 3
        assert out.shape[0] == 2
        assert out.shape[-1] == tiny_video_vit.embed_dim

    def test_embed_dim_attribute(self, tiny_vit):
        assert hasattr(tiny_vit, "embed_dim")
        assert hasattr(tiny_vit, "num_heads")

    def test_single_image_interpolate_pos(self, tiny_vit):
        """Larger image should work via pos-embedding interpolation."""
        img = torch.randn(1, 3, 48, 48)  # different from training size (32)
        out = tiny_vit(img, interpolate_pos_encoding=True)
        assert out.shape[-1] == tiny_vit.embed_dim


# ---- I-JEPA ----

class TestIJEPA:
    def test_forward_returns_preds_and_targets(self):
        """IJEPA forward pass with masked context and targets."""
        model = _tiny_ijepa()
        x = torch.randn(2, 3, 32, 32)

        # Build simple masks: each mask is a flat list of token indices
        # The encoder runs once per context mask (1 in this case)
        # The predictor runs once per (context, target) pair
        n_patches = (32 // 8) ** 2  # 16 tokens
        masks_enc = [torch.arange(8).unsqueeze(0).expand(2, -1)]  # 1 enc mask
        masks_pred = [torch.arange(8, 16).unsqueeze(0).expand(2, -1)]  # 1 pred mask (complementary to enc)

        preds, targets = model(x, masks_enc, masks_pred)
        # With MultiMaskWrapper, preds is a list of predictions per (enc, pred) pair
        assert isinstance(preds, list)
        assert len(preds) >= 1  # at least 1 prediction per context mask
        assert isinstance(targets, list)
        for p, t in zip(preds, targets):
            assert p.ndim == 3, f"pred shape {p.shape} should be (B, K, D)"
            assert t.ndim == 3, f"target shape {t.shape} should be (B, K, D)"

    def test_ema_update(self):
        model = _tiny_ijepa()
        # Just verify EMA update doesn't crash
        model.update_target(momentum=0.99)
        # If we get here, EMA update succeeded
        assert True

    def test_embed(self, small_batch):
        model = _tiny_ijepa()
        emb = model.embed(small_batch)
        # embed() should return patch embeddings (B, N, D)
        assert emb.ndim == 3
        assert emb.shape[-1] == model.encoder.backbone.embed_dim


# ---- LeWM ----

class TestLeWM:
    def test_encode_spatiotemporal(self):
        model = _tiny_lewm(history_size=2)
        pixels = torch.randn(2, 3, 3, 32, 32)   # (B, T, C, H, W)
        action = torch.randn(2, 3, 4)
        info = model.encode({"pixels": pixels, "action": action})
        # emb_dim is 192 (matching vit_tiny.embed_dim)
        assert info["emb"].shape == (2, 3, 192)   # (B, T, emb_dim)
        assert info["act_emb"].shape == (2, 3, 16)

    def test_encode_single_frame_auto_promote(self):
        """LeWM.encode() should auto-promote (B, C, H, W) to (B, 1, C, H, W)."""
        model = _tiny_lewm(history_size=2)
        pixels = torch.randn(2, 3, 32, 32)   # single frame, no T dim
        info = model.encode({"pixels": pixels})
        # emb_dim is 192
        assert info["emb"].shape == (2, 1, 192)

    def test_predict(self):
        model = _tiny_lewm(history_size=2)
        emb = torch.randn(2, 2, 192)  # emb_dim=192
        act_emb = torch.randn(2, 2, 16)
        pred = model.predict(emb, act_emb)
        assert pred.shape == (2, 2, 192)

    def test_rollout_shape(self):
        model = _tiny_lewm(history_size=2)
        model.eval()
        T_ctx = 2
        T_total = 4
        B, S = 1, 2
        pixels = torch.randn(B, S, T_ctx, 3, 32, 32)
        action_seq = torch.randn(B, S, T_total, 4)

        info = {"pixels": pixels}
        with torch.no_grad():
            info = model.rollout(info, action_seq, history_size=2)

        assert "predicted_emb" in info
        assert info["predicted_emb"].shape[0] == B
        assert info["predicted_emb"].shape[1] == S

    def test_build_lewm_history_size_stored(self):
        model = _tiny_lewm(history_size=2)
        assert model.history_size == 2


# ---- ARPredictor pos embedding interpolation ----

class TestARPredictorPosEmbedInterpolation:
    """Regression test: ARPredictor must not crash when T > num_frames."""

    def test_pos_embed_exact_length(self):
        pred = ARPredictor(
            num_frames=3, depth=1, heads=2, mlp_dim=64,
            input_dim=64, hidden_dim=64, cond_dim=16,
        )
        x = torch.randn(2, 3, 64)
        c = torch.randn(2, 3, 16)
        out = pred(x, c)
        assert out.shape == (2, 3, 64)

    def test_pos_embed_longer_sequence_interpolates(self):
        """T=5 > num_frames=3 should trigger interpolation, not crash."""
        pred = ARPredictor(
            num_frames=3, depth=1, heads=2, mlp_dim=64,
            input_dim=64, hidden_dim=64, cond_dim=16,
        )
        x = torch.randn(2, 5, 64)
        c = torch.randn(2, 5, 16)
        out = pred(x, c)   # must not raise
        assert out.shape == (2, 5, 64)

    def test_pos_embed_shorter_sequence(self):
        pred = ARPredictor(
            num_frames=4, depth=1, heads=2, mlp_dim=64,
            input_dim=64, hidden_dim=64, cond_dim=16,
        )
        x = torch.randn(2, 2, 64)
        c = torch.randn(2, 2, 16)
        out = pred(x, c)
        assert out.shape == (2, 2, 64)
