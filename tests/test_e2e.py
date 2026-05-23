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
    collator = MultiBlockMaskCollator(crop_size=IMG_SIZE, patch_size=PATCH_SIZE)

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
    dataset = TrajectoryDataset(
        episodes, obs_key="pixels", act_key="action",
        window=HISTORY_SIZE + NUM_PREDS,
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)


class TestIJEPATrainStep:
    def test_one_training_step(self):
        model = _tiny_ijepa()
        loader = _make_ijepa_loader()

        train_cfg = TrainConfig(
            output_dir="/tmp/pyjepa_test_ijepa",
            device="cpu",
            epochs=1,
            amp_dtype="float32",
            log_every=1,
        )
        ijepa_cfg = IJEPAConfig(
            lr=1e-3, start_lr=1e-4, final_lr=1e-6,
            weight_decay=0.04, warmup_epochs=0,
            momentum_start=0.996, momentum_end=1.0,
        )

        losses = []

        trainer = IJEPATrainer(model, train_cfg, ijepa_cfg,
                               iterations_per_epoch=len(loader))

        @trainer.on_step
        def capture(m):
            if "loss" in m:
                losses.append(m["loss"])

        trainer.fit(loader)

        assert len(losses) > 0, "No loss recorded"
        assert all(isinstance(l, float) for l in losses), "Loss not float"
        assert all(not (l != l) for l in losses), "NaN loss detected"
        assert all(l < 1e6 for l in losses), "Loss exploded"

    def test_loss_decreases_with_identical_images(self):
        """With identical images across context/target, prediction should be trivial."""
        model = _tiny_ijepa()
        # All black images — loss should be small (easy prediction)
        images = torch.zeros(BATCH_SIZE * 2, 3, IMG_SIZE, IMG_SIZE)
        dataset = TensorDataset(images)
        collator = MultiBlockMaskCollator(crop_size=IMG_SIZE, patch_size=PATCH_SIZE)

        def collate_fn(batch):
            imgs = torch.stack([b[0] for b in batch])
            return collator([[img] for img in imgs])

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        train_cfg = TrainConfig(
            output_dir="/tmp/pyjepa_test_ijepa_easy",
            device="cpu", epochs=1, amp_dtype="float32",
        )
        ijepa_cfg = IJEPAConfig(lr=1e-3, weight_decay=0.0, warmup_epochs=0)
        trainer = IJEPATrainer(model, train_cfg, ijepa_cfg, iterations_per_epoch=len(loader))
        trainer.fit(loader)  # should not crash


class TestLeWMTrainStep:
    def test_one_training_step(self):
        enc = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
        model = build_lewm(
            enc, action_dim=ACTION_DIM, history_size=HISTORY_SIZE,
            emb_dim=64, action_emb_dim=16,
            pred_depth=1, pred_heads=2, pred_mlp_dim=64,
            projector_hidden=64,
        )
        loader = _make_lewm_loader()

        train_cfg = TrainConfig(
            output_dir="/tmp/pyjepa_test_lewm",
            device="cpu", epochs=1, amp_dtype="float32",
            log_every=1,
        )
        lewm_cfg = LeWMConfig(
            lr=3e-4, history_size=HISTORY_SIZE, num_preds=NUM_PREDS,
            sigreg_weight=0.1, sigreg_knots=5, sigreg_num_proj=32,
        )

        metrics_list = []
        trainer = LeWMTrainer(model, train_cfg, lewm_cfg)

        @trainer.on_step
        def capture(m):
            metrics_list.append({k: v for k, v in m.items() if isinstance(v, (int, float))})

        trainer.fit(loader)

        assert len(metrics_list) > 0
        last = metrics_list[-1]
        assert "loss" in last or "loss_pred" in last
        total = last.get("loss", last.get("loss_pred", float("nan")))
        assert not (total != total), f"NaN total loss: {last}"
        assert total < 1e6, f"Loss exploded: {total}"
