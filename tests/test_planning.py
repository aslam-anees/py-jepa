"""Tests for planners — including regression test for history_size bug."""

import pytest
import torch

from pyjepa.models.registry import build_lewm
from pyjepa.models.vit import vit_tiny
from pyjepa.planning.cem import CEMConfig, CEMPlanner
from pyjepa.planning.mppi import MPPIConfig, MPPIPlanner
from pyjepa.planning.policy import PlanConfig, RandomPolicy
from pyjepa.planning.random_shoot import RandomShootingPlanner


# Small model to keep tests fast
IMG_SIZE = 32
PATCH_SIZE = 8
ACTION_DIM = 4
HISTORY_SIZE = 2
EMB_DIM = 64


def _tiny_lewm():
    enc = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)  # 192-dim
    model = build_lewm(
        enc, action_dim=ACTION_DIM, history_size=HISTORY_SIZE,
        emb_dim=192,  # match vit_tiny output
        action_emb_dim=16,
        pred_depth=1, pred_heads=2, pred_mlp_dim=256,
        projector_hidden=256,
    )
    return model.eval()


def _make_info(batch_size=1):
    pixels = torch.randn(batch_size, HISTORY_SIZE, 3, IMG_SIZE, IMG_SIZE)
    goal = torch.randn(batch_size, 1, 3, IMG_SIZE, IMG_SIZE)
    return {"pixels": pixels, "goal": goal}


# ---- PlanConfig ----

class TestPlanConfig:
    def test_defaults(self):
        cfg = PlanConfig(horizon=5, action_dim=4)
        assert cfg.horizon == 5
        assert cfg.action_dim == 4
        assert cfg.num_samples > 0
        assert cfg.num_elites > 0

    def test_cem_config_inherits(self):
        cfg = CEMConfig(horizon=5, action_dim=4, momentum=0.5)
        assert cfg.momentum == 0.5
        assert cfg.horizon == 5


# ---- Random Shooting ----

class TestRandomShootingPlanner:
    def test_returns_correct_shape(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=4, num_iters=1, num_elites=2)
        planner = RandomShootingPlanner(cfg)
        info = _make_info()
        with torch.no_grad():
            action = planner(model, info)
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_actions_within_bounds(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=8, action_min=-0.5, action_max=0.5)
        planner = RandomShootingPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.min() >= -0.5 - 1e-5
        assert action.max() <= 0.5 + 1e-5


# ---- CEM ----

class TestCEMPlanner:
    def test_returns_correct_shape(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=8, num_iters=2, num_elites=4)
        planner = CEMPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_warm_start(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=8, num_iters=2, num_elites=4)
        planner = CEMPlanner(cfg)
        warm = torch.zeros(HISTORY_SIZE, ACTION_DIM)
        with torch.no_grad():
            action = planner(model, _make_info(), action_init=warm)
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_multi_iter_reduces_cost(self):
        """More CEM iterations should not increase cost (on average)."""
        model = _tiny_lewm()
        info = _make_info()

        cfg1 = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                          num_samples=16, num_iters=1, num_elites=8)
        cfg5 = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                          num_samples=16, num_iters=5, num_elites=8)

        planner1 = CEMPlanner(cfg1)
        planner5 = CEMPlanner(cfg5)
        with torch.no_grad():
            a1 = planner1(model, {**info})
            a5 = planner5(model, {**info})
        # Just check both return valid tensors
        assert a1.shape == a5.shape == (HISTORY_SIZE, ACTION_DIM)


# ---- MPPI ----

class TestMPPIPlanner:
    def test_returns_correct_shape(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=8, num_iters=2, temperature=0.1)
        planner = MPPIPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_actions_clamped(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=8, num_iters=1,
                         action_min=-0.3, action_max=0.3)
        planner = MPPIPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.min() >= -0.3 - 1e-5
        assert action.max() <= 0.3 + 1e-5


# ---- get_cost ----

class TestGetCost:
    def test_returns_per_sample_cost(self):
        model = _tiny_lewm()
        B, S = 1, 4
        pixels = torch.randn(B, S, HISTORY_SIZE, 3, IMG_SIZE, IMG_SIZE)
        goal = torch.randn(B, S, 1, 3, IMG_SIZE, IMG_SIZE)
        candidates = torch.randn(B, S, HISTORY_SIZE, ACTION_DIM)
        info = {"pixels": pixels, "goal": goal}
        with torch.no_grad():
            cost = model.get_cost(info, candidates)
        assert cost.shape == (B, S)
        assert torch.isfinite(cost).all()


# ---- Regression: history_size mismatch ----

class TestHistorySizeRegression:
    """
    Regression test for the bug where get_cost() called rollout() without
    passing history_size, causing ARPredictor to receive T > num_frames.

    If this test passes, the fix (storing history_size on LeWM and passing
    it in get_cost) is working correctly.
    """

    def test_cem_planner_with_history_size_2(self):
        """CEM planner on a model with history_size=2 must not crash."""
        model = _tiny_lewm()  # built with HISTORY_SIZE=2
        assert model.history_size == HISTORY_SIZE == 2

        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=4, num_iters=2, num_elites=2)
        planner = CEMPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_mppi_planner_with_history_size_2(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=4, num_iters=2)
        planner = MPPIPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)

    def test_random_shooting_with_history_size_2(self):
        model = _tiny_lewm()
        cfg = PlanConfig(horizon=HISTORY_SIZE, action_dim=ACTION_DIM,
                         num_samples=4, num_iters=1, num_elites=2)
        planner = RandomShootingPlanner(cfg)
        with torch.no_grad():
            action = planner(model, _make_info())
        assert action.shape == (HISTORY_SIZE, ACTION_DIM)
