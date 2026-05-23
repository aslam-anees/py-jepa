"""Tests for loss functions: SIGReg, VICReg, JEPA loss, variance regularizer."""

import pytest
import torch

from pyjepa.losses.jepa import JEPALoss, VarianceRegularizer
from pyjepa.losses.sigreg import SIGReg
from pyjepa.losses.vicreg import VICRegLoss


class TestSIGReg:
    def test_forward_returns_scalar(self):
        z = torch.randn(16, 64)
        reg = SIGReg(knots=9, num_proj=64)
        loss = reg(z)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_higher_for_collapsed_representations(self):
        """SIGReg should penalize a constant (collapsed) representation more
        than a diverse random Gaussian."""
        reg = SIGReg(knots=9, num_proj=64)

        torch.manual_seed(42)
        z_random = torch.randn(32, 64)
        z_collapsed = torch.zeros(32, 64) + 5.0   # all identical

        loss_random = reg(z_random).item()
        loss_collapsed = reg(z_collapsed).item()

        assert loss_collapsed > loss_random, (
            f"Collapsed loss {loss_collapsed:.4f} should be > random loss {loss_random:.4f}"
        )

    def test_gradient_flows(self):
        z = torch.randn(8, 32, requires_grad=True)
        reg = SIGReg(knots=5, num_proj=32)
        loss = reg(z)
        loss.backward()
        assert z.grad is not None
        assert not z.grad.isnan().any()

    def test_different_knots_and_projections(self):
        z = torch.randn(16, 128)
        for knots in [5, 17, 33]:
            for proj in [64, 512]:
                reg = SIGReg(knots=knots, num_proj=proj)
                loss = reg(z)
                assert loss.item() >= 0


class TestVICRegLoss:
    def test_forward_returns_scalar(self):
        z = torch.randn(8, 64)
        loss_fn = VICRegLoss()
        loss = loss_fn(z)
        assert loss.ndim == 0

    def test_coefficients(self):
        z = torch.randn(16, 64)
        loss_default = VICRegLoss(sim_coeff=25., std_coeff=25., cov_coeff=1.)
        loss_val = loss_default(z)
        assert torch.isfinite(loss_val)

    def test_two_view(self):
        """VICReg can optionally accept two views."""
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        loss_fn = VICRegLoss()
        loss = loss_fn(z1, z2)
        assert loss.ndim == 0


class TestVarianceRegularizer:
    def test_penalizes_low_variance(self):
        reg = VarianceRegularizer()
        z_low_var = torch.ones(16, 64) * 0.01       # near-constant -> high penalty
        z_high_var = torch.randn(16, 64) * 3.0      # large variance -> low penalty

        loss_low = reg(z_low_var).item()
        loss_high = reg(z_high_var).item()
        assert loss_low > loss_high

    def test_zero_when_sufficient_variance(self):
        """When all feature dims have std > 1, penalty should be 0."""
        reg = VarianceRegularizer(gamma=1.0)
        z = torch.randn(64, 64) * 2.0   # std ~ 2 > gamma=1
        loss = reg(z).item()
        assert loss == pytest.approx(0.0, abs=1e-3)


class TestJEPALoss:
    def test_forward(self):
        loss_fn = JEPALoss(loss_exp=1.0, reg_coeff=0.0)
        preds = [torch.randn(4, 8, 64) for _ in range(3)]
        targets = [torch.randn(4, 8, 64) for _ in range(3)]
        loss = loss_fn(preds, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_with_variance_regularizer(self):
        loss_fn = JEPALoss(loss_exp=1.0, reg_coeff=0.1)
        preds = [torch.randn(4, 8, 64) for _ in range(2)]
        targets = [torch.randn(4, 8, 64) for _ in range(2)]
        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss)

    def test_loss_exp(self):
        """loss_exp=2 uses L4 norm; should still produce finite scalar."""
        loss_fn = JEPALoss(loss_exp=2.0, reg_coeff=0.0)
        preds = [torch.randn(4, 8, 64)]
        targets = [torch.randn(4, 8, 64)]
        loss = loss_fn(preds, targets)
        assert torch.isfinite(loss)
