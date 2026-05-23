"""Tests for device utilities."""

import pytest
import torch

from pyjepa.device.backend import (
    DeviceInfo,
    autocast_dtype_for,
    device_info,
    get_device,
    is_mps_available,
    seed_all,
    supports_compile,
    supports_sdpa,
)
from pyjepa.device.precision import autocast, get_grad_scaler


class TestGetDevice:
    def test_returns_torch_device(self):
        dev = get_device()
        assert isinstance(dev, torch.device)

    def test_explicit_cpu(self):
        dev = get_device("cpu")
        assert dev.type == "cpu"

    def test_auto_falls_back_to_cpu(self):
        import os
        orig = os.environ.get("PYJEPA_DEVICE")
        os.environ["PYJEPA_DEVICE"] = "cpu"
        dev = get_device()
        assert dev.type == "cpu"
        if orig is None:
            del os.environ["PYJEPA_DEVICE"]
        else:
            os.environ["PYJEPA_DEVICE"] = orig


class TestSeedAll:
    def test_does_not_crash(self):
        seed_all(42)
        seed_all(42, deterministic=True)

    def test_reproducibility(self):
        seed_all(123)
        x1 = torch.randn(10)
        seed_all(123)
        x2 = torch.randn(10)
        assert torch.allclose(x1, x2)


class TestSupportsFlags:
    def test_supports_sdpa_bool(self):
        assert isinstance(supports_sdpa(), bool)

    def test_supports_compile_bool(self):
        assert isinstance(supports_compile(), bool)

    def test_is_mps_available_bool(self):
        assert isinstance(is_mps_available(), bool)


class TestDeviceInfo:
    def test_from_current_returns_dataclass(self):
        di = DeviceInfo.from_current()
        assert isinstance(di, DeviceInfo)

    def test_device_info_function(self):
        di = device_info()
        assert hasattr(di, "backend")
        assert hasattr(di, "supports_sdpa")
        assert hasattr(di, "torch_version")

    def test_str_representation(self):
        di = DeviceInfo.from_current()
        s = str(di)
        assert "device=" in s
        assert "backend=" in s

    def test_cpu_device_info(self):
        di = device_info(torch.device("cpu"))
        assert di.backend == "cpu"
        assert di.name == "cpu"


class TestAutocastDtypeFor:
    def test_cpu_returns_none(self):
        result = autocast_dtype_for(torch.device("cpu"), prefer="auto")
        assert result is None

    def test_fp32_returns_none(self):
        result = autocast_dtype_for(torch.device("cuda") if torch.cuda.is_available()
                                    else torch.device("cpu"), prefer="fp32")
        assert result is None


class TestAutocast:
    def test_cpu_context_manager(self):
        """autocast should be a no-op on CPU without raising."""
        with autocast(torch.device("cpu"), dtype=None, enabled=False):
            x = torch.randn(4, 4)
            y = x @ x.T
        assert y.shape == (4, 4)


class TestGetGradScaler:
    def test_cpu_returns_null_scaler(self):
        scaler = get_grad_scaler(torch.device("cpu"), enabled=True)
        # Null scaler has a scale() method that is a no-op
        x = torch.tensor(1.0)
        scaled = scaler.scale(x)
        assert torch.allclose(scaled, x)
