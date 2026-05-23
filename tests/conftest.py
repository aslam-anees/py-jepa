"""Shared pytest fixtures for pyjepa tests."""

import pytest
import torch

from pyjepa.device.backend import get_device
from pyjepa.models.vit import vit_tiny


@pytest.fixture(scope="session")
def device():
    """Use CPU for tests (deterministic, no GPU required)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_vit():
    return vit_tiny(img_size=32, patch_size=8, num_frames=1)


@pytest.fixture(scope="session")
def tiny_video_vit():
    return vit_tiny(img_size=32, patch_size=8, num_frames=4, tubelet_size=2)


@pytest.fixture
def small_batch():
    """(B=2, C=3, H=32, W=32) random image batch."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def small_video_batch():
    """(B=2, C=3, T=4, H=32, W=32) random video clip batch."""
    torch.manual_seed(0)
    return torch.randn(2, 3, 4, 32, 32)
