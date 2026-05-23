"""Autocast / AMP helpers that work the same on CUDA, MPS, and CPU."""

from __future__ import annotations

import contextlib
from typing import Optional

import torch

from .backend import autocast_dtype_for, get_device


@contextlib.contextmanager
def autocast(device: Optional[torch.device] = None, dtype: str = "auto", enabled: bool = True):
    """Device-agnostic ``torch.autocast`` context manager.

    On CUDA picks bf16 (Ampere+) or fp16. On MPS we keep fp32 by default — MPS
    autocast is fp16 only and still has known kernel gaps. Pass ``dtype="fp16"``
    explicitly if you want to opt in.
    """
    device = device or get_device()
    amp_dtype = autocast_dtype_for(device, prefer=dtype)
    if not enabled or amp_dtype is None:
        yield
        return

    device_type = "cuda" if device.type == "cuda" else device.type
    if device_type == "mps":
        # PyTorch supports autocast on MPS as of 2.4; older versions fall back to CPU
        try:
            with torch.autocast(device_type="mps", dtype=amp_dtype):
                yield
            return
        except (RuntimeError, ValueError):
            yield
            return

    with torch.autocast(device_type=device_type, dtype=amp_dtype):
        yield


def get_grad_scaler(device: Optional[torch.device] = None, enabled: bool = True):
    """Return a fp16 GradScaler when running CUDA + fp16, otherwise a no-op."""
    device = device or get_device()
    if not enabled or device.type != "cuda":
        return _NullScaler()
    try:
        return torch.amp.GradScaler(device=device.type)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()


class _NullScaler:
    """A GradScaler that does nothing — keeps training loop code uniform."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def is_enabled(self) -> bool:
        return False
