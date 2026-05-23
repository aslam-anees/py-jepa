"""Device backend utilities — pick the right device for any host (Apple Silicon / Metal,
NVIDIA CUDA on Linux/Windows, AMD ROCm, or plain CPU).

The single entry point most users want is :func:`get_device` which auto-selects the
best accelerator, with safe fallbacks. Mixed precision helpers live in :mod:`precision`.
"""

from .backend import (
    DeviceInfo,
    autocast_dtype_for,
    device_info,
    get_device,
    is_mps_available,
    pin_memory_supported,
    seed_all,
    set_default_device,
    supports_bfloat16,
    supports_compile,
    supports_sdpa,
    to_device,
)
from .precision import autocast, get_grad_scaler

__all__ = [
    "DeviceInfo",
    "autocast",
    "autocast_dtype_for",
    "device_info",
    "get_device",
    "get_grad_scaler",
    "is_mps_available",
    "pin_memory_supported",
    "seed_all",
    "set_default_device",
    "supports_bfloat16",
    "supports_compile",
    "supports_sdpa",
    "to_device",
]
