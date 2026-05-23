"""Cross-platform device selection.

Why a dedicated module? Apple Silicon (Metal) needs MPS, Windows boxes typically run
CUDA, Linux servers can be CUDA or ROCm, and CI runs CPU only. The rest of the
library should never have to ``if torch.cuda.is_available():`` — it just asks
:func:`get_device`.
"""

from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch


def is_mps_available() -> bool:
    """True if Apple Silicon's Metal Performance Shaders backend is usable."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def is_rocm_available() -> bool:
    return getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()


def supports_sdpa() -> bool:
    """torch.nn.functional.scaled_dot_product_attention exists on every torch>=2.0,
    but the fused CUDA / Flash kernels are only available on CUDA. On MPS it falls
    back to a math kernel which is still correct."""
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def supports_compile() -> bool:
    """torch.compile works on CUDA and CPU; MPS support is still experimental and
    has known graph-break issues with attention. We return True only when safe."""
    if not hasattr(torch, "compile"):
        return False
    if is_mps_available() and not is_cuda_available():
        return False
    return True


def supports_bfloat16(device: Optional[torch.device] = None) -> bool:
    """bf16 is supported on CUDA SM80+ (A100/H100), recent CPUs, and (partially) on MPS.
    For safety, we whitelist CUDA only."""
    if device is None:
        device = get_device()
    if device.type == "cuda":
        try:
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
        except Exception:
            return False
    return False


def autocast_dtype_for(device: torch.device, prefer: str = "auto") -> Optional[torch.dtype]:
    """Pick the best autocast dtype for a device.

    ``prefer`` accepts ``"auto"``, ``"bf16"``, ``"fp16"``, ``"fp32"``, ``"none"``.
    Returns ``None`` when autocast should be disabled.
    """
    if prefer in ("none", "fp32"):
        return None
    if prefer == "bf16":
        return torch.bfloat16 if supports_bfloat16(device) else None
    if prefer == "fp16":
        if device.type == "cuda":
            return torch.float16
        return None
    # auto
    if device.type == "cuda":
        return torch.bfloat16 if supports_bfloat16(device) else torch.float16
    if device.type == "mps":
        # MPS autocast is fp16 only and somewhat fragile; keep fp32 by default
        return None
    return None


def pin_memory_supported(device: Optional[torch.device] = None) -> bool:
    """DataLoader.pin_memory only helps for CUDA hosts; on MPS / CPU it can warn."""
    if device is None:
        device = get_device()
    return device.type == "cuda"


def _pick_index() -> int:
    """Honor CUDA_VISIBLE_DEVICES and SLURM_LOCALID if present."""
    if "LOCAL_RANK" in os.environ:
        try:
            return int(os.environ["LOCAL_RANK"])
        except ValueError:
            pass
    if "SLURM_LOCALID" in os.environ:
        try:
            return int(os.environ["SLURM_LOCALID"])
        except ValueError:
            pass
    return 0


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Auto-select the best available device.

    Resolution order: explicit ``prefer`` → ``$PYJEPA_DEVICE`` → CUDA → MPS → CPU.

    ``prefer`` can be one of:
        ``"auto"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``, ``"cpu"``,
        ``"xpu"`` (Intel GPUs), or a fully-formed device string.
    """
    if prefer is None:
        prefer = os.environ.get("PYJEPA_DEVICE", "auto")

    if prefer != "auto":
        return torch.device(prefer)

    if is_cuda_available():
        return torch.device(f"cuda:{_pick_index()}")
    if is_mps_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
        return torch.device("xpu")
    return torch.device("cpu")


def set_default_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """Set the process-wide default device. Returns the resolved device."""
    dev = get_device(device) if isinstance(device, (str, type(None))) else device
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(dev)
    return dev


def seed_all(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA + MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    if is_mps_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def to_device(x: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move tensors in nested containers to ``device``.

    Non-tensor leaves are returned unchanged. Works with tuples, lists, and dicts.
    """
    if isinstance(x, torch.Tensor):
        # non_blocking only helps with pinned-memory CPU tensors copied to CUDA
        if device.type != "cuda":
            non_blocking = False
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        out = [to_device(v, device, non_blocking) for v in x]
        return type(x)(out) if isinstance(x, tuple) else out
    return x


@dataclass
class DeviceInfo:
    device: torch.device
    backend: str          # cuda / mps / cpu / xpu / rocm
    name: str             # human-readable device name
    total_memory_mb: Optional[int]
    supports_bf16: bool
    supports_fp16: bool
    supports_sdpa: bool
    supports_compile: bool
    platform: str
    torch_version: str

    def __str__(self) -> str:
        mem = f"{self.total_memory_mb} MB" if self.total_memory_mb else "n/a"
        return (
            f"[pyjepa] device={self.device} backend={self.backend} name={self.name!r} "
            f"mem={mem} bf16={self.supports_bf16} sdpa={self.supports_sdpa} "
            f"compile={self.supports_compile} platform={self.platform} torch={self.torch_version}"
        )


def device_info(device: Optional[torch.device] = None) -> DeviceInfo:
    """Build a structured snapshot of the active accelerator. Useful for logs."""
    device = device or get_device()
    backend = device.type
    name = "cpu"
    total = None

    if backend == "cuda":
        try:
            name = torch.cuda.get_device_name(device)
            total = int(torch.cuda.get_device_properties(device).total_memory / 1024**2)
        except Exception:
            pass
        if is_rocm_available():
            backend = "rocm"
    elif backend == "mps":
        name = f"Apple Silicon ({platform.processor() or platform.machine()})"
    elif backend == "xpu":
        name = "Intel XPU"

    return DeviceInfo(
        device=device,
        backend=backend,
        name=name,
        total_memory_mb=total,
        supports_bf16=supports_bfloat16(device),
        supports_fp16=backend == "cuda",
        supports_sdpa=supports_sdpa(),
        supports_compile=supports_compile(),
        platform=f"{platform.system()} {platform.release()}",
        torch_version=torch.__version__,
    )
