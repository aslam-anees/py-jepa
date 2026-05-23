"""Pure-PyTorch image/video transforms.

Why not just use torchvision.transforms.v2? Two reasons:
1. We want pyjepa to work without torchvision installed.
2. We want one stack that handles both 2D images and 5D video tensors.

When torchvision IS installed, we transparently delegate to it for the heavy
operations (autoaugment, randaugment) via :func:`build_image_transform`.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from .normalize import IMAGENET_MEAN, IMAGENET_STD


class ToImageTensor:
    """Convert PIL/numpy/uint8 → float tensor in [0, 1].

    Accepts (H, W), (H, W, C), (C, H, W), (T, C, H, W) and PIL images.
    Output always has a channel dim and float32 dtype.
    """

    def __call__(self, x) -> torch.Tensor:
        if hasattr(x, "convert"):  # PIL image
            try:
                from PIL import Image  # noqa: F401
                import numpy as np

                arr = np.array(x.convert("RGB"))
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            except ImportError as e:
                raise ImportError("PIL required for PIL inputs") from e
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            import numpy as np

            t = torch.from_numpy(np.asarray(x))
            if t.ndim == 2:
                t = t.unsqueeze(0)
            elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                t = t.permute(2, 0, 1)

        if t.dtype == torch.uint8:
            t = t.float().div_(255.0)
        else:
            t = t.float()
        return t


class Resize:
    """Bilinear resize for images and videos. Accepts ``(C, H, W)`` or ``(C, T, H, W)``."""

    def __init__(self, size: Union[int, Tuple[int, int]], mode: str = "bilinear") -> None:
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.mode = mode

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = F.interpolate(x.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False).squeeze(0)
        elif x.ndim == 4:  # (C, T, H, W)
            # interp expects (B, C, ...) so wrap and treat T as batch
            C, T, H, W = x.shape
            x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
            x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
            x = x.permute(1, 0, 2, 3)  # back to (C, T, H, W)
        elif x.ndim == 5:  # (B, C, T, H, W)
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)
            x = x.reshape(B, T, C, *self.size).permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"unsupported tensor shape for Resize: {x.shape}")
        return x


class ImageNormalize:
    """Channel-wise normalization. Same mean/std for images and per-frame video."""

    def __init__(
        self,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        if x.ndim == 4:  # (C, T, H, W)
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)
        elif x.ndim == 5:
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)
        return (x - mean) / std


class Compose:
    """Tiny compose — keeps the dependency surface clean."""

    def __init__(self, *fns: Callable) -> None:
        self.fns = fns

    def __call__(self, x):
        for fn in self.fns:
            x = fn(x)
        return x


def build_image_transform(
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    use_torchvision: bool = True,
) -> Callable:
    """Compose a basic image transform (ToTensor → Resize → Normalize).

    If torchvision is available and ``use_torchvision=True`` we route through
    torchvision.transforms.v2 because the conversions are battle-tested for
    PIL/uint8 edge cases.
    """
    if use_torchvision:
        try:
            from torchvision.transforms import v2 as T

            return T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Resize(size=img_size, antialias=True),
                    T.CenterCrop(img_size),
                    T.Normalize(mean=list(mean), std=list(std)),
                ]
            )
        except ImportError:
            pass
    return Compose(ToImageTensor(), Resize(img_size), ImageNormalize(mean=mean, std=std))


def build_video_transform(
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable:
    """Frame-wise image transform applied across a video clip ``(C, T, H, W)``."""
    return Compose(ToImageTensor(), Resize(img_size), ImageNormalize(mean=mean, std=std))
