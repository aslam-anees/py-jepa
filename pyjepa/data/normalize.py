"""Normalization helpers — ImageNet stats, online standard scaler, picklable z-score."""

from __future__ import annotations

from typing import Optional

import torch

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ZScoreNormalizer:
    """Picklable z-score normalizer.

    Why a class instead of a closure? DataLoader workers spawned with ``fork``
    work fine with closures, but ``spawn`` (Windows + macOS default) requires the
    callable to be picklable. Closures are not, classes are.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8) -> None:
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / (self.std + self.eps)).float()


class RunningStandardScaler:
    """Welford's online algorithm for mean and variance.

    Use this when you can't fit the dataset in memory but still want per-feature
    z-score normalization (e.g. for action or state vectors in a robot dataset).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.M2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: torch.Tensor) -> None:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        for row in x:
            self.n += 1
            delta = row.double() - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (row.double() - self.mean)

    @property
    def var(self) -> torch.Tensor:
        if self.n < 2:
            return torch.ones_like(self.mean)
        return (self.M2 / (self.n - 1)).float()

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp_min_(1e-12))

    def to_normalizer(self) -> ZScoreNormalizer:
        return ZScoreNormalizer(self.mean.float(), self.std)
