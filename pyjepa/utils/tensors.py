"""Tensor helpers used across the library — trunc_normal init, mask gather, etc."""

from __future__ import annotations

import math

import torch


def _no_grad_trunc_normal_(tensor: torch.Tensor, mean: float, std: float, a: float, b: float) -> torch.Tensor:
    """Truncated normal initialisation, ported from the PyTorch internal helper.

    Sampling from a truncated normal via the inverse CDF is the same trick the
    original I-JEPA / DINO code uses; we keep it here so the library does not
    depend on private torch._C symbols.
    """

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lo = norm_cdf((a - mean) / std)
        hi = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lo - 1, 2 * hi - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """In-place truncated normal init. Public re-export of the original helper."""
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x: torch.Tensor, masks, concat: bool = True):
    """Gather a subset of tokens from ``x`` according to integer-index ``masks``.

    Args:
        x: ``[B, N, D]`` token tensor.
        masks: a list of index tensors, each ``[B, K]`` with values in ``[0, N)``.
        concat: if True returns one ``[B*len(masks), K, D]`` tensor (along batch),
            else a list of ``[B, K, D]`` tensors.
    """
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    if not concat:
        return all_x
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x: torch.Tensor, B: int, repeat: int) -> torch.Tensor:
    """Repeat each block of ``B`` rows ``repeat`` times.

    Given x of shape ``[N*B, ...]`` this returns ``[N*B*repeat, ...]`` where every
    ``B``-sized chunk is repeated ``repeat`` times back-to-back. Used in I-JEPA
    when multiple target masks share one encoder forward.
    """
    N = len(x) // B
    if N == 0 or repeat == 1:
        return x.repeat_interleave(repeat, dim=0) if repeat != 1 else x
    chunks = []
    for i in range(N):
        block = x[i * B : (i + 1) * B]
        for _ in range(repeat):
            chunks.append(block)
    return torch.cat(chunks, dim=0)
