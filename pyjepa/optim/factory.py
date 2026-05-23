"""Optimizer factories — sane defaults for JEPA training with proper WD groups.

Bias and 1D (norm) parameters get ``weight_decay=0`` so they aren't shrunk
toward zero — that's a standard trick from the ViT / DeiT recipes and matters
for both convergence speed and final accuracy.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_param_groups(
    modules: Iterable[nn.Module],
    weight_decay: float = 0.05,
    exclude_bias_and_norm: bool = True,
) -> list[dict]:
    """Return param groups with WD applied selectively.

    Mirrors the I-JEPA reference: weight matrices get ``weight_decay``, biases
    and norm params get zero WD and a ``WD_exclude=True`` flag so the cosine WD
    schedule can skip them.
    """
    decay_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []

    seen: set[int] = set()
    for mod in modules:
        for n, p in mod.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if exclude_bias_and_norm and ("bias" in n or p.ndim == 1):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

    groups: list[dict] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if nodecay_params:
        groups.append({"params": nodecay_params, "weight_decay": 0.0, "WD_exclude": True})
    return groups


def build_optimizer(
    modules: Iterable[nn.Module],
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    optimizer: str = "adamw",
    exclude_bias_and_norm: bool = True,
) -> torch.optim.Optimizer:
    """Build an optimizer over one or more modules.

    Supported optimizers: ``adamw``, ``adam``, ``sgd``, ``lion`` (if installed).
    """
    groups = build_param_groups(modules, weight_decay=weight_decay, exclude_bias_and_norm=exclude_bias_and_norm)
    optimizer = optimizer.lower()
    if optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if optimizer == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    if optimizer == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=betas[0], nesterov=True)
    if optimizer == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore

            return Lion(groups, lr=lr, betas=betas)
        except ImportError as e:
            raise ImportError("Install lion-pytorch to use Lion: pip install lion-pytorch") from e
    raise ValueError(f"unknown optimizer {optimizer!r}")
