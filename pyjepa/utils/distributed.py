"""Distributed training glue. Safe to call from non-distributed contexts."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist


def init_distributed(backend: Optional[str] = None) -> Tuple[int, int]:
    """Initialize ``torch.distributed`` if env vars indicate distributed launch.

    Returns ``(world_size, rank)``. When not distributed returns ``(1, 0)``.

    Picks ``nccl`` for CUDA, ``gloo`` for CPU/MPS/Windows. Honors RANK / LOCAL_RANK
    / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size <= 1:
        return 1, 0

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return world_size, rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


class AllReduce(torch.autograd.Function):
    """SUM-then-AVERAGE all-reduce that is a no-op outside distributed."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        if not is_distributed():
            return x
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(get_world_size())
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad
