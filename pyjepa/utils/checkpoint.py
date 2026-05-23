"""Checkpoint save/load that keeps the training loop honest about state.

We deliberately save *both* the raw state_dict (portable) and the full training
state (optimizer, scheduler, scaler, EMA momentum, epoch, step) so resumption is
bit-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from .logging import get_logger

logger = get_logger()


def save_checkpoint(
    path: Union[str, Path],
    state: dict,
    keep_last: Optional[int] = None,
) -> Path:
    """Atomically save a checkpoint dict to ``path``.

    If ``keep_last`` is set, older sibling files matching ``{stem}-step*{suffix}``
    are pruned so only the N most recent are kept.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    if keep_last is not None:
        rotate(path, keep_last)
    return path


def rotate(path: Union[str, Path], keep_last: int) -> None:
    """Keep only the ``keep_last`` most recent siblings matching the prefix."""
    path = Path(path)
    parent = path.parent
    prefix = path.stem.split("-step")[0]
    siblings = sorted(parent.glob(f"{prefix}-step*{path.suffix}"))
    while len(siblings) > keep_last:
        try:
            siblings.pop(0).unlink()
        except Exception as e:
            logger.warning(f"failed to rotate checkpoint: {e}")


def load_checkpoint(path: Union[str, Path], map_location: Any = "cpu") -> dict:
    """Load a checkpoint dict. Tries weights_only=True first (safer), falls back."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def load_state_dict(module: torch.nn.Module, state: dict, strict: bool = False, prefix: str = "") -> None:
    """Load ``state`` into ``module``, optionally stripping a leading prefix.

    Useful for restoring weights wrapped in DDP / OptimizedModule containers where
    keys are prefixed with ``module.`` or ``_orig_mod.``.
    """
    if prefix:
        state = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}
    # Strip common wrappers automatically
    stripped = {}
    for k, v in state.items():
        nk = k
        for p in ("module.", "_orig_mod."):
            if nk.startswith(p):
                nk = nk[len(p) :]
        stripped[nk] = v
    missing, unexpected = module.load_state_dict(stripped, strict=strict)
    if missing:
        logger.warning(f"missing keys when loading: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"unexpected keys when loading: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


class CheckpointManager:
    """Convenience wrapper around save/load with a directory-level view.

    Example::

        ckpt = CheckpointManager("runs/ijepa-vits")
        ckpt.save({"encoder": enc.state_dict(), "step": step}, step=step)
        latest = ckpt.latest()
    """

    def __init__(self, root: Union[str, Path], keep_last: int = 3, name: str = "ckpt") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.name = name

    @property
    def latest_path(self) -> Path:
        return self.root / f"{self.name}-latest.pt"

    def step_path(self, step: int) -> Path:
        return self.root / f"{self.name}-step{step:08d}.pt"

    def save(self, state: dict, step: Optional[int] = None, snapshot: bool = True) -> Path:
        save_checkpoint(self.latest_path, state)
        if snapshot and step is not None:
            save_checkpoint(self.step_path(step), state, keep_last=self.keep_last)
            return self.step_path(step)
        return self.latest_path

    def latest(self) -> Optional[dict]:
        if not self.latest_path.exists():
            return None
        return load_checkpoint(self.latest_path)

    def list_steps(self) -> list[int]:
        out: list[int] = []
        for p in self.root.glob(f"{self.name}-step*.pt"):
            try:
                out.append(int(p.stem.split("step")[-1]))
            except ValueError:
                continue
        return sorted(out)
