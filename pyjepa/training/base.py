"""Base trainer shared by I-JEPA / V-JEPA / LeWM training engines.

We deliberately avoid a giant Lightning-style abstraction. The trainer just:
- holds device/dtype/seed state,
- wires up the optimizer + schedulers + AMP scaler,
- runs a basic epoch loop,
- handles checkpoint save/load,
- exposes hooks (``on_step``, ``on_epoch_end``) so users can extend it.

Subclasses implement ``train_step(batch) -> dict``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from ..device.backend import get_device, seed_all, to_device
from ..device.precision import autocast, get_grad_scaler
from ..utils.checkpoint import CheckpointManager
from ..utils.logging import MetricTracker, get_logger

logger = get_logger()


@dataclass
class TrainConfig:
    """All the knobs the base trainer needs.

    Anything model-specific (loss coefficients, mask configs) belongs in the
    subclass-specific config, not here.
    """

    output_dir: str = "runs/exp"
    device: str = "auto"
    seed: int = 0
    epochs: int = 100
    log_every: int = 20
    save_every_epochs: int = 1
    keep_last_ckpts: int = 3
    amp_dtype: str = "auto"   # "auto" | "bf16" | "fp16" | "fp32" | "none"
    grad_clip: Optional[float] = 1.0
    compile_model: bool = False
    deterministic: bool = False
    resume: bool = True


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    extra: dict = field(default_factory=dict)


class BaseTrainer:
    """Lightweight, hook-friendly training engine."""

    def __init__(self, model: nn.Module, config: TrainConfig) -> None:
        self.config = config
        self.device = get_device(config.device)
        seed_all(config.seed, deterministic=config.deterministic)

        self.model = model.to(self.device)
        if config.compile_model:
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back to eager: {e}")

        self.scaler = get_grad_scaler(self.device, enabled=config.amp_dtype not in ("none", "fp32"))
        self.state = TrainState()
        self.ckpt = CheckpointManager(config.output_dir, keep_last=config.keep_last_ckpts)
        self.metrics = MetricTracker(
            csv_path=Path(config.output_dir) / "metrics.csv",
            log_every=config.log_every,
        )

        # Hook lists
        self._on_step: list[Callable[[BaseTrainer, dict], None]] = []
        self._on_epoch_end: list[Callable[[BaseTrainer], None]] = []

    # --- Hooks --------------------------------------------------------------

    def on_step(self, fn: Callable[["BaseTrainer", dict], None]) -> Callable:
        self._on_step.append(fn)
        return fn

    def on_epoch_end(self, fn: Callable[["BaseTrainer"], None]) -> Callable:
        self._on_epoch_end.append(fn)
        return fn

    # --- To be implemented by subclasses ------------------------------------

    def train_step(self, batch: Any) -> dict:
        raise NotImplementedError

    # --- Shared loop --------------------------------------------------------

    def fit(self, train_loader, val_loader=None) -> TrainState:
        if self.config.resume:
            self._maybe_resume()

        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            self._train_epoch(train_loader)
            if val_loader is not None:
                self._validate(val_loader)
            for fn in self._on_epoch_end:
                fn(self)
            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save_checkpoint()
        return self.state

    def _train_epoch(self, loader) -> None:
        self.model.train()
        loader_iter = iter(loader)
        t0 = time.time()
        for batch in loader_iter:
            batch = to_device(batch, self.device)
            metrics = self.train_step(batch)
            self.state.step += 1
            metrics_f = {k: float(v) for k, v in metrics.items() if torch.is_tensor(v) or isinstance(v, (int, float))}
            self.metrics.step(**metrics_f)
            for fn in self._on_step:
                fn(self, metrics)
        logger.info(
            f"epoch {self.state.epoch} done in {time.time() - t0:.1f}s "
            f"avg_loss={self.metrics.meters.get('loss').avg if 'loss' in self.metrics.meters else float('nan'):.4f}"
        )

    def _validate(self, loader) -> None:
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, self.device)
                try:
                    metrics = self.val_step(batch)
                except NotImplementedError:
                    return
                if metrics:
                    self.metrics.step(**{f"val/{k}": float(v) for k, v in metrics.items()})

    def val_step(self, batch: Any) -> dict:
        raise NotImplementedError

    # --- Helpers ------------------------------------------------------------

    @property
    def autocast(self):
        return autocast(self.device, dtype=self.config.amp_dtype)

    def clip_grads(self, params) -> Optional[float]:
        if self.config.grad_clip is None or self.config.grad_clip <= 0:
            return None
        return float(torch.nn.utils.clip_grad_norm_(list(params), self.config.grad_clip))

    def save_checkpoint(self, extra: Optional[dict] = None) -> Path:
        state = self.state_dict()
        if extra:
            state.update(extra)
        path = self.ckpt.save(state, step=self.state.step)
        logger.info(f"saved checkpoint → {path}")
        return path

    def state_dict(self) -> dict:
        return {
            "model": _unwrap(self.model).state_dict(),
            "epoch": self.state.epoch,
            "step": self.state.step,
            "best_loss": self.state.best_loss,
            "scaler": self.scaler.state_dict(),
            "config": self.config.__dict__,
        }

    def _maybe_resume(self) -> None:
        sd = self.ckpt.latest()
        if sd is None:
            return
        try:
            _unwrap(self.model).load_state_dict(sd["model"], strict=False)
            self.state.epoch = sd.get("epoch", 0)
            self.state.step = sd.get("step", 0)
            self.state.best_loss = sd.get("best_loss", float("inf"))
            if "scaler" in sd:
                try:
                    self.scaler.load_state_dict(sd["scaler"])
                except Exception:
                    pass
            logger.info(f"resumed from epoch {self.state.epoch}, step {self.state.step}")
        except Exception as e:
            logger.warning(f"failed to resume checkpoint: {e}")


def _unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP / torch.compile wrappers."""
    for attr in ("_orig_mod", "module"):
        if hasattr(model, attr):
            model = getattr(model, attr)
    return model
