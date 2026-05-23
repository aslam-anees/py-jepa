"""I-JEPA training engine.

Inputs: image batch + ``(masks_enc, masks_pred)`` lists from the mask collator.
Outputs: prediction loss + variance regularizer.

Usage::

    trainer = IJEPATrainer(model, train_cfg, ijepa_cfg)
    trainer.fit(train_loader, val_loader)

Or one-call::

    train_ijepa(model, train_loader, epochs=100, lr=1e-3, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ..losses.jepa import JEPALoss
from ..models.ijepa import IJEPA
from ..optim.ema import EMA
from ..optim.factory import build_optimizer
from ..optim.schedulers import CosineWDSchedule, LinearSchedule, WarmupCosineSchedule
from .base import BaseTrainer, TrainConfig


@dataclass
class IJEPAConfig:
    lr: float = 1e-3
    start_lr: float = 2e-4
    final_lr: float = 1e-6
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    warmup_epochs: int = 15
    iterations_per_epoch: Optional[int] = None  # inferred from loader
    ipe_scale: float = 1.0
    momentum_start: float = 0.996
    momentum_end: float = 1.0
    loss_exp: float = 1.0
    reg_coeff: float = 0.0


class IJEPATrainer(BaseTrainer):
    """Trainer for :class:`IJEPA` (image JEPA) and :class:`VJEPA` (same loss)."""

    def __init__(
        self,
        model: IJEPA,
        train_config: TrainConfig,
        ijepa_config: IJEPAConfig,
        iterations_per_epoch: Optional[int] = None,
    ) -> None:
        super().__init__(model, train_config)
        self.ijepa_config = ijepa_config
        ipe = iterations_per_epoch or ijepa_config.iterations_per_epoch or 1
        total_steps = int(ipe * train_config.epochs * ijepa_config.ipe_scale)

        self.optimizer = build_optimizer(
            [model.encoder, model.predictor],
            lr=ijepa_config.lr,
            weight_decay=ijepa_config.weight_decay,
            optimizer="adamw",
        )
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=int(ijepa_config.warmup_epochs * ipe),
            start_lr=ijepa_config.start_lr,
            ref_lr=ijepa_config.lr,
            final_lr=ijepa_config.final_lr,
            T_max=total_steps,
        )
        self.wd_scheduler = CosineWDSchedule(
            self.optimizer,
            ref_wd=ijepa_config.weight_decay,
            final_wd=ijepa_config.final_weight_decay,
            T_max=total_steps,
        )
        self.ema = EMA(
            online=model.encoder,
            target=model.target_encoder,
            momentum_schedule=LinearSchedule(ijepa_config.momentum_start, ijepa_config.momentum_end, total_steps),
        )
        self.loss_fn = JEPALoss(loss_exp=ijepa_config.loss_exp, reg_coeff=ijepa_config.reg_coeff)

    def train_step(self, batch: Any) -> dict:
        # Two valid input shapes from the mask collator:
        #  (clips_tensor, masks_enc, masks_pred)
        #  ((clips_dict_or_tensor, ...), masks_enc, masks_pred)
        data, masks_enc, masks_pred = _unpack(batch)
        x = _extract_pixels(data)

        self.optimizer.zero_grad(set_to_none=True)
        lr_now = self.scheduler.step()
        wd_now = self.wd_scheduler.step()

        with self.autocast:
            preds, targets = self.model(x, masks_enc, masks_pred)
            out = self.loss_fn(preds, targets)
            loss = out["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = self.clip_grads(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        m = self.ema.update()

        return {
            "loss": loss.detach(),
            **{k: v.detach() for k, v in out.items() if k != "loss"},
            "lr": lr_now,
            "wd": wd_now,
            "ema": m,
            "grad_norm": grad_norm or 0.0,
        }


def _unpack(batch: Any) -> Tuple[Any, Any, Any]:
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(
        "I-JEPA expects a 3-tuple (data, masks_enc, masks_pred) from the mask collator"
    )


def _extract_pixels(data: Any) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, dict):
        for key in ("pixels", "image", "video", "clip"):
            if key in data:
                return data[key]
    if isinstance(data, (list, tuple)) and data:
        return _extract_pixels(data[0])
    raise ValueError(f"could not extract pixel tensor from {type(data)}")


def train_ijepa(
    model: IJEPA,
    train_loader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.04,
    reg_coeff: float = 0.0,
    output_dir: str = "runs/ijepa",
    device: str = "auto",
    amp_dtype: str = "auto",
    val_loader=None,
    **extra,
):
    """One-call I-JEPA pre-training. Returns the final TrainState."""
    train_cfg = TrainConfig(output_dir=output_dir, device=device, epochs=epochs, amp_dtype=amp_dtype, **extra)
    ijepa_cfg = IJEPAConfig(lr=lr, weight_decay=weight_decay, reg_coeff=reg_coeff)
    ipe = len(train_loader) if hasattr(train_loader, "__len__") else 1000
    trainer = IJEPATrainer(model, train_cfg, ijepa_cfg, iterations_per_epoch=ipe)
    return trainer.fit(train_loader, val_loader)
