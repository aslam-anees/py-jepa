"""LeWM training engine — end-to-end JEPA from pixels with two losses.

Forward pass (matches the reference implementation):

    info = model.encode(batch)
    emb = info["emb"]                           # (B, T, D)
    act = info["act_emb"]                       # (B, T, A_emb)

    ctx_emb = emb[:, :H]                        # context window (H = history)
    ctx_act = act[:, :H]
    tgt_emb = emb[:, n_preds:]                  # shifted targets

    pred = model.predict(ctx_emb, ctx_act)
    loss_pred = mse(pred, tgt_emb)
    loss_sig = sigreg(emb.transpose(0, 1))
    loss = loss_pred + lambda * loss_sig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ..losses.sigreg import SIGReg
from ..models.lewm import LeWM
from ..optim.factory import build_optimizer
from ..optim.schedulers import WarmupCosineSchedule
from .base import BaseTrainer, TrainConfig


@dataclass
class LeWMConfig:
    lr: float = 1e-4
    start_lr: float = 1e-5
    final_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    history_size: int = 3
    num_preds: int = 1
    sigreg_weight: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024


class LeWMTrainer(BaseTrainer):
    """Trainer for :class:`LeWM` world model."""

    def __init__(
        self,
        model: LeWM,
        train_config: TrainConfig,
        lewm_config: LeWMConfig,
        iterations_per_epoch: int = 1000,
    ) -> None:
        super().__init__(model, train_config)
        self.lewm_config = lewm_config
        total_steps = iterations_per_epoch * train_config.epochs

        self.optimizer = build_optimizer(
            [model],
            lr=lewm_config.lr,
            weight_decay=lewm_config.weight_decay,
            optimizer="adamw",
        )
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=lewm_config.warmup_epochs * iterations_per_epoch,
            start_lr=lewm_config.start_lr,
            ref_lr=lewm_config.lr,
            final_lr=lewm_config.final_lr,
            T_max=total_steps,
        )
        self.sigreg = SIGReg(knots=lewm_config.sigreg_knots, num_proj=lewm_config.sigreg_num_proj).to(self.device)

    def train_step(self, batch: dict) -> dict:
        cfg = self.lewm_config
        # NaN actions occur at sequence boundaries — clobber them
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        self.optimizer.zero_grad(set_to_none=True)
        lr_now = self.scheduler.step()

        with self.autocast:
            out = self.model.encode(batch)
            emb = out["emb"]
            act_emb = out["act_emb"]

            ctx_emb = emb[:, : cfg.history_size]
            ctx_act = act_emb[:, : cfg.history_size]
            tgt_emb = emb[:, cfg.num_preds :]
            pred_emb = self.model.predict(ctx_emb, ctx_act)

            loss_pred = (pred_emb - tgt_emb).pow(2).mean()
            loss_sig = self.sigreg(emb.transpose(0, 1))
            loss = loss_pred + cfg.sigreg_weight * loss_sig

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = self.clip_grads(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": loss.detach(),
            "loss_pred": loss_pred.detach(),
            "loss_sigreg": loss_sig.detach(),
            "lr": lr_now,
            "grad_norm": grad_norm or 0.0,
        }

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        cfg = self.lewm_config
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        out = self.model.encode(batch)
        emb = out["emb"]
        act_emb = out["act_emb"]
        ctx_emb = emb[:, : cfg.history_size]
        ctx_act = act_emb[:, : cfg.history_size]
        tgt_emb = emb[:, cfg.num_preds :]
        pred_emb = self.model.predict(ctx_emb, ctx_act)
        return {"loss_pred": (pred_emb - tgt_emb).pow(2).mean()}


def train_lewm(
    model: LeWM,
    train_loader,
    *,
    epochs: int = 100,
    lr: float = 1e-4,
    sigreg_weight: float = 1.0,
    history_size: int = 3,
    output_dir: str = "runs/lewm",
    val_loader=None,
    device: str = "auto",
    amp_dtype: str = "auto",
    **extra,
):
    """One-call LeWM training. Returns final TrainState."""
    train_cfg = TrainConfig(output_dir=output_dir, device=device, epochs=epochs, amp_dtype=amp_dtype, **extra)
    lewm_cfg = LeWMConfig(lr=lr, sigreg_weight=sigreg_weight, history_size=history_size)
    ipe = len(train_loader) if hasattr(train_loader, "__len__") else 1000
    trainer = LeWMTrainer(model, train_cfg, lewm_cfg, iterations_per_epoch=ipe)
    return trainer.fit(train_loader, val_loader)
