"""Finetune / linear-probe a pre-trained JEPA encoder for downstream tasks.

Workflow:
    1. Load a pre-trained encoder.
    2. Freeze it (linear/attentive probe) or keep it trainable (end-to-end).
    3. Attach a small head (linear or attentive pooler + linear).
    4. Train on labeled data with standard cross-entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.pooler import AttentiveClassifier
from ..optim.factory import build_optimizer
from ..optim.schedulers import WarmupCosineSchedule
from .base import BaseTrainer, TrainConfig


@dataclass
class FinetuneConfig:
    num_classes: int = 1000
    head: str = "attentive"          # "linear" | "attentive"
    pooling: str = "mean"            # used when head == "linear"; "mean" | "cls" | "max"
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    freeze_encoder: bool = True
    label_smoothing: float = 0.0
    mixup: float = 0.0


class _LinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, pooling: str = "mean") -> None:
        super().__init__()
        self.pooling = pooling
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            pooled = tokens
        elif self.pooling == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pooling == "max":
            pooled = tokens.amax(dim=1)
        elif self.pooling == "cls":
            pooled = tokens[:, 0]
        else:
            raise ValueError(f"unknown pooling {self.pooling}")
        return self.linear(pooled)


class LinearProbeTrainer(BaseTrainer):
    """Supervised classification head on top of a (frozen) JEPA encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        train_config: TrainConfig,
        ft_config: FinetuneConfig,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        iterations_per_epoch: int = 1000,
    ) -> None:
        encoder = encoder
        embed_dim = embed_dim or getattr(encoder, "embed_dim", None) or getattr(getattr(encoder, "backbone", None), "embed_dim", None)
        num_heads = num_heads or getattr(encoder, "num_heads", None) or getattr(getattr(encoder, "backbone", None), "num_heads", None) or 12
        if embed_dim is None:
            raise ValueError("Could not infer embed_dim from encoder; pass it explicitly.")

        if ft_config.head == "linear":
            head = _LinearHead(embed_dim, ft_config.num_classes, pooling=ft_config.pooling)
        elif ft_config.head == "attentive":
            head = AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_classes=ft_config.num_classes,
            )
        else:
            raise ValueError(f"unknown head {ft_config.head!r}")

        class _Composite(nn.Module):
            def __init__(self, enc: nn.Module, hd: nn.Module, freeze: bool) -> None:
                super().__init__()
                self.encoder = enc
                self.head = hd
                if freeze:
                    for p in self.encoder.parameters():
                        p.requires_grad = False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if not any(p.requires_grad for p in self.encoder.parameters()):
                    with torch.no_grad():
                        feats = self.encoder(x)
                else:
                    feats = self.encoder(x)
                return self.head(feats)

        composite = _Composite(encoder, head, freeze=ft_config.freeze_encoder)
        super().__init__(composite, train_config)
        self.ft_config = ft_config

        params = composite.head.parameters()
        if not ft_config.freeze_encoder:
            params = composite.parameters()
        self.optimizer = build_optimizer(
            [nn.ModuleList([composite.head])] if ft_config.freeze_encoder else [composite],
            lr=ft_config.lr,
            weight_decay=ft_config.weight_decay,
            optimizer="adamw",
        )
        total = iterations_per_epoch * train_config.epochs
        self.scheduler = WarmupCosineSchedule(
            self.optimizer,
            warmup_steps=ft_config.warmup_epochs * iterations_per_epoch,
            start_lr=ft_config.lr / 10,
            ref_lr=ft_config.lr,
            T_max=total,
        )

    def train_step(self, batch) -> dict:
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        self.optimizer.zero_grad(set_to_none=True)
        lr_now = self.scheduler.step()

        if self.ft_config.mixup > 0:
            lam = torch.distributions.Beta(self.ft_config.mixup, self.ft_config.mixup).sample().item()
            perm = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[perm]
            y_a, y_b = y, y[perm]
        else:
            y_a = y_b = y
            lam = 1.0

        with self.autocast:
            logits = self.model(x)
            loss_a = F.cross_entropy(logits, y_a, label_smoothing=self.ft_config.label_smoothing)
            loss_b = F.cross_entropy(logits, y_b, label_smoothing=self.ft_config.label_smoothing) if lam < 1.0 else 0.0
            loss = lam * loss_a + (1 - lam) * loss_b if lam < 1.0 else loss_a

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.clip_grads(self.model.parameters())
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()

        return {"loss": loss.detach(), "acc": acc, "lr": lr_now}

    @torch.no_grad()
    def val_step(self, batch) -> dict:
        x, y = batch if isinstance(batch, (list, tuple)) else (batch["image"], batch["label"])
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean()
        return {"loss": loss, "acc": acc}


def finetune_classifier(
    encoder: nn.Module,
    train_loader,
    *,
    num_classes: int,
    head: str = "attentive",
    freeze_encoder: bool = True,
    epochs: int = 30,
    lr: float = 5e-4,
    output_dir: str = "runs/finetune",
    val_loader=None,
    device: str = "auto",
    **extra,
):
    """One-call classifier finetune."""
    train_cfg = TrainConfig(output_dir=output_dir, device=device, epochs=epochs, **extra)
    ft_cfg = FinetuneConfig(num_classes=num_classes, head=head, freeze_encoder=freeze_encoder, lr=lr)
    ipe = len(train_loader) if hasattr(train_loader, "__len__") else 1000
    trainer = LinearProbeTrainer(encoder, train_cfg, ft_cfg, iterations_per_epoch=ipe)
    return trainer.fit(train_loader, val_loader)
