"""LeWorldModel (LeWM) — Stable End-to-End JEPA world model from pixels.

Implementation of the model described in Maes & Le Lidec et al. (2026)
"LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from
Pixels" (https://arxiv.org/abs/2603.19312, https://le-wm.github.io/).

Key idea: rather than an EMA target encoder + a wall of regularizers (DINO/IJEPA),
LeWM trains a single encoder end-to-end with just two losses:

    L = ||predict(emb_t, a_t) - emb_{t+1}||^2  +  λ * SIGReg(emb)

where ``SIGReg`` is the Sketched Isotropic Gaussian Regularizer (in
:mod:`pyjepa.losses.sigreg`) that pushes the latent distribution toward a
standard Gaussian — provably preventing collapse without an EMA or stop-gradient.

The result is a ~15M-param model trainable on a single GPU in a few hours that
matches or beats foundation-model world models on planning benchmarks (PushT,
Cube, TwoRooms, Reacher).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..nn.transformer import AdaLNBlock, Transformer
from .action_encoder import ActionEmbedder, MLPProjector


def _detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class ARPredictor(nn.Module):
    """Autoregressive predictor used by LeWM.

    Takes the last ``T`` context embeddings + corresponding action embeddings and
    predicts the next embedding via a causal Transformer conditioned on actions
    through AdaLN-zero blocks.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim or input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            block_class=AdaLNBlock,
            cond_dim=cond_dim,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, D)``  ``c``: ``(B, T, A_emb)``."""
        T = x.size(1)
        P = self.pos_embedding.size(1)
        if T <= P:
            pos = self.pos_embedding[:, :T]
        else:
            # Interpolate positional embedding to handle longer sequences at inference
            pos = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),  # (1, D, P)
                size=T,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # (1, T, D)
        x = x + pos
        x = self.dropout(x)
        return self.transformer(x, c)


class LeWM(nn.Module):
    """Latent end-to-end world model.

    Composed of:
    - ``encoder``: any image encoder returning either ``[B, D]`` or an HF-style
      object with ``.last_hidden_state[:, 0]`` (cls token).
    - ``predictor``: AR predictor on latent embeddings (typically
      :class:`ARPredictor`).
    - ``action_encoder``: embeds actions to a per-step tensor.
    - ``projector`` / ``pred_proj``: optional MLP heads applied to
      encoder/predictor outputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: Optional[nn.Module] = None,
        pred_proj: Optional[nn.Module] = None,
        history_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.history_size = history_size

    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a flat batch of pixels into a per-frame embedding ``(N, D)``."""
        # Prefer the JEPA-style call signature if available
        try:
            out = self.encoder(pixels, interpolate_pos_encoding=True)
        except TypeError:
            out = self.encoder(pixels)
        # HuggingFace ViTModel returns BaseModelOutput with last_hidden_state
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        if isinstance(out, (list, tuple)):
            out = out[-1]
        # If [B, N, D] take the mean over patch dim, else assume [B, D] already
        if out.ndim == 3:
            return out.mean(dim=1)
        return out

    def encode(self, info: dict) -> dict:
        """Encode observations (and optionally actions) into embeddings.

        Inputs:
            info["pixels"]: ``(B, T, C, H, W)`` or ``(B, C, H, W)`` (single frame)
            info["action"]: optional ``(B, T, A)``
        Outputs (in-place):
            info["emb"]:    ``(B, T, D)``
            info["act_emb"]: ``(B, T, A_emb)`` if action present.
        """
        pixels = info["pixels"].float()
        # Auto-promote single-frame (B, C, H, W) to (B, 1, C, H, W) for goal encoding
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(1)
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        pixels_emb = self._encode_pixels(pixels)
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Predict next-state embeddings autoregressively conditional on actions.

        ``emb``: ``(B, T, D)``   ``act_emb``: ``(B, T, A_emb)``  →  ``(B, T, D)``.
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    @torch.no_grad()
    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int = 3) -> dict:
        """Autoregressive rollout for planning.

        Args:
            info: must contain ``pixels`` of shape ``(B, S, T_ctx, C, H, W)``.
            action_sequence: ``(B, S, T_total, A)`` where ``T_total >= T_ctx``.
            history_size: how many past frames the predictor conditions on.

        Returns the info dict with ``predicted_emb`` of shape ``(B, S, T_total, D)``.
        """
        assert "pixels" in info, "pixels not in info dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: _detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)
            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info: dict) -> torch.Tensor:
        """Per-candidate planning cost: ``||last_pred - last_goal||^2``.

        Tolerates either ``(B, T, D)`` or ``(B, S, T, D)`` goal embeddings —
        any missing sample dim is broadcast from the predicted rollout.
        """
        pred_emb = info["predicted_emb"]   # (B, S, T_total, D)
        goal_emb = info["goal_emb"]
        # Make goal_emb broadcastable to pred_emb shape
        while goal_emb.ndim < pred_emb.ndim:
            goal_emb = goal_emb.unsqueeze(1)
        last_pred = pred_emb[..., -1:, :]
        last_goal = goal_emb[..., -1:, :].expand_as(last_pred).detach()
        cost = F.mse_loss(last_pred, last_goal, reduction="none").sum(
            dim=tuple(range(2, pred_emb.ndim))
        )
        return cost

    @torch.no_grad()
    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute planning cost for a batch of action candidates.

        ``info`` must contain ``pixels`` (current obs window) and ``goal`` (goal img).
        Any ``goal_*`` keys are mapped to ``*`` on the goal-encoding dict.
        """
        assert "goal" in info, "goal not in info dict"
        device = next(self.parameters()).device
        for k in list(info.keys()):
            if torch.is_tensor(info[k]):
                info[k] = info[k].to(device)

        goal = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for k in list(info.keys()):
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k, None)
        goal.pop("action", None)
        goal = self.encode(goal)

        info["goal_emb"] = goal["emb"]
        info = self.rollout(info, action_candidates, history_size=self.history_size)
        return self.criterion(info)
