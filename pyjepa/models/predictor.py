"""ViT-based predictor for I-JEPA / V-JEPA.

Takes context tokens + mask-pred indices and predicts target embeddings for the
masked-out positions. Uses either learned mask tokens (default) or noisy target
diffusion (when ``use_mask_tokens=False``) as the predictor input for target
positions.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from ..nn.attention import Block
from ..nn.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from ..utils.tensors import apply_masks, repeat_interleave_batch, trunc_normal_


class VisionTransformerPredictor(nn.Module):
    """Predictor head used by I-JEPA and V-JEPA."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        use_mask_tokens: bool = True,
        num_mask_tokens: int = 2,
        zero_init_mask_tokens: bool = True,
        use_sdpa: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        self.mask_tokens: Optional[nn.ParameterList] = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)]
            )

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        if self.is_video:
            self.num_patches = (
                (num_frames // tubelet_size) * (img_size // patch_size) * (img_size // patch_size)
            )
        else:
            self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.uniform_power = uniform_power
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, predictor_embed_dim), requires_grad=False
        )

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                )
                for _ in range(depth)
            ]
        )
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self._init_pos_embed(self.predictor_pos_embed.data)
        self.init_std = init_std
        if self.mask_tokens is not None and not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        D = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                D, grid_size, grid_depth, cls_token=False, uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(D, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        for layer_id, layer in enumerate(self.predictor_blocks, start=1):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def diffusion(self, x: torch.Tensor, noise_beta=(0.5, 1.0), steps: int = 1000) -> torch.Tensor:
        """Add diffusion noise to target embeddings as predictor input.

        Used when ``use_mask_tokens=False`` (the "regressor" variant of I-JEPA).
        """
        b1, b2 = noise_beta
        alpha = 1.0
        alpha_scheduler = []
        for i in range(steps):
            beta = b1 + i * (b2 - b1) / steps
            alpha *= 1.0 - beta
            alpha_scheduler.append(alpha)

        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        return alpha ** 0.5 * x + (1.0 - alpha) ** 0.5 * torch.randn(x.shape, device=x.device)

    def forward(
        self,
        ctxt: torch.Tensor,
        tgt: torch.Tensor,
        masks_ctxt,
        masks_tgt,
        mask_index: int = 1,
    ) -> torch.Tensor:
        """Predict masked-target embeddings from masked-context embeddings.

        Args:
            ctxt: ``[B*nctx, Nctx, D]`` encoder output on context tokens.
            tgt: target tokens (only used when ``mask_tokens is None``).
            masks_ctxt / masks_tgt: list of mask index tensors ``[B, K]``.
        """
        if not isinstance(masks_ctxt, list):
            masks_ctxt = [masks_ctxt]
        if not isinstance(masks_tgt, list):
            masks_tgt = [masks_tgt]

        B = len(ctxt) // len(masks_ctxt)
        x = self.predictor_embed(ctxt)
        _, N_ctxt, D = x.shape

        ctxt_pos = self.predictor_pos_embed.repeat(B, 1, 1)
        x = x + apply_masks(ctxt_pos, masks_ctxt)

        if self.mask_tokens is None:
            pred_tokens = self.predictor_embed(tgt)
            pred_tokens = self.diffusion(pred_tokens)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt)

        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks_tgt)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt))
        pred_tokens = pred_tokens + pos_embs

        x = x.repeat(len(masks_tgt), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        x = x[:, N_ctxt:]
        return self.predictor_proj(x)


def vit_predictor(**kwargs) -> VisionTransformerPredictor:
    return VisionTransformerPredictor(
        mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
