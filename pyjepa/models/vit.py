"""Vision Transformer encoder — handles 2D images and 3D video uniformly.

This is the same model used by I-JEPA and V-JEPA — the only difference is the
patch embed (Conv2d vs Conv3d) and the positional embedding (2D vs 3D sincos).
We auto-switch based on ``num_frames``.

Design choices:
- No cls token. JEPA-style models pool with an attentive pooler downstream.
- Fixed (non-learned) sincos positional embedding. Lets us interpolate to new
  resolutions at fine-tune / inference time.
- Position embedding is added *before* masking, so masked indices keep their
  positional information when re-introduced at the predictor.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.attention import Block
from ..nn.patch_embed import PatchEmbed, PatchEmbed3D
from ..nn.pos_embed import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from ..utils.tensors import apply_masks, trunc_normal_


class VisionTransformer(nn.Module):
    """Vision Transformer for images (``num_frames == 1``) or video.

    The forward signature accepts an optional list of integer masks. When given,
    only the selected tokens are processed — this is what makes I-JEPA training
    memory-efficient (we never run the encoder on masked-out patches).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        out_layers: Optional[list[int]] = None,
        uniform_power: bool = False,
        use_sdpa: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
            )
            self.num_patches = (
                (num_frames // tubelet_size) * (img_size // patch_size) * (img_size // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.uniform_power = uniform_power
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
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
        self.norm = norm_layer(embed_dim)

        self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
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
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self) -> None:
        """Following the original ViT init trick: scale each layer by 1/sqrt(2*layer_id)
        to keep activations bounded as depth grows."""
        for layer_id, layer in enumerate(self.blocks, start=1):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def get_num_layers(self) -> int:
        return len(self.blocks)

    def no_weight_decay(self) -> set:
        return set()

    def interpolate_pos_encoding(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """Resize the pre-computed pos-embed to match an unexpected input resolution.

        Lets you pre-train at 224 and fine-tune at 448 without recreating the
        encoder.
        """
        _, N, D = pos_embed.shape
        if self.is_video:
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, "Positional embedding initialized incorrectly"
            scale_factor = (T / N_t, H / N_h, W / N_w)
            pos_embed = F.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, D).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode="trilinear",
                align_corners=False,
            )
            return pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, D)

        _, _, H, W = x.shape
        if H == self.input_size and W == self.input_size:
            return pos_embed
        npatch = (H // self.patch_size) * (W // self.patch_size)
        scale_factor = math.sqrt(npatch / N)
        pos_embed = F.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), D).permute(0, 3, 1, 2),
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return pos_embed.permute(0, 2, 3, 1).view(1, -1, D)

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[list[torch.Tensor]] = None,
        interpolate_pos_encoding: bool = True,
    ) -> torch.Tensor:
        """Encode an image/video into a sequence of token embeddings.

        Args:
            x: ``[B, C, H, W]`` for images or ``[B, C, T, H, W]`` for video.
            masks: optional list of context-mask indices. When provided the
                encoder only processes the kept tokens.
            interpolate_pos_encoding: if True, resize pos_embed to match x.

        Returns:
            ``[B, N, D]`` or ``[B*M, K, D]`` when masks have ``M`` variants.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        pos_embed = self.pos_embed
        if interpolate_pos_encoding:
            pos_embed = self.interpolate_pos_encoding(x, pos_embed)

        x = self.patch_embed(x)
        x = x + pos_embed

        if masks is not None:
            x = apply_masks(x, masks)

        outs: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs  # type: ignore[return-value]
        return self.norm(x)


def _make_vit(embed_dim: int, depth: int, num_heads: int, patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_tiny(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return _make_vit(embed_dim=192, depth=12, num_heads=3, patch_size=patch_size, **kwargs)


def vit_small(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return _make_vit(embed_dim=384, depth=12, num_heads=6, patch_size=patch_size, **kwargs)


def vit_base(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return _make_vit(embed_dim=768, depth=12, num_heads=12, patch_size=patch_size, **kwargs)


def vit_large(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return _make_vit(embed_dim=1024, depth=24, num_heads=16, patch_size=patch_size, **kwargs)


def vit_huge(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return _make_vit(embed_dim=1280, depth=32, num_heads=16, patch_size=patch_size, **kwargs)


def vit_giant(patch_size: int = 16, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


VIT_EMBED_DIMS: dict[str, int] = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
}
