"""Multi-head self-attention + cross-attention blocks.

We use ``torch.nn.functional.scaled_dot_product_attention`` everywhere because:
- CUDA: dispatches to Flash-Attention 2 on supported GPUs;
- MPS: uses a memory-efficient kernel as of torch 2.2;
- CPU: pure math kernel.

The original reference code wrapped SDPA in ``torch.backends.cuda.sdp_kernel`` —
we drop that context because it raises errors on MPS and is no longer needed on
recent torch versions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class Attention(nn.Module):
    """Standard multi-head self-attention. Uses SDPA on every backend."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, Dh]

        if self.use_sdpa and not return_attention:
            attn_mask = None
            if mask is not None and mask.dtype == torch.bool:
                attn_mask = mask
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                is_causal=is_causal,
            )
            attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None and mask.dtype == torch.bool:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Pre-norm transformer block (attention + MLP). The classic ViT layout."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor:
        y, attn = self.attn(self.norm1(x), mask=mask, is_causal=is_causal, return_attention=return_attention)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross-attention used by the attentive pooler."""

    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = False, use_sdpa: bool = True) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        _, N, _ = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if self.use_sdpa:
            q = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            q = attn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return self.proj(q)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with MLP and pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = q + self.xattn(q, self.norm1(x))
        q = q + self.mlp(self.norm2(q))
        return q
