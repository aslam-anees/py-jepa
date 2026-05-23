"""Registry pattern for encoders / predictors / world models.

Lets researchers register a new architecture once and then refer to it everywhere
by string — config files, CLI, examples — without changing library code.

    from pyjepa.models import register_encoder, create_encoder

    @register_encoder("my_funky_vit")
    def my_vit(**kwargs):
        return VisionTransformer(embed_dim=512, depth=10, num_heads=8, **kwargs)

    enc = create_encoder("my_funky_vit", img_size=128)
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .action_encoder import ActionEmbedder, MLPProjector
from .ijepa import IJEPA
from .lewm import ARPredictor, LeWM
from .predictor import vit_predictor
from .vit import VIT_EMBED_DIMS, vit_base, vit_giant, vit_huge, vit_large, vit_small, vit_tiny
from .vjepa import VJEPA
from .wrappers import MultiMaskWrapper, PredictorMultiMaskWrapper

_ENCODERS: dict[str, Callable[..., nn.Module]] = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_huge": vit_huge,
    "vit_giant": vit_giant,
}

_PREDICTORS: dict[str, Callable[..., nn.Module]] = {
    "vit_predictor": vit_predictor,
    "ar_predictor": ARPredictor,
}

_JEPAS: dict[str, Callable[..., nn.Module]] = {}


def register_encoder(name: str, fn: Callable[..., nn.Module] | None = None):
    """Register an encoder factory. Usable as a decorator or direct call."""

    def _wrap(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _ENCODERS[name] = fn
        return fn

    return _wrap if fn is None else _wrap(fn)


def register_predictor(name: str, fn: Callable[..., nn.Module] | None = None):
    def _wrap(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _PREDICTORS[name] = fn
        return fn

    return _wrap if fn is None else _wrap(fn)


def register_jepa(name: str, fn: Callable[..., nn.Module] | None = None):
    def _wrap(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        _JEPAS[name] = fn
        return fn

    return _wrap if fn is None else _wrap(fn)


def available_encoders() -> list[str]:
    return sorted(_ENCODERS.keys())


def available_predictors() -> list[str]:
    return sorted(_PREDICTORS.keys())


def available_jepas() -> list[str]:
    return sorted(_JEPAS.keys())


def create_encoder(name: str, **kwargs: Any) -> nn.Module:
    """Build an encoder by registry name. Unknown names raise a helpful error."""
    if name not in _ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; available: {available_encoders()}")
    return _ENCODERS[name](**kwargs)


def create_predictor(name: str, **kwargs: Any) -> nn.Module:
    if name not in _PREDICTORS:
        raise KeyError(f"unknown predictor {name!r}; available: {available_predictors()}")
    return _PREDICTORS[name](**kwargs)


def create_jepa(
    name: str = "ijepa",
    *,
    encoder: str = "vit_base",
    predictor: str = "vit_predictor",
    img_size: int = 224,
    patch_size: int = 16,
    num_frames: int = 1,
    tubelet_size: int = 2,
    pred_depth: int = 6,
    pred_embed_dim: int = 384,
    num_mask_tokens: int = 2,
    use_mask_tokens: bool = True,
    zero_init_mask_tokens: bool = True,
    use_sdpa: bool = True,
    uniform_power: bool = False,
    wrap_multimask: bool = True,
    encoder_kwargs: dict | None = None,
    predictor_kwargs: dict | None = None,
    **extra: Any,
) -> nn.Module:
    """One-call factory for a fully wired JEPA.

    Args:
        name: ``ijepa`` (image), ``vjepa`` (video), ``lewm`` (world model), or any
            name registered via :func:`register_jepa`.
        encoder: registered encoder name.
        predictor: registered predictor name.
        wrap_multimask: wrap encoder/predictor in MultiMask wrappers (I/V-JEPA only).

    Extra kwargs are forwarded to the constructor of the JEPA wrapper.
    """
    if name in _JEPAS:
        return _JEPAS[name](
            encoder=encoder,
            predictor=predictor,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            **extra,
        )

    enc_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=use_sdpa,
        uniform_power=uniform_power,
    )
    enc_kwargs.update(encoder_kwargs or {})
    enc = create_encoder(encoder, **enc_kwargs)

    pred_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=enc.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=enc.num_heads,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
    )
    pred_kwargs.update(predictor_kwargs or {})
    pred = create_predictor(predictor, **pred_kwargs)

    if wrap_multimask:
        enc = MultiMaskWrapper(enc)
        pred = PredictorMultiMaskWrapper(pred)

    if name == "ijepa":
        return IJEPA(encoder=enc, predictor=pred)
    if name == "vjepa":
        return VJEPA(encoder=enc, predictor=pred)
    if name == "lewm":
        # LeWM has a different topology — see :func:`build_lewm` below
        raise ValueError("Use build_lewm() to construct a LeWM model")
    raise KeyError(f"unknown jepa kind {name!r}; available: {available_jepas() + ['ijepa', 'vjepa', 'lewm']}")


def build_lewm(
    encoder: nn.Module,
    *,
    action_dim: int,
    history_size: int = 3,
    pred_depth: int = 6,
    pred_heads: int = 6,
    pred_dim_head: int = 64,
    pred_mlp_dim: int = 1024,
    pred_hidden_dim: int | None = None,
    emb_dim: int = 384,
    action_emb_dim: int = 64,
    projector_hidden: int = 1024,
    use_projector: bool = True,
    use_pred_proj: bool = True,
    projector_norm: type[nn.Module] | None = nn.LayerNorm,
) -> LeWM:
    """Convenience builder for a LeWM model around any encoder.

    Provide an encoder that returns ``[B, D]`` (or HF-compatible). We auto-derive
    the latent dim from the encoder's output and wire up the predictor.

    Args:
        projector_norm: normalization layer for the projector heads. LayerNorm
            (default) is robust to batch-size-1 inference; pass ``nn.BatchNorm1d``
            to match the original LeWM paper's checkpoint format (only safe when
            you call ``.eval()`` before planning).
    """
    pred_hidden_dim = pred_hidden_dim or emb_dim
    predictor = ARPredictor(
        num_frames=history_size,
        depth=pred_depth,
        heads=pred_heads,
        dim_head=pred_dim_head,
        mlp_dim=pred_mlp_dim,
        input_dim=emb_dim,
        hidden_dim=pred_hidden_dim,
        cond_dim=action_emb_dim,
    )
    action_encoder = ActionEmbedder(input_dim=action_dim, emb_dim=action_emb_dim)
    projector = (
        MLPProjector(input_dim=emb_dim, hidden_dim=projector_hidden, output_dim=emb_dim, norm_fn=projector_norm)
        if use_projector
        else None
    )
    pred_proj = (
        MLPProjector(input_dim=emb_dim, hidden_dim=projector_hidden, output_dim=emb_dim, norm_fn=projector_norm)
        if use_pred_proj
        else None
    )
    return LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
