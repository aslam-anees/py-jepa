"""High-level JEPA models: encoders, predictors, world models.

Factory functions:
    create_encoder(name, ...)        — image or video ViT encoder
    create_predictor(name, ...)      — JEPA predictor (image/video) or AR predictor (LeWM)
    create_jepa(name, ...)           — fully wired I-JEPA / V-JEPA / LeWM model

Direct imports:
    VisionTransformer, VisionTransformerPredictor,
    IJEPA, VJEPA, LeWM,
    ARPredictor, ActionEmbedder, MLPProjector
"""

from .action_encoder import ActionEmbedder, MLPProjector
from .ijepa import IJEPA
from .lewm import ARPredictor, LeWM
from .predictor import VisionTransformerPredictor
from .registry import (
    available_encoders,
    available_jepas,
    available_predictors,
    build_lewm,
    create_encoder,
    create_jepa,
    create_predictor,
    register_encoder,
    register_jepa,
    register_predictor,
)
from .vit import (
    VIT_EMBED_DIMS,
    VisionTransformer,
    vit_base,
    vit_giant,
    vit_huge,
    vit_large,
    vit_small,
    vit_tiny,
)
from .vjepa import VJEPA
from .wrappers import MultiMaskWrapper, PredictorMultiMaskWrapper

__all__ = [
    "ARPredictor",
    "ActionEmbedder",
    "IJEPA",
    "LeWM",
    "MLPProjector",
    "MultiMaskWrapper",
    "PredictorMultiMaskWrapper",
    "VIT_EMBED_DIMS",
    "VJEPA",
    "VisionTransformer",
    "VisionTransformerPredictor",
    "available_encoders",
    "available_jepas",
    "available_predictors",
    "build_lewm",
    "create_encoder",
    "create_jepa",
    "create_predictor",
    "register_encoder",
    "register_jepa",
    "register_predictor",
    "vit_base",
    "vit_giant",
    "vit_huge",
    "vit_large",
    "vit_small",
    "vit_tiny",
]
