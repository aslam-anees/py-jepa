"""pyjepa — a unified library for JEPA, V-JEPA, LeWM, and physical-intelligence research.

Quick start::

    import pyjepa

    # 1. Build a model
    model = pyjepa.create_jepa("ijepa", encoder="vit_small", img_size=224)

    # 2. Train on your data
    pyjepa.train_ijepa(model, train_loader, epochs=100)

    # 3. Embed
    feats = pyjepa.embed_image(model.encoder, images)

For world-modeling / planning, see :func:`build_lewm`, :func:`train_lewm`, and
the :mod:`pyjepa.planning` subpackage.

The library is organized so that every level (models, losses, training, planning)
can be used independently — if you have a custom data pipeline or optimizer,
you can drop only the model + loss into your own loop.
"""

from ._version import __version__

# Re-export the most useful entry points at the top level
from .device import (
    DeviceInfo,
    autocast,
    device_info,
    get_device,
    get_grad_scaler,
    seed_all,
    to_device,
)
from .inference import (
    Encoder,
    Rollout,
    embed_image,
    embed_video,
    rollout_latent,
    surprise_score,
)
from .losses import JEPALoss, SIGReg, VICRegLoss, VarianceRegularizer, prediction_loss
from .masks import (
    MultiBlock3DMaskCollator,
    MultiBlockMaskCollator,
    RandomPatchMaskCollator,
    RandomTubeMaskCollator,
)
from .models import (
    IJEPA,
    LeWM,
    VJEPA,
    ARPredictor,
    ActionEmbedder,
    MLPProjector,
    MultiMaskWrapper,
    PredictorMultiMaskWrapper,
    VisionTransformer,
    VisionTransformerPredictor,
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
    vit_base,
    vit_giant,
    vit_huge,
    vit_large,
    vit_small,
    vit_tiny,
)
from .nn import AttentiveClassifier, AttentivePooler
from .optim import (
    CosineWDSchedule,
    EMA,
    LinearSchedule,
    MomentumScheduler,
    WarmupCosineSchedule,
    build_optimizer,
    build_param_groups,
)
from .planning import (
    CEMConfig,
    CEMPlanner,
    MPPIConfig,
    MPPIPlanner,
    PlanConfig,
    RandomPolicy,
    RandomShootingPlanner,
    WorldModelPolicy,
)
from .training import (
    BaseTrainer,
    IJEPATrainer,
    LeWMTrainer,
    LinearProbeTrainer,
    TrainConfig,
    TrainState,
    VJEPATrainer,
    finetune_classifier,
    train_ijepa,
    train_lewm,
    train_vjepa,
)
from .utils import Config, load_config, save_config

# Avoid re-import for tab completion; import these lazily
from . import data, inference, losses, masks, models, nn, planning, training, utils  # noqa: F401

__all__ = [
    "ARPredictor",
    "ActionEmbedder",
    "AttentiveClassifier",
    "AttentivePooler",
    "BaseTrainer",
    "CEMConfig",
    "CEMPlanner",
    "Config",
    "CosineWDSchedule",
    "DeviceInfo",
    "EMA",
    "Encoder",
    "IJEPA",
    "IJEPATrainer",
    "JEPALoss",
    "LeWM",
    "LeWMTrainer",
    "LinearProbeTrainer",
    "LinearSchedule",
    "MLPProjector",
    "MPPIConfig",
    "MPPIPlanner",
    "MomentumScheduler",
    "MultiBlock3DMaskCollator",
    "MultiBlockMaskCollator",
    "MultiMaskWrapper",
    "PlanConfig",
    "PredictorMultiMaskWrapper",
    "RandomPatchMaskCollator",
    "RandomPolicy",
    "RandomShootingPlanner",
    "RandomTubeMaskCollator",
    "Rollout",
    "SIGReg",
    "TrainConfig",
    "TrainState",
    "VICRegLoss",
    "VJEPA",
    "VJEPATrainer",
    "VarianceRegularizer",
    "VisionTransformer",
    "VisionTransformerPredictor",
    "WarmupCosineSchedule",
    "WorldModelPolicy",
    "__version__",
    "autocast",
    "available_encoders",
    "available_jepas",
    "available_predictors",
    "build_lewm",
    "build_optimizer",
    "build_param_groups",
    "create_encoder",
    "create_jepa",
    "create_predictor",
    "device_info",
    "embed_image",
    "embed_video",
    "finetune_classifier",
    "get_device",
    "get_grad_scaler",
    "load_config",
    "prediction_loss",
    "register_encoder",
    "register_jepa",
    "register_predictor",
    "rollout_latent",
    "save_config",
    "seed_all",
    "surprise_score",
    "to_device",
    "train_ijepa",
    "train_lewm",
    "train_vjepa",
    "vit_base",
    "vit_giant",
    "vit_huge",
    "vit_large",
    "vit_small",
    "vit_tiny",
]
