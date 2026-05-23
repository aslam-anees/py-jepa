"""Command-line interface — wraps the most common workflows.

Subcommands::

    pyjepa info                        # print device / build info
    pyjepa list                        # list registered models/encoders/predictors
    pyjepa train ijepa --config ...    # train I-JEPA from a YAML config
    pyjepa train vjepa --config ...    # train V-JEPA
    pyjepa train lewm  --config ...    # train LeWM world model
    pyjepa finetune    --config ...    # linear/attentive probe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._version import __version__


def _cmd_info(args: argparse.Namespace) -> int:
    from .device import device_info, get_device

    info = device_info(get_device(args.device or "auto"))
    print(info)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    from .models import available_encoders, available_jepas, available_predictors

    print("Encoders:   ", ", ".join(available_encoders()))
    print("Predictors: ", ", ".join(available_predictors()))
    builtins = ["ijepa", "vjepa", "lewm"]
    extras = available_jepas()
    print("Models:     ", ", ".join(builtins + extras))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .utils.config import load_config
    from .utils.logging import get_logger

    logger = get_logger()
    cfg = load_config(args.config)

    if args.kind == "ijepa":
        return _run_ijepa(cfg, logger)
    if args.kind == "vjepa":
        return _run_vjepa(cfg, logger)
    if args.kind == "lewm":
        return _run_lewm(cfg, logger)
    raise ValueError(f"unknown training kind {args.kind!r}")


def _run_ijepa(cfg, logger) -> int:
    import torch
    from torch.utils.data import DataLoader

    from .data import ImageDataset, build_image_transform
    from .masks import MultiBlockMaskCollator
    from .models import create_jepa
    from .training import train_ijepa

    transform = build_image_transform(img_size=cfg.get("img_size", 224))
    dataset = ImageDataset(cfg.data.root, transform=transform)
    collator = MultiBlockMaskCollator(
        cfgs_mask=cfg.get("masks", [{"spatial_scale": [0.15, 0.2]}, {"spatial_scale": [0.15, 0.2]}]),
        crop_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
        drop_last=True,
    )

    model = create_jepa(
        "ijepa",
        encoder=cfg.get("encoder", "vit_small"),
        predictor=cfg.get("predictor", "vit_predictor"),
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        pred_depth=cfg.get("pred_depth", 6),
        pred_embed_dim=cfg.get("pred_embed_dim", 384),
        num_mask_tokens=len(cfg.get("masks", [{}, {}])),
    )
    logger.info(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_ijepa(
        model,
        loader,
        epochs=cfg.get("epochs", 100),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.04),
        reg_coeff=cfg.get("reg_coeff", 0.0),
        output_dir=cfg.get("output_dir", "runs/ijepa"),
        device=cfg.get("device", "auto"),
    )
    return 0


def _run_vjepa(cfg, logger) -> int:
    from torch.utils.data import DataLoader

    from .data import VideoDataset, build_video_transform
    from .masks import MultiBlock3DMaskCollator, RandomTubeMaskCollator
    from .models import create_jepa
    from .training import train_vjepa

    transform = build_video_transform(img_size=cfg.get("img_size", 224))
    dataset = VideoDataset(
        cfg.data.root,
        num_frames=cfg.get("num_frames", 16),
        frame_stride=cfg.get("frame_stride", 4),
        transform=transform,
    )

    mask_type = cfg.get("mask_type", "multiblock3d")
    masks_cfg = cfg.get("masks", [{"spatial_scale": [0.15, 0.2], "temporal_scale": [1.0, 1.0]}])
    if mask_type == "tube":
        collator = RandomTubeMaskCollator(
            cfgs_mask=masks_cfg, crop_size=cfg.get("img_size", 224),
            num_frames=cfg.get("num_frames", 16), patch_size=cfg.get("patch_size", 16),
            tubelet_size=cfg.get("tubelet_size", 2),
        )
    else:
        collator = MultiBlock3DMaskCollator(
            cfgs_mask=masks_cfg, crop_size=cfg.get("img_size", 224),
            num_frames=cfg.get("num_frames", 16), patch_size=cfg.get("patch_size", 16),
            tubelet_size=cfg.get("tubelet_size", 2),
        )

    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collator,
        drop_last=True,
    )

    model = create_jepa(
        "vjepa",
        encoder=cfg.get("encoder", "vit_base"),
        predictor=cfg.get("predictor", "vit_predictor"),
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        num_frames=cfg.get("num_frames", 16),
        tubelet_size=cfg.get("tubelet_size", 2),
        pred_depth=cfg.get("pred_depth", 6),
        pred_embed_dim=cfg.get("pred_embed_dim", 384),
        num_mask_tokens=len(masks_cfg),
    )
    logger.info(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_vjepa(
        model,
        loader,
        epochs=cfg.get("epochs", 100),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.04),
        reg_coeff=cfg.get("reg_coeff", 0.0),
        output_dir=cfg.get("output_dir", "runs/vjepa"),
        device=cfg.get("device", "auto"),
    )
    return 0


def _run_lewm(cfg, logger) -> int:
    from torch.utils.data import DataLoader

    from .data import HDF5TrajectoryDataset
    from .models import build_lewm, create_encoder
    from .training import train_lewm

    dataset = HDF5TrajectoryDataset(
        cfg.data.path,
        window=cfg.get("window", 8),
        frameskip=cfg.get("frameskip", 1),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
    )

    encoder = create_encoder(cfg.get("encoder", "vit_small"), img_size=cfg.get("img_size", 64))
    model = build_lewm(
        encoder,
        action_dim=cfg.action_dim,
        history_size=cfg.get("history_size", 3),
        pred_depth=cfg.get("pred_depth", 6),
        emb_dim=cfg.get("emb_dim", 384),
    )
    logger.info(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    train_lewm(
        model,
        loader,
        epochs=cfg.get("epochs", 100),
        lr=cfg.get("lr", 1e-4),
        sigreg_weight=cfg.get("sigreg_weight", 1.0),
        history_size=cfg.get("history_size", 3),
        output_dir=cfg.get("output_dir", "runs/lewm"),
        device=cfg.get("device", "auto"),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyjepa", description="JEPA / world-model library")
    parser.add_argument("--version", action="version", version=f"pyjepa {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show device info and exit")
    p_info.add_argument("--device", type=str, default=None)
    p_info.set_defaults(func=_cmd_info)

    p_list = sub.add_parser("list", help="List registered encoders / predictors / models")
    p_list.set_defaults(func=_cmd_list)

    p_train = sub.add_parser("train", help="Train a JEPA model from a YAML config")
    p_train.add_argument("kind", choices=["ijepa", "vjepa", "lewm"])
    p_train.add_argument("--config", "-c", required=True, type=Path)
    p_train.set_defaults(func=_cmd_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
