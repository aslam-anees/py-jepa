"""Example 01 — I-JEPA image pre-training quickstart.

Demonstrates:
- Building an I-JEPA model with MultiMask wrappers
- Setting up a MultiBlockMaskCollator for multi-block spatial masking
- Training 2 steps with IJEPATrainer
- Saving and loading a checkpoint
- Embedding images with the trained encoder

This example uses random synthetic data so it runs on any hardware
without requiring a real dataset.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyjepa import (
    IJEPA,
    IJEPAConfig,
    IJEPATrainer,
    MultiBlockMaskCollator,
    MultiMaskWrapper,
    PredictorMultiMaskWrapper,
    TrainConfig,
    get_device,
    vit_small,
)
from pyjepa.inference import Encoder
from pyjepa.models.predictor import vit_predictor


def build_ijepa(img_size: int = 64, patch_size: int = 8) -> IJEPA:
    """Build a small I-JEPA model for demonstration."""
    enc = vit_small(img_size=img_size, patch_size=patch_size, num_frames=1)

    pred = vit_predictor(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=1,
        embed_dim=enc.embed_dim,
        predictor_embed_dim=192,
        depth=4,
        num_heads=enc.num_heads,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=True,
    )

    return IJEPA(
        encoder=MultiMaskWrapper(enc),
        predictor=PredictorMultiMaskWrapper(pred),
    )


def make_fake_loader(img_size: int = 64, batch_size: int = 4, n_batches: int = 4):
    """Create a DataLoader that yields (images, masks_enc, masks_pred) tuples."""
    images = torch.randn(batch_size * n_batches, 3, img_size, img_size)
    dataset = TensorDataset(images)

    collator = MultiBlockMaskCollator(
        crop_size=img_size,
        patch_size=8,
        num_enc_masks=1,
        num_pred_masks=4,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.25),
        aspect_ratio=(0.75, 1.5),
    )

    def collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        return collator([[img] for img in imgs])

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def main():
    device = get_device()
    print(f"Running on device: {device}")

    # Build model
    model = build_ijepa(img_size=64, patch_size=8)
    model = model.to(device)
    print(f"Model built — encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")

    # Data
    loader = make_fake_loader(img_size=64, batch_size=4, n_batches=4)

    # Config
    train_cfg = TrainConfig(
        output_dir="/tmp/pyjepa_ijepa_example",
        device=str(device),
        epochs=1,
        amp_dtype="float32",  # safe default; use float16/bfloat16 on CUDA
        grad_clip=10.0,
        compile_model=False,
        save_every_epochs=1,
        log_every=1,
    )
    ijepa_cfg = IJEPAConfig(
        lr=1e-3,
        start_lr=2e-4,
        final_lr=1e-6,
        weight_decay=0.04,
        warmup_epochs=0,
        momentum_start=0.996,
        momentum_end=1.0,
    )

    trainer = IJEPATrainer(
        model, train_cfg, ijepa_cfg,
        iterations_per_epoch=len(loader),
    )

    # Register a callback to print metrics at each step
    @trainer.on_step
    def print_metrics(metrics):
        print(f"  step {metrics['step']}  loss={metrics.get('loss', float('nan')):.4f}")

    print("\n--- Training ---")
    trainer.fit(loader)

    # Save checkpoint
    ckpt_path = "/tmp/pyjepa_ijepa_example/ijepa_demo.pth"
    trainer.save_checkpoint(ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    # Embed some images using the trained encoder
    print("\n--- Inference ---")
    enc_wrapper = Encoder(model.encoder.backbone, device=device, normalize=True)
    test_imgs = torch.randn(3, 3, 64, 64)
    feats = enc_wrapper.embed_image(test_imgs)
    print(f"Embedding shape: {feats.shape}  (should be [3, {model.encoder.backbone.embed_dim}])")
    print(f"L2 norm (should be ~1 with normalize=True): {feats.norm(dim=-1).mean():.4f}")


if __name__ == "__main__":
    main()
