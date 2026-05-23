"""Example 02 — V-JEPA video pre-training.

Demonstrates:
- Building a video ViT encoder (3-D tubelet embeddings)
- MultiBlock3DMaskCollator for spatiotemporal masking
- Training 2 steps on synthetic video clips

V-JEPA is architecturally identical to I-JEPA; the key differences are:
  - num_frames > 1 activates 3-D patch embedding (tubelets)
  - MultiBlock3DMaskCollator masks 3-D space-time blocks instead of 2-D patches

This example uses random synthetic clips so it runs without a video dataset.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyjepa import (
    IJEPA,
    IJEPAConfig,
    IJEPATrainer,
    MultiBlock3DMaskCollator,
    MultiMaskWrapper,
    PredictorMultiMaskWrapper,
    TrainConfig,
    get_device,
)
from pyjepa.models.predictor import vit_predictor
from pyjepa.models.vit import vit_small


# Video settings
IMG_SIZE = 64
PATCH_SIZE = 8
NUM_FRAMES = 8
TUBELET_SIZE = 2
BATCH_SIZE = 2


def build_vjepa():
    enc = vit_small(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
    )
    pred = vit_predictor(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
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


def make_video_loader():
    """Fake (B, C, T, H, W) video clip dataset."""
    clips = torch.randn(BATCH_SIZE * 4, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE)
    dataset = TensorDataset(clips)

    collator = MultiBlock3DMaskCollator(
        crop_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        num_enc_masks=1,
        num_pred_masks=4,
        enc_mask_scale=(0.9, 1.0),
        pred_mask_scale=(0.15, 0.35),
        aspect_ratio=(0.75, 1.5),
    )

    def collate_fn(batch):
        clips = torch.stack([b[0] for b in batch])
        return collator([[c] for c in clips])

    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def main():
    device = get_device()
    print(f"Device: {device}")

    model = build_vjepa().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"V-JEPA model: {n_params:.1f}M parameters")
    print(f"Temporal tokens per clip: {NUM_FRAMES // TUBELET_SIZE} x ({IMG_SIZE // PATCH_SIZE})^2 = "
          f"{(NUM_FRAMES // TUBELET_SIZE) * (IMG_SIZE // PATCH_SIZE) ** 2} total")

    loader = make_video_loader()

    train_cfg = TrainConfig(
        output_dir="/tmp/pyjepa_vjepa_example",
        device=str(device),
        epochs=1,
        amp_dtype="float32",
        grad_clip=1.0,
        log_every=1,
        save_every_epochs=1,
    )
    ijepa_cfg = IJEPAConfig(
        lr=6e-4, start_lr=1e-4, final_lr=1e-6,
        weight_decay=0.05, warmup_epochs=0,
        momentum_start=0.998, momentum_end=1.0,
    )

    trainer = IJEPATrainer(model, train_cfg, ijepa_cfg, iterations_per_epoch=len(loader))

    @trainer.on_step
    def show(m):
        print(f"  step {m['step']}  loss={m.get('loss', float('nan')):.4f}")

    print("\n--- Training V-JEPA ---")
    trainer.fit(loader)
    print("Done.")


if __name__ == "__main__":
    main()
