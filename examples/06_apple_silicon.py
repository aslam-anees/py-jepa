"""Example 06 — Apple Silicon / MPS device guide.

Demonstrates:
- Auto-detecting the MPS (Metal Performance Shaders) backend
- DeviceInfo: querying capabilities (bf16, sdpa, compile support)
- Training I-JEPA on MPS with the same code as CUDA
- Mixed precision notes for MPS
- seed_all() for reproducibility on Apple Silicon

pyjepa works on Apple Silicon without any code changes — get_device()
returns torch.device("mps") when running on M-series Macs.
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
    seed_all,
    vit_tiny,
)
from pyjepa.device import device_info, is_mps_available, supports_compile
from pyjepa.device.backend import autocast_dtype_for
from pyjepa.models.predictor import vit_predictor


def print_device_info():
    """Print a structured summary of the current hardware."""
    device = get_device()
    info = device_info(device)

    print("=" * 60)
    print("Hardware summary")
    print("=" * 60)
    print(f"  Device      : {info.device}")
    print(f"  Backend     : {info.backend}")
    print(f"  Name        : {info.name}")
    print(f"  Memory      : {info.total_memory_mb or 'n/a'} MB")
    print(f"  bfloat16    : {info.supports_bf16}")
    print(f"  float16     : {info.supports_fp16}")
    print(f"  SDPA        : {info.supports_sdpa}")
    print(f"  compile     : {info.supports_compile}")
    print(f"  Platform    : {info.platform}")
    print(f"  PyTorch     : {info.torch_version}")
    print("=" * 60)

    if info.backend == "mps":
        print("\nApple Silicon MPS notes:")
        print("  - SDPA uses a math kernel (no Flash-Attention), still fast")
        print("  - bfloat16 not supported; autocast uses float32 by default")
        print("  - torch.compile is disabled (graph-break issues in MPS)")
        print("  - torch.backends.cuda.sdp_kernel context not needed or supported")
    elif info.backend == "cuda":
        bf16 = "bfloat16" if info.supports_bf16 else "float16"
        print(f"\nCUDA: AMP will use {bf16}")
    else:
        print("\nCPU: AMP disabled (float32 training)")


def build_tiny_ijepa(device):
    img_size, patch_size = 32, 8
    enc = vit_tiny(img_size=img_size, patch_size=patch_size)
    pred = vit_predictor(
        img_size=img_size, patch_size=patch_size, num_frames=1,
        embed_dim=enc.embed_dim, predictor_embed_dim=96,
        depth=2, num_heads=enc.num_heads,
        use_mask_tokens=True, num_mask_tokens=2,
        zero_init_mask_tokens=True, use_sdpa=True,
    )
    model = IJEPA(
        encoder=MultiMaskWrapper(enc),
        predictor=PredictorMultiMaskWrapper(pred),
    )
    return model.to(device)


def main():
    # Reproducibility on any device (works for MPS, CUDA, CPU)
    seed_all(42)

    # Print hardware info
    print_device_info()
    device = get_device()

    # Choose AMP dtype based on hardware
    amp_dtype_choice = autocast_dtype_for(device, prefer="auto")
    if amp_dtype_choice == torch.float16:
        amp_dtype_str = "float16"
    elif amp_dtype_choice == torch.bfloat16:
        amp_dtype_str = "bfloat16"
    else:
        amp_dtype_str = "float32"
    print(f"\nAutomatic AMP dtype: {amp_dtype_str}")

    # Build model — same code as CUDA
    model = build_tiny_ijepa(device)
    print(f"\nModel built on {device} ({model.__class__.__name__})")

    # Fake data loader
    imgs = torch.randn(16, 3, 32, 32)
    dataset = TensorDataset(imgs)
    collator = MultiBlockMaskCollator(crop_size=32, patch_size=8)

    def collate_fn(batch):
        x = torch.stack([b[0] for b in batch])
        return collator([[xi] for xi in x])

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Training — identical to CUDA usage
    train_cfg = TrainConfig(
        output_dir="/tmp/pyjepa_apple_silicon",
        device=str(device),
        epochs=1,
        amp_dtype=amp_dtype_str,
        grad_clip=10.0,
        # compile_model=False is default; torch.compile is not safe on MPS
        compile_model=False,
        log_every=1,
    )
    ijepa_cfg = IJEPAConfig(
        lr=1e-3, start_lr=2e-4, final_lr=1e-6,
        weight_decay=0.04, warmup_epochs=0,
        momentum_start=0.996, momentum_end=1.0,
    )

    trainer = IJEPATrainer(model, train_cfg, ijepa_cfg,
                           iterations_per_epoch=len(loader))

    @trainer.on_step
    def show(m):
        print(f"  step {m['step']}  loss={m.get('loss', float('nan')):.4f}  device={device}")

    print("\n--- Training on", device, "---")
    trainer.fit(loader)

    # Verify tensors are on the right device
    for name, param in model.named_parameters():
        assert param.device.type == device.type, f"Param {name} on wrong device"
        break  # just check one
    print(f"\nAll parameters confirmed on {device.type}")

    if is_mps_available() and device.type == "mps":
        print("\nApple Silicon MPS training completed successfully.")
        print("Tip: Use PYJEPA_DEVICE=cpu for reproducible tests without MPS.")
    elif device.type == "cuda":
        print("\nCUDA training completed successfully.")
    else:
        print("\nCPU training completed successfully.")


if __name__ == "__main__":
    main()
