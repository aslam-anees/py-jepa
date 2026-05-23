"""Example 03 — LeWM (LeWorldModel) training.

Demonstrates:
- Building a LeWM world model with build_lewm()
- Creating a TrajectoryDataset with synthetic episodes
- Training with LeWMTrainer (MSE prediction loss + SIGReg)
- Showing per-step loss components
- Checkpoint save / load

LeWM loss = ||AR_predict(enc(x_t), a_t) - enc(x_{t+1})||^2 + lambda * SIGReg(enc)

No EMA target encoder is needed — SIGReg prevents collapse on its own.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader

from pyjepa import build_lewm, get_device, vit_tiny
from pyjepa.data import TrajectoryDataset
from pyjepa.training import LeWMConfig, LeWMTrainer, TrainConfig


# ---- Hyper-parameters (tiny for fast demo) ----
IMG_SIZE = 32
PATCH_SIZE = 8
ACTION_DIM = 4
HISTORY_SIZE = 2
NUM_PREDS = 3
WINDOW = HISTORY_SIZE + NUM_PREDS  # 5 frames per sample
EMB_DIM = 192
BATCH_SIZE = 4
N_EPISODES = 6
EP_LEN = 20


def make_fake_episodes():
    """Create synthetic episodes: list of dicts with 'pixels' and 'action'."""
    episodes = []
    for _ in range(N_EPISODES):
        T = EP_LEN
        episodes.append({
            "pixels": np.random.rand(T, 3, IMG_SIZE, IMG_SIZE).astype(np.float32),
            "action": np.random.randn(T, ACTION_DIM).astype(np.float32),
        })
    return episodes


def main():
    device = get_device()
    print(f"Device: {device}")

    # Build LeWM
    encoder = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    model = build_lewm(
        encoder,
        action_dim=ACTION_DIM,
        history_size=HISTORY_SIZE,
        emb_dim=EMB_DIM,
        action_emb_dim=32,
        pred_depth=2,
        pred_heads=3,
        pred_mlp_dim=256,
        projector_hidden=256,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"LeWM: {n_params:.2f}M parameters")
    print(f"Window = history({HISTORY_SIZE}) + preds({NUM_PREDS}) = {WINDOW} frames")

    # Dataset
    episodes = make_fake_episodes()
    dataset = TrajectoryDataset(
        episodes,
        obs_key="pixels",
        act_key="action",
        window=WINDOW,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} windows from {N_EPISODES} episodes")

    # Trainer
    train_cfg = TrainConfig(
        output_dir="/tmp/pyjepa_lewm_example",
        device=str(device),
        epochs=1,
        amp_dtype="float32",
        grad_clip=1.0,
        log_every=1,
        save_every_epochs=1,
    )
    lewm_cfg = LeWMConfig(
        lr=3e-4,
        history_size=HISTORY_SIZE,
        num_preds=NUM_PREDS,
        sigreg_weight=0.1,
        sigreg_knots=9,     # smaller for speed in demo
        sigreg_num_proj=64,
    )

    trainer = LeWMTrainer(model, train_cfg, lewm_cfg)

    @trainer.on_step
    def show(m):
        pred = m.get("loss_pred", float("nan"))
        reg = m.get("loss_sigreg", float("nan"))
        total = m.get("loss", float("nan"))
        print(f"  step {m['step']}  pred={pred:.4f}  sigreg={reg:.4f}  total={total:.4f}")

    print("\n--- Training LeWM ---")
    trainer.fit(loader)

    # Save checkpoint
    ckpt = "/tmp/pyjepa_lewm_example/lewm_demo.pth"
    trainer.save_checkpoint(ckpt)
    print(f"\nCheckpoint saved: {ckpt}")

    # Quick sanity: encode + predict
    model.eval()
    with torch.no_grad():
        px = torch.randn(1, HISTORY_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
        act = torch.randn(1, ACTION_DIM, device=device).unsqueeze(0).expand(1, HISTORY_SIZE, ACTION_DIM)
        info = {"pixels": px, "action": act}
        info = model.encode(info)
        pred = model.predict(info["emb"], info["act_emb"])
    print(f"\nEncode + predict: emb {info['emb'].shape} -> pred {pred.shape}")


if __name__ == "__main__":
    main()
