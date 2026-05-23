"""Example 04 — Linear probe / attentive pooler finetuning.

Demonstrates:
- Training a classification head on top of a frozen I-JEPA encoder
- Both the linear probe and the attentive pooler head
- Using LinearProbeTrainer and FinetuneConfig

In practice you would load a pretrained I-JEPA checkpoint here. This example
simulates that by using a randomly initialized encoder (to show the API).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from pyjepa import TrainConfig, get_device, vit_tiny
from pyjepa.training.finetune import FinetuneConfig, LinearProbeTrainer

NUM_CLASSES = 10
IMG_SIZE = 32
PATCH_SIZE = 8
BATCH_SIZE = 8
N_SAMPLES = 64


def make_classification_loaders():
    imgs = torch.randn(N_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (N_SAMPLES,))
    dataset = TensorDataset(imgs, labels)

    # 80/20 train/val split
    n_train = int(0.8 * len(dataset))
    train_ds = TensorDataset(imgs[:n_train], labels[:n_train])
    val_ds = TensorDataset(imgs[n_train:], labels[n_train:])

    def collate(batch):
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        return x, y

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)
    return train_loader, val_loader


def main():
    device = get_device()
    print(f"Device: {device}")

    # Encoder (normally loaded from a pretrained I-JEPA checkpoint)
    encoder = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE).to(device)
    print(f"Encoder embed_dim: {encoder.embed_dim}")

    train_loader, val_loader = make_classification_loaders()

    # --- Linear Probe ---
    print("\n--- Linear Probe ---")
    train_cfg = TrainConfig(
        output_dir="/tmp/pyjepa_finetune_linear",
        device=str(device),
        epochs=2,
        amp_dtype="float32",
        log_every=1,
    )
    ft_cfg = FinetuneConfig(
        num_classes=NUM_CLASSES,
        head="linear",           # single linear layer on mean-pooled features
        freeze_encoder=True,     # only train the head
        lr=1e-3,
        label_smoothing=0.0,
    )
    trainer = LinearProbeTrainer(encoder, train_cfg, ft_cfg)

    @trainer.on_step
    def show_linear(m):
        print(f"  [linear]  step {m['step']}  loss={m.get('loss', float('nan')):.4f}")

    trainer.fit(train_loader, val_loader)

    # --- Attentive Pooler ---
    print("\n--- Attentive Pooler (stronger than linear probe) ---")
    train_cfg_att = TrainConfig(
        output_dir="/tmp/pyjepa_finetune_attentive",
        device=str(device),
        epochs=2,
        amp_dtype="float32",
        log_every=1,
    )
    ft_cfg_att = FinetuneConfig(
        num_classes=NUM_CLASSES,
        head="attentive",        # cross-attention pooler over patch tokens
        freeze_encoder=True,
        lr=1e-3,
    )
    trainer_att = LinearProbeTrainer(encoder, train_cfg_att, ft_cfg_att)

    @trainer_att.on_step
    def show_att(m):
        print(f"  [attentive]  step {m['step']}  loss={m.get('loss', float('nan')):.4f}")

    trainer_att.fit(train_loader, val_loader)

    # Access the standalone AttentiveClassifier if needed
    from pyjepa import AttentiveClassifier
    cls = AttentiveClassifier(embed_dim=encoder.embed_dim, num_heads=3, num_classes=NUM_CLASSES)
    print(f"\nAttentiveClassifier params: {sum(p.numel() for p in cls.parameters()):,}")


if __name__ == "__main__":
    main()
