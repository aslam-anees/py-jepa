# py-jepa

A unified Python library for Joint Embedding Predictive Architecture (JEPA) models, LeWorldModel (LeWM) world models, and physical intelligence research.

`pyjepa` provides one consistent API covering pre-training (I-JEPA, V-JEPA), end-to-end world-model training (LeWM), finetuning (linear probe, attentive pooler), model-predictive planning (CEM, MPPI), and inference — running identically on Apple Silicon (Metal/MPS), NVIDIA CUDA, and plain CPU.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Supported Hardware](#supported-hardware)
- [Training Guide](#training-guide)
- [Finetuning](#finetuning)
- [Planning](#planning)
- [Inference](#inference)
- [Custom Models](#custom-models)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Performance Notes](#performance-notes)
- [Citation](#citation)

---

## Overview

`pyjepa` unifies three families of models under one clean API:

| Model | Paper | Key idea |
|-------|-------|----------|
| **I-JEPA** | Assran et al., CVPR 2023 | Predict representations of masked image regions in latent space (not pixel space). An EMA target encoder prevents collapse. |
| **V-JEPA** | Bardes et al., arXiv 2024 | Extends I-JEPA to video with 3-D spatiotemporal masking (tube + multi-block). |
| **LeWM** | Maes, Le Lidec et al., arXiv 2026 | End-to-end JEPA world model trained from raw pixels with just two losses: prediction MSE and SIGReg. No EMA encoder. ~15 M parameters. Trains on a single GPU in hours. |

All three share the same `VisionTransformer` encoder backbone, the same optimizer and scheduler utilities, and a common checkpoint format. You can pre-train with I-JEPA, finetune a classifier head, and then adapt the encoder into a LeWM world model — with no code changes beyond swapping the training wrapper.

---

## Installation

### From PyPI

```bash
pip install pyjepa
```

Install with optional extras for your use case:

```bash
pip install pyjepa[vision]    # torchvision + Pillow for image datasets
pip install pyjepa[video]     # av + decord for video loading
pip install pyjepa[data]      # h5py + datasets for trajectory data
pip install pyjepa[train]     # wandb + tensorboard experiment tracking
pip install pyjepa[planning]  # scipy for advanced planning utilities
pip install pyjepa[all]       # everything above
```

### From Source

```bash
git clone https://github.com/your-org/py-jepa.git
cd py-jepa
pip install -e ".[dev]"       # editable install + dev tools (pytest, ruff, black)
```

### Apple Silicon (macOS, M1/M2/M3/M4)

No extra configuration needed. `pyjepa` auto-detects the MPS (Metal Performance Shaders) backend and uses it transparently. Install PyTorch 2.1+ from [pytorch.org](https://pytorch.org) with the standard macOS wheel — the MPS backend is included.

```bash
pip install torch torchvision  # MPS included in standard macOS PyTorch wheel
pip install pyjepa[vision]
```

### Windows (CUDA)

Install the CUDA-enabled PyTorch wheel for your toolkit version first:

```bash
# Example for CUDA 12.1 — see pytorch.org for the correct command for your setup
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyjepa[all]
```

---

## Quick Start

### I-JEPA Image Pre-training

```python
import torch
from torch.utils.data import DataLoader
import pyjepa

# Build model (encoder + EMA target + predictor)
model = pyjepa.create_jepa("ijepa", encoder="vit_small", img_size=224)

# Mask collator: 1 context mask + 4 prediction targets per image
collator = pyjepa.MultiBlockMaskCollator(crop_size=224, patch_size=16)

loader = DataLoader(your_dataset, batch_size=256, collate_fn=collator)

# One-call training
pyjepa.train_ijepa(model, loader, epochs=300, lr=1e-3, output_dir="./output/ijepa")
```

### V-JEPA Video Pre-training

```python
import pyjepa

model = pyjepa.create_jepa(
    "vjepa", encoder="vit_small", img_size=224,
    num_frames=16, tubelet_size=2
)
collator = pyjepa.MultiBlock3DMaskCollator(
    crop_size=224, patch_size=16, num_frames=16, tubelet_size=2
)
loader = DataLoader(your_video_dataset, batch_size=32, collate_fn=collator)
pyjepa.train_vjepa(model, loader, epochs=200, lr=6e-4, output_dir="./output/vjepa")
```

### LeWM World Model Training

```python
from pyjepa import vit_small, build_lewm, train_lewm
from pyjepa.data import TrajectoryDataset

# Build world model
encoder = vit_small(img_size=84, patch_size=12)
model = build_lewm(encoder, action_dim=4, emb_dim=384, history_size=3)

# Trajectory dataset: episodes of (observations, actions)
dataset = TrajectoryDataset(episodes, obs_key="pixels", act_key="action",
                            window=8)  # history_size + num_preds
loader = DataLoader(dataset, batch_size=128)
train_lewm(model, loader, epochs=100, lr=3e-4, sigreg_weight=0.1,
           output_dir="./output/lewm")
```

### LeWM Planning with CEM

```python
import torch
from pyjepa import build_lewm, vit_small, CEMPlanner, PlanConfig

model = build_lewm(vit_small(img_size=84, patch_size=12), action_dim=4,
                   emb_dim=384, history_size=3)
model.eval()

# Current observation window and goal image
info = {
    "pixels": torch.randn(1, 3, 3, 84, 84),   # (B, T_ctx, C, H, W)
    "goal":   torch.randn(1, 1, 3, 84, 84),   # (B, 1, C, H, W)
}

planner = CEMPlanner(PlanConfig(
    horizon=10, action_dim=4,
    num_samples=512, num_elites=64, num_iters=5,
))
best_action_seq = planner(model, info)   # (H, A)
action_to_execute = best_action_seq[0]  # first action
```

---

## Architecture Overview

### I-JEPA

I-JEPA (Image JEPA) trains a `VisionTransformer` encoder end-to-end with a masked prediction objective in latent space. Given an image, a context block is encoded by the online encoder; 4 target blocks are encoded by an EMA (exponential moving average) copy of the encoder. A lightweight cross-attention predictor maps context embeddings to predicted target embeddings.

The training objective is:

```
L_JEPA = (1/M) * sum_m ||predictor(enc(x_ctx), mask_m) - sg(enc_ema(x, mask_m))||^2
```

where `sg(·)` is stop-gradient on the EMA branch. The variance regularizer `VR(z) = mean(relu(1 - sqrt(var(z) + eps)))` can optionally penalize representation collapse.

### V-JEPA

V-JEPA extends I-JEPA to video by replacing 2-D spatial masking with 3-D spatiotemporal masking. Masks are sampled as volumetric blocks (tubes) across the temporal dimension; the predictor must recover embeddings of masked space-time regions conditioned on the visible context. The architecture is otherwise identical — the `VisionTransformer` switches to 3-D tubelet patch embeddings when `num_frames > 1`.

### LeWM (LeWorldModel)

LeWM trains a world model entirely end-to-end from pixels with two losses:

```
L = ||AR_predictor(enc(x_t), a_t) - enc(x_{t+1})||^2  +  lambda * SIGReg(enc(x))
```

**SIGReg** (Sketched Isotropic Gaussian Regularizer) prevents representation collapse by matching the empirical characteristic function of latent embeddings to that of a standard Gaussian, using random 1-D projections:

```
SIGReg(Z) = (1/K) * sum_k | (1/N) * sum_n exp(i * w_k^T z_n) - exp(-||w_k||^2/2) |^2
```

The autoregressive predictor (`ARPredictor`) is a causal Transformer conditioned on action embeddings via AdaLN-zero blocks, making next-state predictions that are used directly for model-predictive planning.

---

## Supported Hardware

| Backend | Device string | AMP dtype | `torch.compile` | Notes |
|---------|--------------|-----------|-----------------|-------|
| NVIDIA CUDA | `cuda` / `cuda:N` | `float16` or `bfloat16` (SM80+) | Supported | Flash-Attention 2 via SDPA |
| Apple Silicon MPS | `mps` | `float32` (MPS autocast experimental) | Not recommended | Math SDPA kernel; no cuda.sdp_kernel context |
| CPU | `cpu` | `float32` | Supported (slow) | Fallback for testing |
| AMD ROCm | `cuda` (HIP) | `float16` | Supported | Detected via `torch.version.hip` |
| Intel XPU | `xpu` | — | — | Basic support via `torch.xpu` |

`pyjepa` auto-selects the device via `PYJEPA_DEVICE` environment variable or `get_device()`:

```python
import pyjepa
device = pyjepa.get_device()              # auto: CUDA → MPS → CPU
device = pyjepa.get_device("mps")        # explicit
device = pyjepa.get_device("cuda:1")     # specific GPU

info = pyjepa.DeviceInfo.from_current()  # structured hardware snapshot
print(info)  # shows memory, bf16, sdpa, compile support
```

---

## Training Guide

### IJEPATrainer

```python
from pyjepa.training import IJEPATrainer, IJEPAConfig, TrainConfig

train_cfg = TrainConfig(
    output_dir="./output/ijepa",
    device="auto",          # or "cuda", "mps", "cpu"
    epochs=300,
    amp_dtype="float16",    # "float16", "bfloat16", "float32"
    grad_clip=10.0,
    compile_model=False,    # True only on CUDA; MPS not supported
    save_every_epochs=10,
    keep_last_ckpts=3,
)

ijepa_cfg = IJEPAConfig(
    lr=1e-3,
    start_lr=2e-4,
    final_lr=1e-6,
    weight_decay=0.04,
    warmup_epochs=40,
    momentum_start=0.996,   # EMA momentum schedule
    momentum_end=1.0,
    loss_exp=1.0,
    reg_coeff=0.0,          # variance regularizer weight
)

trainer = IJEPATrainer(
    model, train_cfg, ijepa_cfg,
    iterations_per_epoch=len(loader),
)
trainer.fit(loader)
```

Hook into the training loop with callbacks:

```python
@trainer.on_step
def log_step(metrics):
    print(f"step {metrics['step']}: loss={metrics['loss']:.4f}")

@trainer.on_epoch_end
def checkpoint_epoch(epoch, metrics):
    if metrics["val_loss"] < best_val:
        trainer.save_checkpoint("best.pth")
```

### LeWMTrainer

```python
from pyjepa.training import LeWMTrainer, LeWMConfig, TrainConfig

lewm_cfg = LeWMConfig(
    lr=3e-4,
    history_size=3,
    num_preds=5,
    sigreg_weight=0.1,
    sigreg_knots=17,
    sigreg_num_proj=1024,
)
trainer = LeWMTrainer(model, TrainConfig(output_dir="./output/lewm"), lewm_cfg)
trainer.fit(loader)
```

Each training batch must contain a `window = history_size + num_preds` observation window. Configure your `TrajectoryDataset` accordingly:

```python
dataset = TrajectoryDataset(episodes, obs_key="pixels", act_key="action",
                            window=history_size + num_preds)
```

### One-Call Training

For simple use cases that do not need custom callbacks:

```python
pyjepa.train_ijepa(model, loader, epochs=300, lr=1e-3, output_dir="./out")
pyjepa.train_vjepa(model, loader, epochs=200, lr=6e-4, output_dir="./out")
pyjepa.train_lewm(model, loader, epochs=100, lr=3e-4, sigreg_weight=0.1,
                  history_size=3, num_preds=5, output_dir="./out")
```

### Config-Driven Training (YAML)

```bash
pyjepa train ijepa --config configs/ijepa_base.yaml
pyjepa train vjepa --config configs/vjepa_base.yaml
pyjepa train lewm  --config configs/lewm_base.yaml
```

Load and extend configs programmatically:

```python
from pyjepa import load_config, save_config

cfg = load_config("configs/ijepa_base.yaml", overrides={"training.lr": 5e-4})
print(cfg.training.lr)     # 0.0005
print(cfg["training.lr"])  # also works
```

### Distributed Training

`pyjepa` uses `torch.distributed` for multi-GPU training. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 train.py --config configs/ijepa_base.yaml
```

Initialize in your script:

```python
from pyjepa.utils.distributed import init_distributed, AllReduce
init_distributed()  # reads RANK/WORLD_SIZE/MASTER_ADDR from environment
```

---

## Finetuning

### Linear Probe

```python
from pyjepa.training.finetune import LinearProbeTrainer, FinetuneConfig

ft_cfg = FinetuneConfig(
    num_classes=1000,
    head="linear",            # "linear" or "attentive"
    freeze_encoder=True,
    lr=1e-3,
    label_smoothing=0.1,
)
trainer = LinearProbeTrainer(ijepa_model.encoder, train_cfg, ft_cfg)
trainer.fit(train_loader, val_loader)
```

### Attentive Pooler

The attentive pooler uses learned query tokens + cross-attention over patch tokens — stronger than a single CLS token:

```python
ft_cfg = FinetuneConfig(num_classes=1000, head="attentive", freeze_encoder=True)
```

### One-Call Finetuning

```python
pyjepa.finetune_classifier(
    encoder=model.encoder,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=1000,
    head="attentive",
    freeze_encoder=True,
    epochs=100,
    lr=1e-3,
    output_dir="./output/finetune",
)
```

The `AttentivePooler` and `AttentiveClassifier` are also available as standalone modules:

```python
from pyjepa import AttentivePooler, AttentiveClassifier

pooler = AttentivePooler(embed_dim=384, num_heads=6, num_query_tokens=1)
cls = AttentiveClassifier(embed_dim=384, num_heads=6, num_classes=1000)
```

---

## Planning

All planners share a unified `PlanConfig` and implement `__call__(model, info, action_init) -> Tensor(H, A)`.

```python
from pyjepa import PlanConfig, CEMPlanner, MPPIPlanner, RandomShootingPlanner

cfg = PlanConfig(
    horizon=10,          # planning horizon (number of steps)
    action_dim=4,        # action space dimension
    num_samples=512,     # candidate trajectories per iteration
    num_elites=64,       # elite fraction (CEM only)
    num_iters=5,         # optimization iterations
    action_min=-1.0,     # action bounds
    action_max=1.0,
    temperature=0.1,     # softmax temperature (MPPI only)
)

cem  = CEMPlanner(cfg)
mppi = MPPIPlanner(cfg)
rs   = RandomShootingPlanner(cfg)
```

**Cross-Entropy Method (CEM):** Iteratively refits a diagonal Gaussian to the top-K lowest-cost candidates. Deterministic convergence, good for short horizons.

**MPPI:** Soft-min reweighting — every candidate contributes to the mean update proportional to `exp(-cost / temperature)`. No hard cutoff; smoother gradient.

**Random Shooting:** Single-shot sampling. Cheap; useful as a warm-start for CEM or as a baseline.

### WorldModelPolicy

`WorldModelPolicy` wraps any world model + planner into a stateful policy that can be stepped at each environment interaction:

```python
from pyjepa.planning import WorldModelPolicy, PlanConfig, CEMPlanner

policy = WorldModelPolicy(
    model=lewm_model,
    solver=CEMPlanner(PlanConfig(horizon=10, action_dim=4, num_samples=512)),
    config=PlanConfig(horizon=10, action_dim=4),
)

# At each environment step:
obs = env.reset()
action = policy.act({"pixels": obs_tensor, "goal": goal_tensor})
```

---

## Inference

### Embedding Images and Video

```python
from pyjepa import Encoder, embed_image, embed_video

# Stateful wrapper with optional pooling and L2 normalization
enc = Encoder(ijepa_model.encoder, device="mps", pooling="mean", normalize=True)
feats = enc.embed_image(images)    # (B, D)
feats = enc.embed_video(video)     # (B, D)

# Functional helpers (no state)
feats = embed_image(ijepa_model.encoder, images)
feats = embed_video(ijepa_model.encoder, video_clips)
```

### Latent Rollout

```python
from pyjepa import rollout_latent
from pyjepa.inference import Rollout

# Functional: roll out a full action sequence at once
initial = {"pixels": obs_tensor}  # (B, T, C, H, W)
predicted_embs = rollout_latent(lewm_model, initial, actions, history_size=3)
# returns (B, T_total, D)

# Stateful: step one action at a time
roll = Rollout(lewm_model, history_size=3)
roll.reset(initial_obs_tensor)
emb_1 = roll.step(action_1)
emb_2 = roll.step(action_2)
```

### Surprise Scoring

The surprise score measures how well the world model predicts the next observation — a proxy for physical plausibility or anomaly detection:

```python
from pyjepa import surprise_score

# Per-step prediction error along a trajectory
scores = surprise_score(lewm_model, info, history_size=3, reduction="per_step")
# (B, T)

# Mean error over whole trajectory
mean_err = surprise_score(lewm_model, info, reduction="mean")
# (B,)
```

---

## Custom Models

### Registering a Custom Encoder

```python
from pyjepa import register_encoder, create_encoder
from pyjepa.models.vit import VisionTransformer

@register_encoder("my_vit_128")
def my_vit(**kwargs):
    return VisionTransformer(
        img_size=128, patch_size=8,
        embed_dim=512, depth=10, num_heads=8,
        **kwargs
    )

# Now usable everywhere by name
enc = create_encoder("my_vit_128")
model = pyjepa.create_jepa("ijepa", encoder="my_vit_128", img_size=128, patch_size=8)
```

### Registering a Custom Predictor

```python
from pyjepa import register_predictor
import torch.nn as nn

@register_predictor("my_mlp_predictor")
class MLPPredictor(nn.Module):
    def __init__(self, embed_dim, **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                 nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x, masks_x=None, masks=None):
        return [self.net(xi) for xi in x]
```

### Listing Available Models

```python
import pyjepa
print(pyjepa.available_encoders())   # ['vit_base', 'vit_giant', 'vit_huge', ...]
print(pyjepa.available_predictors()) # ['ar_predictor', 'vit_predictor']
print(pyjepa.available_jepas())      # user-registered JEPAs
```

---

## Configuration

Example I-JEPA config (`configs/ijepa_base.yaml`):

```yaml
model:
  type: ijepa
  encoder: vit_base
  predictor: vit_predictor
  img_size: 224
  patch_size: 16
  pred_depth: 6
  pred_embed_dim: 384

mask:
  type: multiblock
  num_enc_masks: 1
  num_pred_masks: 4
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  aspect_ratio: [0.75, 1.5]

training:
  output_dir: ./output/ijepa_base
  device: auto
  epochs: 300
  batch_size: 512
  lr: 0.001
  weight_decay: 0.04
  warmup_epochs: 40
  momentum_start: 0.996
  momentum_end: 1.0
  amp_dtype: float16
  grad_clip: 10.0
  save_every_epochs: 10
```

Example LeWM config (`configs/lewm_base.yaml`):

```yaml
model:
  type: lewm
  encoder: vit_small
  img_size: 84
  patch_size: 12
  action_dim: 4
  history_size: 3
  emb_dim: 384
  action_emb_dim: 64
  pred_depth: 6
  pred_heads: 6
  pred_mlp_dim: 1024

training:
  output_dir: ./output/lewm_base
  device: auto
  epochs: 100
  batch_size: 128
  lr: 0.0003
  history_size: 3
  num_preds: 5
  sigreg_weight: 0.1
  amp_dtype: float16
  grad_clip: 1.0
```

---

## CLI Reference

```bash
# Show device and version info
pyjepa info

# List all registered encoders, predictors, and jepas
pyjepa list

# Train with a YAML config
pyjepa train ijepa --config configs/ijepa_base.yaml
pyjepa train vjepa --config configs/vjepa_base.yaml
pyjepa train lewm  --config configs/lewm_base.yaml
```

---

## Performance Notes

### Automatic Mixed Precision

`pyjepa` selects AMP dtype automatically based on hardware:

- **CUDA SM80+ (A100, H100):** `bfloat16` (more stable range, no loss scaling needed)
- **CUDA SM70- (V100, T4):** `float16` with gradient scaler
- **MPS (Apple Silicon):** `float32` (MPS autocast is experimental; disabled by default)
- **CPU:** `float32`

Override with:

```python
trainer = IJEPATrainer(model, TrainConfig(amp_dtype="bfloat16"), ijepa_cfg)
```

### torch.compile

`torch.compile` is enabled on CUDA only. On Apple Silicon MPS, it is disabled by default due to graph-break issues with attention masking. Check at runtime:

```python
from pyjepa.device import supports_compile
print(supports_compile())  # True on CUDA, False on MPS-only machines
```

### DataLoader on Apple Silicon

```python
from pyjepa.device import pin_memory_supported, get_device
device = get_device()

loader = DataLoader(
    dataset, batch_size=256,
    num_workers=4,
    pin_memory=pin_memory_supported(device),  # True only on CUDA
)
```

### Gradient Scaler

`get_grad_scaler()` returns a real `GradScaler` on CUDA with fp16, and a no-op scaler on MPS/CPU/bfloat16, so training loop code is device-agnostic:

```python
from pyjepa.device import get_grad_scaler
scaler = get_grad_scaler(device, enabled=(amp_dtype == "float16"))

with pyjepa.autocast(device, dtype=torch.float16):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## Citation

If you use `pyjepa` in your research, please cite the relevant works:

### I-JEPA

```bibtex
@inproceedings{assran2023ijepa,
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author    = {Mahmoud Assran and Quentin Duval and Ishan Misra and Piotr Bojanowski
               and Pascal Vincent and Michael Rabbat and Yann LeCun and Nicolas Ballas},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023},
}
```

### V-JEPA

```bibtex
@article{bardes2024vjepa,
  title   = {V-JEPA: Latent Video Prediction for Visual Representation Learning},
  author  = {Adrien Bardes and Quentin Garrido and Jean Ponce and Xinlei Chen
             and Michael Rabbat and Yann LeCun and Mido Assran and Nicolas Ballas},
  journal = {arXiv preprint arXiv:2404.08471},
  year    = {2024},
}
```

### LeWM (LeWorldModel)

```bibtex
@article{maes2026lewm,
  title   = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author  = {Quentin Maes and Etienne Le Lidec and others},
  journal = {arXiv preprint arXiv:2603.19312},
  year    = {2026},
  url     = {https://le-wm.github.io/},
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
