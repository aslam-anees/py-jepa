"""Example 05 — Model-Predictive Control with LeWM planners.

Demonstrates:
- Random Shooting, CEM, and MPPI planners
- WorldModelPolicy for stateful per-step planning
- The planning loop (observe -> plan -> act -> observe)

Key design note: build_lewm() history_size must match ARPredictor num_frames.
The planner's horizon should equal the context window (history_size).
"""

import torch

from pyjepa import (
    CEMPlanner,
    MPPIPlanner,
    PlanConfig,
    RandomShootingPlanner,
    build_lewm,
    get_device,
    vit_tiny,
)
from pyjepa.planning.policy import WorldModelPolicy

# Model settings
IMG_SIZE = 32
PATCH_SIZE = 8
ACTION_DIM = 4
HISTORY_SIZE = 2   # ARPredictor is built with num_frames=HISTORY_SIZE
EMB_DIM = 64


def build_model(device):
    encoder = vit_tiny(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    model = build_lewm(
        encoder,
        action_dim=ACTION_DIM,
        history_size=HISTORY_SIZE,
        emb_dim=EMB_DIM,
        action_emb_dim=16,
        pred_depth=1,
        pred_heads=2,
        pred_mlp_dim=128,
        projector_hidden=128,
    )
    return model.to(device).eval()


def make_info(device, batch_size=1):
    """Simulate an observation window + goal image."""
    pixels = torch.randn(batch_size, HISTORY_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
    goal = torch.randn(batch_size, 1, 3, IMG_SIZE, IMG_SIZE, device=device)
    return {"pixels": pixels, "goal": goal}


def main():
    device = get_device()
    print(f"Device: {device}")

    model = build_model(device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"LeWM: {n:.2f}M params  history_size={HISTORY_SIZE}")

    # --- Shared planner config ---
    cfg = PlanConfig(
        horizon=HISTORY_SIZE,   # match the model's context window
        action_dim=ACTION_DIM,
        num_samples=16,
        num_elites=4,
        num_iters=3,
        action_min=-1.0,
        action_max=1.0,
        temperature=0.1,
    )

    info = make_info(device)

    # ---- Random Shooting ----
    print("\n--- Random Shooting ---")
    rs = RandomShootingPlanner(cfg)
    action = rs(model, info)
    print(f"Best action sequence: {action.shape}  (H={cfg.horizon}, A={ACTION_DIM})")

    # ---- CEM ----
    print("\n--- Cross-Entropy Method (CEM) ---")
    cem = CEMPlanner(cfg)
    action_cem = cem(model, info)
    print(f"CEM action: {action_cem.shape}")

    # Warm-start CEM from random shooting result
    action_cem2 = cem(model, info, action_init=action_cem)
    print(f"CEM (warm-started): {action_cem2.shape}")

    # ---- MPPI ----
    print("\n--- MPPI ---")
    mppi = MPPIPlanner(cfg)
    action_mppi = mppi(model, info)
    print(f"MPPI action: {action_mppi.shape}")

    # ---- WorldModelPolicy (stateful) ----
    print("\n--- WorldModelPolicy (stateful MPC loop) ---")
    policy = WorldModelPolicy(
        model=model,
        solver=CEMPlanner(cfg),
        config=cfg,
    )

    print("Simulating 3 environment steps...")
    for step in range(3):
        obs = torch.randn(1, HISTORY_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)
        goal = torch.randn(1, 1, 3, IMG_SIZE, IMG_SIZE, device=device)
        current_info = {"pixels": obs, "goal": goal}

        action_step = policy.act(current_info)
        print(f"  step {step}: action[0]={action_step[0, 0].item():.4f}  "
              f"(shape={action_step.shape})")

    print("\nAll planners completed successfully.")


if __name__ == "__main__":
    main()
