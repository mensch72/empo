#!/usr/bin/env python3
"""
Phase 2 PPO Robot Policy Demo (PufferLib-backed).

Demonstrates computing the robot policy using PufferLib PPO on a MultiGrid
environment.  This is the PPO counterpart to the existing DQN-based
``phase2_robot_policy_demo.py``.

Environment: Tiny 4×6 grid with 1 human, 1 robot, and a rock.
- Human tries to reach goals (uniform random policy prior).
- Robot learns to act using PPO with intrinsic EMPO reward U_r(s).
- Auxiliary networks (V_h^e, X_h, U_r) are trained alongside PPO.

Usage:
    # Inside Docker container (make shell):
    python examples/phase2/phase2_ppo_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_demo.py

    # Quick smoke test (2 iterations):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_demo.py --iters 2

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Path setup (allows running the script directly without ``pip install -e``)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)
for _subdir in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    _path = os.path.join(_PROJECT_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# EMPO / MultiGrid imports
# ---------------------------------------------------------------------------
from gym_multigrid.multigrid import MultiGridEnv, World  # noqa: E402

from empo.learning_based.phase2_ppo.config import PPOPhase2Config  # noqa: E402
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer  # noqa: E402
from empo.learning_based.multigrid.phase2_ppo import (  # noqa: E402
    MultiGridWorldModelEnv,
    create_multigrid_ppo_networks,
)

# ======================================================================
# Environment helpers
# ======================================================================

GRID_MAP = """
We We We We We We
We Ae Ro .. .. We
We We Ay We We We
We We We We We We
"""


def _create_world_model(max_steps: int = 20) -> MultiGridEnv:
    """Create the trivial MultiGrid world model."""
    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=max_steps,
        partial_obs=False,
        objects_set=World,
    )
    return env


def _uniform_human_policy(state, human_idx, goal, world_model):
    """Uniform random policy prior (placeholder)."""
    n = world_model.action_space.n
    return [1.0 / n] * n


def _dummy_goal_sampler(state, human_idx):
    """Returns a dummy goal and unit weight."""
    return f"goal_{human_idx}", 1.0


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 PPO demo (MultiGrid)")
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of PPO iterations (default: 50)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of vectorised environments (default: 4)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="State encoder feature dimension (default: 64)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Actor-critic hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Create a reference environment (used to infer grid size, etc.)
    # ------------------------------------------------------------------
    ref_env = _create_world_model()
    ref_env.reset()

    num_actions = ref_env.action_space.n
    human_indices = ref_env.human_agent_indices
    robot_indices = ref_env.robot_agent_indices
    print(f"Grid: {ref_env.height}×{ref_env.width}")
    print(f"Actions: {num_actions}")
    print(f"Human agents: {human_indices}")
    print(f"Robot agents: {robot_indices}")

    # ------------------------------------------------------------------
    # 2. Configuration
    # ------------------------------------------------------------------
    cfg = PPOPhase2Config(
        # Theory
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,
        xi=1.0,
        eta=1.1,
        # PPO
        num_actions=num_actions,
        num_robots=len(robot_indices),
        hidden_dim=args.hidden_dim,
        ppo_rollout_length=32,
        ppo_num_minibatches=2,
        ppo_update_epochs=2,
        num_envs=args.num_envs,
        num_ppo_iterations=args.iters,
        lr_ppo=3e-4,
        # Auxiliary
        aux_training_steps_per_iteration=5,
        aux_buffer_size=5_000,
        batch_size=32,
        reward_freeze_interval=5,
        # Environment
        steps_per_episode=20,
        # Runtime
        device=args.device,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 3. Create networks (shared encoder for state → observation)
    # ------------------------------------------------------------------
    actor_critic, aux_nets, state_encoder = create_multigrid_ppo_networks(
        env=ref_env,
        config=cfg,
        feature_dim=args.feature_dim,
        use_x_h=True,
        use_u_r=True,
        device=args.device,
    )
    print(f"State encoder feature_dim: {state_encoder.feature_dim}")
    print(f"Actor-critic joint actions: {actor_critic.num_joint_actions}")

    # ------------------------------------------------------------------
    # 4. Build the trainer
    # ------------------------------------------------------------------
    trainer = PPOPhase2Trainer(
        actor_critic=actor_critic,
        auxiliary_networks=aux_nets,
        config=cfg,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 5. Define env_creator (one per vectorised slot)
    # ------------------------------------------------------------------
    # The state encoder is shared (torch module, same parameters across
    # all envs in Serial backend).
    _enc = state_encoder

    def env_creator():
        wm = _create_world_model()
        wm.reset()
        return MultiGridWorldModelEnv(
            world_model=wm,
            human_policy_prior=_uniform_human_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=human_indices,
            robot_agent_indices=robot_indices,
            config=cfg,
            state_encoder=_enc,
            # auxiliary_networks injected by trainer.train()
        )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"\nStarting PPO training for {args.iters} iterations …")
    metrics = trainer.train(env_creator, num_iterations=args.iters)

    # ------------------------------------------------------------------
    # 7. Report results
    # ------------------------------------------------------------------
    if metrics:
        last = metrics[-1]
        print(f"\nTraining complete ({len(metrics)} iterations).")
        print(f"  Final policy_loss : {last.get('policy_loss', 'N/A')}")
        print(f"  Final value_loss  : {last.get('value_loss', 'N/A')}")
        print(f"  Final v_h_e_loss  : {last.get('v_h_e_loss', 'N/A')}")
        print(f"  Final x_h_loss    : {last.get('x_h_loss', 'N/A')}")
        print(f"  Final u_r_loss    : {last.get('u_r_loss', 'N/A')}")
        print(f"  Global env steps  : {trainer.global_env_step}")
    else:
        print("No metrics returned (training may have been too short).")


if __name__ == "__main__":
    main()
