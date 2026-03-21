#!/usr/bin/env python3
"""
Phase 2 PPO Robot Policy Demo — Asymmetric Freeing World.

Demonstrates Phase 2 PPO training on the "simple asymmetric freeing" challenge
from ``multigrid_worlds/jobst_challenges/asymmetric_freeing_simple.yaml``.

Environment:
    An 8×5 grid with two humans locked behind rocks and one robot.
    Human A (left) has fewer reachable goals than Human B (right).
    The robot must decide which human to free first to maximise
    aggregate human power X_h, respecting the equity parameters (ξ, η).

Human behaviour:
    Both humans use the existing ``HeuristicPotentialPolicy`` (deterministic
    goal-directed policy based on shortest-path potentials) instead of a
    uniform random prior.  This makes the demo more realistic: humans
    actively pursue goals, so V_h^e training has a meaningful signal.

Usage:
    # Inside Docker container (make shell):
    python examples/phase2/phase2_ppo_asymmetric_freeing.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_asymmetric_freeing.py

    # Quick smoke test (2 iterations):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_asymmetric_freeing.py --iters 2

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
from gym_multigrid.multigrid import MultiGridEnv, SmallActions  # noqa: E402

from empo.human_policy_prior import HeuristicPotentialPolicy  # noqa: E402
from empo.learning_based.multigrid import PathDistanceCalculator  # noqa: E402
from empo.learning_based.phase2_ppo.config import PPOPhase2Config  # noqa: E402
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer  # noqa: E402
from empo.learning_based.multigrid.phase2_ppo import (  # noqa: E402
    MultiGridWorldModelEnv,
    create_multigrid_ppo_networks,
)

# ======================================================================
# Environment helpers
# ======================================================================

_WORLD_YAML = os.path.join(
    _PROJECT_ROOT,
    "multigrid_worlds",
    "jobst_challenges",
    "asymmetric_freeing_simple.yaml",
)


def _create_world_model(max_steps: int = 20) -> MultiGridEnv:
    """Load the simple asymmetric freeing world from YAML."""
    env = MultiGridEnv(
        config_file=os.path.abspath(_WORLD_YAML),
        partial_obs=False,
        actions_set=SmallActions,
        max_steps=max_steps,
    )
    return env


def _make_heuristic_human_policy(
    ref_env: MultiGridEnv,
    human_indices: list,
    beta: float = 1000.0,
) -> HeuristicPotentialPolicy:
    """Build a heuristic policy prior for the given environment."""
    path_calc = PathDistanceCalculator(
        grid_height=ref_env.height,
        grid_width=ref_env.width,
        world_model=ref_env,
    )
    return HeuristicPotentialPolicy(
        world_model=ref_env,
        human_agent_indices=human_indices,
        path_calculator=path_calc,
        beta=beta,
    )


def _wrap_human_policy(policy: HeuristicPotentialPolicy):
    """Wrap the 3-arg ``HeuristicPotentialPolicy.__call__`` to the 4-arg
    signature expected by ``EMPOWorldModelEnv``.

    Phase 2 PPO env wrapper calls:
        ``human_policy_prior(state, h_idx, goal, world_model)``
    but ``HeuristicPotentialPolicy.__call__`` takes:
        ``(state, human_agent_index, possible_goal)``
    """

    def _policy_prior(state, human_idx, goal, world_model):
        return policy(state, human_idx, goal)

    return _policy_prior


def _make_goal_sampler(ref_env: MultiGridEnv):
    """Create a goal sampler from the YAML-configured possible goals.

    Returns a callable ``(state, human_idx) → (goal, weight)`` that
    wraps the environment's built-in ``possible_goal_sampler``.
    """
    sampler = ref_env.possible_goal_sampler
    if sampler is None:
        raise RuntimeError(
            "The world model has no possible_goal_sampler. "
            "Ensure the YAML file includes a 'possible_goals' section."
        )

    def _sampler(state, human_idx):
        return sampler.sample(state, human_idx)

    return _sampler


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 PPO demo — Asymmetric Freeing (heuristic humans)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of PPO iterations (default: 100)",
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
    # 1. Create a reference environment
    # ------------------------------------------------------------------
    ref_env = _create_world_model()
    ref_env.reset()

    num_actions = ref_env.action_space.n
    human_indices = ref_env.human_agent_indices
    robot_indices = ref_env.robot_agent_indices

    print(f"World: {os.path.basename(_WORLD_YAML)}")
    print(f"Grid : {ref_env.width}×{ref_env.height}")
    print(f"Actions: {num_actions}")
    print(f"Human agents: {human_indices}")
    print(f"Robot agents: {robot_indices}")
    print(f"Max steps: {ref_env.max_steps}")

    # ------------------------------------------------------------------
    # 2. Build heuristic human policy and goal sampler
    # ------------------------------------------------------------------
    human_policy = _make_heuristic_human_policy(ref_env, human_indices)
    human_policy_fn = _wrap_human_policy(human_policy)
    goal_sampler_fn = _make_goal_sampler(ref_env)

    print("Human policy: HeuristicPotentialPolicy (β=1000)")
    print(
        f"Goal sampler: YAML-configured ({len(ref_env.possible_goal_generator.goal_coords)} goals)"
    )

    # ------------------------------------------------------------------
    # 3. Configuration
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
        ppo_rollout_length=64,
        ppo_num_minibatches=4,
        ppo_update_epochs=4,
        num_envs=args.num_envs,
        num_ppo_iterations=args.iters,
        lr_ppo=3e-4,
        # Auxiliary
        aux_training_steps_per_iteration=5,
        aux_buffer_size=10_000,
        batch_size=64,
        reward_freeze_interval=5,
        # Warm-up (skip for this demo)
        warmup_v_h_e_steps=0,
        warmup_x_h_steps=0,
        warmup_u_r_steps=0,
        # Environment
        steps_per_episode=ref_env.max_steps,
        # Runtime
        device=args.device,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 4. Create networks
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
    # 5. Build the trainer
    # ------------------------------------------------------------------
    trainer = PPOPhase2Trainer(
        actor_critic=actor_critic,
        auxiliary_networks=aux_nets,
        config=cfg,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 6. Define env_creator
    # ------------------------------------------------------------------
    shared_encoder = state_encoder

    # Each vectorised env slot creates its own world model and
    # HeuristicPotentialPolicy instance (to avoid shared mutable state).
    def env_creator():
        wm = _create_world_model()
        wm.reset()
        hp = _make_heuristic_human_policy(wm, wm.human_agent_indices)
        hp_fn = _wrap_human_policy(hp)
        gs_fn = _make_goal_sampler(wm)
        return MultiGridWorldModelEnv(
            world_model=wm,
            human_policy_prior=hp_fn,
            goal_sampler=gs_fn,
            human_agent_indices=wm.human_agent_indices,
            robot_agent_indices=wm.robot_agent_indices,
            config=cfg,
            state_encoder=shared_encoder,
            # auxiliary_networks injected by trainer.train()
        )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print(f"\nStarting PPO training for {args.iters} iterations …")
    metrics = trainer.train(env_creator, num_iterations=args.iters)

    # ------------------------------------------------------------------
    # 8. Report results
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
