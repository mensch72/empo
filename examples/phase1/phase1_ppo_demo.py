#!/usr/bin/env python3
"""
Phase 1 PPO Human Policy Prior Demo (PufferLib-backed).

Demonstrates computing a goal-conditioned human policy prior using PufferLib
PPO on a MultiGrid environment.  This is the PPO counterpart to the existing
DQN-based ``neural_policy_prior_demo.py``.

Environment: Tiny 4×6 grid with 1 human, 1 robot, and a rock.
- The training human learns a goal-conditioned policy π_h(a|s,g) via PPO.
- The robot follows a uniform random policy (placeholder).
- Goals are sampled from all reachable cells in the grid.

Output:
    After training, generates a movie of rollouts where the human follows
    the learned policy for randomly sampled goals, annotated with π_h
    action probabilities and V_h value estimates.

Usage:
    # Inside Docker container (make shell):
    python examples/phase1/phase1_ppo_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase1/phase1_ppo_demo.py

    # Quick smoke test (2 iterations, 2 rollouts):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase1/phase1_ppo_demo.py --iters 2 --rollouts 2

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gym_multigrid.multigrid import MultiGridEnv, World  # noqa: E402

from empo.learning_based.phase1_ppo.config import PPOPhase1Config  # noqa: E402
from empo.learning_based.phase1_ppo.trainer import PPOPhase1Trainer  # noqa: E402
from empo.learning_based.multigrid.phase1_ppo import (  # noqa: E402
    MultiGridPhase1PPOEnv,
    create_multigrid_phase1_ppo_networks,
)
from empo.world_specific_helpers.multigrid import ReachCellGoal  # noqa: E402

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


def _get_reachable_cells(env: MultiGridEnv) -> List[Tuple[int, int]]:
    """Return all non-wall, non-object cells in the grid."""
    cells = []
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if cell is None:
                cells.append((x, y))
    return cells


def _make_goal_sampler(
    env: MultiGridEnv,
    reachable_cells: Optional[List[Tuple[int, int]]] = None,
):
    """Create a goal sampler that returns ReachCellGoal objects.

    Uniformly samples from reachable cells.  Each call returns
    ``(goal, weight)`` matching the sampler interface.
    """
    if reachable_cells is None:
        reachable_cells = _get_reachable_cells(env)

    def sampler(state: Any, h_idx: int) -> Tuple[Any, float]:
        pos = reachable_cells[np.random.randint(len(reachable_cells))]
        goal = ReachCellGoal(env, h_idx, pos)
        return goal, 1.0

    return sampler


def _random_agent_policy(state: Any, agent_idx: int) -> int:
    """Uniform random policy for non-training agents."""
    return np.random.randint(4)


# ======================================================================
# Rendering constants
# ======================================================================

RENDER_TILE_SIZE = 96
ANNOTATION_PANEL_WIDTH = 300
ANNOTATION_FONT_SIZE = 12
MOVIE_FPS = 2

ACTION_NAMES = ["still", "left", "right", "forward"]


# ======================================================================
# Rollout with trained PPO policy
# ======================================================================


def run_ppo_rollout(
    env: MultiGridEnv,
    actor_critic: torch.nn.Module,
    state_encoder: torch.nn.Module,
    goal_encoder: torch.nn.Module,
    training_human_index: int,
    other_agent_policies: Dict[int, Any],
    goal: Any,
) -> int:
    """Run a single rollout using the trained PPO actor-critic.

    Uses ``env``'s built-in video recording — frames are captured
    automatically via ``env.render(mode='rgb_array', ...)``.

    Returns the number of environment steps taken.
    """
    actor_critic.eval()

    env.reset()
    state = env.get_state()

    # ----- helpers -----

    def _state_to_obs(s, g):
        """Encode state + goal to flat observation."""
        try:
            encoder_device = next(state_encoder.parameters()).device
        except StopIteration:
            encoder_device = torch.device("cpu")
        with torch.no_grad():
            tensors = state_encoder.tensorize_state(s, env, device=encoder_device)
            state_features = state_encoder(*tensors).squeeze(0)
            goal_tensor = goal_encoder.tensorize_goal(g, device=encoder_device)
            goal_features = goal_encoder(goal_tensor).squeeze(0)
            obs = torch.cat([state_features, goal_features], dim=-1)
        return obs

    def _get_policy(obs_tensor):
        with torch.no_grad():
            logits, value = actor_critic(obs_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs, value.squeeze().item()

    def _select_action(probs):
        return torch.argmax(probs).item()

    def _annotation_text(s, selected_action=None):
        obs = _state_to_obs(s, goal)
        probs, v_h = _get_policy(obs)
        target = getattr(goal, "target_pos", "?")
        lines = [f"Goal: {target}", f"V_h: {v_h:.4f}", ""]
        lines.append("π_h probs:")
        for i, p in enumerate(probs.cpu().tolist()):
            name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"a{i}"
            marker = (
                ">" if selected_action is not None and i == selected_action else " "
            )
            lines.append(f"{marker}{name:>8}: {p:.3f}")
        achieved = goal.is_achieved(s)
        if achieved:
            lines.append("")
            lines.append("★ GOAL ACHIEVED")
        return lines

    # ----- initial frame -----
    obs0 = _state_to_obs(state, goal)
    probs0, _ = _get_policy(obs0)
    action0 = _select_action(probs0)
    env.render(
        mode="rgb_array",
        highlight=False,
        tile_size=RENDER_TILE_SIZE,
        annotation_text=_annotation_text(state, action0),
        annotation_panel_width=ANNOTATION_PANEL_WIDTH,
        annotation_font_size=ANNOTATION_FONT_SIZE,
    )

    # ----- step loop -----
    steps_taken = 0
    for _step in range(env.max_steps):
        state = env.get_state()
        actions = [0] * len(env.agents)

        # Training human uses learned PPO policy
        obs = _state_to_obs(state, goal)
        probs, _ = _get_policy(obs)
        human_action = _select_action(probs)
        actions[training_human_index] = human_action

        # All other agents use their provided policies
        for idx, policy_fn in other_agent_policies.items():
            actions[idx] = int(policy_fn(state, idx))

        _, _, done, _ = env.step(actions)
        steps_taken += 1

        new_state = env.get_state()
        new_obs = _state_to_obs(new_state, goal)
        new_probs, _ = _get_policy(new_obs)
        new_action = _select_action(new_probs)
        env.render(
            mode="rgb_array",
            highlight=False,
            tile_size=RENDER_TILE_SIZE,
            annotation_text=_annotation_text(new_state, new_action),
            annotation_panel_width=ANNOTATION_PANEL_WIDTH,
            annotation_font_size=ANNOTATION_FONT_SIZE,
        )

        if done:
            break

    return steps_taken


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 PPO demo (MultiGrid)")
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
        "--goal-feature-dim",
        type=int,
        default=32,
        help="Goal encoder feature dimension (default: 32)",
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
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Number of rollouts for the output movie (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output movie (default: outputs/phase1_ppo_demo/)",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=MOVIE_FPS,
        help=f"Movie frames per second (default: {MOVIE_FPS})",
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
    training_human = human_indices[0]
    reachable_cells = _get_reachable_cells(ref_env)

    print(f"Grid: {ref_env.height}×{ref_env.width}")
    print(f"Actions: {num_actions}")
    print(f"Human agents: {human_indices}")
    print(f"Robot agents: {robot_indices}")
    print(f"Training human: {training_human}")
    print(f"Reachable cells: {len(reachable_cells)}")

    # ------------------------------------------------------------------
    # 2. Configuration
    # ------------------------------------------------------------------
    cfg = PPOPhase1Config(
        # Theory
        gamma_h=0.99,
        beta_h=1.0,
        # PPO
        num_actions=num_actions,
        hidden_dim=args.hidden_dim,
        ppo_rollout_length=32,
        ppo_num_minibatches=2,
        ppo_update_epochs=2,
        num_envs=args.num_envs,
        num_ppo_iterations=args.iters,
        lr=3e-4,
        # Environment
        steps_per_episode=20,
        # Runtime
        device=args.device,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 3. Create networks (shared encoders for state + goal → observation)
    # ------------------------------------------------------------------
    actor_critic, state_encoder, goal_encoder = create_multigrid_phase1_ppo_networks(
        env=ref_env,
        config=cfg,
        feature_dim=args.feature_dim,
        goal_feature_dim=args.goal_feature_dim,
        device=args.device,
    )
    print(f"State encoder feature_dim: {state_encoder.feature_dim}")
    print(f"Goal encoder feature_dim: {goal_encoder.feature_dim}")
    print(
        f"Actor-critic obs_dim: {state_encoder.feature_dim + goal_encoder.feature_dim}"
    )

    # ------------------------------------------------------------------
    # 4. Build the trainer
    # ------------------------------------------------------------------
    trainer = PPOPhase1Trainer(
        actor_critic=actor_critic,
        config=cfg,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 5. Define env_creator (one per vectorised slot)
    # ------------------------------------------------------------------
    # The encoders are shared (torch modules, same parameters across
    # all envs in Serial backend).
    shared_state_encoder = state_encoder
    shared_goal_encoder = goal_encoder

    # Build other-agent policies: all agents except training human
    # use uniform random policies.
    other_agent_indices = [i for i in range(len(ref_env.agents)) if i != training_human]

    def env_creator():
        wm = _create_world_model()
        wm.reset()
        goal_sampler = _make_goal_sampler(wm, reachable_cells)
        other_policies = {idx: _random_agent_policy for idx in other_agent_indices}
        return MultiGridPhase1PPOEnv(
            world_model=wm,
            goal_sampler=goal_sampler,
            training_human_index=training_human,
            other_agent_policies=other_policies,
            config=cfg,
            state_encoder=shared_state_encoder,
            goal_encoder=shared_goal_encoder,
        )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"\nStarting PPO Phase 1 training for {args.iters} iterations …")
    metrics = trainer.train(env_creator, num_iterations=args.iters)

    # ------------------------------------------------------------------
    # 7. Report results
    # ------------------------------------------------------------------
    if metrics:
        last = metrics[-1]
        print(f"\nTraining complete ({len(metrics)} iterations).")
        print(f"  Final policy_loss : {last.get('policy_loss', 'N/A')}")
        print(f"  Final value_loss  : {last.get('value_loss', 'N/A')}")
        print(f"  Final entropy     : {last.get('entropy', 'N/A')}")
        print(f"  Global env steps  : {trainer.global_env_step}")
    else:
        print("No metrics returned (training may have been too short).")

    # ------------------------------------------------------------------
    # 8. Generate rollout movie
    # ------------------------------------------------------------------
    num_rollouts = args.rollouts
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "outputs", "phase1_ppo_demo"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_rollouts} rollouts with learned policy …")

    rollout_env = _create_world_model()
    rollout_env.reset()
    rollout_env.start_video_recording()

    other_policies_rollout = {idx: _random_agent_policy for idx in other_agent_indices}

    for rollout_idx in range(num_rollouts):
        # Sample a random goal for this rollout
        pos = reachable_cells[np.random.randint(len(reachable_cells))]
        rollout_goal = ReachCellGoal(rollout_env, training_human, pos)
        print(f"  Rollout {rollout_idx + 1}: goal={pos}")

        run_ppo_rollout(
            env=rollout_env,
            actor_critic=actor_critic,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            training_human_index=training_human,
            other_agent_policies=other_policies_rollout,
            goal=rollout_goal,
        )

    movie_path = os.path.join(output_dir, "phase1_ppo_demo.mp4")
    if os.path.exists(movie_path):
        os.remove(movie_path)
    rollout_env.save_video(movie_path, fps=args.movie_fps)

    print(f"\n✓ Movie saved to: {os.path.abspath(movie_path)}")


if __name__ == "__main__":
    main()
