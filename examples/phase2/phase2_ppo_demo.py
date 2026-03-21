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

Output:
    After training, generates a movie of rollouts with the learned policy,
    showing the robot's action probabilities and value estimates.

Usage:
    # Inside Docker container (make shell):
    python examples/phase2/phase2_ppo_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_demo.py

    # Quick smoke test (2 iterations, 2 rollouts):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/phase2_ppo_demo.py --iters 2 --rollouts 2

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import List

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
# Rendering constants
# ======================================================================

RENDER_TILE_SIZE = 96
ANNOTATION_PANEL_WIDTH = 300
ANNOTATION_FONT_SIZE = 12
MOVIE_FPS = 2

SINGLE_ACTION_NAMES = ["still", "left", "right", "forward"]


def _get_joint_action_names(
    num_robots: int, num_actions: int = 4
) -> List[str]:
    """Generate joint action names for ``num_robots`` robots."""
    names = SINGLE_ACTION_NAMES[:num_actions]
    if num_robots == 1:
        return list(names)
    combos = list(itertools.product(names, repeat=num_robots))
    return [", ".join(c) for c in combos]


# ======================================================================
# Rollout with trained PPO policy
# ======================================================================


def run_ppo_rollout(
    env: MultiGridEnv,
    actor_critic,
    state_encoder,
    human_policy_fn,
    goal_sampler,
    human_indices: List[int],
    robot_indices: List[int],
    device: str = "cpu",
) -> int:
    """Run a single rollout using the trained PPO actor-critic.

    Uses ``env``'s built-in video recording — frames are captured
    automatically via ``env.render(mode='rgb_array', ...)``.

    ``human_policy_fn`` is called as ``(state, h_idx, goal, env)``
    and returns a probability list over actions.

    Returns the number of environment steps taken.
    """
    num_actions = env.action_space.n
    joint_action_names = _get_joint_action_names(len(robot_indices), num_actions)
    max_name_len = max(len(n) for n in joint_action_names)
    actor_critic.eval()

    env.reset()
    state = env.get_state()

    # Sample goals for each human
    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler(state, h)
        human_goals[h] = goal

    # ----- helpers -----

    def _state_to_obs(s):
        encoder_device = next(state_encoder.parameters()).device
        with torch.no_grad():
            tensors = state_encoder.tensorize_state(s, env, device=encoder_device)
            features = state_encoder(*tensors)
        return features.squeeze(0)

    def _get_policy(obs_tensor):
        with torch.no_grad():
            logits, value = actor_critic(obs_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs, value.squeeze().item()

    def _select_action(probs):
        return torch.argmax(probs).item()

    def _annotation_text(s, selected_action=None):
        obs = _state_to_obs(s)
        probs, v_r = _get_policy(obs)
        lines = [f"V_r: {v_r:.4f}", ""]
        lines.append("π_r probs:")
        for i, p in enumerate(probs.cpu().tolist()):
            name = joint_action_names[i] if i < len(joint_action_names) else f"a{i}"
            marker = ">" if selected_action is not None and i == selected_action else " "
            lines.append(f"{marker}{name:>{max_name_len}}: {p:.3f}")
        return lines

    # ----- initial frame -----
    obs0 = _state_to_obs(state)
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

        # Humans use policy prior
        for h in human_indices:
            probs_h = human_policy_fn(state, h, human_goals[h], env)
            actions[h] = int(np.random.choice(len(probs_h), p=probs_h))

        # Robot uses trained PPO policy
        obs = _state_to_obs(state)
        probs, _ = _get_policy(obs)
        robot_action = _select_action(probs)

        if len(robot_indices) == 1:
            actions[robot_indices[0]] = robot_action
        else:
            remaining = robot_action
            for r_idx in robot_indices:
                actions[r_idx] = remaining % num_actions
                remaining //= num_actions

        _, _, done, _ = env.step(actions)
        steps_taken += 1

        new_state = env.get_state()
        new_obs = _state_to_obs(new_state)
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
        help="Directory for output movie (default: outputs/phase2_ppo_demo/)",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=MOVIE_FPS,
        help=f"Movie frames per second (default: {MOVIE_FPS})",
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
    shared_encoder = state_encoder

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
            state_encoder=shared_encoder,
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

    # ------------------------------------------------------------------
    # 8. Generate rollout movie
    # ------------------------------------------------------------------
    num_rollouts = args.rollouts
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "outputs", "phase2_ppo_demo"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_rollouts} rollouts with learned policy …")

    rollout_env = _create_world_model()
    rollout_env.reset()
    rollout_env.start_video_recording()

    for rollout_idx in range(num_rollouts):
        steps = run_ppo_rollout(
            env=rollout_env,
            actor_critic=actor_critic,
            state_encoder=state_encoder,
            human_policy_fn=_uniform_human_policy,
            goal_sampler=_dummy_goal_sampler,
            human_indices=list(human_indices),
            robot_indices=list(robot_indices),
            device=args.device,
        )
        if (rollout_idx + 1) % 5 == 0 or rollout_idx == num_rollouts - 1:
            print(
                f"  Completed {rollout_idx + 1}/{num_rollouts} rollouts "
                f"({len(rollout_env._video_frames)} total frames)"
            )

    movie_path = os.path.join(output_dir, "phase2_ppo_demo.mp4")
    if os.path.exists(movie_path):
        os.remove(movie_path)
    rollout_env.save_video(movie_path, fps=args.movie_fps)

    print(f"\n✓ Movie saved to: {os.path.abspath(movie_path)}")


if __name__ == "__main__":
    main()
