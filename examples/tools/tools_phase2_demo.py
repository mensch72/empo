#!/usr/bin/env python3
"""
Phase 2 Robot Policy Learning Demo — Tools Workshop (DQN).

Trains a robot policy using the DQN-based Phase 2 trainer (same learning
algorithm as ``phase2_robot_policy_demo.py`` but for the tools environment).

The demo trains:
- Q_r: Robot state-action value (eq. 4)
- π_r: Robot policy using power-law softmax (eq. 5)
- V_h^e: Human goal achievement under robot policy (eq. 6)
- X_h: Aggregate goal achievement ability (eq. 7)
- U_r: Intrinsic robot reward (eq. 8)
- V_r: Robot state value (eq. 9)

Usage:
    python examples/tools/tools_phase2_demo.py
    python examples/tools/tools_phase2_demo.py --quick
    python examples/tools/tools_phase2_demo.py --steps 50000 --n-agents 3 --n-tools 3
"""

import argparse
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
for p in [
    os.path.join(_REPO_ROOT, "src"),
    os.path.join(_REPO_ROOT, "vendor", "multigrid"),
    os.path.join(_REPO_ROOT, "vendor", "ai_transport"),
    os.path.join(_REPO_ROOT, "multigrid_worlds"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from empo.world_specific_helpers.tools import (  # noqa: E402
    ToolsWorldModel,
    ToolsGoalGenerator,
    ToolsGoalSampler,
    ToolsHeuristicPolicy,
    action_name,
    create_tools_env,
    render_tools_state,
    render_tools_transition,
    save_tools_video,
)
from empo.learning_based.phase2.config import Phase2Config  # noqa: E402
from empo.learning_based.tools.phase2 import (  # noqa: E402
    train_tools_phase2,
)

# ======================================================================
# Rollout with trained Q_r policy
# ======================================================================


def run_dqn_rollout(
    env: ToolsWorldModel,
    q_network,
    human_policy: ToolsHeuristicPolicy,
    goal_sampler: ToolsGoalSampler,
    human_indices: List[int],
    robot_indices: List[int],
    beta_r: float = 10.0,
    device: str = "cpu",
):
    """Run a single rollout using the trained Q_r network.

    Returns a list of rendered frames and the number of steps taken.
    """
    q_network.eval()
    env.reset()
    state = env.get_state()

    # Sample goals for each human
    human_goals: Dict[int, object] = {}
    for h in human_indices:
        goal, _ = goal_sampler.sample(state, h)
        human_goals[h] = goal

    frames = []
    steps_taken = 0

    for _step in range(env.max_steps):
        state = env.get_state()

        # Re-assign goals for humans whose goal is achieved
        for h in human_indices:
            if human_goals[h].is_achieved(state):
                new_goal, _ = goal_sampler.sample(state, h)
                human_goals[h] = new_goal

        goals_list = list(human_goals.values())

        # Render frame
        frame = render_tools_state(env, goals=goals_list)
        if frame is not None:
            frames.append(frame)

        # Build joint action
        actions = [0] * env.n_agents

        # Humans use heuristic policy
        for h in human_indices:
            dist = human_policy(state, h, human_goals[h])
            if dist is not None and dist.sum() > 0:
                actions[h] = np.random.choice(len(dist), p=dist)

        # Robot uses trained Q_r policy (power-law softmax)
        with torch.no_grad():
            q_values = q_network.forward(state, env, device)
            pi_r = q_network.get_policy(q_values, beta_r=beta_r)
            pi_np = pi_r.squeeze().cpu().numpy()

        robot_action = np.random.choice(len(pi_np), p=pi_np)

        # Assign robot actions
        num_robot_actions = env.n_actions_per_agent[robot_indices[0]]
        if len(robot_indices) == 1:
            actions[robot_indices[0]] = robot_action
        else:
            remaining = robot_action
            for r_idx in robot_indices:
                actions[r_idx] = remaining % num_robot_actions
                remaining //= num_robot_actions

        # Print step
        action_strs = [
            action_name(a, env.n_tools, env.give_targets[i])
            for i, a in enumerate(actions)
        ]
        q_np = q_values.squeeze().cpu().numpy()
        best_q = float(q_np.max())
        print(
            f"    Step {_step + 1}/{env.max_steps}  "
            f"Q_r(best)={best_q:.4f}  actions={action_strs}"
        )

        _obs, _reward, terminated, _truncated, _info = env.step(actions)
        new_state = env.get_state()
        steps_taken += 1

        # Transition frames
        transition_frames = render_tools_transition(
            env, state, new_state, actions, goals=goals_list, n_interp=10
        )
        frames.extend(transition_frames)

        if terminated:
            print("    -> TERMINAL")
            break

    # Final frame
    frame = render_tools_state(env, goals=list(human_goals.values()))
    if frame is not None:
        frames.append(frame)

    return frames, steps_taken


# ======================================================================
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 DQN demo — Tools Workshop (heuristic humans)"
    )
    parser.add_argument(
        "--steps",
        type=lambda x: int(float(x)),
        default=20_000,
        help="Number of training steps (default: 20000, supports 1e5 notation)",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick test mode (1000 steps)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of agents (default: 3)",
    )
    parser.add_argument(
        "--n-tools",
        type=int,
        default=3,
        help="Number of tools (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Max environment steps per episode (default: 15)",
    )
    parser.add_argument(
        "--p-failure",
        type=float,
        default=0.1,
        help="Probability of action failure (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="MLP hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=64,
        help="State encoder feature dimension (default: 64)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=5,
        help="Number of evaluation rollouts (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/tools_phase2_demo/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    num_training_steps = 1000 if args.quick else args.steps

    print("=" * 70)
    print("Phase 2 Robot Policy Learning Demo — Tools Workshop (DQN)")
    print("Learning robot policy to maximize aggregate human power")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # 1. Create environment
    # ---------------------------------------------------------------
    env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.max_steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    env.reset(seed=args.seed)

    human_indices = list(env.human_agent_indices)
    robot_indices = list(env.robot_agent_indices)
    num_actions = max(env.n_actions_per_agent[r] for r in robot_indices)

    print(f"  Agents      : {env.n_agents}")
    print(f"  Tools       : {env.n_tools}")
    print(f"  Humans      : {human_indices}")
    print(f"  Robots      : {robot_indices}")
    print(f"  Robot actions: {num_actions}")
    print(f"  Max steps   : {env.max_steps}")
    print()

    # ---------------------------------------------------------------
    # 2. Human policy + goal sampler
    # ---------------------------------------------------------------
    goal_generator = ToolsGoalGenerator(env)
    human_policy = ToolsHeuristicPolicy(env, goal_generator, beta=5.0)
    goal_sampler = ToolsGoalSampler(env)

    print("  Human policy: ToolsHeuristicPolicy (beta=5.0)")
    print("  Goal sampler: ToolsGoalSampler")
    print()

    # ---------------------------------------------------------------
    # 3. Output directory
    # ---------------------------------------------------------------
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(_REPO_ROOT, "outputs", "tools_phase2_demo")
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, "tensorboard")

    # ---------------------------------------------------------------
    # 4. Phase 2 config
    # ---------------------------------------------------------------
    warmup_v_h_e = int(0.1 * num_training_steps)
    warmup_x_h = int(0.1 * num_training_steps)
    warmup_u_r = int(0.05 * num_training_steps)
    warmup_q_r = int(0.1 * num_training_steps)
    warmup_v_r = int(0.05 * num_training_steps)
    beta_r_rampup = int(0.2 * num_training_steps)

    config = Phase2Config(
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,
        xi=1.0,
        eta=1.1,
        beta_r=1000.0,
        epsilon_r_start=1.0,
        epsilon_r_end=0.0,
        epsilon_r_decay_steps=num_training_steps * 2 // 3,
        epsilon_h_start=1.0,
        epsilon_h_end=0.0,
        epsilon_h_decay_steps=num_training_steps * 2 // 3,
        lr_q_r=1e-4,
        lr_v_r=1e-4,
        lr_v_h_e=1e-3,
        lr_x_h=1e-4,
        lr_u_r=1e-4,
        constant_lr_then_1_over_t=True,
        lr_constant_fraction=0.7,
        buffer_size=50_000,
        batch_size=32,
        x_h_batch_size=64,
        num_training_steps=num_training_steps,
        steps_per_episode=args.max_steps,
        training_steps_per_env_step=0.1,
        goal_resample_prob=0.1,
        v_h_e_target_update_interval=100,
        x_h_use_network=True,
        warmup_v_h_e_steps=warmup_v_h_e,
        warmup_x_h_steps=warmup_x_h,
        warmup_u_r_steps=warmup_u_r,
        warmup_q_r_steps=warmup_q_r,
        warmup_v_r_steps=warmup_v_r,
        beta_r_rampup_steps=beta_r_rampup,
        use_z_space_transform=True,
        use_z_based_loss=False,
    )

    # ---------------------------------------------------------------
    # 5. Train
    # ---------------------------------------------------------------
    print("Training Phase 2 robot policy (DQN)...")
    print(f"  Training steps: {config.num_training_steps:,}")
    print(f"  TensorBoard  : {tensorboard_dir}")
    print()

    t0 = time.time()
    q_r, networks, history, trainer = train_tools_phase2(
        world_model=env,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_policy_prior=human_policy,
        goal_sampler=goal_sampler,
        config=config,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        device=args.device,
        verbose=True,
        debug=args.debug,
        tensorboard_dir=tensorboard_dir,
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s")

    # ---------------------------------------------------------------
    # 6. Report results
    # ---------------------------------------------------------------
    if history:
        print("\nLoss history (last 5):")
        for i, losses in enumerate(history[-5:]):
            ep = len(history) - 5 + i
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if v > 0)
            print(f"  Episode {ep}: {loss_str}")

    # ---------------------------------------------------------------
    # 7. Diagnostics at root state
    # ---------------------------------------------------------------
    diag_env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.max_steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    diag_env.reset(seed=args.seed)
    root_state = diag_env.get_state()

    print("\n--- Diagnostics at root state ---")
    with torch.no_grad():
        # Q_r
        q_values = q_r.forward(root_state, diag_env, args.device)
        pi_r = q_r.get_policy(q_values, beta_r=config.beta_r)
        q_np = q_values.squeeze().cpu().numpy()
        pi_np = pi_r.squeeze().cpu().numpy()
        print(f"  Q_r(root)  : {dict(enumerate(q_np.tolist()))}")
        print(f"  pi_r(root) : {dict(enumerate(pi_np.tolist()))}")

        # V_h^e
        print("  V_h^e(root, h, g):")
        for h in human_indices:
            for goal, _w in goal_generator.generate(root_state, h):
                v = networks.v_h_e.forward(root_state, diag_env, h, goal, args.device)
                print(f"    h={h}, g={goal}: {v.item():.6f}")

        # X_h
        if networks.x_h is not None:
            print("  X_h(root, h):")
            for h in human_indices:
                x = networks.x_h.forward(root_state, diag_env, h, args.device)
                print(f"    h={h}: {x.item():.6f}")

        # U_r
        if networks.u_r is not None:
            y, u = networks.u_r.forward(root_state, diag_env, args.device)
            print(f"  U_r(root)  : {u.item():.6f}  (y={y.item():.6f})")

        # V_r
        if networks.v_r is not None:
            v_r = networks.v_r.forward(root_state, diag_env, args.device)
            print(f"  V_r(root)  : {v_r.item():.6f}")

    # ---------------------------------------------------------------
    # 8. Save networks
    # ---------------------------------------------------------------
    networks_path = os.path.join(output_dir, "all_networks.pt")
    print(f"\nSaving networks to: {networks_path}")
    trainer.save_all_networks(networks_path)

    # ---------------------------------------------------------------
    # 9. Rollouts
    # ---------------------------------------------------------------
    num_rollouts = args.rollouts
    print(f"\nGenerating {num_rollouts} rollouts with learned policy …\n")

    rollout_env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.max_steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    rollout_human_policy = ToolsHeuristicPolicy(
        rollout_env, ToolsGoalGenerator(rollout_env), beta=5.0
    )
    rollout_goal_sampler = ToolsGoalSampler(rollout_env)

    all_frames = []
    for rollout_idx in range(num_rollouts):
        print(f"--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
        frames, n_steps = run_dqn_rollout(
            env=rollout_env,
            q_network=q_r,
            human_policy=rollout_human_policy,
            goal_sampler=rollout_goal_sampler,
            human_indices=human_indices,
            robot_indices=robot_indices,
            beta_r=config.beta_r,
            device=args.device,
        )
        all_frames.extend(frames)
        print(f"    {n_steps} steps, {len(frames)} frames")

    # Save video
    if all_frames:
        movie_path = os.path.join(output_dir, "tools_phase2_demo.mp4")
        save_tools_video(all_frames, movie_path, fps=10)
        print(f"\nMovie saved to: {os.path.abspath(movie_path)}")
    else:
        print("\nNo frames recorded.")

    # ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"  TensorBoard: {tensorboard_dir}")
    print(f"  View with: tensorboard --logdir={tensorboard_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
