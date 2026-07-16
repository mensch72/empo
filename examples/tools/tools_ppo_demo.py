#!/usr/bin/env python3
"""
Phase 2 PPO Robot Policy Demo — Tools Workshop.

Demonstrates Phase 2 PPO training on the tools workshop multi-agent
environment.  The robot learns an empowerment-maximising policy via
PPO with auxiliary networks (V_h^e, X_h) while humans follow a
heuristic goal-directed policy.

Environment:
    A tool-exchange workshop with configurable agents and tools.
    Agent 0 is the robot; the remaining agents are humans.  Each agent
    can acquire, give, or request tools.  Communication and reachability
    are governed by Waxman random graphs.

Human behaviour:
    Humans use ``ToolsHeuristicPolicy`` — a goal-conditioned Boltzmann
    policy that acquires needed tools and gives tools to requesters along
    shortest paths.

Output:
    After training, generates a video of rollouts with the learned PPO
    policy, showing how the robot assists human tool exchange.

Usage:
    # Inside Docker (make shell):
    python examples/tools/tools_ppo_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_ppo_demo.py

    # Quick smoke test (2 iterations, 2 rollouts):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_ppo_demo.py --iters 2 --rollouts 2

Requirements:
    pip install pufferlib>=3.0
"""

from __future__ import annotations

import argparse
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
# EMPO / Tools imports
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

from empo.world_specific_helpers.tools import (  # noqa: E402
    ToolsGoalGenerator,
    ToolsGoalSampler,
    ToolsHeuristicPolicy,
    action_name,
    create_tools_env,
    render_tools_state,
    render_tools_transition,
    save_tools_video,
)

from empo.learning_based.phase2_ppo.config import PPOPhase2Config  # noqa: E402
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer  # noqa: E402
from empo.learning_based.tools.phase2_ppo import (  # noqa: E402
    ToolsWorldModelEnv,
    create_tools_ppo_networks,
)

# ======================================================================
# Environment helpers
# ======================================================================


def _make_heuristic_human_policy(env, beta: float = 5.0):
    """Build a heuristic goal-directed policy for humans."""
    goal_gen = ToolsGoalGenerator(env)
    return ToolsHeuristicPolicy(env, goal_gen, beta=beta)


def _wrap_human_policy(policy):
    """Wrap the 3-arg ToolsHeuristicPolicy.__call__ to the 4-arg signature
    expected by EMPOWorldModelEnv.

    Phase 2 PPO env wrapper calls:
        human_policy_prior(state, h_idx, goal, world_model)
    but ToolsHeuristicPolicy.__call__ takes:
        (state, human_agent_index, possible_goal)
    """

    def _policy_prior(state, human_idx, goal, world_model):
        return policy(state, human_idx, goal)

    return _policy_prior


def _make_goal_sampler(env):
    """Create a goal sampler from the tools environment.

    Returns a callable ``(state, human_idx) → (goal, weight)``.
    """
    sampler = ToolsGoalSampler(env)

    def _sampler(state, human_idx):
        return sampler.sample(state, human_idx)

    return _sampler


# ======================================================================
# Rollout with trained PPO policy
# ======================================================================


def run_ppo_rollout(
    env,
    actor_critic,
    state_encoder,
    human_policy,
    goal_sampler,
    human_indices: List[int],
    robot_indices: List[int],
    device: str = "cpu",
):
    """Run a single rollout using the trained PPO actor-critic.

    Returns a list of rendered frames and the number of steps taken.
    """
    actor_critic.eval()
    env.reset()
    state = env.get_state()

    # Sample goals for each human
    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler(state, h)
        human_goals[h] = goal

    def _state_to_obs(s):
        encoder_device = next(state_encoder.parameters()).device
        with torch.no_grad():
            x = state_encoder.tensorize_state(s, env, device=encoder_device)
            features = state_encoder(x)
        return features.squeeze(0)

    def _get_policy(obs_tensor):
        with torch.no_grad():
            logits, value = actor_critic(obs_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs, value.squeeze().item()

    def _select_action(probs):
        return torch.argmax(probs).item()

    frames = []
    steps_taken = 0

    for _step in range(env.max_steps):
        state = env.get_state()

        # Re-assign goals for humans whose goal is achieved
        for h in human_indices:
            if human_goals[h].is_achieved(state):
                new_goal, _ = goal_sampler(state, h)
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

        # Robot uses trained PPO policy
        obs = _state_to_obs(state)
        probs, v_r = _get_policy(obs)
        robot_action = _select_action(probs)

        # Decode joint-action index to per-robot actions
        num_actions = env.action_space.n
        if len(robot_indices) == 1:
            actions[robot_indices[0]] = robot_action
        else:
            remaining = robot_action
            for r_idx in robot_indices:
                actions[r_idx] = remaining % num_actions
                remaining //= num_actions

        # Print step
        action_strs = [
            action_name(a, env.n_tools, env.give_targets[i])
            for i, a in enumerate(actions)
        ]
        print(
            f"    Step {_step + 1}/{env.max_steps}  "
            f"V_r={v_r:.4f}  actions={action_strs}"
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
            print("    → TERMINAL")
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
        description="Phase 2 PPO demo — Tools Workshop (heuristic humans)"
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
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Number of rollouts for the output movie (default: 10)",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of agents — agent 0 is the robot (default: 3)",
    )
    parser.add_argument(
        "--n-tools",
        type=int,
        default=3,
        help="Number of tools (default: 3)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Max steps per episode (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--p-failure",
        type=float,
        default=0.1,
        help="Action failure probability (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output movie (default: outputs/tools_ppo_demo/)",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=10,
        help="Movie frames per second (default: 10)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Create a reference environment
    # ------------------------------------------------------------------
    ref_env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    ref_env.reset(seed=args.seed)

    human_indices = ref_env.human_agent_indices
    robot_indices = ref_env.robot_agent_indices

    # Use the robot's per-agent action count (may be smaller than
    # env.action_space.n which is the max across all agents).
    num_actions = ref_env.n_actions_per_agent[robot_indices[0]]

    print("=" * 60)
    print("  Tools Workshop PPO Demo")
    print("=" * 60)
    print(
        f"  Agents:  {ref_env.n_agents}  "
        f"(robot={robot_indices}, humans={human_indices})"
    )
    print(f"  Tools:   {ref_env.n_tools}")
    print(f"  Actions: {num_actions}")
    print(f"  Steps:   {args.steps}")
    print(f"  p_fail:  {args.p_failure}")
    print()

    # ------------------------------------------------------------------
    # 2. Build heuristic human policy and goal sampler
    # ------------------------------------------------------------------
    print("  Human policy: ToolsHeuristicPolicy (β=5.0)")
    n_goals_per_human = 2 * ref_env.n_tools + 1
    n_humans = len(human_indices)
    print(
        f"  Goal space: {n_goals_per_human} goals/human × "
        f"{n_humans} humans = {n_goals_per_human * n_humans} total"
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
        # Warm-up
        warmup_v_h_e_steps=200,
        warmup_x_h_steps=400,
        warmup_u_r_steps=400,
        # Environment
        steps_per_episode=args.steps,
        # Runtime
        device=args.device,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 4. Create networks
    # ------------------------------------------------------------------
    actor_critic, aux_nets, state_encoder = create_tools_ppo_networks(
        env=ref_env,
        config=cfg,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        use_x_h=True,
        use_u_r=False,
        device=args.device,
    )
    print(f"  State encoder feature_dim: {state_encoder.feature_dim}")
    print(f"  Actor-critic joint actions: {actor_critic.num_joint_actions}")

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

    def env_creator():
        wm = create_tools_env(
            n_agents=args.n_agents,
            n_tools=args.n_tools,
            max_steps=args.steps,
            p_failure=args.p_failure,
            seed=args.seed,
            robot_agent_indices=[0],
        )
        wm.reset(seed=args.seed)
        hp = _make_heuristic_human_policy(wm)
        hp_fn = _wrap_human_policy(hp)
        gs_fn = _make_goal_sampler(wm)
        return ToolsWorldModelEnv(
            world_model=wm,
            human_policy_prior=hp_fn,
            goal_sampler=gs_fn,
            human_agent_indices=list(wm.human_agent_indices),
            robot_agent_indices=list(wm.robot_agent_indices),
            config=cfg,
            state_encoder=shared_encoder,
        )

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print(f"\n  Starting PPO training for {args.iters} iterations …\n")
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
        print(f"  Global env steps  : {trainer.global_env_step}")
    else:
        print("No metrics returned (training may have been too short).")

    # ------------------------------------------------------------------
    # 9. Diagnostics at root state
    # ------------------------------------------------------------------
    diag_env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    diag_env.reset(seed=args.seed)
    root_state = diag_env.get_state()
    _X_H_MIN = 1e-3

    print("\n--- Diagnostics at root state ---")

    actor_critic.eval()
    with torch.no_grad():
        enc_device = next(state_encoder.parameters()).device
        x = state_encoder.tensorize_state(root_state, diag_env, device=enc_device)
        features = state_encoder(x)
        logits, value = actor_critic(features)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    print(f"  V_r(root)         : {value.item():.6f}")
    print("  π_r(root) top-3   : ", end="")
    top_probs, top_idx = torch.topk(probs, min(3, len(probs)))
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        a_name = action_name(i, diag_env.n_tools, diag_env.give_targets[0])
        print(f"{a_name}={p:.3f}  ", end="")
    print()

    # X_h per human
    x_h_net = aux_nets.x_h
    if x_h_net is not None:
        print("  X_h(root, h):")
        for h_idx in human_indices:
            with torch.no_grad():
                x_h = x_h_net(root_state, diag_env, h_idx, args.device)
                x_h_clamped = min(max(float(x_h.item()), _X_H_MIN), 1.0)
            print(f"    h={h_idx}: {x_h.item():.6f} (clamped: {x_h_clamped:.6f})")

    # ------------------------------------------------------------------
    # 10. Generate rollout movie
    # ------------------------------------------------------------------
    num_rollouts = args.rollouts
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "outputs", "tools_ppo_demo"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating {num_rollouts} rollouts with learned policy …\n")

    rollout_env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    rollout_human_policy = _make_heuristic_human_policy(rollout_env)
    rollout_goal_sampler = _make_goal_sampler(rollout_env)

    all_frames = []
    for rollout_idx in range(num_rollouts):
        print(f"--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
        frames, n_steps = run_ppo_rollout(
            env=rollout_env,
            actor_critic=actor_critic,
            state_encoder=state_encoder,
            human_policy=rollout_human_policy,
            goal_sampler=rollout_goal_sampler,
            human_indices=list(rollout_env.human_agent_indices),
            robot_indices=list(rollout_env.robot_agent_indices),
            device=args.device,
        )
        all_frames.extend(frames)
        print(
            f"  Rollout {rollout_idx + 1} complete "
            f"({n_steps} steps, {len(frames)} frames)"
        )

    movie_path = os.path.join(output_dir, "tools_ppo_demo.mp4")
    if all_frames:
        save_tools_video(all_frames, movie_path, fps=args.movie_fps)
        print(f"\nMovie saved to: {os.path.abspath(movie_path)}")
    else:
        print("\nNo frames rendered (render_mode may not be set).")

    print("\nDone.")


if __name__ == "__main__":
    main()
