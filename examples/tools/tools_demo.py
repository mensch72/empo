#!/usr/bin/env python
"""
Tools Workshop Demo
===================

Demonstrates the Tools WorldModel with a robot policy computed via backward
induction (Phase 1 → Phase 2).

Configuration:
    - 1 robot (agent 0, centre), 2 humans (agents 1-2)
    - 3 tools, 15 time steps, p_failure = 0.1

During rollouts, humans pursue sampled goals.  Once a human's goal is
achieved, a new goal is sampled immediately so every human always has an
active goal.  The robot uses the Phase 2 backward-induction policy.

The demo generates a video with multiple rollouts (default 10).

Usage:
    # Inside Docker (make shell):
    python examples/tools/tools_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_demo.py

    # Save video (default location: outputs/tools_demo.mp4):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_demo.py --video outputs/tools_demo.mp4
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from empo.backward_induction import (
    compute_human_policy_prior,
    compute_robot_policy,
)
from empo.world_specific_helpers.tools import (
    HoldGoal,
    IdleGoal,
    ToolsGoalGenerator,
    ToolsGoalSampler,
    ToolsHeuristicPolicy,
    WorkbenchGoal,
    action_name,
    create_tools_env,
    render_tools_state,
    render_tools_transition,
    save_tools_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tools Workshop Demo")
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Number of simulation steps per rollout (default: 15)",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=10,
        help="Number of rollouts for the video (default: 10)",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="outputs/tools_demo.mp4",
        help="Path to save video (default: outputs/tools_demo.mp4)",
    )
    parser.add_argument(
        "--beta-h",
        type=float,
        default=float("inf"),
        help="Human inverse temperature for Phase 1 (default: inf)",
    )
    parser.add_argument(
        "--beta-r",
        type=float,
        default=1000.0,
        help="Robot inverse temperature for Phase 2 (default: 1000.0)",
    )
    parser.add_argument(
        "--gamma-h",
        type=float,
        default=0.95,
        help="Human discount factor (default: 0.95)",
    )
    parser.add_argument(
        "--gamma-r",
        type=float,
        default=0.99,
        help="Robot discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--p-failure",
        type=float,
        default=0.1,
        help="Action failure probability (default: 0.1)",
    )
    parser.add_argument(
        "--phase1-bi",
        action="store_true",
        default=False,
        help="Use backward induction for Phase 1 (default: heuristic policy)",
    )
    return parser.parse_args()


def run_rollout(env, robot_policy, human_policy_prior, goal_sampler, goal_gen, args):
    """Run a single rollout, returning frames and a step log.

    Humans get a new goal whenever their current goal is achieved.
    """
    env.reset(seed=None)
    state = env.get_state()

    # Sample initial goal for each human
    human_goals = {}
    for hi in env.human_agent_indices:
        goal, _ = goal_sampler.sample(state, hi)
        human_goals[hi] = goal

    frames = []
    n_steps = 0

    for step in range(args.steps):
        state = env.get_state()
        remaining, wb, holds, req = state

        # Re-assign goals for humans whose goal is already achieved
        for hi in env.human_agent_indices:
            if human_goals[hi].is_achieved(state):
                new_goal, _ = goal_sampler.sample(state, hi)
                human_goals[hi] = new_goal

        goals_list = list(human_goals.values())

        # Render frame
        frame = render_tools_state(env, goals=goals_list)
        if frame is not None:
            frames.append(frame)

        # Choose actions
        actions = [0] * env.n_agents

        # Robot: use computed policy
        robot_action = robot_policy.sample(state)
        if robot_action is not None:
            for i, r_idx in enumerate(env.robot_agent_indices):
                actions[r_idx] = robot_action[i]

        # Humans: use computed policy prior conditioned on their goal
        for hi in env.human_agent_indices:
            goal = human_goals[hi]
            dist = human_policy_prior(state, hi, goal)
            if dist is not None and dist.sum() > 0:
                actions[hi] = np.random.choice(len(dist), p=dist)

        # Print step summary
        action_strs = [action_name(a, env.n_tools, env.give_targets[i]) for i, a in enumerate(actions)]
        print(
            f"  Step {step + 1}/{args.steps}  remaining={remaining}  "
            f"actions={action_strs}"
        )

        _obs, _reward, terminated, _truncated, _info = env.step(actions)
        new_state = env.get_state()
        n_steps += 1

        # Insert interpolated transition frames showing tool motion
        transition_frames = render_tools_transition(
            env, state, new_state, actions, goals=goals_list, n_interp=10,
        )
        frames.extend(transition_frames)

        if terminated:
            print("  → TERMINAL")
            break

    # Final frame
    frame = render_tools_state(env, goals=list(human_goals.values()))
    if frame is not None:
        frames.append(frame)

    return frames, n_steps


def main():
    args = parse_args()

    print("=" * 60)
    print("  Tools Workshop Demo  (backward induction)")
    print("=" * 60)
    print()

    # Create environment: 1 robot (centre), 2 humans, 3 tools
    env = create_tools_env(
        n_agents=args.n_agents,
        n_tools=args.n_tools,
        max_steps=args.steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    env.reset(seed=args.seed)
    np.random.seed(args.seed)

    print(
        f"  Agents: {env.n_agents}  (robot={env.robot_agent_indices}, "
        f"humans={env.human_agent_indices})"
    )
    print(f"  Tools:  {env.n_tools}")
    print(f"  Steps:  {args.steps}")
    print(f"  Rollouts: {args.rollouts}")
    print(f"  p_fail: {args.p_failure}")
    print()

    # Print adjacency graphs
    print("can_hear adjacency:")
    for i in range(env.n_agents):
        row = " ".join(
            "1" if env.can_hear[i, j] else "." for j in range(env.n_agents)
        )
        print(f"  Agent {i}: [{row}]")
    print()
    print("can_reach adjacency:")
    for i in range(env.n_agents):
        row = " ".join(
            "1" if env.can_reach[i, j] else "." for j in range(env.n_agents)
        )
        print(f"  Agent {i}: [{row}]")
    print()

    # Goal generator (includes HoldGoal, WorkbenchGoal, IdleGoal)
    goal_gen = ToolsGoalGenerator(env)
    goal_sampler = ToolsGoalSampler(env)

    # ---- Phase 1: human policy prior ----
    print("=" * 60)
    if args.phase1_bi:
        print("  Phase 1: computing human policy prior (backward induction) …")
        print("=" * 60)
        human_policy_prior = compute_human_policy_prior(
            env,
            env.human_agent_indices,
            possible_goal_generator=goal_gen,
            beta_h=args.beta_h,
            gamma_h=args.gamma_h,
        )
    else:
        print("  Phase 1: using heuristic human policy prior …")
        print("=" * 60)
        human_policy_prior = ToolsHeuristicPolicy(env, goal_gen, beta=args.beta_h)
    print("  Phase 1 complete.\n")

    # ---- Phase 2: compute robot policy via backward induction ----
    print("=" * 60)
    print("  Phase 2: computing robot policy …")
    print("=" * 60)
    robot_policy = compute_robot_policy(
        env,
        env.human_agent_indices,
        env.robot_agent_indices,
        possible_goal_generator=goal_gen,
        human_policy_prior=human_policy_prior,
        beta_r=args.beta_r,
        gamma_h=args.gamma_h,
        gamma_r=args.gamma_r,
    )
    print("  Phase 2 complete.\n")

    # ---- Rollouts with video ----
    print("=" * 60)
    print(f"  Generating {args.rollouts} rollouts …")
    print("=" * 60)

    all_frames = []
    for rollout_idx in range(args.rollouts):
        print(f"\n--- Rollout {rollout_idx + 1}/{args.rollouts} ---")
        frames, n_steps = run_rollout(
            env, robot_policy, human_policy_prior, goal_sampler, goal_gen, args
        )
        all_frames.extend(frames)
        print(f"  Rollout {rollout_idx + 1} finished after {n_steps} steps.")

    # Save video
    if args.video and all_frames:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        save_tools_video(all_frames, args.video, fps=10)
        print(f"\nVideo saved to {args.video}")

    print("\nDone.")


if __name__ == "__main__":
    main()
