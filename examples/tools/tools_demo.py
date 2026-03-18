#!/usr/bin/env python
"""
Tools Workshop Demo
===================

Demonstrates the Tools WorldModel: agents exchange tools in a shared workshop.

Configuration (from issue):
    - 1 robot (agent 0, centre), 3 humans (agents 1-3)
    - 6 tools, 30 time steps, p_failure = 0.1

The demo runs a short simulation using the heuristic human policy for the
humans and a random policy for the robot, renders each step, and optionally
saves the frames as an mp4/gif video.

Usage:
    # Inside Docker (make shell):
    python examples/tools/tools_demo.py

    # Outside Docker:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_demo.py

    # Quick run (fewer steps):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_demo.py --steps 10

    # Save video:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/tools/tools_demo.py --video outputs/tools_demo.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from empo.world_specific_helpers.tools import (
    ACTION_PASS,
    HoldGoal,
    ToolsGoalGenerator,
    ToolsGoalSampler,
    ToolsHeuristicPolicy,
    ToolsWorldModel,
    WorkbenchGoal,
    action_name,
    create_tools_env,
    render_tools_state,
    save_tools_video,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tools Workshop Demo")
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of simulation steps (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to save video (e.g. outputs/tools_demo.mp4)",
    )
    parser.add_argument(
        "--beta", type=float, default=5.0,
        help="Heuristic policy temperature (default: 5.0)",
    )
    parser.add_argument(
        "--p-failure", type=float, default=0.1,
        help="Action failure probability (default: 0.1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Tools Workshop Demo")
    print("=" * 60)
    print()

    # Create environment: 1 robot (centre), 3 humans, 6 tools
    env = create_tools_env(
        n_agents=4,
        n_tools=6,
        max_steps=args.steps,
        p_failure=args.p_failure,
        seed=args.seed,
        robot_agent_indices=[0],
    )
    state, _ = env.reset(seed=args.seed)
    np.random.seed(args.seed)

    print(f"  Agents: {env.n_agents}  (robot={env.robot_agent_indices}, "
          f"humans={env.human_agent_indices})")
    print(f"  Tools:  {env.n_tools}")
    print(f"  Steps:  {args.steps}")
    print(f"  p_fail: {args.p_failure}")
    print()

    # Print constant-state graphs
    print("can_hear adjacency:")
    for i in range(env.n_agents):
        row = " ".join("1" if env.can_hear[i, j] else "." for j in range(env.n_agents))
        print(f"  Agent {i}: [{row}]")
    print()
    print("can_reach adjacency:")
    for i in range(env.n_agents):
        row = " ".join("1" if env.can_reach[i, j] else "." for j in range(env.n_agents))
        print(f"  Agent {i}: [{row}]")
    print()

    # Assign a random goal to each human for display purposes
    rng = np.random.RandomState(args.seed + 1)
    human_goals = {}
    for hi in env.human_agent_indices:
        k = rng.randint(env.n_tools)
        goal_type = rng.choice(["hold", "workbench"])
        if goal_type == "hold":
            human_goals[hi] = HoldGoal(env, hi, k)
        else:
            human_goals[hi] = WorkbenchGoal(env, hi, k)
    print("Assigned goals:")
    for hi, g in human_goals.items():
        print(f"  Agent {hi}: {g}")
    print()

    # Create heuristic policy for humans
    goal_gen = ToolsGoalGenerator(env)
    heuristic = ToolsHeuristicPolicy(env, goal_gen, beta=args.beta)

    frames = []
    goals_list = list(human_goals.values())

    for step in range(args.steps):
        state = env.get_state()
        remaining, wb, holds, req = state

        # Print state summary
        print(f"--- Step {step + 1}/{args.steps}  (remaining={remaining}) ---")
        for i in range(env.n_agents):
            role = "R" if i in env.robot_agent_indices else "H"
            held = [k for k in range(env.n_tools) if holds[i][k]]
            bench = [k for k in range(env.n_tools) if wb[i][k]]
            reqs = [k for k in range(env.n_tools) if req[i][k]]
            held_str = f"holds T{held[0]}" if held else "empty hand"
            bench_str = ",".join(f"T{k}" for k in bench) if bench else "∅"
            req_str = ",".join(f"T{k}" for k in reqs) if reqs else "–"
            print(f"  {role}{i}: {held_str}  wb=[{bench_str}]  req=[{req_str}]")

        # Render frame
        frame = render_tools_state(env, goals=goals_list)
        if frame is not None:
            frames.append(frame)

        # Choose actions
        actions = []
        for i in range(env.n_agents):
            if i in env.robot_agent_indices:
                # Robot: random action
                a = np.random.randint(env.n_actions)
            else:
                # Human: heuristic conditioned on assigned goal
                goal = human_goals.get(i)
                a = heuristic.sample(state, i, goal)
            actions.append(a)

        action_strs = [
            action_name(a, env.n_tools, env.n_agents) for a in actions
        ]
        print(f"  Actions: {action_strs}")

        # Step
        obs, reward, terminated, truncated, info = env.step(actions)
        if terminated:
            print("  → TERMINAL")
            break
        print()

    # Final frame
    frame = render_tools_state(env, goals=goals_list)
    if frame is not None:
        frames.append(frame)

    # Check goal achievement
    print()
    print("Final goal status:")
    final_state = env.get_state()
    for hi, g in human_goals.items():
        achieved = g.is_achieved(final_state)
        status = "✓ ACHIEVED" if achieved else "✗ not achieved"
        print(f"  Agent {hi}: {g}  →  {status}")

    # Save video
    if args.video and frames:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        save_tools_video(frames, args.video, fps=2)
        print(f"\nVideo saved to {args.video}")

    print("\nDone.")


if __name__ == "__main__":
    main()
