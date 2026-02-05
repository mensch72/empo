#!/usr/bin/env python3
"""
EMPO Learning Demo: Rock Gateway Environment.

Trains the learning-based EMPO pipeline on a medium-complexity 8x8 grid
where a wall divides the grid and a rock blocks the central passage.
The robot learns to maximise the human's empowerment — pushing the rock
to open a shortcut for the human.

After training, the script:
  1. Evaluates random-robot vs EMPO-robot per-goal success rates + time.
  2. Records annotated demo videos (random + trained).

Usage:
    PYTHONPATH=src:vendor/multigrid python examples/phase2/learning_demo.py

    # Faster smoke test:
    PYTHONPATH=src:vendor/multigrid python examples/phase2/learning_demo.py \
        --p1-iters 100 --p2-iters 100 --eval-episodes 30 --video-episodes 2
"""

import argparse
import os
import random as py_random

import numpy as np
import torch

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.ali_learning_based.envs import get_env_path
from empo.ali_learning_based.pipeline import run_empo_learning


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------

GOAL_NAMES = [
    "Top-Left", "Top-Right", "Bot-Left", "Bot-Right",
    "Mid-Left", "Mid-Right", "Goal-7", "Goal-8",
]


def make_robot_greedy_fn(phase2_trainer, state_encoder):
    """Return a function: state_tuple -> greedy robot action (int)."""
    def get_action(state_tuple):
        state_enc = state_encoder.encode(state_tuple).unsqueeze(0)
        with torch.no_grad():
            q = phase2_trainer.q_r_net(state_enc).squeeze(0)
        return q.argmax().item()
    return get_action


def evaluate_per_goal(env, phase1_trainer, robot_action_fn, goals,
                      human_idx, robot_idx, num_agents, num_actions,
                      episodes_per_goal=50):
    """
    Evaluate per-goal: success rate AND average steps to achieve.

    Returns (success_rates, avg_steps) — both are lists of length len(goals).
    avg_steps[i] = mean steps to achieve goal i (only counting successes);
    inf if no successes.
    """
    success_rates = []
    avg_steps_list = []

    for gi, goal in enumerate(goals):
        successes = 0
        step_counts = []
        for _ in range(episodes_per_goal):
            env.reset()
            done = False
            step = 0
            achieved_step = None
            while not done:
                state = env.get_state()
                # Check if goal achieved at this state
                if achieved_step is None and goal.is_achieved(state):
                    achieved_step = step
                # Act
                actions = [0] * num_agents
                human_probs = phase1_trainer.get_policy(state, goal)
                actions[human_idx] = torch.multinomial(human_probs, 1).item()
                if robot_action_fn is None:
                    actions[robot_idx] = py_random.randint(0, num_actions - 1)
                else:
                    actions[robot_idx] = robot_action_fn(state)
                _, _, done, _ = env.step(actions)
                step += 1
            # Final check
            if achieved_step is None and goal.is_achieved(env.get_state()):
                achieved_step = step
            if achieved_step is not None:
                successes += 1
                step_counts.append(achieved_step)

        rate = successes / episodes_per_goal
        success_rates.append(rate)
        avg_steps_list.append(np.mean(step_counts) if step_counts else float('inf'))

    return success_rates, avg_steps_list


def print_eval_table(goal_names, random_rates, empo_rates,
                     random_steps, empo_steps):
    """Print a formatted comparison table."""
    print(f"  {'Goal':<12} {'Rate(R)':>8} {'Rate(E)':>8} "
          f"{'Steps(R)':>9} {'Steps(E)':>9} {'Faster':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    for name, rr, er, rs, es in zip(goal_names, random_rates, empo_rates,
                                     random_steps, empo_steps):
        rs_str = f"{rs:>8.1f}" if rs < float('inf') else f"{'n/a':>8}"
        es_str = f"{es:>8.1f}" if es < float('inf') else f"{'n/a':>8}"
        if rs < float('inf') and es < float('inf') and rs > 0:
            faster = f"{rs - es:>+7.1f}"
        else:
            faster = f"{'':>8}"
        print(f"  {name:<12} {rr:>7.0%} {er:>7.0%} "
              f"{rs_str} {es_str} {faster}")

    rsum = sum(random_rates)
    esum = sum(empo_rates)
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")
    print(f"  {'Empowerment':<12} {rsum:>8.2f} {esum:>8.2f}")


# -------------------------------------------------------------------
# Video recording
# -------------------------------------------------------------------

def record_episodes(env, phase1_trainer, robot_action_fn, goals, empo_result,
                    human_idx, robot_idx, num_agents, num_actions,
                    num_episodes, label, output_path, tile_size=48, fps=3):
    """
    Record annotated episodes. Each episode, a goal is chosen round-robin
    for the human so the video shows the human pursuing different goals.
    """
    env.start_video_recording()

    for ep in range(num_episodes):
        goal_idx = ep % len(goals)
        goal = goals[goal_idx]
        env.reset()
        done = False
        step = 0
        achieved_at = None

        while not done:
            state = env.get_state()
            achieved = [i for i, g in enumerate(goals) if g.is_achieved(state)]
            if achieved_at is None and goal_idx in achieved:
                achieved_at = step
            emp = empo_result.get_empowerment(state)

            annotation = _annotation(
                label, ep + 1, num_episodes, step, env.max_steps,
                emp, achieved, len(goals), goal_idx,
                done=False, achieved_at=achieved_at,
            )
            env.render(mode="rgb_array", annotation_text=annotation,
                       tile_size=tile_size, annotation_panel_width=240)

            actions = [0] * num_agents
            human_probs = phase1_trainer.get_policy(state, goal)
            actions[human_idx] = torch.multinomial(human_probs, 1).item()
            if robot_action_fn is None:
                actions[robot_idx] = py_random.randint(0, num_actions - 1)
            else:
                actions[robot_idx] = robot_action_fn(state)

            _, _, done, _ = env.step(actions)
            step += 1

        # Final frame (held)
        state = env.get_state()
        achieved = [i for i, g in enumerate(goals) if g.is_achieved(state)]
        if achieved_at is None and goal_idx in achieved:
            achieved_at = step
        emp = empo_result.get_empowerment(state)
        annotation = _annotation(
            label, ep + 1, num_episodes, step, env.max_steps,
            emp, achieved, len(goals), goal_idx,
            done=True, achieved_at=achieved_at,
        )
        for _ in range(fps * 2):
            env.render(mode="rgb_array", annotation_text=annotation,
                       tile_size=tile_size, annotation_panel_width=240)

    env.save_video(output_path, fps=fps)
    print(f"  -> {output_path}")


def _annotation(label, ep, total_eps, step, max_steps,
                empowerment, achieved_indices, num_goals,
                pursuing_goal_idx, done=False, achieved_at=None):
    lines = [
        f"  {label}",
        f"  Episode {ep}/{total_eps}",
        f"  Step {step}/{max_steps}",
        "",
        f"  Empowerment: {empowerment:.3f}",
        "",
        f"  Human pursuing:",
        f"    -> {GOAL_NAMES[pursuing_goal_idx]}",
        "",
        "  Goals in range:",
    ]
    for i in range(num_goals):
        name = GOAL_NAMES[i] if i < len(GOAL_NAMES) else f"Goal {i}"
        marker = "X" if i in achieved_indices else " "
        arrow = " <-" if i == pursuing_goal_idx else ""
        lines.append(f"    [{marker}] {name}{arrow}")

    if done:
        lines.append("")
        if achieved_at is not None:
            lines.append(f"  >> REACHED step {achieved_at} <<")
        else:
            lines.append("  >> NOT REACHED <<")

    return lines


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train EMPO on Rock Gateway and record demo videos.",
    )
    parser.add_argument("--p1-iters", type=int, default=800)
    parser.add_argument("--p2-iters", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Episodes per goal for evaluation")
    parser.add_argument("--video-episodes", type=int, default=8,
                        help="Video episodes (cycles through goals)")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_random.seed(args.seed)

    config = get_env_path("rock_gateway_demo.yaml")

    # --- Train ---
    print("=" * 60)
    print("EMPO LEARNING DEMO — Rock Gateway (8x8, UCB)")
    print("=" * 60)

    result = run_empo_learning(
        config,
        num_phase1_iters=args.p1_iters,
        phase1_episodes_per_iter=4,
        phase1_train_steps_per_iter=8,
        phase1_exploration="ucb",
        phase1_ucb_c=2.0,
        num_phase2_iters=args.p2_iters,
        phase2_episodes_per_iter=4,
        phase2_train_steps_per_iter=8,
        phase2_eps_decay=2000,
        log_interval=200,
    )

    env = result.env
    goals = result.goals
    robot_idx = result.robot_agent_idx
    human_idx = result.human_agent_idx
    num_agents = len(env.agents)
    num_actions = 4

    # --- Per-goal evaluation ---
    print("\n" + "=" * 60)
    print("PER-GOAL EVALUATION")
    print("=" * 60)

    robot_greedy = make_robot_greedy_fn(result.phase2_trainer, result.state_encoder)

    print(f"\n  (R) = Random robot  (E) = EMPO robot")
    print(f"  Rate = success rate, Steps = avg steps to reach goal")
    print(f"  {args.eval_episodes} episodes per goal\n")

    random_rates, random_steps = evaluate_per_goal(
        env, result.phase1_trainer, None, goals,
        human_idx, robot_idx, num_agents, num_actions,
        episodes_per_goal=args.eval_episodes,
    )
    empo_rates, empo_steps = evaluate_per_goal(
        env, result.phase1_trainer, robot_greedy, goals,
        human_idx, robot_idx, num_agents, num_actions,
        episodes_per_goal=args.eval_episodes,
    )

    print_eval_table(GOAL_NAMES[:len(goals)], random_rates, empo_rates,
                     random_steps, empo_steps)

    # --- V_h^e values at initial state ---
    print(f"\n  V_h^e at initial state:")
    env.reset()
    state = env.get_state()
    for i, g in enumerate(goals):
        v = result.get_vhe(state, g)
        print(f"    {GOAL_NAMES[i]}: {v:.4f}")
    print(f"  Empowerment (sum): {result.get_empowerment(state):.4f}")

    # --- Record videos ---
    print("\n" + "=" * 60)
    print("VIDEO RECORDING")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Recording random robot episodes...")
    record_episodes(
        env, result.phase1_trainer, None, goals, result,
        human_idx, robot_idx, num_agents, num_actions,
        num_episodes=args.video_episodes, label="Random Robot",
        output_path=os.path.join(args.output_dir, "demo_random.mp4"),
    )

    print("Recording EMPO robot episodes...")
    record_episodes(
        env, result.phase1_trainer, robot_greedy, goals, result,
        human_idx, robot_idx, num_agents, num_actions,
        num_episodes=args.video_episodes, label="EMPO Robot",
        output_path=os.path.join(args.output_dir, "demo_empo.mp4"),
    )

    print("\nDone! Videos saved to:", args.output_dir)


if __name__ == "__main__":
    main()
