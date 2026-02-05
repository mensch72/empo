#!/usr/bin/env python3
"""
MCTS Test - Test EMPO framework with heuristic policies.

This script:
1. Loads environment with rocks_can_kill enabled
2. Lets humans settle (escape danger, then stay still)
3. Runs MCTS with robot to push rocks and maximize human power
4. Outputs video only (no visualization plots)

Usage:
    PYTHONPATH=src:vendor/multigrid python examples/mcts/mcts_test.py
    PYTHONPATH=src:vendor/multigrid python examples/mcts/mcts_test.py --config path/to/config.yaml
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig, MinRobotRiskHumanPolicy


ACTION_NAMES = {
    0: "Still",
    1: "Left",
    2: "Right",
    3: "Forward",
    4: "Pickup",
    5: "Drop",
    6: "Toggle",
    7: "Done",
}


def convert_gif_to_mp4(gif_path: str, mp4_path: str, fps: int = 2) -> bool:
    """Convert GIF to MP4 using ffmpeg."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", gif_path,
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r", str(fps),
            mp4_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def settle_humans(
    env: MultiGridEnv,
    human_prior: MinRobotRiskHumanPolicy,
    human_agent_indices: list,
    robot_agent_indices: list,
    max_settle_steps: int = 20,
    verbose: bool = True,
) -> int:
    """
    Let humans settle by taking actions until they stop moving.

    Returns number of steps taken to settle.
    """
    if verbose:
        print("\n[SETTLE] Letting humans escape danger and settle...")

    for step in range(max_settle_steps):
        state = env.get_state()

        # Get human actions
        human_actions = []
        any_movement = False
        for idx in human_agent_indices:
            action = human_prior.best_action(state, idx)
            human_actions.append(action)
            if action != 0:  # 0 = Still
                any_movement = True

        # If all humans staying still, we're settled
        if not any_movement:
            if verbose:
                print(f"[SETTLE] Humans settled after {step} steps")
            return step

        # Combine with robot staying still
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
        full_actions = [0] * num_agents
        for i, idx in enumerate(human_agent_indices):
            full_actions[idx] = human_actions[i]
        for idx in robot_agent_indices:
            full_actions[idx] = 0  # Robot stays still during settling

        # Take step
        step_result = env.step(full_actions)
        env.render(mode='rgb_array')  # Capture frame for video

        # Check if any human got terminated
        if len(step_result) == 5:
            _, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            _, _, done, _ = step_result

        if done:
            if verbose:
                print(f"[SETTLE] Episode ended during settling at step {step}")
            return step

    if verbose:
        print(f"[SETTLE] Max settle steps ({max_settle_steps}) reached")
    return max_settle_steps


def run_mcts_episode(
    env: MultiGridEnv,
    planner: MCTSPlanner,
    human_prior: MinRobotRiskHumanPolicy,
    human_agent_indices: list,
    robot_agent_indices: list,
    max_steps: int = 50,
    verbose: bool = True,
) -> dict:
    """Run MCTS episode after humans have settled."""

    state = env.get_state()
    total_reward = 0.0
    step_count = 0
    action_history = []
    humans_terminated = []

    if verbose:
        print(f"\n[MCTS] Running robot policy with MCTS...")

    for step in range(max_steps):
        if verbose and step % 5 == 0:
            print(f"[MCTS] Step {step}/{max_steps}")

        # Run MCTS search
        result = planner.search_with_result(state)
        best_robot_action = result.best_action

        # Get human actions (should mostly be Still now)
        human_actions = []
        for idx in human_agent_indices:
            action = human_prior.best_action(state, idx)
            human_actions.append(action)

        # Combine actions
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
        full_actions = [0] * num_agents
        for i, idx in enumerate(human_agent_indices):
            full_actions[idx] = human_actions[i]
        for i, idx in enumerate(robot_agent_indices):
            full_actions[idx] = best_robot_action[i]

        robot_action_name = ACTION_NAMES.get(best_robot_action[0], str(best_robot_action[0]))
        action_history.append(robot_action_name)

        # Take step
        step_result = env.step(full_actions)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        env.render(mode='rgb_array')

        # Check for terminated humans
        for idx in human_agent_indices:
            if hasattr(env.agents[idx], 'terminated') and env.agents[idx].terminated:
                if idx not in humans_terminated:
                    humans_terminated.append(idx)
                    if verbose:
                        print(f"[MCTS] Human {idx} was terminated by rock!")

        state = env.get_state()
        total_reward += reward if isinstance(reward, (int, float)) else sum(reward)
        step_count = step + 1

        if done:
            if verbose:
                print(f"[MCTS] Episode finished at step {step + 1}")
            break

    return {
        'steps': step_count,
        'total_reward': total_reward,
        'done': done if 'done' in dir() else False,
        'action_history': action_history,
        'humans_terminated': humans_terminated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS Test - EMPO Framework Test")
    default_config = (
        Path(__file__).resolve().parents[2]
        / "multigrid_worlds"
        / "puzzles"
        / "ali_challenges"
        / "mcts_simple.yaml"
    )
    parser.add_argument("--config", default=str(default_config), help="Path to YAML environment")
    parser.add_argument("--steps", type=int, default=40, help="Maximum MCTS steps")
    parser.add_argument("--settle-steps", type=int, default=20, help="Max steps for human settling")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per step")
    parser.add_argument("--depth", type=int, default=20, help="Max rollout depth")
    parser.add_argument("--output", default="outputs/mcts_test.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=3, help="Video frames per second")

    # EMPO parameters
    parser.add_argument("--zeta", type=float, default=2.0, help="Risk aversion")
    parser.add_argument("--xi", type=float, default=1.0, help="Inequality aversion")
    parser.add_argument("--eta", type=float, default=1.1, help="Intertemporal aversion")
    parser.add_argument("--beta-r", type=float, default=5.0, help="Robot rationality")

    args = parser.parse_args()

    print("=" * 70)
    print("MCTS Test - EMPO Framework with Heuristic Policies")
    print("=" * 70)

    # Load environment
    print(f"\n[INIT] Loading environment: {Path(args.config).stem}")
    env = MultiGridEnv(config_file=args.config, partial_obs=False)
    env.reset()
    print(f"[INIT] Grid size: {env.width}x{env.height}")
    print(f"[INIT] rocks_can_kill: {getattr(env, 'rocks_can_kill', False)}")

    # Determine human/robot indices
    human_agent_indices = []
    robot_agent_indices = []
    if hasattr(env, 'agents'):
        for idx, agent in enumerate(env.agents):
            if getattr(agent, 'color', None) == 'grey':
                robot_agent_indices.append(idx)
            else:
                human_agent_indices.append(idx)

    if not human_agent_indices:
        human_agent_indices = [0]
    if not robot_agent_indices:
        robot_agent_indices = [i for i in range(len(env.agents)) if i not in human_agent_indices]

    print(f"[INIT] Humans: {human_agent_indices}, Robots: {robot_agent_indices}")

    # Create human policy prior
    human_prior = MinRobotRiskHumanPolicy(
        env, human_agent_indices, robot_agent_indices,
        beta=5.0, danger_penalty=10.0
    )

    # Create MCTS planner
    config = MCTSConfig(
        num_simulations=args.sims,
        max_depth=args.depth,
        verbose=False,
        use_transition_probabilities=False,
        zeta=args.zeta,
        xi=args.xi,
        eta=args.eta,
        beta_r=args.beta_r,
    )

    planner = MCTSPlanner(
        world_model=env,
        human_policy_prior=human_prior,
        human_agent_indices=human_agent_indices,
        robot_agent_indices=robot_agent_indices,
        config=config,
    )

    print(f"\n[CONFIG] EMPO: zeta={args.zeta}, xi={args.xi}, eta={args.eta}, beta_r={args.beta_r}")
    print(f"[CONFIG] MCTS: sims={args.sims}, depth={args.depth}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start video recording
    print(f"\n[VIDEO] Recording to {output_path}")
    env.start_video_recording()
    env.render(mode='rgb_array')  # Initial frame

    # Phase 1: Let humans settle
    settle_steps = settle_humans(
        env, human_prior, human_agent_indices, robot_agent_indices,
        max_settle_steps=args.settle_steps, verbose=True
    )

    # Phase 2: Run MCTS
    results = run_mcts_episode(
        env, planner, human_prior, human_agent_indices, robot_agent_indices,
        max_steps=args.steps, verbose=True
    )

    # Save video
    print(f"\n[VIDEO] Saving video...")
    gif_path = output_path.with_suffix('.gif')
    env.save_video(str(gif_path), fps=args.fps)

    # Convert to MP4
    if output_path.suffix.lower() == '.mp4':
        if convert_gif_to_mp4(str(gif_path), str(output_path), fps=args.fps):
            print(f"[VIDEO] MP4 created: {output_path}")
        else:
            print(f"[VIDEO] MP4 conversion failed, GIF: {gif_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Settle steps: {settle_steps}")
    print(f"  MCTS steps: {results['steps']}")
    print(f"  Total reward: {results['total_reward']:.2f}")
    print(f"  Humans terminated: {len(results['humans_terminated'])} {results['humans_terminated']}")
    print(f"  Robot actions: {' -> '.join(results.get('action_history', [])[:20])}")
    if len(results.get('action_history', [])) > 20:
        print(f"  ... ({len(results['action_history'])} total)")
    print(f"\n  Video: {output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
