#!/usr/bin/env python3
"""
Phase 2 Robot Policy Learning Demo.

This script demonstrates Phase 2 of the EMPO framework - learning a robot policy
that maximizes aggregate human power. Based on equations (4)-(9) from the paper.

Environment:
- 5x5 grid with 2 human agents (yellow) and 2 robot agents (grey)
- Humans follow HeuristicPotentialPolicy toward their goals
- Robots learn to help humans achieve their goals

The demo trains:
- Q_r: Robot state-action value (eq. 4)
- π_r: Robot policy using power-law softmax (eq. 5)
- V_h^e: Human goal achievement under robot policy (eq. 6)
- X_h: Aggregate goal achievement ability (eq. 7)
- U_r: Intrinsic robot reward (eq. 8)
- V_r: Robot state value (eq. 9)

Usage:
    python phase2_robot_policy_demo.py           # Full run (1000 episodes)
    python phase2_robot_policy_demo.py --quick   # Quick test (100 episodes)
    python phase2_robot_policy_demo.py --debug   # Enable verbose debug output

Output:
- TensorBoard logs in outputs/phase2_demo/
- Movie of 100 rollouts with the learned policy
"""

import sys
import os
import time
import random
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.multigrid import MultiGridGoalSampler, ReachCellGoal
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid import PathDistanceCalculator
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.multigrid.phase2 import train_multigrid_phase2


# ============================================================================
# Environment Definition
# ============================================================================

GRID_MAP = """
We We We We We We We
We .. .. .. .. .. We
We .. Ae .. Ay .. We
We .. .. .. .. .. We
We .. Ay .. Ae .. We
We .. .. .. .. .. We
We We We We We We We
"""

MAX_STEPS = 20
NUM_ROLLOUTS = 100  # Rollouts for final movie
MOVIE_FPS = 3


class Phase2DemoEnv(MultiGridEnv):
    """
    A simple 5x5 grid environment for Phase 2 demo.
    
    - 2 yellow agents (humans)
    - 2 grey agents (robots)
    """
    
    def __init__(self, max_steps: int = MAX_STEPS):
        super().__init__(
            map=GRID_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')


# ============================================================================
# Rollout and Movie Generation
# ============================================================================

def run_policy_rollout(
    env: MultiGridEnv,
    robot_q_network,
    human_policy,
    goal_sampler,
    human_indices,
    robot_indices,
    device: str = 'cpu'
):
    """
    Run a single rollout with the learned robot policy.
    
    Returns list of frames for movie generation.
    """
    env.reset()
    frames = []
    num_actions = env.action_space.n
    
    # Sample initial goals for humans
    state = env.get_state()
    human_goals = {}
    for h in human_indices:
        goal, _ = goal_sampler.sample(state, h)
        human_goals[h] = goal
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Render frame
        frame = env.render(mode='rgb_array', highlight=False)
        frames.append(frame)
        
        # Get actions
        actions = [0] * len(env.agents)
        
        # Humans use heuristic policy
        for h in human_indices:
            actions[h] = human_policy.sample(state, h, human_goals[h])
        
        # Robots use learned Q-network
        with torch.no_grad():
            q_values = robot_q_network.encode_and_forward(state, env, device)
            # Use greedy action (epsilon=0)
            robot_action = robot_q_network.sample_action(q_values, epsilon=0.0)
            
            # Assign actions to robots
            for i, r in enumerate(robot_indices):
                if i < len(robot_action):
                    actions[r] = robot_action[i]
        
        # Step environment
        _, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Final frame
    frame = env.render(mode='rgb_array', highlight=False)
    frames.append(frame)
    
    return frames


def create_rollout_movie(
    all_frames,
    output_path: str,
    num_rollouts: int
):
    """Create a movie from rollout frames."""
    print(f"Creating movie with {len(all_frames)} rollouts...")
    
    frames = []
    rollout_info = []
    
    for rollout_idx, rollout_frames in enumerate(all_frames):
        for frame_idx, frame in enumerate(rollout_frames):
            frames.append(frame)
            rollout_info.append((rollout_idx, frame_idx))
    
    if len(frames) == 0:
        print("No frames to create movie!")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        rollout_idx, step_idx = rollout_info[frame_idx]
        im.set_array(frames[frame_idx])
        title.set_text(
            f'Rollout {rollout_idx + 1}/{num_rollouts} | Step {step_idx}\n'
            f'Humans: heuristic policy | Robots: learned Q_r policy'
        )
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=300, blit=True, repeat=True
    )
    
    try:
        writer = animation.FFMpegWriter(fps=MOVIE_FPS, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✓ Movie saved to {output_path}")
    except Exception as e:
        print(f"Could not save MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=MOVIE_FPS)
            print(f"✓ Movie saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"Error saving movie: {e2}")
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(quick_mode: bool = False, debug: bool = False):
    """Run Phase 2 demo."""
    print("=" * 70)
    print("Phase 2 Robot Policy Learning Demo")
    print("Learning robot policy to maximize aggregate human power")
    print("Based on EMPO paper equations (4)-(9)")
    print("=" * 70)
    print()
    
    # Configuration
    if quick_mode:
        num_episodes = 100
        num_rollouts = 10
        print("[QUICK MODE] Running with reduced episodes and rollouts")
    else:
        num_episodes = 1000
        num_rollouts = NUM_ROLLOUTS
    
    device = 'cpu'
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'phase2_demo')
    os.makedirs(output_dir, exist_ok=True)
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    
    # Create environment
    print("Creating environment...")
    env = Phase2DemoEnv(max_steps=MAX_STEPS)
    env.reset()
    
    # Identify agents
    human_indices = []
    robot_indices = []
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_indices.append(i)
            print(f"  Human {i}: pos={tuple(agent.pos)}")
        elif agent.color == 'grey':
            robot_indices.append(i)
            print(f"  Robot {i}: pos={tuple(agent.pos)}")
    
    print(f"  Grid: {env.width}x{env.height}")
    print(f"  Humans: {len(human_indices)}")
    print(f"  Robots: {len(robot_indices)}")
    print()
    
    # Create path calculator for heuristic policy
    print("Creating path calculator and human policy...")
    path_calc = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env
    )
    
    # Create human policy using existing HeuristicPotentialPolicy
    human_policy = HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_indices,
        path_calculator=path_calc,
        softness=10.0  # Moderately deterministic
    )
    
    # Create goal sampler using existing MultiGridGoalSampler
    goal_sampler = MultiGridGoalSampler(env)
    
    # Wrapper to adapt goal sampler to trainer's expected interface
    def goal_sampler_fn(state, human_idx):
        goal, _ = goal_sampler.sample(state, human_idx)
        return goal
    
    # Wrapper to adapt human policy to trainer's expected interface
    def human_policy_fn(state, human_idx, goal):
        return human_policy.sample(state, human_idx, goal)
    
    print()
    
    # Phase 2 configuration
    config = Phase2Config(
        gamma_r=0.99,
        gamma_h=0.99,
        zeta=2.0,      # Risk preference
        xi=1.0,        # Inter-human inequality aversion
        eta=1.1,       # Intertemporal inequality aversion
        beta_r=5.0,    # Robot policy concentration
        epsilon_r_start=1.0,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=num_episodes * 10,
        lr_q_r=1e-3,
        lr_v_r=1e-3,
        lr_v_h_e=1e-3,
        lr_x_h=1e-3,
        lr_u_r=1e-3,
        buffer_size=10000,
        batch_size=32,
        num_episodes=num_episodes,
        steps_per_episode=env.max_steps,
        updates_per_step=1,
        goal_resample_prob=0.1,
    )
    
    # Train Phase 2
    print("Training Phase 2 robot policy...")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Steps per episode: {config.steps_per_episode}")
    print(f"  TensorBoard: {tensorboard_dir}")
    print()
    
    t0 = time.time()
    robot_q_network, networks, history = train_multigrid_phase2(
        world_model=env,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_policy_prior=human_policy_fn,
        goal_sampler=goal_sampler_fn,
        config=config,
        hidden_dim=128,
        device=device,
        verbose=True,
        debug=debug,
        tensorboard_dir=tensorboard_dir
    )
    elapsed = time.time() - t0
    
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Show loss history
    if history and len(history) > 0:
        print("\nLoss history (last 5 episodes):")
        for i, losses in enumerate(history[-5:]):
            episode_num = len(history) - 5 + i
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if v > 0)
            print(f"  Episode {episode_num}: {loss_str}")
    
    # Generate rollout movie
    print(f"\nGenerating {num_rollouts} rollouts with learned policy...")
    all_frames = []
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        frames = run_policy_rollout(
            env=env,
            robot_q_network=robot_q_network,
            human_policy=human_policy,
            goal_sampler=goal_sampler,
            human_indices=human_indices,
            robot_indices=robot_indices,
            device=device
        )
        all_frames.append(frames)
        if (rollout_idx + 1) % 10 == 0:
            print(f"  Completed {rollout_idx + 1}/{num_rollouts} rollouts")
    
    # Create movie
    movie_path = os.path.join(output_dir, 'phase2_robot_policy_demo.mp4')
    create_rollout_movie(all_frames, movie_path, num_rollouts)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"  TensorBoard logs: {tensorboard_dir}")
    print(f"  View with: tensorboard --logdir={tensorboard_dir}")
    print(f"  Movie: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2 Robot Policy Learning Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable verbose debug output')
    args = parser.parse_args()
    main(quick_mode=args.quick, debug=args.debug)
