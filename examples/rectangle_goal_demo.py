#!/usr/bin/env python3
"""
Rectangle Goal Training Demo.

This script demonstrates training a neural policy prior with rectangle goals,
where the goal is to reach any cell within a target rectangular region.

The demo:
1. Trains a Q-network with rectangle goal support
2. Uses path-based reward shaping with rectangle goals
3. Produces a movie showing agents navigating to rectangle goal regions

Key features:
- All goals are bounding boxes (x1, y1, x2, y2) with inclusive coordinates
- Point goals are represented as (x, y, x, y)
- PathDistanceCalculator computes shortest path to any cell in the rectangle
- Goal encoder encodes bounding box coordinates directly

Usage:
    python rectangle_goal_demo.py           # Full run (300 episodes)
    python rectangle_goal_demo.py --quick   # Quick test run (30 episodes)

Requirements:
    - torch
    - matplotlib
    - ffmpeg (optional, for MP4 output; falls back to GIF)
"""

import sys
import os
import time
import random
import argparse
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.world_specific_helpers.multigrid import (
    ReachRectangleGoal,
    MultiGridGoalSampler,
    render_goal_overlay,
)
from empo.learning_based.multigrid import (
    MultiGridQNetwork as QNetwork,
    train_multigrid_neural_policy_prior as train_neural_policy_prior,
    PathDistanceCalculator,
)


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 9           # 9x9 grid (including outer walls)
NUM_HUMANS = 2          # 2 human agents (yellow)
NUM_ROBOTS = 1          # 1 robot agent (grey)
MAX_STEPS = 30          # Maximum steps per episode

# Full training configuration (default)
NUM_TRAINING_EPISODES_FULL = 300
NUM_ROLLOUTS_FULL = 8

# Quick test configuration (for --quick flag)
NUM_TRAINING_EPISODES_QUICK = 30
NUM_ROLLOUTS_QUICK = 3

# Movie settings
MOVIE_FPS = 3

# Wall probability
WALL_PROBABILITY = 0.1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rectangle Goal Training Demo"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced training'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=None,
        help='Number of training episodes'
    )
    return parser.parse_args()


# ============================================================================
# Environment
# ============================================================================

class RectangleGoalEnv(MultiGridEnv):
    """
    Simple environment for rectangle goal training.
    """
    
    def __init__(
        self,
        grid_size: int = 9,
        num_humans: int = 2,
        num_robots: int = 1,
        max_steps: int = 30,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        map_str = self._generate_map()
        
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
    
    def _generate_map(self) -> str:
        """Generate a random map."""
        lines = []
        available_cells = []
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')  # Wall
                else:
                    if np.random.random() < WALL_PROBABILITY:
                        row.append('We')
                    else:
                        row.append('..')
                        available_cells.append((x, y))
            lines.append(' '.join(row))
        
        grid_lines = [line.split() for line in lines]
        
        # Ensure enough cells
        num_agents = self.num_humans + self.num_robots
        while len(available_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid_lines[y][x] == 'We' and (x, y) not in available_cells:
                        grid_lines[y][x] = '..'
                        available_cells.append((x, y))
                        if len(available_cells) >= num_agents:
                            break
                if len(available_cells) >= num_agents:
                    break
        
        random.shuffle(available_cells)
        agent_positions = available_cells[:num_agents]
        
        # Place human agents (yellow)
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid_lines[y][x] = 'Ay'
        
        # Place robot agents (grey)
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid_lines[y][x] = 'Ae'
        
        return '\n'.join(' '.join(row) for row in grid_lines)


# ============================================================================
# Training
# ============================================================================

def train_rectangle_goal_policy(
    num_episodes: int,
    device: str = 'cpu',
    verbose: bool = True
):
    """Train a neural policy prior with rectangle goals."""
    print("=" * 60)
    print("Training Neural Policy Prior with Rectangle Goals")
    print(f"  Episodes: {num_episodes}")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Agents: {NUM_HUMANS} humans + {NUM_ROBOTS} robot")
    print("=" * 60)
    
    # Create base environment
    base_env = RectangleGoalEnv(
        grid_size=GRID_SIZE,
        num_humans=NUM_HUMANS,
        num_robots=NUM_ROBOTS,
        max_steps=MAX_STEPS,
        seed=42
    )
    base_env.reset()
    
    # Identify human agents
    human_agent_indices = [i for i, a in enumerate(base_env.agents) if a.color == 'yellow']
    
    print(f"  Human agent indices: {human_agent_indices}")
    
    # Create rectangle goal sampler with weight-proportional sampling
    # This samples goals with P(goal) ∝ (1+x2-x1)*(1+y2-y1)
    goal_sampler = MultiGridGoalSampler(base_env)
    
    # World model generator for ensemble training
    def world_model_generator(episode: int):
        env = RectangleGoalEnv(
            grid_size=GRID_SIZE,
            num_humans=NUM_HUMANS,
            num_robots=NUM_ROBOTS,
            max_steps=MAX_STEPS,
            seed=42 + episode
        )
        env.reset()
        return env
    
    # Train (all goals are rectangles now)
    t0 = time.time()
    neural_prior = train_neural_policy_prior(
        world_model=base_env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=num_episodes,
        steps_per_episode=MAX_STEPS,
        beta=50.0,
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        replay_buffer_size=10000,
        updates_per_episode=4,
        reward_shaping=True,
        epsilon=0.3,
        device=device,
        verbose=verbose,
        world_model_generator=world_model_generator,
        episodes_per_model=1
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    
    return neural_prior, base_env, human_agent_indices, goal_sampler


# ============================================================================
# Visualization
# ============================================================================

def render_with_rectangle_overlay(
    env,
    rectangle_goals: Dict[int, ReachRectangleGoal],
    value_dict: Optional[Dict[Tuple[int, int], float]] = None
):
    """
    Render environment with rectangle goal overlays.
    
    Uses render_goal_overlay from empo.multigrid for dashed blue rectangle
    boundaries and agent-to-goal connection lines.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Render base environment
    img = env.render(mode='rgb_array')
    ax.imshow(img)
    
    # Get cell size in pixels
    tile_size = img.shape[1] // env.width
    
    # Get agent positions from environment state
    state = env.get_state()
    _, agent_states, _, _ = state
    
    # Draw rectangle goals using rendering from empo.multigrid
    for agent_idx, goal in rectangle_goals.items():
        if agent_idx < len(agent_states):
            agent_state = agent_states[agent_idx]
            agent_pos = (float(agent_state[0]), float(agent_state[1]))
            
            render_goal_overlay(
                ax=ax,
                goal=goal,
                agent_pos=agent_pos,
                agent_idx=agent_idx,
                tile_size=tile_size,
                goal_color=(0.0, 0.4, 1.0, 0.7),  # Blue, semi-transparent
                line_width=2.5,
                inset=0.08
            )
    
    ax.axis('off')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    
    fig.tight_layout(pad=0)
    
    # Convert to array
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = frame[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    
    return frame


def run_rollouts(
    neural_prior,
    env,
    human_agent_indices: List[int],
    goal_sampler: MultiGridGoalSampler,
    num_rollouts: int,
    device: str = 'cpu'
):
    """Run rollouts and create animation."""
    print()
    print("=" * 60)
    print(f"Running {num_rollouts} Rollouts with Rectangle Goals")
    print("=" * 60)
    
    all_frames = []
    num_actions = env.action_space.n
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        state = env.get_state()
        
        # Update goal sampler
        goal_sampler.set_world_model(env)
        
        # Sample rectangle goals for each human
        human_goals = {}
        for h_idx in human_agent_indices:
            goal, _ = goal_sampler.sample(state, h_idx)
            human_goals[h_idx] = goal
        
        print(f"\nRollout {rollout_idx + 1}:")
        for h_idx, goal in human_goals.items():
            print(f"  Agent {h_idx}: {goal}")
        
        goals_achieved = {h_idx: False for h_idx in human_agent_indices}
        
        for step in range(env.max_steps):
            state = env.get_state()
            
            # Render frame
            frame = render_with_rectangle_overlay(env, human_goals)
            all_frames.append(frame)
            
            # Check achievements
            for h_idx in human_agent_indices:
                if human_goals[h_idx].is_achieved(state):
                    goals_achieved[h_idx] = True
            
            # Get actions
            actions = []
            for agent_idx in range(len(env.agents)):
                if agent_idx in human_agent_indices:
                    goal = human_goals[agent_idx]
                    
                    # Get policy from neural prior
                    action_dist = neural_prior(state, agent_idx, goal)
                    probs = np.array([action_dist.get(i, 0.0) for i in range(num_actions)])
                    probs = probs / probs.sum()
                    action = np.random.choice(num_actions, p=probs)
                else:
                    # Robot uses random policy
                    action = np.random.randint(num_actions)
                
                actions.append(action)
            
            # Step
            _, _, done, _ = env.step(actions)
            if done:
                break
        
        # Final frame
        frame = render_with_rectangle_overlay(env, human_goals)
        all_frames.append(frame)
        
        # Report
        achieved = sum(1 for v in goals_achieved.values() if v)
        print(f"  Goals achieved: {achieved}/{len(human_agent_indices)}")
    
    return all_frames


def save_animation(frames: List[np.ndarray], output_path: str, fps: int = 3):
    """Save animation as MP4 or GIF."""
    print(f"\nSaving animation to {output_path}...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000/fps, blit=True
    )
    
    # Try MP4 first, fall back to GIF
    try:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output_path, writer=writer)
    except Exception as e:
        print(f"  MP4 failed ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=fps)
        output_path = gif_path
    
    plt.close(fig)
    print(f"  Saved to {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Configure based on mode
    if args.quick:
        num_episodes = args.episodes or NUM_TRAINING_EPISODES_QUICK
        num_rollouts = NUM_ROLLOUTS_QUICK
        mode_str = "QUICK TEST MODE"
    else:
        num_episodes = args.episodes or NUM_TRAINING_EPISODES_FULL
        num_rollouts = NUM_ROLLOUTS_FULL
        mode_str = "FULL MODE"
    
    print()
    print("=" * 70)
    print("Rectangle Goal Training Demo")
    print(f"  [{mode_str}]")
    print("Goals are rectangular regions, not just single cells")
    print("=" * 70)
    print()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    
    # Train
    neural_prior, env, human_agent_indices, goal_sampler = train_rectangle_goal_policy(
        num_episodes=num_episodes,
        device=device,
        verbose=True
    )
    
    # Run rollouts
    frames = run_rollouts(
        neural_prior,
        env,
        human_agent_indices,
        goal_sampler,
        num_rollouts=num_rollouts,
        device=device
    )
    
    # Save animation
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rectangle_goal_demo.mp4')
    save_animation(frames, output_path, fps=MOVIE_FPS)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  - All goals are bounding boxes (x1, y1, x2, y2) with inclusive coordinates")
    print("  - Point goals are represented as (x, y, x, y)")
    print("  - PathDistanceCalculator computes distances to rectangles via BFS")
    print("  - Goal encoder directly encodes bounding box coordinates")
    print("  - Reward shaping uses potential based on distance to rectangle")
    print()


if __name__ == "__main__":
    main()
