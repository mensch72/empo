#!/usr/bin/env python3
"""
Phi Network Training Demo with Random Multigrid Ensemble.

This script demonstrates training a direct phi network (h_phi) that learns to predict
marginal policy priors without needing goals at inference time. The demo:

1. Trains BOTH a Q-network and a phi network jointly on a random multigrid ensemble
2. The phi network learns to approximate E_g[π(a|s,g)] - the marginal policy over goals
3. At inference time, the phi network can predict actions without specifying a goal
4. Produces a movie showing rollouts using the learned phi network

Key differences from Q-network-only training:
- The phi network provides goal-independent behavior
- Faster inference (no goal sampling or averaging needed)
- Useful when the "typical goal distribution" is what you want to model

Usage:
    python phi_network_ensemble_demo.py           # Full run (500 episodes)
    python phi_network_ensemble_demo.py --quick   # Quick test run (50 episodes)

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

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.nn_based.multigrid import (
    MultiGridQNetwork as QNetwork,
    DirectPhiNetwork,
    train_multigrid_neural_policy_prior as train_neural_policy_prior,
    MultiGridNeuralHumanPolicyPrior,
)


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 7           # 7x7 grid (including outer walls)
NUM_HUMANS = 3          # 3 human agents (yellow)
NUM_ROBOTS = 1          # 1 robot agent (grey)
MAX_STEPS = 20          # Maximum steps per episode

# Full training configuration (default)
NUM_TRAINING_EPISODES_FULL = 500
NUM_ROLLOUTS_FULL = 10

# Quick test configuration (for --quick flag)
NUM_TRAINING_EPISODES_QUICK = 50
NUM_ROLLOUTS_QUICK = 3

# Movie settings
MOVIE_FPS = 3           # Frames per second for movie output

# Object placement probabilities
WALL_PROBABILITY = 0.15
KEY_PROBABILITY = 0.05
BALL_PROBABILITY = 0.05
BOX_PROBABILITY = 0.03
DOOR_PROBABILITY = 0.03
LAVA_PROBABILITY = 0.02
BLOCK_PROBABILITY = 0.03


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phi Network Training Demo with Random Multigrid Ensemble"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced training episodes'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=None,
        help='Number of training episodes (overrides default)'
    )
    parser.add_argument(
        '--rollouts', '-r',
        type=int,
        default=None,
        help='Number of rollouts for the movie (overrides default)'
    )
    return parser.parse_args()


# ============================================================================
# Random Multigrid Environment Generator (same as ensemble demo)
# ============================================================================

class RandomMultigridEnv(MultiGridEnv):
    """
    A randomly generated multigrid environment with configurable agents and objects.
    """
    
    def __init__(
        self,
        grid_size: int = 7,
        num_humans: int = 3,
        num_robots: int = 1,
        max_steps: int = 20,
        seed: Optional[int] = None,
        wall_prob: float = 0.15,
        key_prob: float = 0.05,
        ball_prob: float = 0.05,
        box_prob: float = 0.03,
        door_prob: float = 0.03,
        lava_prob: float = 0.02,
        block_prob: float = 0.03
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        self.wall_prob = wall_prob
        self.key_prob = key_prob
        self.ball_prob = ball_prob
        self.box_prob = box_prob
        self.door_prob = door_prob
        self.lava_prob = lava_prob
        self.block_prob = block_prob
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        map_str = self._generate_random_map()
        
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
    
    def _generate_random_map(self) -> str:
        """Generate a random map string for the environment."""
        lines = []
        colors = ['r', 'g', 'b', 'p']
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')
                else:
                    r = random.random()
                    cumulative = 0
                    
                    cumulative += self.wall_prob
                    if r < cumulative:
                        row.append('We')
                        continue
                    
                    cumulative += self.lava_prob
                    if r < cumulative:
                        row.append('La')
                        continue
                    
                    cumulative += self.key_prob
                    if r < cumulative:
                        row.append(f'K{random.choice(colors)}')
                        continue
                    
                    cumulative += self.ball_prob
                    if r < cumulative:
                        row.append(f'B{random.choice(colors)}')
                        continue
                    
                    cumulative += self.box_prob
                    if r < cumulative:
                        row.append(f'X{random.choice(colors)}')
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')
                        continue
                    
                    row.append('..')
            
            lines.append(' '.join(row))
        
        grid_lines = [line.split() for line in lines]
        num_agents = self.num_humans + self.num_robots
        
        empty_cells = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if grid_lines[y][x] == '..':
                    empty_cells.append((x, y))
        
        while len(empty_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid_lines[y][x] != '..' and (x, y) not in empty_cells:
                        grid_lines[y][x] = '..'
                        empty_cells.append((x, y))
                        if len(empty_cells) >= num_agents:
                            break
                if len(empty_cells) >= num_agents:
                    break
        
        random.shuffle(empty_cells)
        agent_positions = empty_cells[:num_agents]
        
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid_lines[y][x] = 'Ay'
        
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid_lines[y][x] = 'Ae'
        
        return '\n'.join(' '.join(row) for row in grid_lines)


# ============================================================================
# Goal Definitions (needed for training)
# ============================================================================

class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = tuple(target_pos)
    
    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            if agent_state[0] == self.target_pos[0] and agent_state[1] == self.target_pos[1]:
                return 1
        return 0
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                self.target_pos == other.target_pos)


class DynamicGoalSampler(PossibleGoalSampler):
    """A goal sampler that dynamically adapts to changing environments."""
    
    def __init__(self, base_env: RandomMultigridEnv):
        super().__init__(base_env)
        self._update_goal_cells()
    
    def _update_goal_cells(self):
        self._goal_cells = []
        env = self.world_model
        for x in range(1, env.width - 1):
            for y in range(1, env.height - 1):
                cell = env.grid.get(x, y)
                if cell is None:
                    self._goal_cells.append((x, y))
                elif hasattr(cell, 'can_overlap') and cell.can_overlap():
                    self._goal_cells.append((x, y))
    
    def set_world_model(self, world_model):
        self.world_model = world_model
        self._update_goal_cells()
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        if not self._goal_cells:
            target_pos = (self.world_model.width // 2, self.world_model.height // 2)
        else:
            target_pos = random.choice(self._goal_cells)
        goal = ReachCellGoal(self.world_model, human_agent_index, target_pos)
        return goal, 1.0


# ============================================================================
# Training with Phi Network
# ============================================================================

def create_random_env(seed: int) -> RandomMultigridEnv:
    """Create a new random environment with the given seed."""
    return RandomMultigridEnv(
        grid_size=GRID_SIZE,
        num_humans=NUM_HUMANS,
        num_robots=NUM_ROBOTS,
        max_steps=MAX_STEPS,
        seed=seed,
        wall_prob=WALL_PROBABILITY,
        key_prob=KEY_PROBABILITY,
        ball_prob=BALL_PROBABILITY,
        box_prob=BOX_PROBABILITY,
        door_prob=DOOR_PROBABILITY,
        lava_prob=LAVA_PROBABILITY,
        block_prob=BLOCK_PROBABILITY
    )


def train_with_phi_network(
    human_agent_indices: List[int],
    num_episodes: int,
    device: str = 'cpu',
    verbose: bool = True,
    base_seed: int = 42
) -> MultiGridNeuralHumanPolicyPrior:
    """
    Train BOTH a Q-network and a phi network jointly on an ensemble.
    
    The phi network learns to predict marginal policies E_g[π(a|s,g)] and can
    be used for goal-independent inference.
    
    Args:
        human_agent_indices: Indices of human agents to model.
        num_episodes: Total number of training episodes.
        device: Torch device.
        verbose: Whether to print training progress.
        base_seed: Base random seed.
    
    Returns:
        MultiGridNeuralHumanPolicyPrior with trained Q-network AND phi network.
    """
    if verbose:
        print(f"Training Q-network AND Phi network on random environment ensemble...")
        print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
        print(f"  Agents: {NUM_HUMANS} humans + {NUM_ROBOTS} robot")
        print(f"  Training episodes: {num_episodes}")
        print(f"  Phi network: ENABLED (joint training)")
    
    # Create base environment
    base_env = create_random_env(seed=base_seed)
    base_env.reset()
    
    # Goal sampler that updates when environment changes
    goal_sampler = DynamicGoalSampler(base_env)
    
    # World model generator for ensemble training
    def world_model_generator(episode: int) -> RandomMultigridEnv:
        env = create_random_env(seed=base_seed + episode)
        env.reset()
        return env
    
    # Train with phi network ENABLED
    neural_prior = train_neural_policy_prior(
        world_model=base_env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=num_episodes,
        steps_per_episode=MAX_STEPS,
        beta=100.0,
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=128,
        replay_buffer_size=10000,
        updates_per_episode=4,
        # KEY: Enable phi network training!
        train_phi_network=True,
        phi_learning_rate=1e-3,
        phi_num_goal_samples=10,
        epsilon=0.3,
        exploration_policy=np.array([0.06, 0.19, 0.19, 0.56]),
        use_path_based_shaping=False,
        device=device,
        verbose=verbose,
        world_model_generator=world_model_generator,
        episodes_per_model=1
    )
    
    return neural_prior


# ============================================================================
# Rollout with Phi Network (Goal-Free!)
# ============================================================================

def get_phi_action(
    neural_prior: MultiGridNeuralHumanPolicyPrior,
    state,
    env: RandomMultigridEnv,
    human_idx: int,
    device: str = 'cpu'
) -> int:
    """
    Sample an action from the learned phi network (marginal policy).
    
    Unlike Q-network inference, this does NOT require a goal!
    The phi network has learned to predict E_g[π(a|s,g)] directly.
    """
    # Use the public __call__ API with goal=None for marginal policy
    action_probs = neural_prior(state, human_idx, goal=None)
    
    # Convert to tensor and sample
    probs = torch.tensor([action_probs[i] for i in range(len(action_probs))], device=device)
    action = torch.multinomial(probs.unsqueeze(0), 1).item()
    
    return action


def render_with_phi_overlay(
    env: RandomMultigridEnv,
    first_human_idx: int,
    tile_size: int = 32
) -> np.ndarray:
    """
    Render the environment with phi network indicator.
    
    Shows a purple circle around the first human to indicate
    goal-free phi network control.
    """
    img = env.render(mode='rgb_array', highlight=False)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Mark the first human with a purple circle (phi = goal-free)
    state = env.get_state()
    _, agent_states, _, _ = state
    if first_human_idx < len(agent_states):
        human_pos = agent_states[first_human_idx]
        hx = int(human_pos[0]) * tile_size + tile_size // 2
        hy = int(human_pos[1]) * tile_size + tile_size // 2
        # Purple ring indicates phi network (goal-free)
        ring = plt.Circle((hx, hy), tile_size * 0.45, fill=False,
                          color='purple', linewidth=3)
        ax.add_patch(ring)
        ax.text(hx, hy - tile_size * 0.35, 'φ', ha='center', va='center',
                fontsize=12, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
    
    ax.axis('off')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = buf[:, :, :3]
    
    plt.close(fig)
    return buf


def run_phi_rollout(
    env: RandomMultigridEnv,
    neural_prior: MultiGridNeuralHumanPolicyPrior,
    human_agent_indices: List[int],
    robot_index: int,
    first_human_idx: int,
    device: str = 'cpu'
) -> List[np.ndarray]:
    """
    Run a rollout using the PHI NETWORK (no goals needed!).
    
    The phi network provides goal-independent behavior by predicting
    the marginal policy E_g[π(a|s,g)].
    """
    env.reset()
    frames = []
    num_actions = env.action_space.n
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Render with phi indicator
        frame = render_with_phi_overlay(env, first_human_idx)
        frames.append(frame)
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(len(env.agents)):
            if agent_idx in human_agent_indices:
                # USE PHI NETWORK - NO GOAL NEEDED!
                action = get_phi_action(
                    neural_prior, state, env, agent_idx, device
                )
            else:
                # Robot uses random policy
                action = random.randint(0, num_actions - 1)
            actions.append(action)
        
        # Take step
        _, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Final frame
    frame = render_with_phi_overlay(env, first_human_idx)
    frames.append(frame)
    
    return frames


def create_phi_movie(
    all_frames: List[List[np.ndarray]],
    env_indices: List[int],
    output_path: str,
    num_rollouts: int,
    num_test_envs: int
):
    """Create a movie from phi network rollout frames."""
    print(f"Creating movie with {len(all_frames)} rollouts...")
    
    frames = []
    rollout_info = []
    
    for rollout_idx, (rollout_frames, env_idx) in enumerate(zip(all_frames, env_indices)):
        for frame_idx, frame in enumerate(rollout_frames):
            frames.append(frame)
            rollout_info.append((rollout_idx, frame_idx, env_idx))
    
    if len(frames) == 0:
        print("No frames to create movie!")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('', fontsize=12, fontweight='bold')
    
    def update(frame_idx):
        rollout_idx, step_idx, env_idx = rollout_info[frame_idx]
        im.set_array(frames[frame_idx])
        title.set_text(
            f'Rollout {rollout_idx + 1}/{num_rollouts} | Env {env_idx + 1}/{num_test_envs} | Step {step_idx}\n'
            f'φ = Phi Network (goal-free marginal policy) | Robot: random'
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

def main():
    args = parse_args()
    
    # Determine configuration
    if args.quick:
        num_episodes = args.episodes if args.episodes is not None else NUM_TRAINING_EPISODES_QUICK
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS_QUICK
        mode_str = "QUICK TEST MODE"
    else:
        num_episodes = args.episodes if args.episodes is not None else NUM_TRAINING_EPISODES_FULL
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS_FULL
        mode_str = "FULL MODE"
    
    print("=" * 70)
    print("Phi Network Training Demo with Random Multigrid Ensemble")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    print("This demo trains a PHI NETWORK that predicts marginal policies")
    print("E_g[π(a|s,g)] - enabling goal-free inference at test time!")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    
    # Create sample environment to identify agents
    sample_env = create_random_env(seed=42)
    sample_env.reset()
    
    human_agent_indices = []
    robot_index = None
    for i, agent in enumerate(sample_env.agents):
        if agent.color == 'yellow':
            human_agent_indices.append(i)
        elif agent.color == 'grey':
            robot_index = i
    
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Human agents (yellow): {human_agent_indices}")
    print(f"Robot agent (grey): {robot_index}")
    print()
    
    # Train with phi network
    t0 = time.time()
    neural_prior = train_with_phi_network(
        human_agent_indices=human_agent_indices,
        num_episodes=num_episodes,
        device=device,
        verbose=True,
        base_seed=42
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Check if phi network was trained
    if neural_prior.direct_phi_network is not None:
        print("✓ Phi network successfully trained!")
    else:
        print("⚠ Phi network not available - falling back to goal sampling")
    print()
    
    # Generate test environments
    # Use at least 3 environments, or half the number of rollouts (for variety)
    MIN_TEST_ENVS = 3
    num_test_envs = max(MIN_TEST_ENVS, num_rollouts // 2)
    print(f"Generating {num_test_envs} test environments for rollouts...")
    test_environments = [create_random_env(seed=1000 + i) for i in range(num_test_envs)]
    for env in test_environments:
        env.reset()
    
    # Run rollouts using PHI NETWORK (goal-free!)
    print(f"Running {num_rollouts} rollouts using PHI NETWORK (no goals needed!)...")
    all_frames = []
    env_indices = []
    
    first_human_idx = human_agent_indices[0] if human_agent_indices else 0
    
    for rollout_idx in range(num_rollouts):
        env_idx = rollout_idx % len(test_environments)
        env = test_environments[env_idx]
        env.reset()
        
        # Update neural_prior's world_model for this environment
        neural_prior.world_model = env
        if neural_prior.goal_sampler is not None:
            neural_prior.goal_sampler.set_world_model(env)
        
        print(f"  Rollout {rollout_idx + 1}: Env {env_idx + 1} (using phi network - no goal!)")
        
        frames = run_phi_rollout(
            env=env,
            neural_prior=neural_prior,
            human_agent_indices=human_agent_indices,
            robot_index=robot_index,
            first_human_idx=first_human_idx,
            device=device
        )
        all_frames.append(frames)
        env_indices.append(env_idx)
        print(f"    Captured {len(frames)} frames")
    
    print()
    
    # Create movie
    movie_path = os.path.join(output_dir, 'phi_network_ensemble_demo.mp4')
    create_phi_movie(all_frames, env_indices, movie_path, num_rollouts, num_test_envs)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print()
    print("Key takeaway: The phi network predicts actions WITHOUT needing goals!")
    print("It has learned E_g[π(a|s,g)] - the marginal policy over all goals.")
    print("=" * 70)


if __name__ == "__main__":
    main()
