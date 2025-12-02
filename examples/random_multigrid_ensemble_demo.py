#!/usr/bin/env python3
"""
Random Multigrid Ensemble Policy Prior Learning Demo.

This script demonstrates neural network-based policy prior learning on an ensemble
of randomly generated small multigrids. The demo:

1. Generates random 7x7 multigrids with:
   - 3 human agents (yellow)
   - 1 robot agent (grey)
   - Random internal walls
   - Random objects (keys, balls, boxes, doors, lava, blocks)
   
2. Trains a neural network policy prior on this ensemble by:
   - Cycling through environments during training
   - Learning goal-conditioned policies that generalize across layouts
   
3. Produces a movie with 10 rollouts across different environments to show
   generalization of the learned policy.

Usage:
    python random_multigrid_ensemble_demo.py           # Full run (500 episodes)
    python random_multigrid_ensemble_demo.py --quick   # Quick test run (50 episodes)

Requirements:
    - torch
    - matplotlib
    - ffmpeg (optional, for MP4 output; falls back to GIF)

NOTE FOR FUTURE EXAMPLES:
    When creating long-running example scripts, always include a command-line
    parameter (e.g., --quick, --test, --fast) that shortens the run time for
    testing purposes. This allows developers to quickly verify the script works
    without waiting for the full training run.
"""

import sys
import os
import time
import random
import argparse
from typing import List, Tuple, Dict, Any, Optional

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.nn_based import (
    QNetwork,
    train_neural_policy_prior,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
)


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 7           # 7x7 grid (including outer walls)
NUM_HUMANS = 3          # 3 human agents (yellow)
NUM_ROBOTS = 1          # 1 robot agent (grey)
MAX_STEPS = 20          # Maximum steps per episode
NUM_TEST_ENVS = 10      # Number of test environments for rollout evaluation
NUM_ROLLOUTS = 10       # Number of rollouts for the movie

# Full training configuration (default)
NUM_TRAINING_EPISODES_FULL = 500

# Quick test configuration (for --quick flag)
NUM_TRAINING_EPISODES_QUICK = 50
NUM_TEST_ENVS_QUICK = 3
NUM_ROLLOUTS_QUICK = 3

# Object placement probabilities
WALL_PROBABILITY = 0.15      # Probability of placing internal walls
KEY_PROBABILITY = 0.05       # Probability of placing a key
BALL_PROBABILITY = 0.05      # Probability of placing a ball
BOX_PROBABILITY = 0.03       # Probability of placing a box
DOOR_PROBABILITY = 0.03      # Probability of placing a door
LAVA_PROBABILITY = 0.02      # Probability of placing lava
BLOCK_PROBABILITY = 0.03     # Probability of placing a block


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Random Multigrid Ensemble Policy Prior Learning Demo"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced training episodes and environments'
    )
    parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=None,
        help='Number of training episodes (overrides default)'
    )
    parser.add_argument(
        '--envs', '-n',
        type=int,
        default=None,
        help='Number of test environments for rollout evaluation (overrides default)'
    )
    parser.add_argument(
        '--rollouts', '-r',
        type=int,
        default=None,
        help='Number of rollouts for the movie (overrides default)'
    )
    return parser.parse_args()


# ============================================================================
# Random Multigrid Environment Generator
# ============================================================================

class RandomMultigridEnv(MultiGridEnv):
    """
    A randomly generated multigrid environment with configurable agents and objects.
    
    The environment creates a grid with:
    - Outer walls on all edges
    - Random internal walls and obstacles
    - Random objects (keys, balls, boxes, doors, lava, blocks)
    - Specified number of human (yellow) and robot (grey) agents
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
        """
        Initialize the random multigrid environment.
        
        Args:
            grid_size: Size of the grid (including outer walls).
            num_humans: Number of human agents (yellow).
            num_robots: Number of robot agents (grey).
            max_steps: Maximum steps per episode.
            seed: Random seed for reproducibility.
            wall_prob: Probability of internal walls.
            key_prob: Probability of placing keys.
            ball_prob: Probability of placing balls.
            box_prob: Probability of placing boxes.
            door_prob: Probability of placing doors.
            lava_prob: Probability of placing lava.
            block_prob: Probability of placing blocks.
        """
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
        
        # Build the map string
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
        
        # Valid color codes: r=red, g=green, b=blue, p=purple, y=yellow, e=grey
        colors = ['r', 'g', 'b', 'p']  # Not using y or e as they're for agents
        
        # Track which cells are available for agents
        available_cells = []
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                # Outer walls
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')  # Grey wall
                else:
                    # Inner cells - randomly place objects
                    r = random.random()
                    cumulative = 0
                    
                    cumulative += self.wall_prob
                    if r < cumulative:
                        row.append('We')  # Grey wall
                        continue
                    
                    cumulative += self.lava_prob
                    if r < cumulative:
                        row.append('La')  # Lava
                        continue
                    
                    cumulative += self.key_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'K{color}')  # Key
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.ball_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'B{color}')  # Ball
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.box_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'X{color}')  # Box
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')  # Block (no color code)
                        available_cells.append((x, y))
                        continue
                    
                    # Empty cell
                    row.append('..')
                    available_cells.append((x, y))
            
            lines.append(' '.join(row))
        
        # Convert to grid for agent placement
        grid_lines = [line.split() for line in lines]
        
        # Ensure we have enough cells for agents
        num_agents = self.num_humans + self.num_robots
        
        # Find all empty cells (not walls, lava, or objects)
        empty_cells = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if grid_lines[y][x] == '..':
                    empty_cells.append((x, y))
        
        # If not enough empty cells, clear some wall/object cells
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
        
        # Place agents randomly
        random.shuffle(empty_cells)
        agent_positions = empty_cells[:num_agents]
        
        # Place human agents (yellow)
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid_lines[y][x] = 'Ay'  # Yellow agent
        
        # Place robot agents (grey)
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid_lines[y][x] = 'Ae'  # Grey agent (e = grey)
        
        # Rebuild map string
        return '\n'.join(' '.join(row) for row in grid_lines)


# ============================================================================
# Goal Definitions
# ============================================================================

class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = tuple(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            pos_x, pos_y = agent_state[0], agent_state[1]
            if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell({self.target_pos[0]},{self.target_pos[1]})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                self.target_pos == other.target_pos)


# ============================================================================
# Training on Ensemble (Using world_model_generator)
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


class DynamicGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that dynamically adapts to changing environments.
    
    When set_world_model() is called (by train_neural_policy_prior when using
    world_model_generator), it updates its internal goal cells list.
    """
    
    def __init__(self, base_env: RandomMultigridEnv):
        """Initialize with a base environment."""
        super().__init__(base_env)
        self._update_goal_cells()
    
    def _update_goal_cells(self):
        """Update the list of walkable goal cells from current world_model."""
        self._goal_cells = []
        env = self.world_model
        for x in range(1, env.width - 1):
            for y in range(1, env.height - 1):
                cell = env.grid.get(x, y)
                if cell is None:
                    self._goal_cells.append((x, y))
                elif hasattr(cell, 'can_overlap') and cell.can_overlap():
                    self._goal_cells.append((x, y))
                elif hasattr(cell, 'type') and cell.type in ('goal', 'floor', 'switch'):
                    self._goal_cells.append((x, y))
    
    def set_world_model(self, world_model):
        """Called by train_neural_policy_prior when environment changes."""
        self.world_model = world_model
        self._update_goal_cells()
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """Sample a random goal from walkable cells."""
        if not self._goal_cells:
            target_pos = (self.world_model.width // 2, self.world_model.height // 2)
        else:
            target_pos = random.choice(self._goal_cells)
        goal = ReachCellGoal(self.world_model, human_agent_index, target_pos)
        return goal, 1.0


def train_on_ensemble(
    human_agent_indices: List[int],
    num_episodes: int = NUM_TRAINING_EPISODES_FULL,
    episodes_per_env: int = 1,
    device: str = 'cpu',
    verbose: bool = True,
    base_seed: int = 42
) -> Tuple[QNetwork, RandomMultigridEnv]:
    """
    Train a SINGLE Q-network on an ensemble of randomly generated environments.
    
    Uses the world_model_generator feature of train_neural_policy_prior to
    generate new environments during training, enabling generalization.
    
    Args:
        human_agent_indices: Indices of human agents to model.
        num_episodes: Total number of training episodes.
        episodes_per_env: Number of episodes per generated environment.
        device: Torch device ('cpu' or 'cuda').
        verbose: Whether to print training progress.
        base_seed: Base random seed for reproducibility.
    
    Returns:
        Tuple of (trained_q_network, sample_environment_for_testing)
    """
    if verbose:
        print(f"Training SINGLE Q-network on random environment ensemble...")
        print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
        print(f"  Agents: {NUM_HUMANS} humans + {NUM_ROBOTS} robot")
        print(f"  Training episodes: {num_episodes}")
        print(f"  Episodes per environment: {episodes_per_env}")
    
    # Create base environment for initialization
    base_env = create_random_env(seed=base_seed)
    base_env.reset()
    
    # Goal sampler that updates when environment changes
    goal_sampler = DynamicGoalSampler(base_env)
    
    # World model generator: creates new random environment for each batch of episodes
    def world_model_generator(episode: int) -> RandomMultigridEnv:
        env = create_random_env(seed=base_seed + episode)
        env.reset()
        return env
    
    # Use train_neural_policy_prior with world_model_generator
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
        train_phi_network=False,
        epsilon=0.3,
        exploration_policy=np.array([0.06, 0.19, 0.19, 0.56]),
        use_path_based_shaping=False,  # Use Manhattan distance for ensemble (simpler)
        device=device,
        verbose=verbose,
        world_model_generator=world_model_generator,
        episodes_per_model=episodes_per_env
    )
    
    return neural_prior.q_network, base_env


# ============================================================================
# Rollout and Visualization
# ============================================================================

def state_to_grid_tensor(
    state, 
    env: RandomMultigridEnv,
    query_agent_index: int,
    human_agent_indices: List[int],
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a state to tensor representation for the neural network.
    
    Uses the same channel structure as StateEncoder:
    - num_object_types: explicit object type channels
    - 3: "other" object channels (overlappable, immobile, mobile)
    - 1: per-color agent channel (backward compatibility mode)
    - 1: query agent channel
    """
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    grid_width = env.width
    grid_height = env.height
    num_agents = len(env.agents)
    num_object_types = NUM_OBJECT_TYPE_CHANNELS
    
    # Channel structure (matching StateEncoder with num_agents_per_color=None)
    num_other_object_channels = 3
    num_color_channels = 1  # Backward compatibility: single channel for all agents
    num_channels = num_object_types + num_other_object_channels + num_color_channels + 1
    
    # Channel indices
    other_overlappable_idx = num_object_types
    other_immobile_idx = num_object_types + 1
    other_mobile_idx = num_object_types + 2
    color_channels_start = num_object_types + num_other_object_channels
    query_agent_channel_idx = color_channels_start + num_color_channels
    
    grid_tensor = torch.zeros(1, num_channels, grid_height, grid_width, device=device)
    
    # 1. Encode object-type channels from the persistent world grid
    for y in range(grid_height):
        for x in range(grid_width):
            cell = env.grid.get(x, y)
            if cell is not None:
                cell_type = getattr(cell, 'type', None)
                if cell_type is not None:
                    if cell_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                        if channel_idx < num_object_types:
                            grid_tensor[0, channel_idx, y, x] = 1.0
                    else:
                        # Object type not in explicit channels - use "other" channels
                        if cell_type in OVERLAPPABLE_OBJECTS:
                            grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                            grid_tensor[0, other_mobile_idx, y, x] = 1.0
                        else:
                            grid_tensor[0, other_immobile_idx, y, x] = 1.0
    
    # 2. Encode all agent positions in single color channel (backward compatibility)
    for i, agent_state in enumerate(agent_states):
        x, y = int(agent_state[0]), int(agent_state[1])
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid_tensor[0, color_channels_start, y, x] = 1.0
    
    # 3. Encode query agent channel
    if query_agent_index < len(agent_states):
        agent_state = agent_states[query_agent_index]
        x, y = int(agent_state[0]), int(agent_state[1])
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid_tensor[0, query_agent_channel_idx, y, x] = 1.0
    
    # Normalize step count
    step_tensor = torch.tensor([[step_count / env.max_steps]], device=device, dtype=torch.float32)
    
    return grid_tensor, step_tensor


def get_agent_tensors(
    state,
    human_idx: int,
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract agent position, direction, and index tensors from state."""
    _, agent_states, _, _ = state
    agent_state = agent_states[human_idx]
    
    position = torch.tensor([[
        agent_state[0] / grid_width,
        agent_state[1] / grid_height
    ]], device=device, dtype=torch.float32)
    
    direction = torch.zeros(1, 4, device=device)
    dir_idx = int(agent_state[2]) % 4
    direction[0, dir_idx] = 1.0
    
    agent_idx_tensor = torch.tensor([human_idx], device=device)
    
    return position, direction, agent_idx_tensor


def get_goal_tensor(
    goal_pos: Tuple[int, int],
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """Convert goal position to normalized tensor."""
    return torch.tensor([[
        goal_pos[0] / grid_width,
        goal_pos[1] / grid_height,
        goal_pos[0] / grid_width,
        goal_pos[1] / grid_height
    ]], device=device, dtype=torch.float32)


def get_boltzmann_action(
    q_network: QNetwork,
    state,
    env: RandomMultigridEnv,
    human_idx: int,
    human_agent_indices: List[int],
    goal_pos: Tuple[int, int],
    beta: float = 100.0,
    device: str = 'cpu'
) -> int:
    """Sample an action from the learned Boltzmann policy."""
    grid_tensor, step_tensor = state_to_grid_tensor(
        state, env, human_idx, human_agent_indices, device
    )
    position, direction, agent_idx_t = get_agent_tensors(
        state, human_idx, env.width, env.height, device
    )
    goal_coords = get_goal_tensor(goal_pos, env.width, env.height, device)
    
    with torch.no_grad():
        q_values = q_network(
            grid_tensor, step_tensor,
            position, direction, agent_idx_t,
            goal_coords
        )
        if beta == float('inf'):
            action = torch.argmax(q_values, dim=1).item()
        else:
            q_values -= torch.max(q_values, dim=1, keepdim=True).values
            policy = F.softmax(beta * q_values, dim=1)
            action = torch.multinomial(policy, 1).item()
    
    return action


def render_with_goal_overlay(
    env: RandomMultigridEnv,
    first_human_idx: int,
    first_human_goal: Tuple[int, int],
    tile_size: int = 32
) -> np.ndarray:
    """
    Render the environment with goal and human indicators.
    
    - Blue circle around the first human agent
    - Blue star marking the first human's goal
    """
    img = env.render(mode='rgb_array', highlight=False)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Mark the first human's goal with a blue star
    if first_human_goal:
        gx = first_human_goal[0] * tile_size + tile_size // 2
        gy = first_human_goal[1] * tile_size + tile_size // 2
        ax.plot(gx, gy, marker='*', markersize=20, color='blue',
                markeredgecolor='white', markeredgewidth=2)
    
    # Mark the first human with a blue circle/ring
    state = env.get_state()
    _, agent_states, _, _ = state
    if first_human_idx < len(agent_states):
        human_pos = agent_states[first_human_idx]
        hx = int(human_pos[0]) * tile_size + tile_size // 2
        hy = int(human_pos[1]) * tile_size + tile_size // 2
        # Draw a blue ring around the first human
        ring = plt.Circle((hx, hy), tile_size * 0.45, fill=False,
                          color='blue', linewidth=3)
        ax.add_patch(ring)
        # Add "H1" label
        ax.text(hx, hy - tile_size * 0.35, 'H1', ha='center', va='center',
                fontsize=9, fontweight='bold', color='blue',
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


def run_rollout(
    env: RandomMultigridEnv,
    q_network: QNetwork,
    human_agent_indices: List[int],
    human_goals: Dict[int, Tuple[int, int]],
    robot_index: int,
    first_human_idx: int,
    beta: float = 100.0,
    device: str = 'cpu'
) -> List[np.ndarray]:
    """
    Run a single rollout and return frames for animation.
    
    Visualization includes:
    - Blue circle around the first human agent (H1)
    - Blue star marking the first human's goal
    """
    env.reset()
    frames = []
    num_actions = env.action_space.n
    first_human_goal = human_goals.get(first_human_idx)
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Render current frame with goal overlay
        frame = render_with_goal_overlay(env, first_human_idx, first_human_goal)
        frames.append(frame)
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(len(env.agents)):
            if agent_idx in human_agent_indices:
                goal_pos = human_goals[agent_idx]
                action = get_boltzmann_action(
                    q_network, state, env, agent_idx, human_agent_indices,
                    goal_pos, beta, device
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
    frame = render_with_goal_overlay(env, first_human_idx, first_human_goal)
    frames.append(frame)
    
    return frames


def create_rollout_movie(
    all_frames: List[List[np.ndarray]],
    env_indices: List[int],
    output_path: str,
    num_rollouts: int,
    num_test_envs: int
):
    """Create a movie from rollout frames."""
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
            f'★ = H1 goal | ○ = H1 agent | Humans: learned policy | Robot: random'
        )
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=300, blit=True, repeat=True
    )
    
    try:
        writer = animation.FFMpegWriter(fps=3, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✓ Movie saved to {output_path}")
    except Exception as e:
        print(f"Could not save MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=3)
            print(f"✓ Movie saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"Error saving movie: {e2}")
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Determine configuration based on --quick flag or explicit overrides
    if args.quick:
        num_test_envs = args.envs if args.envs is not None else NUM_TEST_ENVS_QUICK
        num_episodes = args.episodes if args.episodes is not None else NUM_TRAINING_EPISODES_QUICK
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS_QUICK
        mode_str = "QUICK TEST MODE"
    else:
        num_test_envs = args.envs if args.envs is not None else NUM_TEST_ENVS
        num_episodes = args.episodes if args.episodes is not None else NUM_TRAINING_EPISODES_FULL
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS
        mode_str = "FULL MODE"
    
    print("=" * 70)
    print("Random Multigrid Ensemble Policy Prior Learning Demo")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    
    # Create a sample environment to identify agent types
    sample_env = create_random_env(seed=42)
    sample_env.reset()
    
    # Identify agent types
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
    
    # Train on ensemble (new env generated each episode)
    t0 = time.time()
    q_network, base_env = train_on_ensemble(
        human_agent_indices=human_agent_indices,
        num_episodes=num_episodes,
        episodes_per_env=1,  # New environment each episode for maximum diversity
        device=device,
        verbose=True,
        base_seed=42
    )
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print()
    
    # Generate test environments for rollouts
    print(f"Generating {num_test_envs} test environments for rollouts...")
    test_environments = [create_random_env(seed=1000 + i) for i in range(num_test_envs)]
    for env in test_environments:
        env.reset()
    
    # Run rollouts across different test environments
    print(f"Running {num_rollouts} rollouts across test environments...")
    all_frames = []
    env_indices = []
    
    # First human is the one whose goal we'll visualize
    first_human_idx = human_agent_indices[0] if human_agent_indices else 0
    
    for rollout_idx in range(num_rollouts):
        # Select a test environment for this rollout
        env_idx = rollout_idx % len(test_environments)
        env = test_environments[env_idx]
        env.reset()
        
        # Get walkable cells for goal sampling
        goal_cells = []
        for x in range(1, env.width - 1):
            for y in range(1, env.height - 1):
                cell = env.grid.get(x, y)
                if cell is None or (hasattr(cell, 'can_overlap') and cell.can_overlap()):
                    goal_cells.append((x, y))
        
        # Assign random goals to humans
        human_goals = {}
        for h_idx in human_agent_indices:
            if goal_cells:
                human_goals[h_idx] = random.choice(goal_cells)
            else:
                human_goals[h_idx] = (env.width // 2, env.height // 2)
        
        first_human_goal = human_goals.get(first_human_idx, None)
        print(f"  Rollout {rollout_idx + 1}: Env {env_idx + 1}, H1 Goal: {first_human_goal}")
        
        frames = run_rollout(
            env=env,
            q_network=q_network,
            human_agent_indices=human_agent_indices,
            human_goals=human_goals,
            robot_index=robot_index,
            first_human_idx=first_human_idx,
            beta=100.0,
            device=device
        )
        all_frames.append(frames)
        env_indices.append(env_idx)
        print(f"    Captured {len(frames)} frames")
    
    print()
    
    # Create movie
    movie_path = os.path.join(output_dir, 'random_multigrid_ensemble_demo.mp4')
    create_rollout_movie(all_frames, env_indices, movie_path, num_rollouts, num_test_envs)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
