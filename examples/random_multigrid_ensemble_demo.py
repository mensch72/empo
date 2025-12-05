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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.multigrid import (
    ReachRectangleGoal,
    MultiGridGoalSampler,
    RandomPolicy,
    render_goal_overlay,
)
from empo.nn_based.multigrid import (
    MultiGridNeuralHumanPolicyPrior,
    train_multigrid_neural_policy_prior as train_neural_policy_prior,
)


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 7           # 7x7 grid (including outer walls)
NUM_HUMANS = 2          # 2 human agents (yellow)
NUM_ROBOTS = 1          # 1 robot agent (grey)
MAX_STEPS = 50          # Maximum steps per episode
ROLLOUT_STEPS = 15      # Steps per rollout (shorter than training episodes)
NUM_TEST_ENVS = 50      # Number of test environments for rollout evaluation
NUM_ROLLOUTS = 50       # Number of rollouts for the movie

# Full training configuration (default)
NUM_TRAINING_EPISODES_FULL = 10000

# Quick test configuration (for --quick flag)
NUM_TRAINING_EPISODES_QUICK = 50
NUM_TEST_ENVS_QUICK = 3
NUM_ROLLOUTS_QUICK = 3

# Object placement probabilities
WALL_PROBABILITY = 0.15      # Probability of placing internal walls
DOOR_PROBABILITY = 0.03      # Probability of placing a door (also places matching key)
BALL_PROBABILITY = 0.0       # Probability of placing a ball
BOX_PROBABILITY = 0.0        # Probability of placing a box
LAVA_PROBABILITY = 0.0       # Probability of placing lava
BLOCK_PROBABILITY = 0.03     # Probability of placing a block
ROCK_PROBABILITY = 0.02      # Probability of placing a rock
UNSTEADY_GROUND_PROBABILITY = 0.10  # Probability of placing unsteady ground

# Door/Key color (single color for both)
DOOR_KEY_COLOR = 'r'  # Red

# Maximum attempts for rejection sampling
MAX_REJECTION_SAMPLING_ATTEMPTS = 1000


# ============================================================================
# Custom Goal Sampler (Small Goals for Better Learning)
# ============================================================================

class SmallGoalSampler(PossibleGoalSampler):
    """
    Goal sampler with modified weight function that samples goals of at most size (3,3).
    
    Goals are sampled by rejection sampling: draw x1,y1,x2,y2 uniformly at random,
    reject if x2 < x1 or y2 < y1 or x2-x1 > 2 or y2-y1 > 2.
    This produces goals of at most size (3,3) cells to improve learning efficiency.
    
    Args:
        world_model: The multigrid environment.
        valid_x_range: Optional (x_min, x_max) for valid goal coordinates.
                       Defaults to (1, width-2) to exclude outer walls.
        valid_y_range: Optional (y_min, y_max) for valid goal coordinates.
                       Defaults to (1, height-2) to exclude outer walls.
        seed: Optional random seed for reproducibility.
    """
    
    def __init__(
        self,
        world_model,
        valid_x_range: Optional[Tuple[int, int]] = None,
        valid_y_range: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None
    ):
        super().__init__(world_model)
        self._rng = np.random.default_rng(seed)
        self._custom_x_range = valid_x_range
        self._custom_y_range = valid_y_range
        self._update_valid_range()
    
    def _update_valid_range(self):
        """Update valid coordinate ranges for goal placement."""
        env = self.world_model
        if self._custom_x_range is not None:
            self._x_range = self._custom_x_range
        else:
            self._x_range = (1, env.width - 2)
        
        if self._custom_y_range is not None:
            self._y_range = self._custom_y_range
        else:
            self._y_range = (1, env.height - 2)
    
    def set_world_model(self, world_model):
        """Update world model and refresh valid ranges."""
        self.world_model = world_model
        self._update_valid_range()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
    
    def sample_rectangle(self) -> Tuple[int, int, int, int]:
        """
        Sample a rectangle (x1, y1, x2, y2) with rejection sampling.
        
        Draws x1, y1, x2, y2 uniformly at random and rejects if:
        - x2 < x1 or y2 < y1 (invalid rectangle)
        - x2 - x1 > 2 or y2 - y1 > 2 (goal too large, at most size 3x3)
        
        Returns:
            Tuple (x1, y1, x2, y2) with valid small goal coordinates.
        """
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range
        
        for _ in range(MAX_REJECTION_SAMPLING_ATTEMPTS):
            # Draw all coordinates uniformly at random
            x1 = self._rng.integers(x_min, x_max + 1)
            y1 = self._rng.integers(y_min, y_max + 1)
            x2 = self._rng.integers(x_min, x_max + 1)
            y2 = self._rng.integers(y_min, y_max + 1)
            
            # Rejection conditions
            if x2 < x1 or y2 < y1:
                continue
            if x2 - x1 > 2 or y2 - y1 > 2:
                continue
            
            return (x1, y1, x2, y2)
        
        # Fallback: return a point goal
        x = self._rng.integers(x_min, x_max + 1)
        y = self._rng.integers(y_min, y_max + 1)
        return (x, y, x, y)
    
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        """
        Sample a small rectangle goal with uniform weight.
        
        The returned weight is 1.0 since all sampled goals have equal probability
        under this rejection sampling scheme.
        
        Args:
            state: Current world state (not used for sampling).
            human_agent_index: Index of the human agent for the goal.
        
        Returns:
            Tuple of (goal, weight) where weight is always 1.0.
        """
        x1, y1, x2, y2 = self.sample_rectangle()
        goal = ReachRectangleGoal(self.world_model, human_agent_index, (x1, y1, x2, y2))
        return goal, 1.0


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
    parser.add_argument(
        '--load-policy',
        type=str,
        default=None,
        help='Path to a saved policy file to load instead of training from scratch'
    )
    parser.add_argument(
        '--save-policy',
        type=str,
        default=None,
        help='Path to save the trained policy (default: outputs/random_multigrid_policy.pt)'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip training (useful when loading a saved policy for rollouts only)'
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
    - Random objects (keys, balls, boxes, doors, lava, blocks, rocks, unsteady ground)
    - Specified number of human (yellow) and robot (grey) agents
    
    Doors and keys are paired: when a door is placed, a matching key is also placed.
    """
    
    def __init__(
        self,
        grid_size: int = 7,
        num_humans: int = 2,
        num_robots: int = 1,
        max_steps: int = 50,
        seed: Optional[int] = None,
        wall_prob: float = 0.15,
        ball_prob: float = 0.0,
        box_prob: float = 0.0,
        door_prob: float = 0.03,
        lava_prob: float = 0.0,
        block_prob: float = 0.03,
        rock_prob: float = 0.02,
        unsteady_prob: float = 0.10,
        door_key_color: str = 'r'
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
            ball_prob: Probability of placing balls.
            box_prob: Probability of placing boxes.
            door_prob: Probability of placing doors (also places matching key).
            lava_prob: Probability of placing lava.
            block_prob: Probability of placing blocks.
            rock_prob: Probability of placing rocks.
            unsteady_prob: Probability of placing unsteady ground.
            door_key_color: Color code for doors and keys (single color).
        """
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        self.wall_prob = wall_prob
        self.ball_prob = ball_prob
        self.box_prob = box_prob
        self.door_prob = door_prob
        self.lava_prob = lava_prob
        self.block_prob = block_prob
        self.rock_prob = rock_prob
        self.unsteady_prob = unsteady_prob
        self.door_key_color = door_key_color
        
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
        """
        Generate a random map string for the environment.
        
        Doors and keys are paired: when a door is placed, a matching key
        is also placed in a random available cell. They use the same color.
        """
        # Track which cells are available for agents
        available_cells = []
        
        # Track cells where we'll place keys (for doors placed)
        pending_keys = []
        
        # First pass: generate the grid without keys
        grid = []
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
                    
                    cumulative += self.rock_prob
                    if r < cumulative:
                        row.append('Ro')  # Rock
                        continue
                    
                    cumulative += self.door_prob
                    if r < cumulative:
                        # Place closed door and schedule a matching key
                        row.append(f'C{self.door_key_color}')  # Closed door
                        pending_keys.append(self.door_key_color)
                        # Doors can be passed through (when open), so add to available
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.ball_prob
                    if r < cumulative:
                        color = random.choice(['r', 'g', 'b', 'p'])
                        row.append(f'B{color}')  # Ball
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.box_prob
                    if r < cumulative:
                        color = random.choice(['r', 'g', 'b', 'p'])
                        row.append(f'X{color}')  # Box
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')  # Block (non-overlappable)
                        # Blocks cannot be overlapped, don't add to available cells
                        continue
                    
                    cumulative += self.unsteady_prob
                    if r < cumulative:
                        row.append('Un')  # Unsteady ground (overlappable)
                        available_cells.append((x, y))
                        continue
                    
                    # Empty cell
                    row.append('..')
                    available_cells.append((x, y))
            
            grid.append(row)
        
        # Second pass: place keys for each door in random empty cells
        # Find all empty cells for key placement
        empty_cells_for_keys = [(x, y) for (x, y) in available_cells 
                                if grid[y][x] == '..']
        
        random.shuffle(empty_cells_for_keys)
        for i, key_color in enumerate(pending_keys):
            if i < len(empty_cells_for_keys):
                kx, ky = empty_cells_for_keys[i]
                grid[ky][kx] = f'K{key_color}'  # Place key
        
        # Ensure we have enough cells for agents
        num_agents = self.num_humans + self.num_robots
        
        # Find all empty cells (not walls, lava, or blocking objects)
        empty_cells = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if grid[y][x] == '..':
                    empty_cells.append((x, y))
        
        # If not enough empty cells, clear some wall/object cells
        while len(empty_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid[y][x] != '..' and (x, y) not in empty_cells:
                        grid[y][x] = '..'
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
            grid[y][x] = 'Ay'  # Yellow agent
        
        # Place robot agents (grey)
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid[y][x] = 'Ae'  # Grey agent (e = grey)
        
        # Build map string
        return '\n'.join(' '.join(row) for row in grid)


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
        ball_prob=BALL_PROBABILITY,
        box_prob=BOX_PROBABILITY,
        door_prob=DOOR_PROBABILITY,
        lava_prob=LAVA_PROBABILITY,
        block_prob=BLOCK_PROBABILITY,
        rock_prob=ROCK_PROBABILITY,
        unsteady_prob=UNSTEADY_GROUND_PROBABILITY,
        door_key_color=DOOR_KEY_COLOR
    )


def train_on_ensemble(
    human_agent_indices: List[int],
    num_episodes: int = NUM_TRAINING_EPISODES_FULL,
    episodes_per_env: int = 1,
    device: str = 'cpu',
    verbose: bool = True,
    base_seed: int = 42
) -> Tuple['MultiGridNeuralHumanPolicyPrior', RandomMultigridEnv]:
    """
    Train a neural policy prior on an ensemble of randomly generated environments.
    
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
        Tuple of (trained_policy, sample_environment_for_testing)
    """
    if verbose:
        print(f"Training neural policy prior on random environment ensemble...")
        print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")
        print(f"  Agents: {NUM_HUMANS} humans + {NUM_ROBOTS} robot")
        print(f"  Training episodes: {num_episodes}")
        print(f"  Episodes per environment: {episodes_per_env}")
    
    # Create base environment for initialization
    base_env = create_random_env(seed=base_seed)
    base_env.reset()
    
    # Use SmallGoalSampler for small goals (at most 3x3) to improve learning
    goal_sampler = SmallGoalSampler(base_env)
    
    # World model generator: creates new random environment for each batch of episodes
    def world_model_generator(episode: int) -> RandomMultigridEnv:
        env = create_random_env(seed=base_seed + episode)
        env.reset()
        return env
    
    # Use train_neural_policy_prior with world_model_generator
    policy = train_neural_policy_prior(
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
        reward_shaping=True,  # Use path-based reward shaping with passing costs
        device=device,
        verbose=verbose,
        world_model_generator=world_model_generator,
        episodes_per_model=episodes_per_env
    )
    
    return policy, base_env


# ============================================================================
# Rollout and Visualization
# ============================================================================

def render_with_goal_overlay(
    env: RandomMultigridEnv,
    first_human_idx: int,
    first_human_goal: ReachRectangleGoal,
    tile_size: int = 32
) -> np.ndarray:
    """
    Render the environment with goal and human indicators.
    
    Uses render_goal_overlay from empo.multigrid for dashed blue rectangle
    boundaries and agent-to-goal connection lines.
    
    - Blue dashed rectangle around the goal area (slightly inside cell bounds)
    - Blue dashed line connecting the agent to the closest point on the goal boundary
    """
    img = env.render(mode='rgb_array', highlight=False)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Get agent positions from environment state
    state = env.get_state()
    _, agent_states, _, _ = state
    
    # Render goal using the goal overlay function from empo.multigrid
    if first_human_goal is not None and first_human_idx < len(agent_states):
        human_pos = agent_states[first_human_idx]
        agent_pos = (float(human_pos[0]), float(human_pos[1]))
        
        # Use the goal object directly - it has target_rect attribute
        render_goal_overlay(
            ax=ax,
            goal=first_human_goal,
            agent_pos=agent_pos,
            agent_idx=first_human_idx,
            tile_size=tile_size,
            goal_color=(0.0, 0.4, 1.0, 0.7),  # Blue, semi-transparent
            line_width=2.5,
            inset=0.08
        )
    
    # Mark the first human with a label
    if first_human_idx < len(agent_states):
        human_pos = agent_states[first_human_idx]
        hx = int(human_pos[0]) * tile_size + tile_size // 2
        hy = int(human_pos[1]) * tile_size + tile_size // 2
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
    policy: 'MultiGridNeuralHumanPolicyPrior',
    human_agent_indices: List[int],
    human_goals: Dict[int, ReachRectangleGoal],
    robot_index: int,
    first_human_idx: int,
    robot_policy: Optional[RandomPolicy] = None,
    max_steps: int = ROLLOUT_STEPS
) -> List[np.ndarray]:
    """
    Run a single rollout and return frames for animation.
    
    Args:
        env: The environment.
        policy: Trained neural policy prior.
        human_agent_indices: List of human agent indices.
        human_goals: Dict mapping agent index to ReachRectangleGoal.
        robot_index: Index of the robot agent.
        first_human_idx: Index of the first human (for visualization).
        robot_policy: Optional RandomPolicy for robot actions.
                     If None, creates one with default distribution.
        max_steps: Maximum number of steps for the rollout (default: ROLLOUT_STEPS).
    
    Returns:
        List of frames for animation.
    """
    env.reset()
    frames = []
    first_human_goal = human_goals.get(first_human_idx)
    
    # Create robot policy if not provided
    if robot_policy is None:
        robot_policy = RandomPolicy()
    
    for step in range(max_steps):
        state = env.get_state()
        
        # Render current frame with goal overlay for first human
        frame = render_with_goal_overlay(env, first_human_idx, first_human_goal)
        frames.append(frame)
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(len(env.agents)):
            if agent_idx in human_agent_indices:
                goal = human_goals[agent_idx]
                # Use policy.sample() directly with the goal
                action = policy.sample(state, agent_idx, goal)
            else:
                # Robot uses random policy
                action = robot_policy.sample()
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
    
    # Identify agent types (yellow = human, grey = robot)
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
    
    # Load or train policy
    if args.load_policy:
        # Load existing policy using MultiGridNeuralHumanPolicyPrior.load()
        print(f"Loading policy from {args.load_policy}...")
        goal_sampler = SmallGoalSampler(sample_env)
        policy = MultiGridNeuralHumanPolicyPrior.load(
            filepath=args.load_policy,
            world_model=sample_env,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            device=device
        )
        base_env = sample_env
        print(f"Policy loaded from {args.load_policy}")
        
        # Optionally train more episodes on top of loaded policy
        if num_episodes > 0 and not args.no_train:
            print(f"Continuing training for {num_episodes} additional episodes...")
            t0 = time.time()
            policy, base_env = train_on_ensemble(
                human_agent_indices=human_agent_indices,
                num_episodes=num_episodes,
                episodes_per_env=1,
                device=device,
                verbose=True,
                base_seed=42
            )
            elapsed = time.time() - t0
            print(f"\nAdditional training completed in {elapsed:.2f} seconds")
            print()
    elif args.no_train:
        print("Error: --no-train flag requires --load-policy to specify a saved policy file to load.")
        return
    else:
        # Train from scratch
        t0 = time.time()
        policy, base_env = train_on_ensemble(
            human_agent_indices=human_agent_indices,
            num_episodes=num_episodes,
            episodes_per_env=1,
            device=device,
            verbose=True,
            base_seed=42
        )
        elapsed = time.time() - t0
        print(f"\nTraining completed in {elapsed:.2f} seconds")
        print()
    
    # Save policy if requested
    policy_save_path = args.save_policy
    if policy_save_path is None:
        policy_save_path = os.path.join(output_dir, 'random_multigrid_policy.pt')
    
    # Always save the policy (unless loading without additional training)
    if not (args.load_policy and args.no_train):
        policy.save(policy_save_path)
        print(f"Policy saved to {policy_save_path}")
    
    # Generate test environments for rollouts
    print(f"Generating {num_test_envs} test environments for rollouts...")
    test_environments = [create_random_env(seed=1000 + i) for i in range(num_test_envs)]
    for env in test_environments:
        env.reset()
    
    # Run rollouts across different test environments
    print(f"Running {num_rollouts} rollouts across test environments...")
    print(f"  Rollout steps: {ROLLOUT_STEPS}")
    print(f"  Yellow (human) agents: learned policy")
    print(f"  Grey (robot) agent: random policy")
    all_frames = []
    env_indices = []
    
    # First human is the one whose goal we'll visualize
    first_human_idx = human_agent_indices[0] if human_agent_indices else 0
    
    for rollout_idx in range(num_rollouts):
        # Select a test environment for this rollout
        env_idx = rollout_idx % len(test_environments)
        env = test_environments[env_idx]
        env.reset()
        
        # Use SmallGoalSampler to sample small rectangle goals (at most 3x3)
        goal_sampler = SmallGoalSampler(env)
        state = env.get_state()
        
        # Assign random rectangle goals to humans using the sampler
        human_goals = {}
        for h_idx in human_agent_indices:
            # sample() returns (ReachRectangleGoal, weight)
            goal, _ = goal_sampler.sample(state, h_idx)
            human_goals[h_idx] = goal
        
        first_human_goal = human_goals.get(first_human_idx, None)
        goal_rect = first_human_goal.target_rect if first_human_goal else None
        print(f"  Rollout {rollout_idx + 1}: Env {env_idx + 1}, H1 Goal rect: {goal_rect}")
        
        frames = run_rollout(
            env=env,
            policy=policy,
            human_agent_indices=human_agent_indices,
            human_goals=human_goals,
            robot_index=robot_index,
            first_human_idx=first_human_idx,
            max_steps=ROLLOUT_STEPS
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
    print(f"Policy saved: {os.path.abspath(policy_save_path)}")
    print(f"Movie output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
