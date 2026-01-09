#!/usr/bin/env python3
"""
Random Multigrid Ensemble with Heuristic Humans and Exploration Robots.

This script demonstrates a configurable random multigrid environment with:
- 3 human agents using heuristic potential-based policies
- 2 robot agents using multi-step exploration policies
- All possible object types in configurable quantities
- Long rollouts (100 steps) with video recording

The demo:
1. Generates a 20x20 random multigrid with configurable objects
2. Uses HeuristicPotentialPolicy for humans (gradient-based path following)
3. Uses MultiGridMultiStepExplorationPolicy for robots (directed exploration)
4. Records a movie of the rollout without annotation panel

Usage:
    python random_ensemble_heuristic_exploration_demo.py              # Default settings
    python random_ensemble_heuristic_exploration_demo.py --steps 200  # Longer rollout
    python random_ensemble_heuristic_exploration_demo.py --walls 0.2  # More walls
    python random_ensemble_heuristic_exploration_demo.py --seed 123   # Different random map

Object Configuration:
    --walls      Probability of wall placement (default: 0.10)
    --doors      Probability of door placement (default: 0.02)
    --keys       Probability of key placement (default: 0.03)
    --balls      Probability of ball placement (default: 0.03)
    --boxes      Probability of box placement (default: 0.02)
    --lava       Probability of lava placement (default: 0.01)
    --blocks     Probability of block placement (default: 0.02)
    --rocks      Probability of rock placement (default: 0.02)
    --unsteady   Probability of unsteady ground (default: 0.05)

Output:
    - Movie in outputs/random_ensemble_heuristic_exploration_demo.mp4
"""

import argparse
import os
import random
from typing import List, Tuple, Optional, Dict

import numpy as np

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions,
    Key, Ball, Box, Door, Lava, Block, Rock, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.world_specific_helpers.multigrid import (
    ReachRectangleGoal,
    RandomPolicy,
)
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.learning_based.multigrid.phase2.exploration_policies import (
    MultiGridMultiStepExplorationPolicy
)
from empo.learning_based.multigrid.path_distance import PathDistanceCalculator


# ============================================================================
# Configuration
# ============================================================================

# Grid and agents
GRID_SIZE = 20
NUM_HUMANS = 3
NUM_ROBOTS = 2
MAX_STEPS = 100

# Object placement probabilities (configurable via CLI)
DEFAULT_WALL_PROB = 0.10
DEFAULT_DOOR_PROB = 0.02
DEFAULT_KEY_PROB = 0.03
DEFAULT_BALL_PROB = 0.03
DEFAULT_BOX_PROB = 0.02
DEFAULT_LAVA_PROB = 0.01
DEFAULT_BLOCK_PROB = 0.02
DEFAULT_ROCK_PROB = 0.02
DEFAULT_UNSTEADY_PROB = 0.05

# Policy parameters
DEFAULT_BETA_HUMAN = 1000.0  # High temperature for more deterministic heuristic
DEFAULT_GAMMA_HUMAN = 0.95

# Rendering
RENDER_TILE_SIZE = 32
MOVIE_FPS = 10

# Maximum attempts for rejection sampling
MAX_REJECTION_SAMPLING_ATTEMPTS = 1000


# ============================================================================
# Goal Sampler
# ============================================================================

class SmallGoalSampler(PossibleGoalSampler):
    """
    Goal sampler that samples rectangle goals of at most size (3,3).
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
        self.world_model = world_model
        self._update_valid_range()
    
    def sample_rectangle(self) -> Tuple[int, int, int, int]:
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range
        
        for _ in range(MAX_REJECTION_SAMPLING_ATTEMPTS):
            x1 = self._rng.integers(x_min, x_max + 1)
            y1 = self._rng.integers(y_min, y_max + 1)
            x2 = self._rng.integers(x_min, x_max + 1)
            y2 = self._rng.integers(y_min, y_max + 1)
            
            if x2 < x1 or y2 < y1:
                continue
            if x2 - x1 > 2 or y2 - y1 > 2:
                continue
            
            return (x1, y1, x2, y2)
        
        # Fallback to single cell
        x = self._rng.integers(x_min, x_max + 1)
        y = self._rng.integers(y_min, y_max + 1)
        return (x, y, x, y)
    
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        x1, y1, x2, y2 = self.sample_rectangle()
        goal = ReachRectangleGoal(self.world_model, human_agent_index, (x1, y1, x2, y2))
        return goal, 1.0


# ============================================================================
# Random Multigrid Environment Generator
# ============================================================================

class RandomMultigridEnv(MultiGridEnv):
    """
    A randomly generated 20x20 multigrid with configurable object types.
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        num_humans: int = 3,
        num_robots: int = 2,
        max_steps: int = 100,
        seed: Optional[int] = None,
        wall_prob: float = 0.10,
        door_prob: float = 0.02,
        key_prob: float = 0.03,
        ball_prob: float = 0.03,
        box_prob: float = 0.02,
        lava_prob: float = 0.01,
        block_prob: float = 0.02,
        rock_prob: float = 0.02,
        unsteady_prob: float = 0.05,
        door_key_color: str = 'r'
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.num_robots = num_robots
        self.wall_prob = wall_prob
        self.door_prob = door_prob
        self.key_prob = key_prob
        self.ball_prob = ball_prob
        self.box_prob = box_prob
        self.lava_prob = lava_prob
        self.block_prob = block_prob
        self.rock_prob = rock_prob
        self.unsteady_prob = unsteady_prob
        self.door_key_color = door_key_color
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        map_str = self._generate_random_map()
        
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions  # Use SmallActions for simplicity
        )
    
    def _generate_random_map(self) -> str:
        """Generate a random map string with all object types."""
        available_cells = []
        pending_keys = []
        colors = ['r', 'g', 'b', 'p', 'y']
        
        # Create grid
        grid_lines = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                # Outer walls
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')
                else:
                    # Randomly place objects
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
                    
                    cumulative += self.door_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'D{color}')
                        # Track that we need a matching key
                        pending_keys.append(color)
                        continue
                    
                    cumulative += self.key_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'K{color}')
                        continue
                    
                    cumulative += self.ball_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'B{color}')
                        continue
                    
                    cumulative += self.box_prob
                    if r < cumulative:
                        color = random.choice(colors)
                        row.append(f'X{color}')
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')
                        continue
                    
                    cumulative += self.rock_prob
                    if r < cumulative:
                        row.append('Ro')
                        continue
                    
                    cumulative += self.unsteady_prob
                    if r < cumulative:
                        row.append('Un')
                        continue
                    
                    # Empty cell
                    row.append('..')
                    available_cells.append((x, y))
            
            grid_lines.append(row)
        
        # Ensure we have keys for all doors
        for door_color in pending_keys:
            if available_cells:
                x, y = random.choice(available_cells)
                available_cells.remove((x, y))
                grid_lines[y][x] = f'K{door_color}'
        
        # Place agents
        num_agents = self.num_humans + self.num_robots
        
        # Ensure we have enough empty cells for agents
        while len(available_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid_lines[y][x] != '..' and (x, y) not in available_cells:
                        grid_lines[y][x] = '..'
                        available_cells.append((x, y))
                        if len(available_cells) >= num_agents:
                            break
                if len(available_cells) >= num_agents:
                    break
        
        # Randomly select agent positions
        random.shuffle(available_cells)
        agent_positions = available_cells[:num_agents]
        
        # Place humans (yellow agents)
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid_lines[y][x] = 'Ay'
        
        # Place robots (grey agents)
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid_lines[y][x] = 'Ae'
        
        return '\n'.join(' '.join(row) for row in grid_lines)


# ============================================================================
# Policy Creation
# ============================================================================

def create_heuristic_policy(
    env: MultiGridEnv,
    human_agent_indices: List[int],
    beta: float,
    gamma: float
) -> HeuristicPotentialPolicy:
    """Create a heuristic potential-based policy for humans."""
    # Create path distance calculator
    print("    Computing path distances...")
    path_calc = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env
    )
    
    # Create heuristic policy
    policy = HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_agent_indices,
        path_calculator=path_calc,
        beta=beta,
        num_actions=4  # SmallActions: still, left, right, forward
    )
    
    return policy


def create_robot_exploration_policy(
    env: MultiGridEnv,
    robot_agent_indices: List[int]
) -> MultiGridMultiStepExplorationPolicy:
    """Create a multi-step exploration policy for robots."""
    policy = MultiGridMultiStepExplorationPolicy(
        agent_indices=robot_agent_indices,
        sequence_probs={
            'still': 0.05,
            'forward': 0.50,
            'left_forward': 0.18,
            'right_forward': 0.18,
            'back_forward': 0.09
        },
        expected_k=2.5,  # Average sequence length
        world_model=env
    )
    return policy


# ============================================================================
# Rollout Execution
# ============================================================================

def get_agent_indices(env: MultiGridEnv) -> Tuple[List[int], List[int]]:
    """Get human and robot agent indices."""
    human_indices = env.human_agent_indices
    robot_indices = env.robot_agent_indices
    return human_indices, robot_indices


def run_rollout(
    env: MultiGridEnv,
    human_policy: HeuristicPotentialPolicy,
    robot_policy: MultiGridMultiStepExplorationPolicy,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_goals: Dict[int, PossibleGoal],
    max_steps: int
) -> int:
    """
    Run a single rollout with heuristic humans and exploration robots.
    
    Returns:
        Number of steps taken.
    """
    env.reset()
    steps_taken = 0
    
    # Render initial frame (no annotation panel)
    env.render(mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE,
               goal_overlays=human_goals)
    
    for step in range(max_steps):
        state = env.get_state()
        
        # Get actions for all agents
        actions = [0] * len(env.agents)
        
        # Humans use heuristic policy
        for h_idx in human_agent_indices:
            human_goal = human_goals[h_idx]
            action_dist = human_policy(state, h_idx, human_goal)
            if action_dist is not None:
                actions[h_idx] = np.random.choice(len(action_dist), p=action_dist)
        
        # Robots use exploration policy
        robot_actions = robot_policy.sample(state)
        if robot_actions is not None:
            for i, r_idx in enumerate(robot_agent_indices):
                actions[r_idx] = robot_actions[i]
        
        # Step environment
        _, _, done, _ = env.step(actions)
        steps_taken += 1
        
        # Render frame (no annotation panel)
        env.render(mode='rgb_array', highlight=False, tile_size=RENDER_TILE_SIZE,
                   goal_overlays=human_goals)
        
        if done:
            break
    
    return steps_taken


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the demo."""
    args = parse_args()
    
    print("=" * 70)
    print("Random Multigrid: Heuristic Humans + Exploration Robots")
    print("=" * 70)
    print()
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Agents: {NUM_HUMANS} humans (heuristic) + {NUM_ROBOTS} robots (exploration)")
    print(f"Max steps: {args.steps}")
    print(f"Random seed: {args.seed}")
    print()
    print("Object probabilities:")
    print(f"  Walls:         {args.walls:.3f}")
    print(f"  Doors:         {args.doors:.3f}")
    print(f"  Keys:          {args.keys:.3f}")
    print(f"  Balls:         {args.balls:.3f}")
    print(f"  Boxes:         {args.boxes:.3f}")
    print(f"  Lava:          {args.lava:.3f}")
    print(f"  Blocks:        {args.blocks:.3f}")
    print(f"  Rocks:         {args.rocks:.3f}")
    print(f"  Unsteady:      {args.unsteady:.3f}")
    print()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    print("Generating random environment...")
    env = RandomMultigridEnv(
        grid_size=GRID_SIZE,
        num_humans=NUM_HUMANS,
        num_robots=NUM_ROBOTS,
        max_steps=args.steps,
        seed=args.seed,
        wall_prob=args.walls,
        door_prob=args.doors,
        key_prob=args.keys,
        ball_prob=args.balls,
        box_prob=args.boxes,
        lava_prob=args.lava,
        block_prob=args.blocks,
        rock_prob=args.rocks,
        unsteady_prob=args.unsteady
    )
    env.reset()
    
    # Get agent indices
    human_indices, robot_indices = get_agent_indices(env)
    print(f"  Human agents: {human_indices}")
    print(f"  Robot agents: {robot_indices}")
    print()
    
    # Create policies
    print("Creating policies...")
    print("  Human policy: HeuristicPotentialPolicy")
    human_policy = create_heuristic_policy(env, human_indices, args.beta, args.gamma)
    print(f"    Beta: {args.beta}, Gamma: {args.gamma}")
    
    print("  Robot policy: MultiGridMultiStepExplorationPolicy")
    robot_policy = create_robot_exploration_policy(env, robot_indices)
    print("    Multi-step directed exploration")
    print()
    
    # Sample goals
    print("Sampling goals for humans...")
    goal_sampler = SmallGoalSampler(env, seed=args.seed)
    state = env.get_state()
    
    human_goals = {}
    for h_idx in human_indices:
        goal, _ = goal_sampler.sample(state, h_idx)
        human_goals[h_idx] = goal
        rect = goal.target_rect
        print(f"  Human {h_idx}: goal rectangle {rect}")
    print()
    
    # Start video recording
    print(f"Running rollout ({args.steps} max steps)...")
    env.start_video_recording()
    
    steps = run_rollout(
        env=env,
        human_policy=human_policy,
        robot_policy=robot_policy,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_goals=human_goals,
        max_steps=args.steps
    )
    
    print(f"  Completed in {steps} steps")
    print(f"  Total frames: {len(env._video_frames)}")
    print()
    
    # Save movie
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    movie_path = os.path.join(output_dir, 'random_ensemble_heuristic_exploration_demo.mp4')
    if os.path.exists(movie_path):
        os.remove(movie_path)
    
    print("Saving movie...")
    env.save_video(movie_path, fps=args.fps)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Movie output: {os.path.abspath(movie_path)}")
    print("=" * 70)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Random Multigrid with Heuristic Humans and Exploration Robots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation parameters
    parser.add_argument('--steps', '-s', type=int, default=MAX_STEPS,
                        help='Maximum steps per rollout')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for environment generation')
    
    # Policy parameters
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA_HUMAN,
                        help='Softmax temperature for human heuristic policy')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA_HUMAN,
                        help='Discount factor for human policy')
    
    # Object placement probabilities
    parser.add_argument('--walls', type=float, default=DEFAULT_WALL_PROB,
                        help='Probability of wall placement')
    parser.add_argument('--doors', type=float, default=DEFAULT_DOOR_PROB,
                        help='Probability of door placement')
    parser.add_argument('--keys', type=float, default=DEFAULT_KEY_PROB,
                        help='Probability of key placement')
    parser.add_argument('--balls', type=float, default=DEFAULT_BALL_PROB,
                        help='Probability of ball placement')
    parser.add_argument('--boxes', type=float, default=DEFAULT_BOX_PROB,
                        help='Probability of box placement')
    parser.add_argument('--lava', type=float, default=DEFAULT_LAVA_PROB,
                        help='Probability of lava placement')
    parser.add_argument('--blocks', type=float, default=DEFAULT_BLOCK_PROB,
                        help='Probability of block placement')
    parser.add_argument('--rocks', type=float, default=DEFAULT_ROCK_PROB,
                        help='Probability of rock placement')
    parser.add_argument('--unsteady', type=float, default=DEFAULT_UNSTEADY_PROB,
                        help='Probability of unsteady ground placement')
    
    # Output parameters
    parser.add_argument('--fps', type=int, default=MOVIE_FPS,
                        help='Frames per second for output video')
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
