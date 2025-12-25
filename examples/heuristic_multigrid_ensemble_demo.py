#!/usr/bin/env python3
"""
Heuristic Multigrid Ensemble Demo.

This script demonstrates the heuristic potential-based policy for human agents
on an ensemble of randomly generated multigrids. Unlike the neural policy demo,
this uses no learning - the policy is purely based on potential function gradients
computed from precomputed shortest paths.

The demo:
1. Generates random 7x7 multigrids with:
   - 2 human agents (yellow)
   - 1 robot agent (grey)
   - Random internal walls and objects
   
2. Creates a heuristic policy using PathDistanceCalculator:
   - Precomputes shortest paths between all cell pairs
   - At each step, evaluates potential at neighboring cells
   - Produces soft probability distribution favoring moves toward goal
   
3. Produces a movie with rollouts across different environments showing
   the heuristic policy in action (humans use heuristic, robot uses random).

Usage:
    python heuristic_multigrid_ensemble_demo.py           # Full run (100 rollouts)
    python heuristic_multigrid_ensemble_demo.py --quick   # Quick test (10 rollouts)
    python heuristic_multigrid_ensemble_demo.py --beta 50  # More deterministic

Requirements:
    - matplotlib
    - ffmpeg (optional, for MP4 output; falls back to GIF)
"""

import sys
import os
import random
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, Actions, #SmallActions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.multigrid import (
    ReachRectangleGoal,
    RandomPolicy,
    render_goals_on_frame,
)
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid.path_distance import PathDistanceCalculator


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 8           # grid (including outer walls)
NUM_HUMANS = 2          # human agents (yellow)
NUM_ROBOTS = 1          # robot agents (grey)
MAX_STEPS = 50          # Maximum steps per episode
ROLLOUT_STEPS = 30      # Steps per rollout
NUM_TEST_ENVS = 50      # Number of test environments
NUM_ROLLOUTS = 100      # Number of rollouts for the movie (default)

# Quick test configuration
NUM_TEST_ENVS_QUICK = 5
NUM_ROLLOUTS_QUICK = 10

# Object placement probabilities
WALL_PROBABILITY = 0.05 # 0.1
DOOR_PROBABILITY = 0.03 # 0.03
BALL_PROBABILITY = 0.0
BOX_PROBABILITY = 0.0
LAVA_PROBABILITY = 0.0
BLOCK_PROBABILITY = 0.00 #0.03
ROCK_PROBABILITY = 0.00 # 0.02
UNSTEADY_GROUND_PROBABILITY = 0.10

# Door/Key color
DOOR_KEY_COLOR = 'r'

# Heuristic policy parameters
DEFAULT_BETA = 1000.0  # Softmax temperature (higher = more deterministic)

# Maximum attempts for rejection sampling
MAX_REJECTION_SAMPLING_ATTEMPTS = 1000


# ============================================================================
# Goal Sampler (copied from random_multigrid_ensemble_demo.py)
# ============================================================================

class SmallGoalSampler(PossibleGoalSampler):
    """
    Goal sampler that samples goals of at most size (3,3).
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
        
        x = self._rng.integers(x_min, x_max + 1)
        y = self._rng.integers(y_min, y_max + 1)
        return (x, y, x, y)
    
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        x1, y1, x2, y2 = self.sample_rectangle()
        goal = ReachRectangleGoal(self.world_model, human_agent_index, (x1, y1, x2, y2))
        return goal, 1.0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Heuristic Multigrid Ensemble Demo"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced environments and rollouts'
    )
    parser.add_argument(
        '--envs', '-n',
        type=int,
        default=None,
        help='Number of test environments (overrides default)'
    )
    parser.add_argument(
        '--rollouts', '-r',
        type=int,
        default=None,
        help='Number of rollouts for the movie (overrides default)'
    )
    parser.add_argument(
        '--beta', '-s',
        type=float,
        default=DEFAULT_BETA,
        help=f'Softmax temperature for heuristic policy (default: {DEFAULT_BETA})'
    )
    parser.add_argument(
        '--steps', '-t',
        type=int,
        default=ROLLOUT_STEPS,
        help=f'Steps per rollout (default: {ROLLOUT_STEPS})'
    )
    return parser.parse_args()


# ============================================================================
# Random Multigrid Environment Generator
# ============================================================================

class RandomMultigridEnv(MultiGridEnv):
    """
    A randomly generated multigrid environment.
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
        
        map_str = self._generate_random_map()
        
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=Actions #SmallActions
        )
    
    def _generate_random_map(self) -> str:
        """Generate a random map string."""
        available_cells = []
        pending_keys = []
        
        grid = []
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
                    
                    cumulative += self.rock_prob
                    if r < cumulative:
                        row.append('Ro')
                        continue
                    
                    cumulative += self.door_prob
                    if r < cumulative:
                        row.append(f'L{self.door_key_color}')  # 'L' = Locked door (requires key)
                        pending_keys.append(self.door_key_color)
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.ball_prob
                    if r < cumulative:
                        color = random.choice(['r', 'g', 'b', 'p'])
                        row.append(f'B{color}')
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.box_prob
                    if r < cumulative:
                        color = random.choice(['r', 'g', 'b', 'p'])
                        row.append(f'X{color}')
                        available_cells.append((x, y))
                        continue
                    
                    cumulative += self.block_prob
                    if r < cumulative:
                        row.append('Bl')
                        continue
                    
                    cumulative += self.unsteady_prob
                    if r < cumulative:
                        row.append('Un')
                        available_cells.append((x, y))
                        continue
                    
                    row.append('..')
                    available_cells.append((x, y))
            
            grid.append(row)
        
        # Place keys for doors
        empty_cells_for_keys = [(x, y) for (x, y) in available_cells if grid[y][x] == '..']
        random.shuffle(empty_cells_for_keys)
        for i, key_color in enumerate(pending_keys):
            if i < len(empty_cells_for_keys):
                kx, ky = empty_cells_for_keys[i]
                grid[ky][kx] = f'K{key_color}'
        
        # Ensure enough cells for agents
        num_agents = self.num_humans + self.num_robots
        empty_cells = []
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if grid[y][x] == '..':
                    empty_cells.append((x, y))
        
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
        
        random.shuffle(empty_cells)
        agent_positions = empty_cells[:num_agents]
        
        for i in range(self.num_humans):
            x, y = agent_positions[i]
            grid[y][x] = 'Ay'
        
        for i in range(self.num_robots):
            x, y = agent_positions[self.num_humans + i]
            grid[y][x] = 'Ae'
        
        return '\n'.join(' '.join(row) for row in grid)


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


def get_agent_indices(env: RandomMultigridEnv) -> Tuple[List[int], Optional[int]]:
    """Identify human (yellow) and robot (grey) agent indices."""
    human_agent_indices = []
    robot_index = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_agent_indices.append(i)
        elif agent.color == 'grey':
            robot_index = i
    return human_agent_indices, robot_index


# ============================================================================
# Heuristic Policy Creation
# ============================================================================

def create_heuristic_policy(
    env: RandomMultigridEnv,
    human_agent_indices: List[int],
    beta: float = DEFAULT_BETA
) -> HeuristicPotentialPolicy:
    """
    Create a heuristic potential-based policy for the given environment.
    
    Args:
        env: The multigrid environment.
        human_agent_indices: Indices of human agents.
        beta: Softmax temperature (higher = more deterministic).
    
    Returns:
        HeuristicPotentialPolicy ready for use.
    """
    # Create path distance calculator with precomputed shortest paths
    path_calculator = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env
    )
    
    # Create the heuristic policy
    policy = HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_agent_indices,
        path_calculator=path_calculator,
        beta=beta,
        num_actions=8  # Full Actions: still, left, right, forward, pickup, drop, toggle, done
    )
    
    return policy


# ============================================================================
# Rollout and Visualization
# ============================================================================

def run_rollout(
    env: RandomMultigridEnv,
    policy: HeuristicPotentialPolicy,
    human_agent_indices: List[int],
    human_goals: Dict[int, ReachRectangleGoal],
    robot_index: int,
    robot_policy: Optional[RandomPolicy] = None,
    max_steps: int = ROLLOUT_STEPS,
    frame_buffer: Optional[List] = None
) -> int:
    """
    Run a single rollout and record frames with goal overlays.
    
    Args:
        env: The environment.
        policy: Heuristic potential policy.
        human_agent_indices: List of human agent indices.
        human_goals: Dict mapping agent index to ReachRectangleGoal.
        robot_index: Index of the robot agent.
        robot_policy: Optional RandomPolicy for robot actions.
        max_steps: Maximum number of steps for the rollout.
        frame_buffer: List to append frames to (if None, uses env._video_frames).
    
    Returns:
        Number of steps taken.
    """
    if robot_policy is None:
        robot_policy = RandomPolicy()
    
    if frame_buffer is None:
        frame_buffer = env._video_frames
    
    # Temporarily disable env's auto-recording to avoid duplicate frames
    was_recording = getattr(env, '_recording', False)
    env._recording = False
    
    steps_taken = 0
    for step in range(max_steps):
        state = env.get_state()
        
        # Render frame with goal overlay and capture it
        frame = render_goals_on_frame(env, human_goals)
        frame_buffer.append(frame)
        
        # Get actions
        actions = []
        for agent_idx in range(len(env.agents)):
            if agent_idx in human_agent_indices:
                goal = human_goals[agent_idx]
                action = policy.sample(state, agent_idx, goal)
            else:
                action = robot_policy.sample()
            actions.append(action)
        
        _, _, done, _ = env.step(actions)
        steps_taken += 1
        
        if done:
            break
    
    # Capture final frame
    frame = render_goals_on_frame(env, human_goals)
    frame_buffer.append(frame)
    
    # Restore recording state
    env._recording = was_recording
    
    return steps_taken


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Determine configuration
    if args.quick:
        num_test_envs = args.envs if args.envs is not None else NUM_TEST_ENVS_QUICK
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS_QUICK
        mode_str = "QUICK TEST MODE"
    else:
        num_test_envs = args.envs if args.envs is not None else NUM_TEST_ENVS
        num_rollouts = args.rollouts if args.rollouts is not None else NUM_ROLLOUTS
        mode_str = "FULL MODE"
    
    beta = args.beta
    rollout_steps = args.steps
    
    print("=" * 70)
    print("Heuristic Multigrid Ensemble Demo")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Agents: {NUM_HUMANS} humans + {NUM_ROBOTS} robot")
    print(f"Test environments: {num_test_envs}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Beta (β): {beta}")
    print(f"Steps per rollout: {rollout_steps}")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Generate test environments
    print(f"Generating {num_test_envs} test environments...")
    test_environments = [create_random_env(seed=1000 + i) for i in range(num_test_envs)]
    for env in test_environments:
        env.reset()
    
    # Use a single environment for video recording (accumulates all frames)
    # We'll use the first test environment as our "recording" environment
    recording_env = test_environments[0]
    recording_env.start_video_recording()
    
    # Run rollouts
    print(f"Running {num_rollouts} rollouts...")
    print(f"  Humans: heuristic potential policy (beta={beta})")
    print(f"  Robot: random policy")
    print()
    
    robot_policy = RandomPolicy()
    
    for rollout_idx in range(num_rollouts):
        # Select environment
        env_idx = rollout_idx % len(test_environments)
        env = test_environments[env_idx]
        env.reset()
        
        # Get agent indices for this environment
        env_human_indices, env_robot_index = get_agent_indices(env)
        first_human_idx = env_human_indices[0] if env_human_indices else 0
        
        # Create heuristic policy for this environment
        # (needs to be recreated per env since PathDistanceCalculator is env-specific)
        policy = create_heuristic_policy(env, env_human_indices, beta)
        
        # Sample goals
        goal_sampler = SmallGoalSampler(env)
        state = env.get_state()
        
        human_goals = {}
        for h_idx in env_human_indices:
            goal, _ = goal_sampler.sample(state, h_idx)
            human_goals[h_idx] = goal
        
        first_human_goal = human_goals.get(first_human_idx, None)
        goal_rect = first_human_goal.target_rect if first_human_goal else None
        
        print(f"  Rollout {rollout_idx + 1}/{num_rollouts}: Env {env_idx + 1}, "
              f"H1 Goal: {goal_rect}, humans={env_human_indices}")
        
        # Start video recording for this env if it's not the recording env
        # We'll collect frames directly into recording_env's frame buffer
        env._recording = True
        env._video_frames = recording_env._video_frames  # Share frame buffer
        
        steps = run_rollout(
            env=env,
            policy=policy,
            human_agent_indices=env_human_indices,
            human_goals=human_goals,
            robot_index=env_robot_index,
            robot_policy=robot_policy,
            max_steps=rollout_steps
        )
        print(f"    {steps} steps, {len(recording_env._video_frames)} total frames")
    
    print()
    
    # Save movie using environment's save_video method
    movie_path = os.path.join(output_dir, 'heuristic_multigrid_ensemble_demo.mp4')
    # Remove existing file to ensure clean overwrite
    if os.path.exists(movie_path):
        os.remove(movie_path)
    recording_env.save_video(movie_path, fps=10)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Movie output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
