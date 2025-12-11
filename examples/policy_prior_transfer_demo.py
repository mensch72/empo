#!/usr/bin/env python3
"""
Policy Prior Transfer Demo: Train on SmallActions, Reuse with Full Actions.

This script demonstrates the save/load functionality of NeuralHumanPolicyPrior
to enable policy transfer across different action spaces and agent configurations.

The demo:
1. Trains a neural network policy prior on an ensemble of random environments with:
   - SmallActions (4 actions: still, left, right, forward)
   - 2 human agents (yellow)
   - 1 robot agent (grey)
   
2. Saves the trained model to a file

3. Loads the model for use with a different configuration:
   - Full Actions (8 actions: still, left, right, forward, pickup, drop, toggle, done)
   - 3 human agents (yellow)
   - 2 robot agents (grey)
   
4. Runs rollouts with the transferred policy to verify it works

Key Features Demonstrated:
- Save/load of trained neural networks with metadata
- Action space compatibility checking
- Handling of actions not present in original training
- Agent count flexibility (more agents than trained on)

Usage:
    python policy_prior_transfer_demo.py           # Full run
    python policy_prior_transfer_demo.py --quick   # Quick test run

Requirements:
    - torch
    - matplotlib (for movie output)
"""

import sys
import os
import time
import random
import argparse
import tempfile
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions, Actions,
    Key, Ball, Box, Door, Lava, Block, Goal
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.multigrid import ReachCellGoal, MultiGridGoalSampler, RandomPolicy
from empo.nn_based.multigrid import (
    MultiGridNeuralHumanPolicyPrior as NeuralHumanPolicyPrior,
    MultiGridQNetwork as QNetwork,
    train_multigrid_neural_policy_prior as train_neural_policy_prior,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
)


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 7           # 7x7 grid (including outer walls)
MAX_STEPS = 15          # Maximum steps per episode

# Training configuration (SmallActions, 2 humans, 1 robot)
TRAIN_NUM_HUMANS = 2
TRAIN_NUM_ROBOTS = 1

# Test configuration (Full Actions, 3 humans, 2 robots)
TEST_NUM_HUMANS = 3
TEST_NUM_ROBOTS = 2

# Episode counts
NUM_TRAINING_EPISODES_FULL = 300
NUM_TRAINING_EPISODES_QUICK = 30
NUM_ROLLOUTS_FULL = 5
NUM_ROLLOUTS_QUICK = 2


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Policy Prior Transfer Demo: Train SmallActions -> Use Full Actions"
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
    return parser.parse_args()


# ============================================================================
# Environment Definitions
# ============================================================================

class SmallActionsEnv(MultiGridEnv):
    """
    Environment with SmallActions (4 actions) for training.
    
    Agents: 2 humans (yellow) + 1 robot (grey) = 3 total
    """
    
    def __init__(
        self,
        grid_size: int = 7,
        num_humans: int = 2,
        num_robots: int = 1,
        max_steps: int = 15,
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
        """Generate a simple map with some obstacles."""
        lines = []
        
        # Valid positions for agents
        available_cells = []
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')  # Wall
                else:
                    # Add some random walls
                    if random.random() < 0.1:
                        row.append('We')
                    else:
                        row.append('..')
                        available_cells.append((x, y))
            lines.append(' '.join(row))
        
        # Convert to grid for agent placement
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


class FullActionsEnv(MultiGridEnv):
    """
    Environment with Full Actions (8 actions) for testing transfer.
    
    Agents: 3 humans (yellow) + 2 robots (grey) = 5 total
    """
    
    def __init__(
        self,
        grid_size: int = 7,
        num_humans: int = 3,
        num_robots: int = 2,
        max_steps: int = 15,
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
            actions_set=Actions  # Full 8 actions
        )
    
    def _generate_map(self) -> str:
        """Generate a map with some objects for pickup/toggle actions."""
        lines = []
        colors = ['r', 'g', 'b', 'p']
        available_cells = []
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    row.append('We')
                else:
                    r = random.random()
                    if r < 0.08:
                        row.append('We')  # Wall
                    elif r < 0.12:
                        c = random.choice(colors)
                        row.append(f'K{c}')  # Key
                        available_cells.append((x, y))
                    elif r < 0.14:
                        c = random.choice(colors)
                        row.append(f'B{c}')  # Ball
                        available_cells.append((x, y))
                    else:
                        row.append('..')
                        available_cells.append((x, y))
            lines.append(' '.join(row))
        
        grid_lines = [line.split() for line in lines]
        
        num_agents = self.num_humans + self.num_robots
        
        # Find empty cells
        empty_cells = [(x, y) for (x, y) in available_cells 
                       if grid_lines[y][x] == '..']
        
        while len(empty_cells) < num_agents:
            for y in range(1, self.grid_size - 1):
                for x in range(1, self.grid_size - 1):
                    if grid_lines[y][x] not in ['..', 'Ay', 'Ae'] and (x, y) not in empty_cells:
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
# Training and Transfer
# ============================================================================

def train_small_actions_policy(
    num_episodes: int,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[NeuralHumanPolicyPrior, str]:
    """
    Train a policy on SmallActions environment and save it.
    
    Returns:
        Tuple of (trained_prior, save_path)
    """
    print("=" * 60)
    print("Phase 1: Training on SmallActions (4 actions)")
    print(f"  Agents: {TRAIN_NUM_HUMANS} humans + {TRAIN_NUM_ROBOTS} robot")
    print(f"  Episodes: {num_episodes}")
    print("=" * 60)
    
    # Create base environment
    base_env = SmallActionsEnv(
        grid_size=GRID_SIZE,
        num_humans=TRAIN_NUM_HUMANS,
        num_robots=TRAIN_NUM_ROBOTS,
        max_steps=MAX_STEPS,
        seed=42
    )
    base_env.reset()
    
    # Identify human agents
    human_agent_indices = [i for i, a in enumerate(base_env.agents) if a.color == 'yellow']
    
    print(f"  Human agent indices: {human_agent_indices}")
    print(f"  Action space: {base_env.actions.available}")
    
    # Use MultiGridGoalSampler for weight-proportional goal sampling
    goal_sampler = MultiGridGoalSampler(base_env)
    
    # World model generator for ensemble training
    def world_model_generator(episode: int):
        env = SmallActionsEnv(
            grid_size=GRID_SIZE,
            num_humans=TRAIN_NUM_HUMANS,
            num_robots=TRAIN_NUM_ROBOTS,
            max_steps=MAX_STEPS,
            seed=42 + episode
        )
        env.reset()
        return env
    
    # Train
    t0 = time.time()
    neural_prior = train_neural_policy_prior(
        world_model=base_env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=num_episodes,
        steps_per_episode=MAX_STEPS,
        beta=100.0,
        gamma=0.99,
        learning_rate=1e-3,
        batch_size=64,
        replay_buffer_size=5000,
        updates_per_episode=4,
        train_phi_network=False,
        epsilon=0.3,
        reward_shaping=True,  # Use path-based reward shaping with passing costs
        device=device,
        verbose=verbose,
        world_model_generator=world_model_generator,
        episodes_per_model=1
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    
    # Save the model
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'small_actions_policy.pt')
    
    neural_prior.save(save_path)
    print(f"  Model saved to: {save_path}")
    
    return neural_prior, save_path


def load_and_transfer_policy(
    save_path: str,
    device: str = 'cpu'
) -> Tuple[NeuralHumanPolicyPrior, FullActionsEnv]:
    """
    Load a saved policy and adapt it for FullActions environment.
    
    Returns:
        Tuple of (loaded_prior, test_environment)
    """
    print()
    print("=" * 60)
    print("Phase 2: Loading and Transferring to Full Actions (8 actions)")
    print(f"  Agents: {TEST_NUM_HUMANS} humans + {TEST_NUM_ROBOTS} robots")
    print("=" * 60)
    
    # Create test environment with full actions
    test_env = FullActionsEnv(
        grid_size=GRID_SIZE,
        num_humans=TEST_NUM_HUMANS,
        num_robots=TEST_NUM_ROBOTS,
        max_steps=MAX_STEPS,
        seed=100
    )
    test_env.reset()
    
    # Identify human agents
    human_agent_indices = [i for i, a in enumerate(test_env.agents) if a.color == 'yellow']
    
    print(f"  Human agent indices: {human_agent_indices}")
    print(f"  Action space: {test_env.actions.available}")
    
    # Use MultiGridGoalSampler for weight-proportional goal sampling
    goal_sampler = MultiGridGoalSampler(test_env)
    
    # Load with action transfer
    # Actions not in SmallActions (pickup, drop, toggle, done) will get 0 probability
    loaded_prior = NeuralHumanPolicyPrior.load(
        save_path,
        world_model=test_env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        infeasible_actions_become=None,  # Mask out new actions (they get 0 probability)
        device=device
    )
    
    print("  Policy loaded successfully!")
    
    return loaded_prior, test_env


def run_transfer_rollouts(
    prior: NeuralHumanPolicyPrior,
    env: FullActionsEnv,
    goal_sampler: MultiGridGoalSampler,
    num_rollouts: int,
    device: str = 'cpu'
) -> None:
    """Run rollouts using the transferred policy."""
    print()
    print("=" * 60)
    print(f"Phase 3: Running {num_rollouts} Rollouts with Transferred Policy")
    print("=" * 60)
    
    human_agent_indices = [i for i, a in enumerate(env.agents) if a.color == 'yellow']
    
    for rollout_idx in range(num_rollouts):
        env.reset()
        state = env.get_state()
        
        # Sample goals for each human
        human_goals = {}
        for h_idx in human_agent_indices:
            goal, _ = goal_sampler.sample(state, h_idx)
            human_goals[h_idx] = goal
        
        print(f"\nRollout {rollout_idx + 1}:")
        print(f"  Human goals: {[str(g) for g in human_goals.values()]}")
        
        total_rewards = {h_idx: 0 for h_idx in human_agent_indices}
        
        # Robot uses RandomPolicy (biased toward forward movement)
        robot_policy = RandomPolicy()
        
        for step in range(env.max_steps):
            state = env.get_state()
            
            # Get actions for all agents
            actions = []
            for agent_idx in range(len(env.agents)):
                if agent_idx in human_agent_indices:
                    goal = human_goals[agent_idx]
                    # Use prior.sample() directly
                    action = prior.sample(state, agent_idx, goal)
                else:
                    # Robot uses random policy
                    action = robot_policy.sample()
                
                actions.append(action)
            
            # Step
            _, _, done, _ = env.step(actions)
            next_state = env.get_state()
            
            # Check goal achievements
            for h_idx in human_agent_indices:
                if human_goals[h_idx].is_achieved(next_state):
                    total_rewards[h_idx] = 1
            
            if done:
                break
        
        # Report results
        goals_reached = sum(1 for r in total_rewards.values() if r > 0)
        print(f"  Steps: {step + 1}, Goals reached: {goals_reached}/{len(human_agent_indices)}")


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
    print("Policy Prior Transfer Demo")
    print(f"  [{mode_str}]")
    print("Train on SmallActions (4) -> Transfer to Full Actions (8)")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    
    # Phase 1: Train on SmallActions
    neural_prior, save_path = train_small_actions_policy(
        num_episodes=num_episodes,
        device=device,
        verbose=True
    )
    
    # Phase 2: Load and transfer to FullActions
    loaded_prior, test_env = load_and_transfer_policy(
        save_path=save_path,
        device=device
    )
    
    # Phase 3: Run rollouts
    goal_sampler = MultiGridGoalSampler(test_env)
    run_transfer_rollouts(
        prior=loaded_prior,
        env=test_env,
        goal_sampler=goal_sampler,
        num_rollouts=num_rollouts,
        device=device
    )
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  - Policy trained on SmallActions (4 actions) successfully loaded")
    print("  - Compatible actions (still, left, right, forward) work correctly")
    print("  - New actions (pickup, drop, toggle, done) get 0 probability")
    print("  - Agent count can be different between training and deployment")
    print()


if __name__ == "__main__":
    main()
