#!/usr/bin/env python3
"""
Example: Cross-Grid Policy Loading for Multigrid

This example demonstrates how to:
1. Train a policy on a large grid (15x15)
2. Save the trained policy
3. Load the policy for use on a smaller grid (10x10)

The policy trained on the larger grid is automatically adapted to work on the
smaller grid by padding the encoded state with walls. This enables transfer
learning and efficient policy reuse across different grid sizes.
"""

import os

import tempfile
import torch
from empo.learning_based.multigrid import (
    MultiGridNeuralHumanPolicyPrior,
    train_multigrid_neural_policy_prior,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from typing import Tuple
import numpy as np


class SimpleReachGoal(PossibleGoal):
    """A simple goal to reach a target position."""
    
    def __init__(self, world_model, target_pos):
        super().__init__(world_model)
        self.target_pos = target_pos
        self.target_rect = (target_pos[0], target_pos[1], target_pos[0], target_pos[1])
    
    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        if len(agent_states) > 0 and agent_states[0][0] is not None:
            x, y = int(agent_states[0][0]), int(agent_states[0][1])
            if x == self.target_pos[0] and y == self.target_pos[1]:
                return 1
        return 0
    
    def __hash__(self):
        return hash(self.target_pos)
    
    def __eq__(self, other):
        return isinstance(other, SimpleReachGoal) and self.target_pos == other.target_pos


class SimpleGoalSampler(PossibleGoalSampler):
    """A simple goal sampler that samples random reachable positions."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        # Sample a random position in the grid
        x = np.random.randint(1, self.world_model.width - 1)
        y = np.random.randint(1, self.world_model.height - 1)
        return SimpleReachGoal(self.world_model, (x, y)), 1.0


class SimpleMockWorld:
    """A simple mock world for demonstration."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.max_steps = 100
        self.grid = SimpleMockGrid(width, height)
        self.agents = [SimpleMockAgent('agent_0', 'grey')]
        self.stumble_prob = 0.0
        self.magic_entry_prob = 1.0
        self.magic_solidify_prob = 0.0
        self._state = self._create_initial_state()
    
    def _create_initial_state(self):
        step_count = 0
        agent_states = [(self.width // 2, self.height // 2, 0, False, True)]
        mobile_objects = []
        mutable_objects = []
        return (step_count, agent_states, mobile_objects, mutable_objects)
    
    def reset(self):
        self._state = self._create_initial_state()
        return self._state
    
    def get_state(self):
        return self._state
    
    def step(self, actions):
        # Simple step: just move agent randomly for demo
        step_count, agent_states, mobile_objects, mutable_objects = self._state
        new_agent_states = []
        for i, (x, y, d, t, s) in enumerate(agent_states):
            # Random walk
            dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
            new_x = max(0, min(self.width - 1, x + dx))
            new_y = max(0, min(self.height - 1, y + dy))
            new_agent_states.append((new_x, new_y, d, t, s))
        
        self._state = (step_count + 1, new_agent_states, mobile_objects, mutable_objects)
        return self._state


class SimpleMockGrid:
    """A simple mock grid."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def get(self, x, y):
        return None  # Empty grid for demo


class SimpleMockAgent:
    """A simple mock agent."""
    
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.can_enter_magic_walls = False
        self.can_push_rocks = False
        self.carrying = None
        self.paused = False
        self.terminated = False
        self.forced_next_action = None


def main():
    print("=" * 70)
    print("Cross-Grid Policy Loading Example")
    print("=" * 70)
    print()
    
    # Step 1: Create a large world (15x15) and train a policy
    print("Step 1: Training policy on LARGE grid (15x15)...")
    print("-" * 70)
    
    large_world = SimpleMockWorld(width=15, height=15)
    goal_sampler_large = SimpleGoalSampler(large_world)
    
    # Train a simple policy (minimal training for demo speed)
    # Note: Using very few episodes/steps for demonstration purposes only.
    # In practice, you would use much larger values (e.g., 1000 episodes, 100 steps each).
    prior_large = train_multigrid_neural_policy_prior(
        env=large_world,
        human_agent_indices=[0],
        goal_sampler=goal_sampler_large,
        num_episodes=10,  # Minimal for demo speed
        steps_per_episode=20,  # Minimal for demo speed
        batch_size=16,
        learning_rate=0.001,
        device='cpu',
        verbose=False
    )
    
    print(f"✓ Policy trained on 15x15 grid")
    print(f"  Grid dimensions: {prior_large.q_network.state_encoder.grid_height} x "
          f"{prior_large.q_network.state_encoder.grid_width}")
    print()
    
    # Step 2: Save the policy
    print("Step 2: Saving trained policy...")
    print("-" * 70)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        policy_path = f.name
    
    prior_large.save(policy_path)
    print(f"✓ Policy saved to: {policy_path}")
    print()
    
    # Step 3: Create a small world (10x10) and load the policy
    print("Step 3: Loading policy on SMALL grid (10x10)...")
    print("-" * 70)
    
    small_world = SimpleMockWorld(width=10, height=10)
    goal_sampler_small = SimpleGoalSampler(small_world)
    
    # Load the policy trained on the large grid for use on the small grid
    prior_small = MultiGridNeuralHumanPolicyPrior.load(
        policy_path,
        world_model=small_world,
        human_agent_indices=[0],
        goal_sampler=goal_sampler_small,
        device='cpu'
    )
    
    print(f"✓ Policy loaded successfully!")
    print(f"  Original training grid: 15 x 15")
    print(f"  Current world grid:     10 x 10")
    print(f"  Encoder grid (padded):  {prior_small.q_network.state_encoder.grid_height} x "
          f"{prior_small.q_network.state_encoder.grid_width}")
    print()
    print("  How it works:")
    print("  - The encoder maintains the 15x15 grid from training")
    print("  - The actual 10x10 world is encoded in the top-left")
    print("  - The remaining area (10-15 in x and y) is padded with walls")
    print("  - All coordinates are absolute integers, so they remain valid")
    print()
    
    # Step 4: Test the loaded policy
    print("Step 4: Testing loaded policy...")
    print("-" * 70)
    
    state = small_world.get_state()
    goal = SimpleReachGoal(small_world, (5, 5))
    
    prior_small.q_network.eval()
    with torch.no_grad():
        q_values = prior_small.q_network.forward(
            state, small_world, 0, goal, device='cpu'
        )
        policy = prior_small.q_network.get_policy(q_values)
    
    print(f"✓ Policy produces valid output")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Policy shape:   {policy.shape}")
    print(f"  Policy sum:     {policy.sum().item():.4f} (should be 1.0)")
    print()
    
    # Cleanup
    os.unlink(policy_path)
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("- Policies trained on larger grids work on smaller grids")
    print("- The encoder automatically pads smaller grids with walls")
    print("- This enables efficient transfer learning across grid sizes")
    print("- No retraining needed when deploying to smaller environments")


if __name__ == '__main__':
    main()
