#!/usr/bin/env python3
"""
Test cross-grid policy loading for multigrid environments.

Tests that policies trained on larger grids can be loaded and used on smaller grids,
with automatic padding using grey walls.
"""

import sys
import os

import numpy as np
import torch
import tempfile

from empo.nn_based.multigrid import (
    MultiGridStateEncoder,
    MultiGridQNetwork,
    MultiGridNeuralHumanPolicyPrior,
    OBJECT_TYPE_TO_CHANNEL,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from typing import Tuple


class SimpleReachGoal(PossibleGoal):
    """A simple goal to reach a target position."""
    
    def __init__(self, world_model, target_pos):
        super().__init__(world_model)
        self.target_pos = target_pos
    
    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        if len(agent_states) > 0:
            x, y = agent_states[0][0], agent_states[0][1]
            if x == self.target_pos[0] and y == self.target_pos[1]:
                return 1
        return 0
    
    def __hash__(self):
        return hash(self.target_pos)
    
    def __eq__(self, other):
        return isinstance(other, SimpleReachGoal) and self.target_pos == other.target_pos


class SimpleGoalSampler(PossibleGoalSampler):
    """A simple goal sampler that samples random target positions."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        x = np.random.randint(0, self.world_model.width)
        y = np.random.randint(0, self.world_model.height)
        return SimpleReachGoal(self.world_model, (x, y)), 1.0


class MockWorldModel:
    """Mock world model for testing."""
    
    def __init__(self, width=10, height=10, num_agents=2):
        self.width = width
        self.height = height
        self.max_steps = 100
        self.grid = MockGrid(width, height)
        self.agents = [MockAgent(f'agent_{i}', 'grey') for i in range(num_agents)]
        self.stumble_prob = 0.0
        self.magic_entry_prob = 1.0
        self.magic_solidify_prob = 0.0


class MockGrid:
    """Mock grid for testing."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._cells = {}
    
    def get(self, x, y):
        return self._cells.get((x, y), None)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, color='grey'):
        self.name = name
        self.color = color
        self.can_enter_magic_walls = False
        self.can_push_rocks = False
        self.carrying = None
        self.paused = False
        self.terminated = False
        self.forced_next_action = None


def create_mock_state(num_agents=2):
    """Create a mock state tuple."""
    step_count = 0
    agent_states = [(i, i, 0, False, False) for i in range(num_agents)]  # x, y, dir, term, started
    mobile_objects = []
    mutable_objects = []
    return (step_count, agent_states, mobile_objects, mutable_objects)


def test_state_encoder_padding():
    """Test that state encoder pads smaller grids with walls."""
    print("Testing state encoder padding...")
    
    # Create encoder configured for 15x15 grid
    encoder = MultiGridStateEncoder(
        grid_height=15,
        grid_width=15,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        feature_dim=256
    )
    
    # Create world model with smaller 10x10 grid
    world_model = MockWorldModel(width=10, height=10)
    state = create_mock_state()
    
    # Encode state (agent-agnostic)
    grid_tensor, global_features, agent_features, interactive_features = \
        encoder.tensorize_state(state, world_model)
    
    # Check dimensions
    assert grid_tensor.shape == (1, encoder.num_grid_channels, 15, 15)
    print(f"  ✓ Grid tensor shape correct: {grid_tensor.shape}")
    
    # Check that padding area has walls
    wall_channel = OBJECT_TYPE_TO_CHANNEL['wall']
    
    # Right padding (x >= 10)
    right_padding = grid_tensor[0, wall_channel, :, 10:]
    assert torch.all(right_padding == 1.0), "Right padding should be all walls"
    print("  ✓ Right padding has walls")
    
    # Bottom padding (y >= 10)
    bottom_padding = grid_tensor[0, wall_channel, 10:, :]
    assert torch.all(bottom_padding == 1.0), "Bottom padding should be all walls"
    print("  ✓ Bottom padding has walls")
    
    # Inner area (actual world) should not have walls (unless explicitly placed)
    inner_area = grid_tensor[0, wall_channel, :10, :10]
    # Inner area should be all zeros since mock world has no walls
    assert torch.all(inner_area == 0.0), "Inner area should not have walls in mock world"
    print("  ✓ Inner area clear (no walls)")
    
    print("  ✓ State encoder padding test passed!")


def test_cross_grid_save_load():
    """Test saving policy on large grid and loading on small grid."""
    print("Testing cross-grid save/load...")
    
    # Train on large grid (15x15)
    large_world = MockWorldModel(width=15, height=15)
    goal_sampler_large = SimpleGoalSampler(large_world)
    
    q_network_large = MultiGridQNetwork(
        grid_height=15,
        grid_width=15,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    prior_large = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network_large,
        world_model=large_world,
        human_agent_indices=[0],
        goal_sampler=goal_sampler_large,
        action_encoding={0: 'still', 1: 'forward', 2: 'left', 3: 'right'},
        device='cpu'
    )
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior_large.save(filepath)
    print(f"  ✓ Saved policy trained on 15x15 grid")
    
    # Load on smaller grid (10x10)
    small_world = MockWorldModel(width=10, height=10)
    goal_sampler_small = SimpleGoalSampler(small_world)
    
    loaded = MultiGridNeuralHumanPolicyPrior.load(
        filepath,
        world_model=small_world,
        human_agent_indices=[0],
        goal_sampler=goal_sampler_small,
        device='cpu'
    )
    
    print("  ✓ Successfully loaded policy on 10x10 grid")
    
    # Test that loaded model can be used on small world
    state = create_mock_state()
    goal = SimpleReachGoal(small_world, (5, 5))
    
    loaded.q_network.eval()
    
    with torch.no_grad():
        q_values = loaded.q_network.forward(state, small_world, 0, goal)
    
    assert q_values.shape == (1, 4), "Q-values should have correct shape"
    print("  ✓ Loaded policy produces valid Q-values on small grid")
    
    # Cleanup
    os.unlink(filepath)
    print("  ✓ Cross-grid save/load test passed!")


def test_reject_larger_grid():
    """Test that loading policy from small grid to large grid is rejected."""
    print("Testing rejection of larger grid...")
    
    # Train on small grid (10x10)
    small_world = MockWorldModel(width=10, height=10)
    
    q_network_small = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    prior_small = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network_small,
        world_model=small_world,
        human_agent_indices=[0],
        device='cpu'
    )
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior_small.save(filepath)
    
    # Try to load on larger grid (15x15) - should fail
    large_world = MockWorldModel(width=15, height=15)
    
    try:
        MultiGridNeuralHumanPolicyPrior.load(
            filepath,
            world_model=large_world,
            human_agent_indices=[0],
            device='cpu'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot load policy trained on smaller grid" in str(e)
        print(f"  ✓ Correctly rejected: {e}")
    
    # Cleanup
    os.unlink(filepath)
    print("  ✓ Reject larger grid test passed!")


def test_equal_grid_still_works():
    """Test that equal grid dimensions still work as before."""
    print("Testing equal grid dimensions...")
    
    world_model = MockWorldModel(width=10, height=10)
    goal_sampler = SimpleGoalSampler(world_model)
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    prior = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=world_model,
        human_agent_indices=[0],
        goal_sampler=goal_sampler,
        device='cpu'
    )
    
    # Save and load with same dimensions
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    
    loaded = MultiGridNeuralHumanPolicyPrior.load(
        filepath,
        world_model=world_model,
        human_agent_indices=[0],
        goal_sampler=goal_sampler,
        device='cpu'
    )
    
    # Test that loaded model produces same outputs
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    q_network.eval()
    loaded.q_network.eval()
    
    with torch.no_grad():
        original_q = q_network.forward(state, world_model, 0, goal)
        loaded_q = loaded.q_network.forward(state, world_model, 0, goal)
    
    assert torch.allclose(original_q, loaded_q, atol=1e-5)
    print("  ✓ Equal dimensions still work correctly")
    
    # Cleanup
    os.unlink(filepath)
    print("  ✓ Equal grid test passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("Cross-Grid Policy Loading Tests")
    print("=" * 60)
    
    test_state_encoder_padding()
    print()
    test_cross_grid_save_load()
    print()
    test_reject_larger_grid()
    print()
    test_equal_grid_still_works()
    print()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
