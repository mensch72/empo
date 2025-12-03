#!/usr/bin/env python3
"""
Test script for the neural network-based policy prior.

Tests the modular architecture with:
- Base classes in nn_based/
- Multigrid-specific implementations in nn_based/multigrid/
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import tempfile

from empo.nn_based import (
    BaseStateEncoder,
    BaseGoalEncoder,
    BaseQNetwork,
    BasePolicyPriorNetwork,
    BaseNeuralHumanPolicyPrior,
    ReplayBuffer,
    Trainer,
)
from empo.nn_based.multigrid import (
    MultiGridStateEncoder,
    MultiGridGoalEncoder,
    MultiGridQNetwork,
    MultiGridPolicyPriorNetwork,
    MultiGridNeuralHumanPolicyPrior,
    NUM_OBJECT_TYPE_CHANNELS,
    AGENT_FEATURE_SIZE,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler, PossibleGoalGenerator
from typing import Iterator, Tuple


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


def test_multigrid_state_encoder():
    """Test the unified MultiGridStateEncoder."""
    print("Testing MultiGridStateEncoder...")
    
    num_agents_per_color = {'grey': 2}
    
    encoder = MultiGridStateEncoder(
        grid_height=10,
        grid_width=10,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=7,
        feature_dim=256
    )
    
    # Create dummy inputs
    batch_size = 4
    grid_tensor = torch.randn(batch_size, encoder.num_grid_channels, 10, 10)
    global_features = torch.randn(batch_size, 4)
    
    # Calculate agent feature size
    total_agents = sum(num_agents_per_color.values())
    agent_feature_size = AGENT_FEATURE_SIZE * (1 + total_agents)  # query + per-color
    agent_features = torch.randn(batch_size, agent_feature_size)
    
    interactive_features = torch.randn(batch_size, encoder._interactive_input_size)
    
    # Forward pass
    features = encoder(grid_tensor, global_features, agent_features, interactive_features)
    
    assert features.shape == (batch_size, 256), f"Expected (4, 256), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print(f"  ✓ Num grid channels: {encoder.num_grid_channels}")
    print("  ✓ MultiGridStateEncoder test passed!")


def test_multigrid_state_encoder_encode_state():
    """Test state encoding from actual state tuple."""
    print("Testing MultiGridStateEncoder.encode_state...")
    
    world_model = MockWorldModel()
    state = create_mock_state(num_agents=2)
    
    encoder = MultiGridStateEncoder(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        feature_dim=256
    )
    
    grid_tensor, global_features, agent_features, interactive_features = \
        encoder.encode_state(state, world_model, query_agent_idx=0)
    
    assert grid_tensor.shape[0] == 1
    assert global_features.shape == (1, 4)
    print(f"  ✓ Grid tensor shape: {grid_tensor.shape}")
    print(f"  ✓ Global features shape: {global_features.shape}")
    print(f"  ✓ Agent features shape: {agent_features.shape}")
    print(f"  ✓ Interactive features shape: {interactive_features.shape}")
    print("  ✓ MultiGridStateEncoder.encode_state test passed!")


def test_multigrid_goal_encoder():
    """Test the MultiGridGoalEncoder."""
    print("Testing MultiGridGoalEncoder...")
    
    encoder = MultiGridGoalEncoder(
        grid_height=10,
        grid_width=10,
        feature_dim=32
    )
    
    # Create dummy input
    batch_size = 4
    goal_coords = torch.randn(batch_size, 2)
    
    # Forward pass
    features = encoder(goal_coords)
    
    assert features.shape == (batch_size, 32), f"Expected (4, 32), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ MultiGridGoalEncoder test passed!")


def test_multigrid_q_network():
    """Test the MultiGridQNetwork."""
    print("Testing MultiGridQNetwork...")
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        state_feature_dim=256,
        goal_feature_dim=32,
        hidden_dim=256,
        beta=1.0
    )
    
    # Test with mock state
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    q_values = q_network.encode_and_forward(state, world_model, 0, goal)
    
    assert q_values.shape == (1, 4), f"Expected (1, 4), got {q_values.shape}"
    print(f"  ✓ Q-values shape: {q_values.shape}")
    
    # Test policy computation
    policy = q_network.get_policy(q_values)
    assert policy.shape == (1, 4)
    assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
    print(f"  ✓ Policy shape: {policy.shape}")
    print(f"  ✓ Policy sums to 1: {policy.sum().item():.4f}")
    print("  ✓ MultiGridQNetwork test passed!")


def test_replay_buffer():
    """Test the ReplayBuffer."""
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    # Add transitions
    for i in range(50):
        buffer.push(state, i % 4, state, 0, goal)
    
    assert len(buffer) == 50
    
    # Sample batch
    batch = buffer.sample(16)
    assert len(batch) == 16
    
    print(f"  ✓ Buffer length: {len(buffer)}")
    print(f"  ✓ Batch size: {len(batch)}")
    print("  ✓ ReplayBuffer test passed!")


def test_trainer():
    """Test the Trainer class."""
    print("Testing Trainer...")
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    target_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.Adam(q_network.parameters())
    buffer = ReplayBuffer(capacity=100)
    
    trainer = Trainer(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=buffer,
        gamma=0.99,
        target_update_freq=10,
        device='cpu'
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    # Store some transitions
    for i in range(32):
        trainer.store_transition(state, i % 4, state, 0, goal)
    
    # Train step
    loss = trainer.train_step(batch_size=16)
    assert loss is not None
    
    print(f"  ✓ Training loss: {loss:.4f}")
    print(f"  ✓ Total steps: {trainer.total_steps}")
    print("  ✓ Trainer test passed!")


def test_save_load():
    """Test save and load functionality."""
    print("Testing save/load...")
    
    world_model = MockWorldModel()
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
        action_encoding={0: 'still', 1: 'forward', 2: 'left', 3: 'right'},
        device='cpu'
    )
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    print(f"  ✓ Saved to {filepath}")
    
    # Load
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
        original_q = q_network.encode_and_forward(state, world_model, 0, goal)
        loaded_q = loaded.q_network.encode_and_forward(state, world_model, 0, goal)
    
    assert torch.allclose(original_q, loaded_q, atol=1e-5)
    print("  ✓ Loaded model produces same Q-values")
    
    # Cleanup
    os.unlink(filepath)
    print("  ✓ save/load test passed!")


def test_load_dimension_mismatch():
    """Test that load fails with mismatched dimensions."""
    print("Testing load with dimension mismatch...")
    
    world_model = MockWorldModel(width=10, height=10)
    
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
        device='cpu'
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    
    # Try to load with different grid dimensions
    different_world = MockWorldModel(width=15, height=15)
    
    try:
        MultiGridNeuralHumanPolicyPrior.load(
            filepath,
            world_model=different_world,
            human_agent_indices=[0],
            device='cpu'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Grid dimensions mismatch" in str(e)
        print(f"  ✓ Correctly raised: {e}")
    
    os.unlink(filepath)
    print("  ✓ load dimension mismatch test passed!")


def test_load_action_conflict():
    """Test that load fails with conflicting action encodings."""
    print("Testing load with action conflict...")
    
    world_model = MockWorldModel()
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    # Save with one action encoding
    prior = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=world_model,
        human_agent_indices=[0],
        action_encoding={0: 'still', 1: 'forward', 2: 'left', 3: 'right'},
        device='cpu'
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    
    # Create world model with conflicting actions
    class MockAction:
        def __init__(self, name):
            self.name = name
    
    world_model_conflict = MockWorldModel()
    world_model_conflict.actions = [
        MockAction('BACKWARD'),  # Conflicts with 'still' at index 0
        MockAction('FORWARD'),
        MockAction('LEFT'),
        MockAction('RIGHT'),
    ]
    
    try:
        MultiGridNeuralHumanPolicyPrior.load(
            filepath,
            world_model=world_model_conflict,
            human_agent_indices=[0],
            device='cpu'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Action encoding conflict" in str(e)
        print(f"  ✓ Correctly raised: {e}")
    
    os.unlink(filepath)
    print("  ✓ load action conflict test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Neural Policy Prior Tests")
    print("=" * 60)
    
    test_multigrid_state_encoder()
    print()
    
    test_multigrid_state_encoder_encode_state()
    print()
    
    test_multigrid_goal_encoder()
    print()
    
    test_multigrid_q_network()
    print()
    
    test_replay_buffer()
    print()
    
    test_trainer()
    print()
    
    test_save_load()
    print()
    
    test_load_dimension_mismatch()
    print()
    
    test_load_action_conflict()
    print()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
