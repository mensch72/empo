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
    direction = torch.zeros(batch_size, 4)
    direction[:, 0] = 1
    agent_idx = torch.zeros(batch_size, dtype=torch.long)
    goal_coords = torch.randn(batch_size, 4)
    global_world_features = torch.randn(batch_size, 4)
    query_abilities = torch.zeros(batch_size, 2)
    query_carried = torch.zeros(batch_size, 2)
    query_status = torch.zeros(batch_size, 3)
    query_status[:, 2] = -1.0  # No forced action
    
    # Forward pass
    q_values = q_network(
        state_tensor, remaining_time,
        position, direction, agent_idx,
        goal_coords, global_world_features,
        query_abilities, query_carried, query_status
    )
    
    assert q_values.shape == (batch_size, 4), f"Expected (4, 4), got {q_values.shape}"
    print(f"  ✓ Q-values shape: {q_values.shape}")
    
    # Test policy computation
    policy = q_network.get_policy(q_values, beta=1.0)
    assert policy.shape == (batch_size, 4), f"Expected (4, 4), got {policy.shape}"
    assert torch.allclose(policy.sum(dim=1), torch.ones(batch_size)), "Policy should sum to 1"
    print(f"  ✓ Policy shape: {policy.shape}")
    print("  ✓ QNetwork test passed!")


def test_policy_prior_network():
    """Test the PolicyPriorNetwork."""
    print("Testing PolicyPriorNetwork...")
    
    num_object_types = TEST_NUM_OBJECT_TYPES  # Updated for per-color doors/keys and magic wall channel
    num_agents = 2
    # New channel structure: num_object_types + 3 (other) + 1 (color) + 1 (query)
    num_color_channels = 1
    num_channels = num_object_types + 3 + num_color_channels + 1  # = 34
    
    # Create encoders
    state_encoder = StateEncoder(grid_width=10, grid_height=10, num_object_types=num_object_types, num_agents=num_agents)
    agent_encoder = AgentEncoder(grid_width=10, grid_height=10, num_agents=num_agents)
    
    phi_network = PolicyPriorNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        num_actions=4
    )
    
    # Create dummy input with correct number of channels
    batch_size = 4
    state_tensor = torch.randn(batch_size, num_channels, 10, 10)
    remaining_time = torch.randn(batch_size, 1)
    position = torch.randn(batch_size, 2)
    direction = torch.zeros(batch_size, 4)
    direction[:, 0] = 1
    agent_idx = torch.zeros(batch_size, dtype=torch.long)
    global_world_features = torch.randn(batch_size, 4)
    query_abilities = torch.zeros(batch_size, 2)
    query_carried = torch.zeros(batch_size, 2)
    query_status = torch.zeros(batch_size, 3)
    query_status[:, 2] = -1.0  # No forced action
    
    # Forward pass
    policy = phi_network(
        state_tensor, remaining_time,
        position, direction, agent_idx,
        global_world_features, query_abilities, query_carried, query_status
    )
    
    assert policy.shape == (batch_size, 4), f"Expected (4, 4), got {policy.shape}"
    assert torch.allclose(policy.sum(dim=1), torch.ones(batch_size)), "Policy should sum to 1"
    print(f"  ✓ Policy shape: {policy.shape}")
    print("  ✓ PolicyPriorNetwork test passed!")


def test_create_networks():
    """Test the create_policy_prior_networks utility function."""
    print("Testing create_policy_prior_networks...")
    
    # Create a mock world model
    class MockWorldModel:
        width = 10
        height = 10
        agents = [None, None]  # 2 agents
        
        class action_space:
            n = 4
    
    world_model = MockWorldModel()
    
    q_network, phi_network = create_policy_prior_networks(world_model)
    
    assert isinstance(q_network, QNetwork), "Should return QNetwork"
    assert isinstance(phi_network, PolicyPriorNetwork), "Should return PolicyPriorNetwork"
    print("  ✓ Networks created successfully")
    print("  ✓ create_policy_prior_networks test passed!")


def test_save_load():
    """Test the save and load functionality of NeuralHumanPolicyPrior."""
    print("Testing save/load functionality...")
    import os
    import tempfile
    
    # Create a mock world model with SmallActions
    class MockWorldModel:
        width = 7
        height = 7
        max_steps = 20
        
        class grid:
            @staticmethod
            def get(x, y):
                return None  # Empty grid
        
        class actions:
            available = ['still', 'left', 'right', 'forward']
            still = 0
            left = 1
            right = 2
            forward = 3
        
        class action_space:
            n = 4
        
        agents = [None, None, None]  # 3 agents
    
    world_model = MockWorldModel()
    
    # Create encoders and network
    num_agents = len(world_model.agents)
    state_encoder = StateEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    agent_encoder = AgentEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    goal_encoder = GoalEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height
    )
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=4
    )
    
    # Create the prior
    prior = NeuralHumanPolicyPrior(
        world_model=world_model,
        human_agent_indices=[0, 1],
        q_network=q_network,
        beta=10.0
    )
    
    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_prior.pt')
        prior.save(filepath)
        assert os.path.exists(filepath), "Save file should be created"
        print("  ✓ Save successful")
        
        # Load back - same environment
        loaded_prior = NeuralHumanPolicyPrior.load(
            filepath,
            world_model=world_model,
            human_agent_indices=[0, 1],
            device='cpu'
        )
        assert loaded_prior is not None, "Loaded prior should not be None"
        assert loaded_prior.beta == 10.0, "Beta should be preserved"
        print("  ✓ Load successful with same environment")
        
        # Test with a different world model that has more actions
        class MockWorldModelFullActions:
            width = 7
            height = 7
            max_steps = 20
            
            class grid:
                @staticmethod
                def get(x, y):
                    return None
            
            class actions:
                available = ['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
                still = 0
                left = 1
                right = 2
                forward = 3
                pickup = 4
                drop = 5
                toggle = 6
                done = 7
            
            class action_space:
                n = 8
            
            agents = [None, None, None, None, None]  # More agents
        
        world_model_full = MockWorldModelFullActions()
        
        # Load with full actions environment
        loaded_prior_full = NeuralHumanPolicyPrior.load(
            filepath,
            world_model=world_model_full,
            human_agent_indices=[0, 1, 2],
            infeasible_actions_become=None,  # Mask out non-existent actions
            device='cpu'
        )
        assert loaded_prior_full is not None
        assert hasattr(loaded_prior_full, '_action_mapping')
        print("  ✓ Load successful with different action space")
        
    print("  ✓ save/load test passed!")


def test_load_conflicting_dimensions():
    """Test that load raises ValueError when grid dimensions don't match."""
    print("Testing load with conflicting dimensions...")
    import os
    import tempfile
    
    # Create a mock world model with 7x7 grid
    class MockWorldModel:
        width = 7
        height = 7
        max_steps = 20
        
        class grid:
            @staticmethod
            def get(x, y):
                return None
        
        class actions:
            available = ['still', 'left', 'right', 'forward']
        
        class action_space:
            n = 4
        
        agents = [None, None, None]
    
    world_model = MockWorldModel()
    
    # Create and save a prior
    num_agents = len(world_model.agents)
    state_encoder = StateEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    agent_encoder = AgentEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    goal_encoder = GoalEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height
    )
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=4
    )
    
    prior = NeuralHumanPolicyPrior(
        world_model=world_model,
        human_agent_indices=[0, 1],
        q_network=q_network,
        beta=10.0
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_prior.pt')
        prior.save(filepath)
        
        # Create a world model with different dimensions (10x10)
        class MockWorldModelDifferentSize:
            width = 10
            height = 10
            max_steps = 20
            
            class grid:
                @staticmethod
                def get(x, y):
                    return None
            
            class actions:
                available = ['still', 'left', 'right', 'forward']
            
            class action_space:
                n = 4
            
            agents = [None, None, None]
        
        world_model_different = MockWorldModelDifferentSize()
        
        # Try to load with different dimensions - should raise ValueError
        try:
            NeuralHumanPolicyPrior.load(
                filepath,
                world_model=world_model_different,
                human_agent_indices=[0, 1],
                device='cpu'
            )
            assert False, "Should have raised ValueError for dimension mismatch"
        except ValueError as e:
            assert "Grid dimensions mismatch" in str(e)
            print(f"  ✓ Correctly raised ValueError: {e}")
    
    print("  ✓ load_conflicting_dimensions test passed!")


def test_load_conflicting_action_space():
    """Test that load raises ValueError when action encodings conflict."""
    print("Testing load with conflicting action space...")
    import os
    import tempfile
    
    # Create a mock world model with specific action encoding
    class MockWorldModel:
        width = 7
        height = 7
        max_steps = 20
        
        class grid:
            @staticmethod
            def get(x, y):
                return None
        
        class actions:
            available = ['still', 'left', 'right', 'forward']
        
        class action_space:
            n = 4
        
        agents = [None, None, None]
    
    world_model = MockWorldModel()
    
    # Create and save a prior
    num_agents = len(world_model.agents)
    state_encoder = StateEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    agent_encoder = AgentEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height,
        num_agents=num_agents
    )
    goal_encoder = GoalEncoder(
        grid_width=world_model.width,
        grid_height=world_model.height
    )
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=4
    )
    
    prior = NeuralHumanPolicyPrior(
        world_model=world_model,
        human_agent_indices=[0, 1],
        q_network=q_network,
        beta=10.0
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_prior.pt')
        prior.save(filepath)
        
        # Create a world model with conflicting action encoding
        # (same indices but different action meanings)
        class MockWorldModelConflictingActions:
            width = 7
            height = 7
            max_steps = 20
            
            class grid:
                @staticmethod
                def get(x, y):
                    return None
            
            class actions:
                # Conflict: index 1 is 'left' in saved but 'pickup' here
                available = ['still', 'pickup', 'drop', 'toggle']
            
            class action_space:
                n = 4
            
            agents = [None, None, None]
        
        world_model_conflict = MockWorldModelConflictingActions()
        
        # Try to load with conflicting action space - should raise ValueError
        try:
            NeuralHumanPolicyPrior.load(
                filepath,
                world_model=world_model_conflict,
                human_agent_indices=[0, 1],
                device='cpu'
            )
            assert False, "Should have raised ValueError for action encoding conflict"
        except ValueError as e:
            assert "Action encoding conflict" in str(e)
            print(f"  ✓ Correctly raised ValueError: {e}")
    
    print("  ✓ load_conflicting_action_space test passed!")


# =============================================================================
# Tests for new nn_based.multigrid subpackage
# =============================================================================

def test_multigrid_state_encoder():
    """Test the MultiGridStateEncoder from the new subpackage."""
    from empo.nn_based.multigrid import MultiGridStateEncoder, NUM_OBJECT_TYPE_CHANNELS, NUM_GLOBAL_WORLD_FEATURES
    print("Testing MultiGridStateEncoder...")
    
    encoder = MultiGridStateEncoder(
        grid_height=10,
        grid_width=10,
        num_object_types=NUM_OBJECT_TYPE_CHANNELS,
        num_agent_colors=7,
        feature_dim=128
    )
    
    # Expected channels: 29 object types + 3 other + 7 colors + 1 query = 40
    assert encoder.num_channels == 40, f"Expected 40 channels, got {encoder.num_channels}"
    
    # Create dummy input
    batch_size = 4
    grid_tensor = torch.randn(batch_size, encoder.num_channels, 10, 10)
    global_features = torch.randn(batch_size, NUM_GLOBAL_WORLD_FEATURES)
    
    features = encoder(grid_tensor, global_features)
    assert features.shape == (batch_size, 128), f"Expected (4, 128), got {features.shape}"
    
    print(f"  ✓ Channels: {encoder.num_channels}")
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ MultiGridStateEncoder test passed!")


def test_multigrid_agent_encoder():
    """Test the MultiGridAgentEncoder from the new subpackage."""
    from empo.nn_based.multigrid import MultiGridAgentEncoder, AGENT_FEATURE_SIZE, COLOR_TO_IDX
    print("Testing MultiGridAgentEncoder...")
    
    encoder = MultiGridAgentEncoder(
        num_agents_per_color={'yellow': 2, 'grey': 1},
        feature_dim=64
    )
    
    # Input dim: query(13) + yellow(2*13) + grey(1*13) = 13 + 26 + 13 = 52
    assert encoder.input_dim == 52, f"Expected input_dim 52, got {encoder.input_dim}"
    
    batch_size = 4
    query_pos = torch.randn(batch_size, 2)
    query_dir = torch.zeros(batch_size, 4)
    query_dir[:, 0] = 1
    query_abil = torch.zeros(batch_size, 2)
    query_carr = torch.zeros(batch_size, 2)
    query_stat = torch.zeros(batch_size, 3)
    query_stat[:, 2] = -1
    
    # All agents data
    num_agents = 3
    all_pos = torch.randn(batch_size, num_agents, 2)
    all_dir = torch.zeros(batch_size, num_agents, 4)
    all_dir[:, :, 0] = 1
    all_abil = torch.zeros(batch_size, num_agents, 2)
    all_carr = torch.zeros(batch_size, num_agents, 2)
    all_stat = torch.zeros(batch_size, num_agents, 3)
    all_stat[:, :, 2] = -1
    # Agent colors: 2 yellow, 1 grey
    colors = torch.tensor([[COLOR_TO_IDX['yellow'], COLOR_TO_IDX['yellow'], COLOR_TO_IDX['grey']]] * batch_size)
    
    features = encoder(
        query_pos, query_dir, query_abil, query_carr, query_stat,
        all_pos, all_dir, all_abil, all_carr, all_stat, colors
    )
    
    assert features.shape == (batch_size, 64), f"Expected (4, 64), got {features.shape}"
    print(f"  ✓ Input dim: {encoder.input_dim}")
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ MultiGridAgentEncoder test passed!")


def test_multigrid_interactive_encoder():
    """Test the MultiGridInteractiveObjectEncoder from the new subpackage."""
    from empo.nn_based.multigrid import (
        MultiGridInteractiveObjectEncoder,
        KILLBUTTON_FEATURE_SIZE,
        PAUSESWITCH_FEATURE_SIZE,
        DISABLINGSWITCH_FEATURE_SIZE,
        CONTROLBUTTON_FEATURE_SIZE,
    )
    print("Testing MultiGridInteractiveObjectEncoder...")
    
    encoder = MultiGridInteractiveObjectEncoder(
        max_kill_buttons=4,
        max_pause_switches=4,
        max_disabling_switches=4,
        max_control_buttons=4,
        feature_dim=64
    )
    
    # Expected input dim: 4*5 + 4*6 + 4*6 + 4*7 = 20 + 24 + 24 + 28 = 96
    expected_input_dim = (
        4 * KILLBUTTON_FEATURE_SIZE +
        4 * PAUSESWITCH_FEATURE_SIZE +
        4 * DISABLINGSWITCH_FEATURE_SIZE +
        4 * CONTROLBUTTON_FEATURE_SIZE
    )
    assert encoder.input_dim == expected_input_dim, f"Expected {expected_input_dim}, got {encoder.input_dim}"
    
    # Verify ControlButton has 7 features (including _awaiting_action)
    assert CONTROLBUTTON_FEATURE_SIZE == 7, f"Expected 7, got {CONTROLBUTTON_FEATURE_SIZE}"
    
    batch_size = 4
    kb = torch.randn(batch_size, 4, KILLBUTTON_FEATURE_SIZE)
    ps = torch.randn(batch_size, 4, PAUSESWITCH_FEATURE_SIZE)
    ds = torch.randn(batch_size, 4, DISABLINGSWITCH_FEATURE_SIZE)
    cb = torch.randn(batch_size, 4, CONTROLBUTTON_FEATURE_SIZE)
    
    features = encoder(kb, ps, ds, cb)
    assert features.shape == (batch_size, 64), f"Expected (4, 64), got {features.shape}"
    
    print(f"  ✓ Input dim: {encoder.input_dim}")
    print(f"  ✓ ControlButton features: {CONTROLBUTTON_FEATURE_SIZE} (includes _awaiting_action)")
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ MultiGridInteractiveObjectEncoder test passed!")


def test_multigrid_q_network():
    """Test the full MultiGridQNetwork from the new subpackage."""
    from empo.nn_based.multigrid import (
        MultiGridStateEncoder,
        MultiGridAgentEncoder,
        MultiGridGoalEncoder,
        MultiGridInteractiveObjectEncoder,
        MultiGridQNetwork,
        COLOR_TO_IDX,
    )
    print("Testing MultiGridQNetwork...")
    
    # Create encoders
    state_enc = MultiGridStateEncoder(10, 10, feature_dim=128)
    agent_enc = MultiGridAgentEncoder({'yellow': 2, 'grey': 1}, feature_dim=64)
    goal_enc = MultiGridGoalEncoder(feature_dim=32)
    interactive_enc = MultiGridInteractiveObjectEncoder(feature_dim=64)
    
    q_network = MultiGridQNetwork(
        state_encoder=state_enc,
        agent_encoder=agent_enc,
        goal_encoder=goal_enc,
        interactive_encoder=interactive_enc,
        num_actions=8,
        hidden_dim=256
    )
    
    batch_size = 4
    
    # State inputs
    grid_tensor = torch.randn(batch_size, state_enc.num_channels, 10, 10)
    global_features = torch.randn(batch_size, 4)
    
    # Agent inputs
    query_pos = torch.randn(batch_size, 2)
    query_dir = torch.zeros(batch_size, 4)
    query_dir[:, 0] = 1
    query_abil = torch.zeros(batch_size, 2)
    query_carr = torch.zeros(batch_size, 2)
    query_stat = torch.zeros(batch_size, 3)
    query_stat[:, 2] = -1
    
    num_agents = 3
    all_pos = torch.randn(batch_size, num_agents, 2)
    all_dir = torch.zeros(batch_size, num_agents, 4)
    all_dir[:, :, 0] = 1
    all_abil = torch.zeros(batch_size, num_agents, 2)
    all_carr = torch.zeros(batch_size, num_agents, 2)
    all_stat = torch.zeros(batch_size, num_agents, 3)
    all_stat[:, :, 2] = -1
    colors = torch.tensor([[COLOR_TO_IDX['yellow'], COLOR_TO_IDX['yellow'], COLOR_TO_IDX['grey']]] * batch_size)
    
    # Goal inputs
    goal_coords = torch.randn(batch_size, 4)
    
    # Interactive object inputs
    kb = torch.randn(batch_size, 4, 5)
    ps = torch.randn(batch_size, 4, 6)
    ds = torch.randn(batch_size, 4, 6)
    cb = torch.randn(batch_size, 4, 7)
    
    q_values = q_network(
        grid_tensor, global_features,
        query_pos, query_dir, query_abil, query_carr, query_stat,
        all_pos, all_dir, all_abil, all_carr, all_stat, colors,
        goal_coords,
        kb, ps, ds, cb
    )
    
    assert q_values.shape == (batch_size, 8), f"Expected (4, 8), got {q_values.shape}"
    
    # Test policy computation
    policy = q_network.get_policy(q_values, beta=1.0)
    assert policy.shape == (batch_size, 8)
    assert torch.allclose(policy.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    print(f"  ✓ Q-values shape: {q_values.shape}")
    print(f"  ✓ Policy sums to 1.0: {policy.sum(dim=1).mean():.4f}")
    print("  ✓ MultiGridQNetwork test passed!")


def test_multigrid_feature_extraction():
    """Test feature extraction functions."""
    from empo.nn_based.multigrid.feature_extraction import (
        extract_agent_features,
        extract_interactive_objects,
        extract_global_world_features,
    )
    from empo.nn_based.multigrid.constants import CONTROLBUTTON_FEATURE_SIZE
    print("Testing multigrid feature extraction...")
    
    # Create a mock state tuple
    # Format: (step_count, agent_states, mobile_objects, mutable_objects)
    # Agent state: (pos_x, pos_y, dir, terminated, started, paused, carrying_type, carrying_color, forced_next_action)
    agent_states = [
        (3, 4, 0, False, True, False, 'key', 'red', None),
        (5, 6, 1, True, True, True, None, None, 2),
    ]
    mutable_objects = [
        ('door', 1, 1, True, False),  # open door
        ('controlbutton', 2, 2, True, 0, 3),  # enabled, controlled_agent=0, triggered_action=3
    ]
    state = (10, tuple(agent_states), (), tuple(mutable_objects))
    
    # Mock world model
    class MockAgent:
        def __init__(self, color, can_magic=False, can_push=False):
            self.color = color
            self.can_enter_magic_walls = can_magic
            self.can_push_rocks = can_push
    
    class MockWorldModel:
        width = 10
        height = 10
        max_steps = 100
        stumble_prob = 0.1
        magic_entry_prob = 0.5
        magic_solidify_prob = 0.3
        agents = [MockAgent('yellow', True, False), MockAgent('grey', False, True)]
        grid = None
    
    world_model = MockWorldModel()
    
    # Test agent feature extraction
    positions, directions, abilities, carried, status, colors = extract_agent_features(
        state, world_model, device='cpu'
    )
    
    assert positions.shape == (2, 2), f"Expected (2, 2), got {positions.shape}"
    assert directions.shape == (2, 4), f"Expected (2, 4), got {directions.shape}"
    assert abilities.shape == (2, 2), f"Expected (2, 2), got {abilities.shape}"
    assert carried.shape == (2, 2), f"Expected (2, 2), got {carried.shape}"
    assert status.shape == (2, 3), f"Expected (2, 3), got {status.shape}"
    
    # Verify agent 0 values
    assert positions[0, 0] == 3.0, "Agent 0 x should be 3"
    assert positions[0, 1] == 4.0, "Agent 0 y should be 4"
    assert directions[0, 0] == 1.0, "Agent 0 should face right"
    assert abilities[0, 0] == 1.0, "Agent 0 should have can_enter_magic_walls"
    assert abilities[0, 1] == 0.0, "Agent 0 should not have can_push_rocks"
    assert status[0, 0] == 0.0, "Agent 0 should not be paused"
    assert status[0, 1] == 0.0, "Agent 0 should not be terminated"
    
    # Verify agent 1 values
    assert positions[1, 0] == 5.0, "Agent 1 x should be 5"
    assert positions[1, 1] == 6.0, "Agent 1 y should be 6"
    assert status[1, 0] == 1.0, "Agent 1 should be paused"
    assert status[1, 1] == 1.0, "Agent 1 should be terminated"
    assert status[1, 2] == 2.0, "Agent 1 should have forced_next_action=2"
    
    print("  ✓ Agent features extracted correctly")
    
    # Test global feature extraction
    global_features = extract_global_world_features(state, world_model, device='cpu')
    assert global_features.shape == (4,)
    assert global_features[0] == 0.1, "stumble_prob should be 0.1"
    assert global_features[1] == 0.5, "magic_entry_prob should be 0.5"
    assert global_features[2] == 0.3, "magic_solidify_prob should be 0.3"
    assert global_features[3] == 90.0, "remaining_time should be 90 (100-10)"
    
    print("  ✓ Global features extracted correctly")
    print("  ✓ multigrid feature extraction test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Neural Policy Prior Tests")
    print("=" * 60)
    print()
    
    test_state_encoder()
    print()
    
    test_agent_encoder()
    print()
    
    test_goal_encoder()
    print()
    
    test_q_network()
    print()
    
    test_policy_prior_network()
    print()
    
    test_create_networks()
    print()
    
    test_save_load()
    print()
    
    test_load_conflicting_dimensions()
    print()
    
    test_load_conflicting_action_space()
    print()
    
    # New multigrid subpackage tests
    print("=" * 60)
    print("nn_based.multigrid Subpackage Tests")
    print("=" * 60)
    print()
    
    test_multigrid_state_encoder()
    print()
    
    test_multigrid_agent_encoder()
    print()
    
    test_multigrid_interactive_encoder()
    print()
    
    test_multigrid_q_network()
    print()
    
    test_multigrid_feature_extraction()
    print()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
