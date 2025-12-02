#!/usr/bin/env python3
"""
Test script for the neural network-based policy prior.

This script tests the basic functionality of the NeuralHumanPolicyPrior
and its components (StateEncoder, AgentEncoder, GoalEncoder, QNetwork).
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

from empo.nn_based import (
    StateEncoder,
    AgentEncoder,
    GoalEncoder,
    QNetwork,
    PolicyPriorNetwork,
    NeuralHumanPolicyPrior,
    create_policy_prior_networks,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler, PossibleGoalGenerator
from typing import Iterator, Tuple


class SimpleReachGoal(PossibleGoal):
    """A simple goal to reach a target position."""
    
    def __init__(self, world_model, target_pos):
        super().__init__(world_model)
        self.target_pos = target_pos
    
    def is_achieved(self, state) -> int:
        # Simplified check - actual implementation depends on state format
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


class SimpleGoalGenerator(PossibleGoalGenerator):
    """A simple goal generator that yields all reachable cells."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
        self.goals = []
        for x in range(world_model.width):
            for y in range(world_model.height):
                self.goals.append((x, y))
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        weight = 1.0 / len(self.goals)
        for pos in self.goals:
            yield SimpleReachGoal(self.world_model, pos), weight


def test_state_encoder():
    """Test the StateEncoder network."""
    print("Testing StateEncoder...")
    
    num_object_types = 8
    num_agents = 2
    # New channel structure:
    # - num_object_types: explicit object type channels
    # - 3: "other" object channels (overlappable, immobile, mobile)
    # - num_agents: per-agent position channels
    # - 1: query agent channel
    # - 1: "other humans" channel
    num_channels = num_object_types + 3 + num_agents + 1 + 1  # = 15
    
    encoder = StateEncoder(
        grid_width=10,
        grid_height=10,
        num_object_types=num_object_types,
        num_agents=num_agents,
        feature_dim=128
    )
    
    # Create dummy input with correct number of channels
    batch_size = 4
    state_tensor = torch.randn(batch_size, num_channels, 10, 10)
    step_count = torch.randn(batch_size, 1)
    
    # Forward pass
    features = encoder(state_tensor, step_count)
    
    assert features.shape == (batch_size, 128), f"Expected (4, 128), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ StateEncoder test passed!")


def test_agent_encoder():
    """Test the AgentEncoder network."""
    print("Testing AgentEncoder...")
    
    encoder = AgentEncoder(
        grid_width=10,
        grid_height=10,
        num_agents=2,
        feature_dim=32
    )
    
    # Create dummy input
    batch_size = 4
    position = torch.randn(batch_size, 2)
    direction = torch.zeros(batch_size, 4)
    direction[:, 0] = 1  # All facing right
    agent_idx = torch.zeros(batch_size, dtype=torch.long)
    
    # Forward pass
    features = encoder(position, direction, agent_idx)
    
    assert features.shape == (batch_size, 32), f"Expected (4, 32), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ AgentEncoder test passed!")


def test_goal_encoder():
    """Test the GoalEncoder network."""
    print("Testing GoalEncoder...")
    
    encoder = GoalEncoder(
        grid_width=10,
        grid_height=10,
        feature_dim=32
    )
    
    # Create dummy input
    batch_size = 4
    goal_coords = torch.randn(batch_size, 4)  # x1, y1, x2, y2
    
    # Forward pass
    features = encoder(goal_coords)
    
    assert features.shape == (batch_size, 32), f"Expected (4, 32), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print("  ✓ GoalEncoder test passed!")


def test_q_network():
    """Test the QNetwork."""
    print("Testing QNetwork...")
    
    num_object_types = 8
    num_agents = 2
    # New channel structure
    num_channels = num_object_types + 3 + num_agents + 1 + 1  # = 15
    
    # Create encoders
    state_encoder = StateEncoder(grid_width=10, grid_height=10, num_object_types=num_object_types, num_agents=num_agents)
    agent_encoder = AgentEncoder(grid_width=10, grid_height=10, num_agents=num_agents)
    goal_encoder = GoalEncoder(grid_width=10, grid_height=10)
    
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=4
    )
    
    # Create dummy input with correct number of channels
    batch_size = 4
    state_tensor = torch.randn(batch_size, num_channels, 10, 10)
    step_count = torch.randn(batch_size, 1)
    position = torch.randn(batch_size, 2)
    direction = torch.zeros(batch_size, 4)
    direction[:, 0] = 1
    agent_idx = torch.zeros(batch_size, dtype=torch.long)
    goal_coords = torch.randn(batch_size, 4)
    
    # Forward pass
    q_values = q_network(
        state_tensor, step_count,
        position, direction, agent_idx,
        goal_coords
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
    
    num_object_types = 8
    num_agents = 2
    # New channel structure
    num_channels = num_object_types + 3 + num_agents + 1 + 1  # = 15
    
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
    step_count = torch.randn(batch_size, 1)
    position = torch.randn(batch_size, 2)
    direction = torch.zeros(batch_size, 4)
    direction[:, 0] = 1
    agent_idx = torch.zeros(batch_size, dtype=torch.long)
    
    # Forward pass
    policy = phi_network(
        state_tensor, step_count,
        position, direction, agent_idx
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
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
