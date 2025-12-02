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
    # Now includes +1 for "other humans" channel
    num_channels = num_object_types + num_agents + 1  # = 11
    
    encoder = StateEncoder(
        grid_width=10,
        grid_height=10,
        num_object_types=num_object_types,
        num_agents=num_agents,
        feature_dim=128
    )
    
    # Create dummy input with correct number of channels
    batch_size = 4
    state_tensor = torch.randn(batch_size, num_channels, 10, 10)  # 11 channels now
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
    # Now includes +1 for "other humans" channel
    num_channels = num_object_types + num_agents + 1  # = 11
    
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
    # Now includes +1 for "other humans" channel
    num_channels = num_object_types + num_agents + 1  # = 11
    
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
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
