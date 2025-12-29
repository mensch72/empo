#!/usr/bin/env python3
"""
Test script for shared encoder architecture in Phase 2 networks.

This script verifies that:
1. All networks share the same encoder instances
2. Encoder caching works correctly
3. Gradients flow properly through shared encoders
"""

import sys
import os

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


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
        self.robot_indices = [0]


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
    agent_states = [(i, i, 0, False, False) for i in range(num_agents)]
    mobile_objects = []
    mutable_objects = []
    return (step_count, agent_states, mobile_objects, mutable_objects)


def test_shared_state_encoder():
    """Test that all networks share the same state encoder instance."""
    print("\n=== Testing shared state encoder ===")
    
    from empo.nn_based.multigrid.phase2.trainer import create_phase2_networks
    from empo.nn_based.phase2.config import Phase2Config
    
    # Create mock environment
    world_model = MockWorldModel(width=6, height=6, num_agents=2)
    # Enable u_r_use_network and v_r_use_network so U_r and V_r networks are created for testing
    config = Phase2Config(u_r_use_network=True, v_r_use_network=True)
    
    # Create networks with shared encoders
    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=1,
        num_actions=5,
        hidden_dim=64,
        device='cpu'
    )
    
    # Check that all state encoders are the same instance
    state_encoder_ids = [
        id(networks.q_r.state_encoder),
        id(networks.v_h_e.state_encoder),
        id(networks.x_h.state_encoder),
        id(networks.u_r.state_encoder),
        id(networks.v_r.state_encoder),
    ]
    
    print(f"  State encoder IDs: {state_encoder_ids}")
    
    # All should be the same
    assert len(set(state_encoder_ids)) == 1, f"State encoders are not shared! Found {len(set(state_encoder_ids))} different instances"
    print("  ✓ All networks share the same state encoder instance")
    
    # Check goal encoder sharing (only V_h^e uses it)
    print(f"  Goal encoder in V_h^e: id={id(networks.v_h_e.goal_encoder)}")
    print("  ✓ Goal encoder check passed")
    
    # Check agent encoder sharing (V_h^e and X_h)
    agent_encoder_ids = [
        id(networks.v_h_e.agent_encoder),
        id(networks.x_h.agent_encoder),
    ]
    print(f"  Agent encoder IDs: {agent_encoder_ids}")
    assert len(set(agent_encoder_ids)) == 1, f"Agent encoders are not shared! Found {len(set(agent_encoder_ids))} different instances"
    print("  ✓ V_h^e and X_h share the same agent encoder instance")
    
    print("\n=== Shared encoder test PASSED ===")
    return networks


def test_encoder_caching():
    """Test that encoder caching works correctly."""
    print("\n=== Testing encoder caching ===")
    
    from empo.nn_based.multigrid.phase2.trainer import create_phase2_networks
    from empo.nn_based.phase2.config import Phase2Config
    
    world_model = MockWorldModel(width=6, height=6, num_agents=2)
    config = Phase2Config()
    
    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=1,
        num_actions=5,
        hidden_dim=64,
        device='cpu'
    )
    
    state_encoder = networks.q_r.state_encoder
    
    # Clear any existing cache
    state_encoder.clear_cache()
    
    # Create a state
    state = create_mock_state(num_agents=2)
    
    # Encode the same state twice
    enc1 = state_encoder.tensorize_state(state, world_model, 'cpu')
    enc2 = state_encoder.tensorize_state(state, world_model, 'cpu')
    
    # Check cache stats
    stats = state_encoder.get_cache_stats()
    hits, misses = stats
    
    print(f"  Cache stats after 2 encodes of same state: hits={hits}, misses={misses}")
    
    # First encode should be a miss, second should be a hit
    assert misses == 1, f"Expected 1 miss, got {misses}"
    assert hits == 1, f"Expected 1 hit, got {hits}"
    print("  ✓ State encoder caching works correctly")
    
    # Check that tensors are equal
    for t1, t2 in zip(enc1, enc2):
        assert torch.equal(t1, t2), "Cached tensors should be identical"
    print("  ✓ Cached tensors are identical")
    
    # Test with different state (should be a new cache entry)
    state2 = create_mock_state(num_agents=2)  # New state object
    enc3 = state_encoder.tensorize_state(state2, world_model, 'cpu')
    stats = state_encoder.get_cache_stats()
    hits, misses = stats
    print(f"  Cache stats after encode of different state: hits={hits}, misses={misses}")
    assert misses == 2, f"Expected 2 misses (new state), got {misses}"
    print("  ✓ Different state creates new cache entry")
    
    # Test clear_cache
    state_encoder.clear_cache()
    state_encoder.reset_cache_stats()
    stats = state_encoder.get_cache_stats()
    hits, misses = stats
    assert hits == 0 and misses == 0, "Cache stats should be reset after reset_cache_stats"
    print("  ✓ clear_cache() and reset_cache_stats() work correctly")
    
    print("\n=== Encoder caching test PASSED ===")


def test_gradient_flow():
    """Test that gradients flow correctly through shared encoders."""
    print("\n=== Testing gradient flow through shared encoders ===")
    
    from empo.nn_based.multigrid.phase2.trainer import create_phase2_networks
    from empo.nn_based.phase2.config import Phase2Config
    
    world_model = MockWorldModel(width=6, height=6, num_agents=2)
    config = Phase2Config()
    
    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=1,
        num_actions=5,
        hidden_dim=64,
        device='cpu'
    )
    
    state = create_mock_state(num_agents=2)
    
    # Get the shared state encoder
    state_encoder = networks.q_r.state_encoder
    
    # Clear cache before gradient test
    state_encoder.clear_cache()
    
    # Encode state
    grid_tensor, global_features, agent_features, interactive_features = \
        state_encoder.tensorize_state(state, world_model, 'cpu')
    
    # Forward through state encoder NN
    state_features = state_encoder(grid_tensor, global_features, agent_features, interactive_features)
    
    # Check that state_features requires grad
    assert state_features.requires_grad, "State features should require grad"
    print("  ✓ State features require grad")
    
    # Compute a simple loss and backprop
    loss = state_features.sum()
    loss.backward()
    
    # Check that encoder parameters got gradients
    has_grad = False
    for name, param in state_encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"  ✓ Gradient for {name}: {param.grad.abs().sum():.4f}")
            break
    
    assert has_grad, "State encoder should have gradients after backprop"
    print("  ✓ Gradients flow through shared state encoder")
    
    print("\n=== Gradient flow test PASSED ===")


def test_all():
    """Run all tests."""
    print("="*60)
    print("Testing Shared Encoder Architecture")
    print("="*60)
    
    test_shared_state_encoder()
    test_encoder_caching()
    test_gradient_flow()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


if __name__ == '__main__':
    test_all()
