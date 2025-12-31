#!/usr/bin/env python3
"""
Test script for Phase 2 robot policy learning neural networks.

Tests the modular architecture with:
- Base classes in nn_based/phase2/
- Multigrid-specific implementations in nn_based/multigrid/phase2/
"""

import sys
import os

import numpy as np
import torch
import tempfile

from empo.nn_based.phase2 import (
    Phase2Config,
    BaseRobotQNetwork,
    BaseHumanGoalAchievementNetwork,
    BaseAggregateGoalAbilityNetwork,
    BaseIntrinsicRewardNetwork,
    BaseRobotValueNetwork,
    Phase2ReplayBuffer,
    Phase2Transition,
)
from empo.nn_based.multigrid.phase2 import (
    MultiGridRobotQNetwork,
    MultiGridHumanGoalAchievementNetwork,
    MultiGridAggregateGoalAbilityNetwork,
    MultiGridIntrinsicRewardNetwork,
    MultiGridRobotValueNetwork,
)
from empo.nn_based.soft_clamp import SoftClamp


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


class MockGoal:
    """Mock goal for testing."""
    
    def __init__(self, target_pos):
        self.target_pos = target_pos
        self.target_rect = (target_pos[0], target_pos[1], target_pos[0], target_pos[1])


def create_mock_state(num_agents=2):
    """Create a mock state tuple."""
    step_count = 0
    agent_states = [(i, i, 0, False, False) for i in range(num_agents)]
    mobile_objects = []
    mutable_objects = []
    return (step_count, agent_states, mobile_objects, mutable_objects)


def test_phase2_config():
    """Test the Phase2Config dataclass."""
    print("Testing Phase2Config...")
    
    # Default config
    config = Phase2Config()
    assert config.gamma_r == 0.99
    assert config.gamma_h == 0.99
    assert config.zeta == 2.0
    assert config.xi == 1.0
    assert config.eta == 1.1
    assert config.beta_r == 10.0
    print("  ✓ Default config values")
    
    # Epsilon decay
    assert config.get_epsilon(0) == config.epsilon_r_start
    assert config.get_epsilon(config.epsilon_r_decay_steps) == config.epsilon_r_end
    assert config.get_epsilon(config.epsilon_r_decay_steps // 2) < config.epsilon_r_start
    print("  ✓ Epsilon decay works")
    
    # Custom config
    custom_config = Phase2Config(
        gamma_r=0.95,
        zeta=3.0,
        xi=2.0,
        eta=1.5,
        beta_r=5.0
    )
    assert custom_config.gamma_r == 0.95
    assert custom_config.zeta == 3.0
    print("  ✓ Custom config values")
    
    print("  ✓ Phase2Config test passed!")


def test_phase2_replay_buffer():
    """Test the Phase2ReplayBuffer."""
    print("Testing Phase2ReplayBuffer...")
    
    buffer = Phase2ReplayBuffer(capacity=100)
    
    state = create_mock_state()
    robot_action = (0, 1)  # Two robots
    goals = {0: MockGoal((5, 5)), 1: MockGoal((3, 3))}  # Two humans with goals
    goal_weights = {0: 1.0, 1: 1.0}  # Equal weights for both humans
    human_actions = [2, 3]
    next_state = create_mock_state()
    
    # Add transitions
    for i in range(50):
        buffer.push(state, robot_action, goals, goal_weights, human_actions, next_state)
    
    assert len(buffer) == 50
    print(f"  ✓ Buffer length: {len(buffer)}")
    
    # Sample batch
    batch = buffer.sample(16)
    assert len(batch) == 16
    assert all(isinstance(t, Phase2Transition) for t in batch)
    print(f"  ✓ Batch size: {len(batch)}")
    
    # Test transition fields
    t = batch[0]
    assert t.robot_action == robot_action
    assert t.human_actions == human_actions
    print("  ✓ Transition fields correct")
    
    # Test capacity overflow
    buffer2 = Phase2ReplayBuffer(capacity=10)
    for i in range(20):
        buffer2.push(state, (i % 4, i % 4), goals, goal_weights, human_actions, next_state)
    assert len(buffer2) == 10
    print("  ✓ Capacity overflow handled correctly")
    
    # Test clear
    buffer.clear()
    assert len(buffer) == 0
    print("  ✓ Clear works")
    
    print("  ✓ Phase2ReplayBuffer test passed!")


def test_multigrid_robot_q_network():
    """Test the MultiGridRobotQNetwork."""
    print("Testing MultiGridRobotQNetwork...")
    
    q_network = MultiGridRobotQNetwork(
        grid_height=10,
        grid_width=10,
        num_robot_actions=4,
        num_robots=2,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        beta_r=10.0
    )
    
    # Test action space size
    assert q_network.num_action_combinations == 4 ** 2  # 16 joint actions
    print(f"  ✓ Num action combinations: {q_network.num_action_combinations}")
    
    # Test forward pass
    world_model = MockWorldModel()
    state = create_mock_state()
    q_values = q_network.forward(state, world_model, device='cpu')
    
    assert q_values.shape == (1, 16)
    assert (q_values < 0).all(), "Q_r must be negative"
    print(f"  ✓ Q-values shape: {q_values.shape}")
    print(f"  ✓ Q-values are all negative: {q_values.max().item():.4f} < 0")
    
    # Test policy computation (power-law softmax)
    policy = q_network.get_policy(q_values)
    assert policy.shape == (1, 16)
    assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
    assert (policy > 0).all(), "Policy probabilities must be positive"
    print(f"  ✓ Policy shape: {policy.shape}")
    print(f"  ✓ Policy sums to 1: {policy.sum().item():.4f}")
    
    # Test action index conversion
    idx = q_network.action_tuple_to_index((2, 3))
    assert idx == 2 + 3 * 4  # Mixed-radix representation
    actions = q_network.action_index_to_tuple(idx)
    assert actions == (2, 3)
    print("  ✓ Action index conversion works")
    
    # Test action sampling
    action = q_network.sample_action(q_values)
    assert len(action) == 2  # Two robots
    assert all(0 <= a < 4 for a in action)
    print(f"  ✓ Sampled action: {action}")
    
    # Test sampling with different beta_r (deterministic with high beta_r)
    action_det = q_network.sample_action(q_values, beta_r=1000.0)
    assert len(action_det) == 2
    print(f"  ✓ High beta_r action: {action_det}")
    
    # Test get_value (expected Q under policy)
    value = q_network.get_value(q_values)
    assert value.shape == (1,)
    assert value.item() < 0, "Expected value must be negative"
    print(f"  ✓ Expected value: {value.item():.4f}")
    
    print("  ✓ MultiGridRobotQNetwork test passed!")


def test_multigrid_human_goal_achievement_network():
    """Test the MultiGridHumanGoalAchievementNetwork."""
    print("Testing MultiGridHumanGoalAchievementNetwork...")
    
    v_h_e_network = MultiGridHumanGoalAchievementNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        gamma_h=0.99
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = MockGoal((5, 5))
    
    # Test forward pass
    v_h_e = v_h_e_network.forward(
        state, world_model, human_agent_idx=0, goal=goal, device='cpu'
    )
    
    assert v_h_e.shape == (1,)
    # After soft clamp, values should be in reasonable range
    print(f"  ✓ V_h^e shape: {v_h_e.shape}")
    print(f"  ✓ V_h^e value: {v_h_e.item():.4f}")
    
    # Test TD target computation (truncated Bellman)
    # Case 1: Goal achieved (reward=1, no continuation)
    goal_achieved = torch.tensor([1.0])
    next_v = torch.tensor([0.5])
    target = v_h_e_network.compute_td_target(goal_achieved, next_v)
    assert target.item() == 1.0, "When goal achieved, target should be 1"
    print(f"  ✓ TD target when goal achieved: {target.item()}")
    
    # Case 2: Goal not achieved (reward=0, with continuation)
    goal_not_achieved = torch.tensor([0.0])
    target2 = v_h_e_network.compute_td_target(goal_not_achieved, next_v)
    expected = 0.99 * 0.5
    assert torch.isclose(target2, torch.tensor(expected), atol=1e-5)
    print(f"  ✓ TD target when goal not achieved: {target2.item():.4f}")
    
    # Case 3: Goal not achieved AND terminal (episode ended without goal)
    terminal_flag = torch.tensor([1.0])
    target3 = v_h_e_network.compute_td_target(goal_not_achieved, next_v, terminal=terminal_flag)
    assert target3.item() == 0.0, "When terminal and goal not achieved, target should be 0"
    print(f"  ✓ TD target when terminal: {target3.item()}")
    
    # Case 4: Goal achieved AND terminal (terminal flag should be ignored)
    target4 = v_h_e_network.compute_td_target(goal_achieved, next_v, terminal=terminal_flag)
    assert target4.item() == 1.0, "When goal achieved, target should be 1 regardless of terminal"
    print(f"  ✓ TD target when goal achieved + terminal: {target4.item()}")
    
    # Case 5: Batched computation with mixed terminal flags
    goal_batch = torch.tensor([0.0, 0.0, 1.0, 0.0])
    next_v_batch = torch.tensor([0.5, 0.5, 0.5, 0.5])
    terminal_batch = torch.tensor([0.0, 1.0, 1.0, 0.0])
    target5 = v_h_e_network.compute_td_target(goal_batch, next_v_batch, terminal=terminal_batch)
    expected_batch = torch.tensor([0.99 * 0.5, 0.0, 1.0, 0.99 * 0.5])
    assert torch.allclose(target5, expected_batch, atol=1e-5)
    print(f"  ✓ Batched TD target with mixed terminal flags: {target5.tolist()}")
    
    # Test soft clamp preserves gradients
    raw = torch.tensor([0.5], requires_grad=True)
    clamped = v_h_e_network.apply_clamp(raw)
    clamped.backward()
    assert raw.grad is not None
    print("  ✓ Soft clamp preserves gradients")
    
    # Test hard clamp for inference
    hard_clamped = v_h_e_network.apply_hard_clamp(torch.tensor([1.5]))
    assert hard_clamped.item() == 1.0
    print("  ✓ Hard clamp bounds to [0, 1]")
    
    print("  ✓ MultiGridHumanGoalAchievementNetwork test passed!")


def test_multigrid_aggregate_goal_ability_network():
    """Test the MultiGridAggregateGoalAbilityNetwork."""
    print("Testing MultiGridAggregateGoalAbilityNetwork...")
    
    x_h_network = MultiGridAggregateGoalAbilityNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        zeta=2.0
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    
    # Test forward pass
    x_h = x_h_network.forward(
        state, world_model, human_agent_idx=0, device='cpu'
    )
    
    assert x_h.shape == (1,)
    print(f"  ✓ X_h shape: {x_h.shape}")
    print(f"  ✓ X_h value: {x_h.item():.4f}")
    
    # Test target computation from V_h^e
    v_h_e = torch.tensor([0.8])
    target = x_h_network.compute_target(v_h_e)
    expected = 0.8 ** 2.0  # V_h^e^zeta
    assert torch.isclose(target, torch.tensor(expected), atol=1e-5)
    print(f"  ✓ Target X_h from V_h^e=0.8: {target.item():.4f}")
    
    # Test compute from multiple V_h^e samples
    v_h_e_samples = torch.tensor([[0.5, 0.6, 0.7, 0.8]])  # batch=1, 4 goals
    x_h_computed = x_h_network.compute_from_v_h_e_samples(v_h_e_samples)
    # Expected: mean(V^zeta) = mean([0.5^2, 0.6^2, 0.7^2, 0.8^2])
    expected_vals = torch.tensor([0.5, 0.6, 0.7, 0.8]) ** 2.0
    expected_mean = expected_vals.mean()
    assert torch.isclose(x_h_computed.squeeze(), expected_mean, atol=1e-5)
    print(f"  ✓ X_h from multiple V_h^e samples: {x_h_computed.item():.4f}")
    
    # Test zeta validation
    try:
        bad_network = MultiGridAggregateGoalAbilityNetwork(
            grid_height=10,
            grid_width=10,
            num_agents_per_color={'grey': 2},
            num_agent_colors=7,
            zeta=0.5  # Invalid: must be >= 1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejects zeta < 1: {e}")
    
    print("  ✓ MultiGridAggregateGoalAbilityNetwork test passed!")


def test_multigrid_intrinsic_reward_network():
    """Test the MultiGridIntrinsicRewardNetwork."""
    print("Testing MultiGridIntrinsicRewardNetwork...")
    
    u_r_network = MultiGridIntrinsicRewardNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        xi=1.0,
        eta=1.1
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    
    # Test forward pass
    y, u_r = u_r_network.forward(
        state, world_model, device='cpu'
    )
    
    assert y.shape == (1,)
    assert u_r.shape == (1,)
    assert y.item() > 1, "y must be > 1"
    assert u_r.item() < 0, "U_r must be < 0"
    print(f"  ✓ y shape: {y.shape}, value: {y.item():.4f}")
    print(f"  ✓ U_r shape: {u_r.shape}, value: {u_r.item():.4f}")
    
    # Test y to U_r conversion
    test_y = torch.tensor([2.0])
    test_u_r = u_r_network.y_to_u_r(test_y)
    expected_u_r = -(2.0 ** 1.1)
    assert torch.isclose(test_u_r, torch.tensor(expected_u_r), atol=1e-5)
    print(f"  ✓ y_to_u_r(2.0) = {test_u_r.item():.4f}")
    
    # Test log(y-1) to y conversion
    log_y_minus_1 = torch.tensor([0.0])  # y = 1 + exp(0) = 2
    y_converted = u_r_network.log_y_minus_1_to_y(log_y_minus_1)
    assert torch.isclose(y_converted, torch.tensor(2.0), atol=1e-5)
    print(f"  ✓ log_y_minus_1_to_y(0) = {y_converted.item():.4f}")
    
    # Test target y from X_h
    x_h = torch.tensor([0.5])
    target_y = u_r_network.compute_target_y(x_h)
    expected_target = 0.5 ** (-1.0)  # X_h^{-xi} = 2.0
    assert torch.isclose(target_y, torch.tensor(expected_target), atol=1e-5)
    print(f"  ✓ Target y from X_h=0.5: {target_y.item():.4f}")
    
    # Test compute from X_h values for all humans
    x_h_values = torch.tensor([[0.5, 0.4]])  # batch=1, 2 humans
    y_computed, u_r_computed = u_r_network.compute_from_x_h(x_h_values)
    # y = mean([0.5^{-1}, 0.4^{-1}]) = mean([2, 2.5]) = 2.25
    # U_r = -2.25^1.1
    expected_y = (2.0 + 2.5) / 2
    assert torch.isclose(y_computed.squeeze(), torch.tensor(expected_y), atol=1e-5)
    print(f"  ✓ y from X_h values: {y_computed.item():.4f}")
    print(f"  ✓ U_r from X_h values: {u_r_computed.item():.4f}")
    
    # Test parameter validation
    try:
        bad_network = MultiGridIntrinsicRewardNetwork(
            grid_height=10,
            grid_width=10,
            num_agents_per_color={'grey': 2},
            num_agent_colors=7,
            xi=0.5  # Invalid: must be >= 1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejects xi < 1: {e}")
    
    try:
        bad_network = MultiGridIntrinsicRewardNetwork(
            grid_height=10,
            grid_width=10,
            num_agents_per_color={'grey': 2},
            num_agent_colors=7,
            eta=0.5  # Invalid: must be >= 1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly rejects eta < 1: {e}")
    
    print("  ✓ MultiGridIntrinsicRewardNetwork test passed!")


def test_multigrid_robot_value_network():
    """Test the MultiGridRobotValueNetwork."""
    print("Testing MultiGridRobotValueNetwork...")
    
    v_r_network = MultiGridRobotValueNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        gamma_r=0.99
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    
    # Test forward pass
    v_r = v_r_network.forward(
        state, world_model, device='cpu'
    )
    
    assert v_r.shape == (1,)
    assert v_r.item() < 0, "V_r must be < 0"
    print(f"  ✓ V_r shape: {v_r.shape}")
    print(f"  ✓ V_r value: {v_r.item():.4f} < 0")
    
    # Test ensure_negative
    raw_values = torch.tensor([1.0, 0.0, -1.0, 5.0, -5.0])
    negative_values = v_r_network.ensure_negative(raw_values)
    assert (negative_values < 0).all(), "ensure_negative should make all values negative"
    print(f"  ✓ ensure_negative: all values negative")
    
    # Test compute from components
    u_r = torch.tensor([-1.5])
    q_r = torch.tensor([[-0.5, -0.7, -0.3, -0.9]])  # 4 actions
    pi_r = torch.tensor([[0.1, 0.2, 0.4, 0.3]])  # Policy
    
    v_r_computed = v_r_network.compute_from_components(u_r, q_r, pi_r)
    expected_eq = (pi_r * q_r).sum(dim=-1)  # E[Q_r]
    expected_v_r = u_r + expected_eq
    assert torch.isclose(v_r_computed, expected_v_r, atol=1e-5)
    print(f"  ✓ V_r from components: {v_r_computed.item():.4f}")
    
    # Test TD target
    u_r_cur = torch.tensor([-2.0])
    next_v_r = torch.tensor([-1.0])
    td_target = v_r_network.compute_td_target(u_r_cur, next_v_r)
    expected_target = -2.0 + 0.99 * (-1.0)
    assert torch.isclose(td_target, torch.tensor(expected_target), atol=1e-5)
    print(f"  ✓ TD target: {td_target.item():.4f}")
    
    print("  ✓ MultiGridRobotValueNetwork test passed!")


def test_power_law_policy():
    """Test the power-law softmax policy derivation (equation 5)."""
    print("Testing power-law policy...")
    
    # Create network with different beta values
    for beta_r in [1.0, 5.0, 10.0, 50.0]:
        q_network = MultiGridRobotQNetwork(
            grid_height=10,
            grid_width=10,
            num_robot_actions=4,
            num_robots=1,  # Single robot for simplicity
            num_agents_per_color={'grey': 2},
            num_agent_colors=7,
            beta_r=beta_r
        )
        
        # Create Q-values with one clear winner
        q_values = torch.tensor([[-1.0, -2.0, -5.0, -10.0]])
        policy = q_network.get_policy(q_values)
        
        # Higher beta should make policy more deterministic
        # The best action (Q=-1.0, so -Q=1.0 is smallest, but raised to -beta
        # means 1^{-beta}=1 while 2^{-beta}, 5^{-beta}, 10^{-beta} are smaller
        # Actually: (-Q)^{-beta} = 1/(-Q)^beta
        # For Q=-1: (-Q)^{-beta} = 1
        # For Q=-2: (-Q)^{-beta} = 2^{-beta}
        # Higher beta means 2^{-beta} -> 0 faster
        print(f"  Beta={beta_r}: policy = {policy.squeeze().tolist()}")
    
    # Verify that higher beta concentrates probability on best action
    q_network_low_beta = MultiGridRobotQNetwork(
        grid_height=10, grid_width=10, num_robot_actions=4, num_robots=1,
        num_agents_per_color={'grey': 2}, num_agent_colors=7, beta_r=1.0
    )
    q_network_high_beta = MultiGridRobotQNetwork(
        grid_height=10, grid_width=10, num_robot_actions=4, num_robots=1,
        num_agents_per_color={'grey': 2}, num_agent_colors=7, beta_r=50.0
    )
    
    q_values = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    policy_low = q_network_low_beta.get_policy(q_values)
    policy_high = q_network_high_beta.get_policy(q_values)
    
    # Best action (index 0) should have higher prob with higher beta
    assert policy_high[0, 0] > policy_low[0, 0], "Higher beta should give more weight to best action"
    print(f"  ✓ Higher beta concentrates probability: {policy_low[0, 0]:.4f} < {policy_high[0, 0]:.4f}")
    
    print("  ✓ Power-law policy test passed!")


def test_soft_clamp_in_phase2():
    """Test that SoftClamp is used correctly in Phase 2 networks."""
    print("Testing SoftClamp usage in Phase 2...")
    
    # Create networks
    v_h_e = MultiGridHumanGoalAchievementNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7
    )
    
    x_h = MultiGridAggregateGoalAbilityNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7
    )
    
    # Test that SoftClamp preserves gradients in linear region
    test_values = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
    
    # For V_h^e
    clamped_v = v_h_e.apply_clamp(test_values)
    loss_v = clamped_v.sum()
    loss_v.backward()
    assert test_values.grad is not None
    # In linear region [0, 1], gradient should be ~1
    assert torch.allclose(test_values.grad, torch.ones(3), atol=0.01), \
        f"Gradients in linear region should be ~1, got {test_values.grad}"
    print("  ✓ V_h^e SoftClamp preserves gradients in [0, 1]")
    
    test_values2 = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
    clamped_x = x_h.apply_clamp(test_values2)
    loss_x = clamped_x.sum()
    loss_x.backward()
    assert torch.allclose(test_values2.grad, torch.ones(3), atol=0.01)
    print("  ✓ X_h SoftClamp preserves gradients in [0, 1]")
    
    # Test that values outside range are bounded but gradients still flow
    test_outside = torch.tensor([-0.5, 1.5], requires_grad=True)
    clamped_out = v_h_e.apply_clamp(test_outside)
    
    # Values should be bounded more than original
    assert clamped_out[0] > -0.5, "Below-range should be bounded"
    assert clamped_out[1] < 1.5, "Above-range should be bounded"
    
    loss_out = clamped_out.sum()
    loss_out.backward()
    assert test_outside.grad is not None
    assert test_outside.grad[0] > 0, "Gradient should still flow for below-range"
    assert test_outside.grad[1] > 0, "Gradient should still flow for above-range"
    print("  ✓ SoftClamp allows gradients outside range")
    
    # Test hard clamp for inference
    inference_values = torch.tensor([-0.5, 0.5, 1.5])
    hard_v = v_h_e.apply_hard_clamp(inference_values)
    assert hard_v[0] == 0.0
    assert hard_v[1] == 0.5
    assert hard_v[2] == 1.0
    print("  ✓ Hard clamp strictly bounds values for inference")
    
    print("  ✓ SoftClamp usage test passed!")


def test_network_configs():
    """Test that networks properly return configs for save/load."""
    print("Testing network configs...")
    
    q_r = MultiGridRobotQNetwork(
        grid_height=10,
        grid_width=10,
        num_robot_actions=4,
        num_robots=2,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        beta_r=10.0
    )
    config = q_r.get_config()
    assert config['num_robot_actions'] == 4
    assert config['num_robots'] == 2
    assert config['beta_r'] == 10.0
    print(f"  ✓ Q_r config: {config}")
    
    v_h_e = MultiGridHumanGoalAchievementNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        gamma_h=0.95
    )
    config = v_h_e.get_config()
    assert config['gamma_h'] == 0.95
    print(f"  ✓ V_h^e config keys: {list(config.keys())}")
    
    x_h = MultiGridAggregateGoalAbilityNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        zeta=3.0
    )
    config = x_h.get_config()
    assert config['zeta'] == 3.0
    print(f"  ✓ X_h config keys: {list(config.keys())}")
    
    u_r = MultiGridIntrinsicRewardNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        xi=2.0,
        eta=1.5
    )
    config = u_r.get_config()
    assert config['xi'] == 2.0
    assert config['eta'] == 1.5
    print(f"  ✓ U_r config keys: {list(config.keys())}")
    
    v_r = MultiGridRobotValueNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        gamma_r=0.95
    )
    config = v_r.get_config()
    assert config['gamma_r'] == 0.95
    print(f"  ✓ V_r config keys: {list(config.keys())}")
    
    print("  ✓ Network configs test passed!")


def test_ensure_negative():
    """Test the ensure_negative function used in Q_r and V_r."""
    print("Testing ensure_negative...")
    
    q_network = MultiGridRobotQNetwork(
        grid_height=10,
        grid_width=10,
        num_robot_actions=4,
        num_robots=1,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7
    )
    
    v_network = MultiGridRobotValueNetwork(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7
    )
    
    # Test with various input values
    test_values = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    
    # Both use -softplus to ensure negative
    neg_q = q_network.ensure_negative(test_values)
    neg_v = v_network.ensure_negative(test_values)
    
    assert (neg_q < 0).all(), "Q_r ensure_negative should give all negative values"
    assert (neg_v < 0).all(), "V_r ensure_negative should give all negative values"
    
    # Note: Q_r uses -softplus(x), V_r uses -softplus(-x)
    # These have different behaviors but both ensure negative output
    print(f"  ✓ Q_r ensure_negative: all values negative")
    print(f"  ✓ V_r ensure_negative: all values negative")
    
    # Test gradient flow
    test_grad = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)
    neg_out = q_network.ensure_negative(test_grad)
    neg_out.sum().backward()
    assert test_grad.grad is not None
    print("  ✓ Gradients flow through ensure_negative")
    
    print("  ✓ ensure_negative test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Phase 2 Tests")
    print("=" * 60)
    
    test_phase2_config()
    print()
    
    test_phase2_replay_buffer()
    print()
    
    test_multigrid_robot_q_network()
    print()
    
    test_multigrid_human_goal_achievement_network()
    print()
    
    test_multigrid_aggregate_goal_ability_network()
    print()
    
    test_multigrid_intrinsic_reward_network()
    print()
    
    test_multigrid_robot_value_network()
    print()
    
    test_power_law_policy()
    print()
    
    test_soft_clamp_in_phase2()
    print()
    
    test_network_configs()
    print()
    
    test_ensure_negative()
    print()
    
    print("=" * 60)
    print("All Phase 2 tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
