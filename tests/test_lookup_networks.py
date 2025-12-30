"""
Tests for Phase 2 lookup table networks.

Tests verify that lookup table implementations:
1. Have the same API as neural network versions
2. Store and retrieve values correctly
3. Support gradient computation
4. Handle batching properly
5. Can be saved and loaded
"""

import pytest
import torch
import torch.nn as nn

from empo.nn_based.phase2.lookup import (
    LookupTableRobotQNetwork,
    LookupTableRobotValueNetwork,
    LookupTableHumanGoalAbilityNetwork,
    LookupTableAggregateGoalAbilityNetwork,
    LookupTableIntrinsicRewardNetwork,
)
from empo.nn_based.phase2.config import Phase2Config


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def simple_states():
    """Create simple hashable test states."""
    return [
        (0, 0, 0),  # state 1
        (0, 0, 1),  # state 2
        (1, 0, 0),  # state 3
        (0, 0, 0),  # duplicate of state 1
    ]


@pytest.fixture
def simple_goals():
    """Create simple hashable test goals."""
    # Goals need to be hashable, use tuples
    return [
        ('goal', 0, 0),  # goal 1
        ('goal', 1, 0),  # goal 2
        ('goal', 0, 1),  # goal 3
        ('goal', 0, 0),  # duplicate of goal 1
    ]


# =============================================================================
# LookupTableRobotQNetwork tests
# =============================================================================

class TestLookupTableRobotQNetwork:
    """Tests for LookupTableRobotQNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = LookupTableRobotQNetwork(
            num_actions=4,
            num_robots=2,
            beta_r=10.0,
            default_q_r=-1.0
        )
        
        assert network.num_actions == 4
        assert network.num_robots == 2
        assert network.num_action_combinations == 16  # 4^2
        assert network.beta_r == 10.0
        assert network.default_q_r == -1.0
        assert len(network.table) == 0  # Empty initially
    
    def test_forward_creates_entries(self, simple_states):
        """Test that forward pass creates table entries for unseen states."""
        network = LookupTableRobotQNetwork(
            num_actions=3,
            num_robots=1,
            default_q_r=-0.5
        )
        
        # Forward pass should create entries
        output = network.forward(simple_states, device='cpu')
        
        assert output.shape == (4, 3)  # batch_size=4, num_actions=3
        # Should have 3 unique entries (state 1 and 4 are duplicates)
        assert len(network.table) == 3
    
    def test_output_is_negative(self, simple_states):
        """Test that Q-values are always negative."""
        network = LookupTableRobotQNetwork(
            num_actions=4,
            num_robots=1,
            default_q_r=-1.0
        )
        
        output = network.forward(simple_states, device='cpu')
        
        assert (output < 0).all(), "Q_r values must be negative"
    
    def test_duplicate_states_share_entry(self, simple_states):
        """Test that duplicate states return the same values."""
        network = LookupTableRobotQNetwork(
            num_actions=3,
            num_robots=1,
            default_q_r=-1.0
        )
        
        output = network.forward(simple_states, device='cpu')
        
        # States 0 and 3 are identical, should have same output
        assert torch.allclose(output[0], output[3])
    
    def test_gradient_flow(self, simple_states):
        """Test that gradients flow through lookup table."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
            default_q_r=-1.0
        )
        
        output = network.forward(simple_states[:1], device='cpu')
        loss = output.mean()
        loss.backward()
        
        # Check that the accessed parameter has gradients
        state_key = hash(simple_states[0])
        assert state_key in network.table
        assert network.table[state_key].grad is not None
    
    def test_encode_and_forward(self, simple_states):
        """Test encode_and_forward for single state."""
        network = LookupTableRobotQNetwork(
            num_actions=4,
            num_robots=2
        )
        
        output = network.encode_and_forward(
            state=simple_states[0],
            world_model=None,  # Not used for lookup tables
            device='cpu'
        )
        
        assert output.shape == (1, 16)  # (1, num_action_combinations)
        assert (output < 0).all()
    
    def test_get_policy(self, simple_states):
        """Test policy computation from Q-values."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
            beta_r=5.0
        )
        
        output = network.forward(simple_states[:1], device='cpu')
        policy = network.get_policy(output)
        
        # Policy should sum to 1
        assert torch.allclose(policy.sum(dim=-1), torch.tensor([1.0]))
        # All probabilities should be positive
        assert (policy > 0).all()
    
    def test_get_config(self):
        """Test configuration retrieval."""
        network = LookupTableRobotQNetwork(
            num_actions=4,
            num_robots=2,
            beta_r=10.0,
            default_q_r=-2.0
        )
        
        config = network.get_config()
        
        assert config['type'] == 'lookup_table'
        assert config['num_actions'] == 4
        assert config['num_robots'] == 2
        assert config['beta_r'] == 10.0
        assert config['default_q_r'] == -2.0
    
    def test_state_dict_and_load(self, simple_states):
        """Test save/load functionality."""
        network1 = LookupTableRobotQNetwork(
            num_actions=3,
            num_robots=1,
            default_q_r=-1.0
        )
        
        # Create some entries
        _ = network1.forward(simple_states, device='cpu')
        
        # Modify one entry
        key = hash(simple_states[0])
        network1.table[key].data.fill_(5.0)  # Raw value before ensure_negative
        
        # Save
        state_dict = network1.state_dict()
        
        # Load into new network
        network2 = LookupTableRobotQNetwork(
            num_actions=3,
            num_robots=1,
            default_q_r=-1.0
        )
        network2.load_state_dict(state_dict)
        
        # Verify
        assert len(network2.table) == len(network1.table)
        assert torch.allclose(network2.table[key], network1.table[key])


# =============================================================================
# LookupTableRobotValueNetwork tests
# =============================================================================

class TestLookupTableRobotValueNetwork:
    """Tests for LookupTableRobotValueNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = LookupTableRobotValueNetwork(
            gamma_r=0.99,
            default_v_r=-1.0
        )
        
        assert network.gamma_r == 0.99
        assert network.default_v_r == -1.0
        assert len(network.table) == 0
    
    def test_forward_output_shape(self, simple_states):
        """Test forward pass output shape."""
        network = LookupTableRobotValueNetwork()
        
        output = network.forward(simple_states, device='cpu')
        
        assert output.shape == (4,)
        assert (output < 0).all(), "V_r values must be negative"
    
    def test_encode_and_forward(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableRobotValueNetwork()
        
        output = network.encode_and_forward(
            state=simple_states[0],
            world_model=None,
            device='cpu'
        )
        
        assert output.shape == (1,)
        assert output.item() < 0


# =============================================================================
# LookupTableHumanGoalAbilityNetwork tests
# =============================================================================

class TestLookupTableHumanGoalAbilityNetwork:
    """Tests for LookupTableHumanGoalAbilityNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = LookupTableHumanGoalAbilityNetwork(
            gamma_h=0.99,
            default_v_h_e=0.5
        )
        
        assert network.gamma_h == 0.99
        assert network.default_v_h_e == 0.5
        assert len(network.table) == 0
    
    def test_forward_output_shape(self, simple_states, simple_goals):
        """Test forward pass output shape."""
        network = LookupTableHumanGoalAbilityNetwork()
        
        output = network.forward(simple_states, simple_goals, device='cpu')
        
        assert output.shape == (4,)
    
    def test_output_in_bounds(self, simple_states, simple_goals):
        """Test that output is in [0, 1]."""
        network = LookupTableHumanGoalAbilityNetwork(
            feasible_range=(0.0, 1.0)
        )
        
        output = network.forward(simple_states, simple_goals, device='cpu')
        
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_different_goals_different_entries(self, simple_states):
        """Test that different goals create different entries."""
        network = LookupTableHumanGoalAbilityNetwork()
        
        states = [simple_states[0], simple_states[0]]
        goals = [('goal', 'A'), ('goal', 'B')]
        
        output = network.forward(states, goals, device='cpu')
        
        # Same state but different goals should create 2 entries
        assert len(network.table) == 2
    
    def test_encode_and_forward(self, simple_states, simple_goals):
        """Test single-state forward pass."""
        network = LookupTableHumanGoalAbilityNetwork()
        
        output = network.encode_and_forward(
            state=simple_states[0],
            world_model=None,
            human_agent_idx=0,
            goal=simple_goals[0],
            device='cpu'
        )
        
        assert output.shape == (1,)
        assert 0 <= output.item() <= 1


# =============================================================================
# LookupTableAggregateGoalAbilityNetwork tests
# =============================================================================

class TestLookupTableAggregateGoalAbilityNetwork:
    """Tests for LookupTableAggregateGoalAbilityNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = LookupTableAggregateGoalAbilityNetwork(
            zeta=2.0,
            default_x_h=0.5
        )
        
        assert network.zeta == 2.0
        assert network.default_x_h == 0.5
        assert len(network.table) == 0
    
    def test_forward_output_shape(self, simple_states):
        """Test forward pass output shape."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        human_indices = [0, 1, 0, 1]  # Mix of human indices
        output = network.forward(simple_states, human_indices, device='cpu')
        
        assert output.shape == (4,)
    
    def test_output_in_bounds(self, simple_states):
        """Test that output is in (0, 1]."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        human_indices = [0, 1, 2, 3]
        output = network.forward(simple_states, human_indices, device='cpu')
        
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_different_humans_different_entries(self, simple_states):
        """Test that different human indices create different entries."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        # Same state, different humans
        states = [simple_states[0], simple_states[0]]
        human_indices = [0, 1]
        
        output = network.forward(states, human_indices, device='cpu')
        
        # Should create 2 entries (same state but different human indices)
        assert len(network.table) == 2
    
    def test_encode_and_forward(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        output = network.encode_and_forward(
            state=simple_states[0],
            world_model=None,
            human_agent_idx=0,
            device='cpu'
        )
        
        assert output.shape == (1,)
        assert 0 <= output.item() <= 1


# =============================================================================
# LookupTableIntrinsicRewardNetwork tests
# =============================================================================

class TestLookupTableIntrinsicRewardNetwork:
    """Tests for LookupTableIntrinsicRewardNetwork."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = LookupTableIntrinsicRewardNetwork(
            xi=1.0,
            eta=1.1,
            default_y=2.0
        )
        
        assert network.xi == 1.0
        assert network.eta == 1.1
        assert network.default_y == 2.0
        assert len(network.table) == 0
    
    def test_invalid_default_y(self):
        """Test that default_y < 1 raises error."""
        with pytest.raises(ValueError):
            LookupTableIntrinsicRewardNetwork(default_y=0.5)
    
    def test_forward_returns_tuple(self, simple_states):
        """Test forward pass returns (y, U_r) tuple."""
        network = LookupTableIntrinsicRewardNetwork()
        
        y, u_r = network.forward(simple_states, device='cpu')
        
        assert y.shape == (4,)
        assert u_r.shape == (4,)
    
    def test_y_greater_than_one(self, simple_states):
        """Test that y > 1 always."""
        network = LookupTableIntrinsicRewardNetwork()
        
        y, _ = network.forward(simple_states, device='cpu')
        
        assert (y > 1).all(), "y must be > 1"
    
    def test_u_r_negative(self, simple_states):
        """Test that U_r < 0 always."""
        network = LookupTableIntrinsicRewardNetwork()
        
        _, u_r = network.forward(simple_states, device='cpu')
        
        assert (u_r < 0).all(), "U_r must be negative"
    
    def test_y_to_u_r_relationship(self, simple_states):
        """Test that U_r = -y^η."""
        network = LookupTableIntrinsicRewardNetwork(eta=1.5)
        
        y, u_r = network.forward(simple_states, device='cpu')
        
        expected_u_r = -(y ** 1.5)
        assert torch.allclose(u_r, expected_u_r)
    
    def test_encode_and_forward(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableIntrinsicRewardNetwork()
        
        y, u_r = network.encode_and_forward(
            state=simple_states[0],
            world_model=None,
            device='cpu'
        )
        
        assert y.shape == (1,)
        assert u_r.shape == (1,)
        assert y.item() > 1
        assert u_r.item() < 0


# =============================================================================
# Phase2Config lookup table settings tests
# =============================================================================

class TestPhase2ConfigLookupTables:
    """Tests for Phase2Config lookup table settings."""
    
    def test_default_lookup_disabled(self):
        """Test that lookup tables are disabled by default."""
        config = Phase2Config()
        
        assert config.use_lookup_tables is False
    
    def test_should_use_lookup_table_when_disabled(self):
        """Test should_use_lookup_table when lookup tables are disabled."""
        config = Phase2Config(use_lookup_tables=False)
        
        assert config.should_use_lookup_table('q_r') is False
        assert config.should_use_lookup_table('v_h_e') is False
    
    def test_should_use_lookup_table_when_enabled(self):
        """Test should_use_lookup_table when lookup tables are enabled."""
        config = Phase2Config(
            use_lookup_tables=True,
            use_lookup_q_r=True,
            use_lookup_v_h_e=True,
            use_lookup_x_h=False,  # Selectively disabled
        )
        
        assert config.should_use_lookup_table('q_r') is True
        assert config.should_use_lookup_table('v_h_e') is True
        assert config.should_use_lookup_table('x_h') is False
    
    def test_v_r_lookup_requires_network(self):
        """Test that V_r lookup requires v_r_use_network=True."""
        config = Phase2Config(
            use_lookup_tables=True,
            use_lookup_v_r=True,
            v_r_use_network=False,  # V_r computed from Q_r, U_r
        )
        
        # Should be False because v_r_use_network=False
        assert config.should_use_lookup_table('v_r') is False
    
    def test_u_r_lookup_requires_network(self):
        """Test that U_r lookup requires u_r_use_network=True."""
        config = Phase2Config(
            use_lookup_tables=True,
            use_lookup_u_r=True,
            u_r_use_network=False,  # U_r computed from X_h
        )
        
        # Should be False because u_r_use_network=False
        assert config.should_use_lookup_table('u_r') is False
    
    def test_get_lookup_default(self):
        """Test getting default values for lookup table entries."""
        config = Phase2Config(
            lookup_default_q_r=-2.0,
            lookup_default_v_h_e=0.7,
        )
        
        assert config.get_lookup_default('q_r') == -2.0
        assert config.get_lookup_default('v_h_e') == 0.7
    
    def test_should_recreate_optimizer_disabled(self):
        """Test optimizer recreation when lookup tables disabled."""
        config = Phase2Config(use_lookup_tables=False)
        
        assert config.should_recreate_optimizer(1000) is False
    
    def test_should_recreate_optimizer_at_interval(self):
        """Test optimizer recreation at regular intervals."""
        config = Phase2Config(
            use_lookup_tables=True,
            lookup_optimizer_recreate_interval=500,
        )
        
        assert config.should_recreate_optimizer(500) is True
        assert config.should_recreate_optimizer(1000) is True
        assert config.should_recreate_optimizer(499) is False
    
    def test_should_recreate_optimizer_at_warmup_boundaries(self):
        """Test optimizer recreation at warmup stage boundaries."""
        config = Phase2Config(
            use_lookup_tables=True,
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=100,
            warmup_q_r_steps=100,
            lookup_optimizer_recreate_interval=10000,  # Large interval
        )
        
        # Should recreate at warmup boundaries
        assert config.should_recreate_optimizer(100) is True   # _warmup_v_h_e_end
        assert config.should_recreate_optimizer(200) is True   # _warmup_x_h_end


# =============================================================================
# Integration tests
# =============================================================================

class TestLookupTableIntegration:
    """Integration tests for lookup table networks."""
    
    def test_optimizer_can_update_entries(self, simple_states):
        """Test that standard optimizers can update lookup table entries."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create entries
        output = network.forward(simple_states[:1], device='cpu')
        
        # Create optimizer with current parameters
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        
        # Compute loss and update
        loss = output.mean()
        loss.backward()
        
        # Get parameter before update
        key = hash(simple_states[0])
        old_value = network.table[key].data.clone()
        
        optimizer.step()
        
        # Parameter should have changed
        new_value = network.table[key].data
        assert not torch.allclose(old_value, new_value)
    
    def test_device_transfer(self, simple_states):
        """Test moving lookup table to different device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create some entries on CPU
        _ = network.forward(simple_states, device='cpu')
        
        # Move to GPU
        network = network.to('cuda')
        
        # All entries should be on GPU
        for key, param in network.table.items():
            assert param.device.type == 'cuda'
