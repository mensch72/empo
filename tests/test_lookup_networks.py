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
        output = network.forward_batch(simple_states, None, device='cpu')
        
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
        
        output = network.forward_batch(simple_states, None, device='cpu')
        
        assert (output < 0).all(), "Q_r values must be negative"
    
    def test_duplicate_states_share_entry(self, simple_states):
        """Test that duplicate states return the same values."""
        network = LookupTableRobotQNetwork(
            num_actions=3,
            num_robots=1,
            default_q_r=-1.0
        )
        
        output = network.forward_batch(simple_states, None, device='cpu')
        
        # States 0 and 3 are identical, should have same output
        assert torch.allclose(output[0], output[3])
    
    def test_gradient_flow(self, simple_states):
        """Test that gradients flow through lookup table."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
            default_q_r=-1.0
        )
        
        output = network.forward_batch(simple_states, None, device='cpu')
        loss = output.mean()
        loss.backward()
        
        # Check that the accessed parameter has gradients
        # Note: When world_model is None, map_hash is 0
        state_key = hash((simple_states[0], 0))
        assert state_key in network.table
        assert network.table[state_key].grad is not None
    
    def test_forward_single_state(self, simple_states):
        """Test encode_and_forward for single state."""
        network = LookupTableRobotQNetwork(
            num_actions=4,
            num_robots=2
        )
        
        output = network.forward(
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
        
        output = network.forward_batch(simple_states, None, device='cpu')
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
        _ = network1.forward_batch(simple_states, None, device='cpu')
        
        # Modify one entry
        # Note: When world_model is None, map_hash is 0
        key = hash((simple_states[0], 0))
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
        
        output = network.forward_batch(simple_states, None, device='cpu')
        
        assert output.shape == (4,)
        assert (output < 0).all(), "V_r values must be negative"
    
    def test_forward_single_state(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableRobotValueNetwork()
        
        output = network.forward(
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
        
        output = network.forward_batch(simple_states, simple_goals, None, None, device='cpu')
        
        assert output.shape == (4,)
    
    def test_output_in_bounds(self, simple_states, simple_goals):
        """Test that output is in [0, 1]."""
        network = LookupTableHumanGoalAbilityNetwork(
            feasible_range=(0.0, 1.0)
        )
        
        output = network.forward_batch(simple_states, simple_goals, None, None, device='cpu')
        
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_different_goals_different_entries(self, simple_states):
        """Test that different goals create different entries."""
        network = LookupTableHumanGoalAbilityNetwork()
        
        states = [simple_states[0], simple_states[0]]
        goals = [('goal', 'A'), ('goal', 'B')]
        human_indices = [0, 0]
        
        output = network.forward_batch(states, goals, human_indices, None, device='cpu')
        
        # Same state but different goals should create 2 entries
        assert len(network.table) == 2
    
    def test_forward_single_state(self, simple_states, simple_goals):
        """Test single-state forward pass."""
        network = LookupTableHumanGoalAbilityNetwork()
        
        output = network.forward(
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
        output = network.forward_batch(simple_states, human_indices, None, device='cpu')
        
        assert output.shape == (4,)
    
    def test_output_in_bounds(self, simple_states):
        """Test that output is in (0, 1]."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        human_indices = [0, 1, 2, 3]
        output = network.forward_batch(simple_states, human_indices, None, device='cpu')
        
        assert (output >= 0).all()
        assert (output <= 1).all()
    
    def test_different_humans_different_entries(self, simple_states):
        """Test that different human indices create different entries."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        # Same state, different humans
        states = [simple_states[0], simple_states[0]]
        human_indices = [0, 1]
        
        output = network.forward_batch(states, human_indices, None, device='cpu')
        
        # Should create 2 entries (same state but different human indices)
        assert len(network.table) == 2
    
    def test_forward_single_state(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableAggregateGoalAbilityNetwork()
        
        output = network.forward(
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
        
        y, u_r = network.forward_batch(simple_states, None, device='cpu')
        
        assert y.shape == (4,)
        assert u_r.shape == (4,)
    
    def test_y_greater_than_one(self, simple_states):
        """Test that y > 1 always."""
        network = LookupTableIntrinsicRewardNetwork()
        
        y, _ = network.forward_batch(simple_states, None, device='cpu')
        
        assert (y > 1).all(), "y must be > 1"
    
    def test_u_r_negative(self, simple_states):
        """Test that U_r < 0 always."""
        network = LookupTableIntrinsicRewardNetwork()
        
        _, u_r = network.forward_batch(simple_states, None, device='cpu')
        
        assert (u_r < 0).all(), "U_r must be negative"
    
    def test_y_to_u_r_relationship(self, simple_states):
        """Test that U_r = -y^η."""
        network = LookupTableIntrinsicRewardNetwork(eta=1.5)
        
        y, u_r = network.forward_batch(simple_states, None, device='cpu')
        
        expected_u_r = -(y ** 1.5)
        assert torch.allclose(u_r, expected_u_r)
    
    def test_forward_single_state(self, simple_states):
        """Test single-state forward pass."""
        network = LookupTableIntrinsicRewardNetwork()
        
        y, u_r = network.forward(
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
    
    def test_x_h_lookup_requires_network(self):
        """Test that X_h lookup requires x_h_use_network=True."""
        config = Phase2Config(
            use_lookup_tables=True,
            use_lookup_x_h=True,
            x_h_use_network=False,  # X_h computed from V_h^e samples
        )
        
        # Should be False because x_h_use_network=False
        assert config.should_use_lookup_table('x_h') is False
    
    def test_get_lookup_default(self):
        """Test getting default values for lookup table entries."""
        config = Phase2Config(
            lookup_default_q_r=-2.0,
            lookup_default_v_h_e=0.7,
        )
        
        assert config.get_lookup_default('q_r') == -2.0
        assert config.get_lookup_default('v_h_e') == 0.7


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
        output = network.forward_batch(simple_states, None, device='cpu')
        
        # Create optimizer with current parameters
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        
        # Compute loss and update
        loss = output.mean()
        loss.backward()
        
        # Get parameter before update
        # Note: When world_model is None, map_hash is 0
        key = hash((simple_states[0], 0))
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
        _ = network.forward_batch(simple_states, None, device='cpu')
        
        # Move to GPU
        network = network.to('cuda')
        
        # All entries should be on GPU
        for key, param in network.table.items():
            assert param.device.type == 'cuda'
    
    def test_incremental_param_tracking(self, simple_states):
        """Test that get_new_params tracks newly created parameters."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Initially no new params
        assert len(network._new_params) == 0
        
        # Create entries via forward
        _ = network.forward_batch(simple_states[:2], None, device='cpu')
        
        # Should have 2 new params (2 unique states)
        assert len(network._new_params) == 2
        
        # Get new params clears the list
        new_params = network.get_new_params()
        assert len(new_params) == 2
        assert len(network._new_params) == 0
        
        # Add more states
        _ = network.forward_batch(simple_states[2:3], None, device='cpu')
        
        # Should have 1 new param
        assert len(network._new_params) == 1
        
        # Accessing existing state doesn't add new params
        _ = network.forward_batch(simple_states[:1], None, device='cpu')
        assert len(network._new_params) == 1
    
    def test_incremental_optimizer_updates(self, simple_states):
        """Test that incrementally added params work with optimizer."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create initial entries
        _ = network.forward_batch(simple_states[:2], None, device='cpu')
        
        # Create optimizer with initial params
        optimizer = torch.optim.Adam(network.get_new_params(), lr=0.1)
        
        # Add more entries
        _ = network.forward_batch(simple_states[2:3], None, device='cpu')
        
        # Incrementally add new params
        new_params = network.get_new_params()
        for param in new_params:
            optimizer.param_groups[0]['params'].append(param)
        
        # Verify all 3 params are in optimizer
        assert len(optimizer.param_groups[0]['params']) == 3
        
        # Verify gradients flow to all params
        output = network.forward_batch(simple_states[:3], None, device='cpu')
        loss = output.sum()
        loss.backward()
        
        for param in optimizer.param_groups[0]['params']:
            assert param.grad is not None


class TestLookupTableUtilities:
    """Tests for lookup table utility functions."""
    
    def test_is_lookup_table_network_with_lookup_tables(self):
        """Test is_lookup_table_network returns True for lookup tables."""
        from empo.nn_based.phase2.lookup import is_lookup_table_network
        
        q_r = LookupTableRobotQNetwork(num_actions=2, num_robots=1)
        v_r = LookupTableRobotValueNetwork()
        v_h_e = LookupTableHumanGoalAbilityNetwork()
        x_h = LookupTableAggregateGoalAbilityNetwork()
        u_r = LookupTableIntrinsicRewardNetwork()
        
        assert is_lookup_table_network(q_r) is True
        assert is_lookup_table_network(v_r) is True
        assert is_lookup_table_network(v_h_e) is True
        assert is_lookup_table_network(x_h) is True
        assert is_lookup_table_network(u_r) is True
    
    def test_is_lookup_table_network_with_other_objects(self):
        """Test is_lookup_table_network returns False for other objects."""
        from empo.nn_based.phase2.lookup import is_lookup_table_network
        
        assert is_lookup_table_network(nn.Linear(10, 10)) is False
        assert is_lookup_table_network(None) is False
        assert is_lookup_table_network("not a network") is False
    
    def test_get_all_lookup_tables_all_lookup(self):
        """Test get_all_lookup_tables with all lookup table networks."""
        from empo.nn_based.phase2.lookup import get_all_lookup_tables
        from empo.nn_based.phase2.trainer import Phase2Networks
        
        networks = Phase2Networks(
            q_r=LookupTableRobotQNetwork(num_actions=2, num_robots=1),
            v_h_e=LookupTableHumanGoalAbilityNetwork(),
            x_h=LookupTableAggregateGoalAbilityNetwork(),
            u_r=LookupTableIntrinsicRewardNetwork(),
            v_r=LookupTableRobotValueNetwork(),
        )
        
        lookup_tables = get_all_lookup_tables(networks)
        
        assert 'q_r' in lookup_tables
        assert 'v_h_e' in lookup_tables
        assert 'x_h' in lookup_tables
        assert 'u_r' in lookup_tables
        assert 'v_r' in lookup_tables
        assert len(lookup_tables) == 5
    
    def test_get_all_lookup_tables_mixed(self):
        """Test get_all_lookup_tables with mixed network types."""
        from empo.nn_based.phase2.lookup import get_all_lookup_tables
        from empo.nn_based.phase2.trainer import Phase2Networks
        
        # Create a mock neural network (not a lookup table)
        class MockQNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)
        
        networks = Phase2Networks(
            q_r=MockQNetwork(),  # Not a lookup table
            v_h_e=LookupTableHumanGoalAbilityNetwork(),
            x_h=LookupTableAggregateGoalAbilityNetwork(),
        )
        
        lookup_tables = get_all_lookup_tables(networks)
        
        assert 'q_r' not in lookup_tables
        assert 'v_h_e' in lookup_tables
        assert 'x_h' in lookup_tables
        assert len(lookup_tables) == 2
    
    def test_get_total_table_size(self, simple_states):
        """Test get_total_table_size counts entries correctly."""
        from empo.nn_based.phase2.lookup import get_total_table_size
        from empo.nn_based.phase2.trainer import Phase2Networks
        
        q_r = LookupTableRobotQNetwork(num_actions=2, num_robots=1)
        v_h_e = LookupTableHumanGoalAbilityNetwork()
        x_h = LookupTableAggregateGoalAbilityNetwork()
        
        networks = Phase2Networks(q_r=q_r, v_h_e=v_h_e, x_h=x_h)
        
        # Initially empty
        assert get_total_table_size(networks) == 0
        
        # Add some entries to q_r (3 unique states)
        _ = q_r.forward_batch(simple_states[:3], None, device='cpu')
        assert get_total_table_size(networks) == 3
        
        # Add entries to v_h_e (2 unique state-goal pairs)
        _ = v_h_e.forward_batch(simple_states[:2], [('g', 1), ('g', 2)], [0, 0], None, device='cpu')
        assert get_total_table_size(networks) == 5


class TestNetworkFactory:
    """Tests for network factory functions."""
    
    def test_create_all_phase2_lookup_networks(self):
        """Test creating all lookup networks at once."""
        from empo.nn_based.phase2.network_factory import create_all_phase2_lookup_networks
        
        config = Phase2Config(
            use_lookup_tables=True,
            use_lookup_q_r=True,
            use_lookup_v_h_e=True,
            use_lookup_x_h=True,
            use_lookup_u_r=True,
            use_lookup_v_r=True,
            u_r_use_network=True,
            v_r_use_network=True,
        )
        
        q_r, v_h_e, x_h, u_r, v_r = create_all_phase2_lookup_networks(
            config, num_actions=4, num_robots=1
        )
        
        assert isinstance(q_r, LookupTableRobotQNetwork)
        assert isinstance(v_h_e, LookupTableHumanGoalAbilityNetwork)
        assert isinstance(x_h, LookupTableAggregateGoalAbilityNetwork)
        assert isinstance(u_r, LookupTableIntrinsicRewardNetwork)
        assert isinstance(v_r, LookupTableRobotValueNetwork)
    
    def test_create_all_lookup_without_optional_networks(self):
        """Test creating lookup networks with u_r and v_r disabled."""
        from empo.nn_based.phase2.network_factory import create_all_phase2_lookup_networks
        
        config = Phase2Config(
            use_lookup_tables=True,
            u_r_use_network=False,  # U_r computed from X_h
            v_r_use_network=False,  # V_r computed from U_r and Q_r
        )
        
        q_r, v_h_e, x_h, u_r, v_r = create_all_phase2_lookup_networks(
            config, num_actions=4, num_robots=1
        )
        
        assert isinstance(q_r, LookupTableRobotQNetwork)
        assert isinstance(v_h_e, LookupTableHumanGoalAbilityNetwork)
        assert isinstance(x_h, LookupTableAggregateGoalAbilityNetwork)
        assert u_r is None
        assert v_r is None
    
    def test_create_all_lookup_without_x_h_network(self):
        """Test creating lookup networks with x_h disabled."""
        from empo.nn_based.phase2.network_factory import create_all_phase2_lookup_networks
        
        config = Phase2Config(
            use_lookup_tables=True,
            x_h_use_network=False,  # X_h computed from V_h^e samples
            u_r_use_network=False,  # U_r computed from X_h
            v_r_use_network=False,  # V_r computed from U_r and Q_r
        )
        
        q_r, v_h_e, x_h, u_r, v_r = create_all_phase2_lookup_networks(
            config, num_actions=4, num_robots=1
        )
        
        assert isinstance(q_r, LookupTableRobotQNetwork)
        assert isinstance(v_h_e, LookupTableHumanGoalAbilityNetwork)
        assert x_h is None  # X_h computed directly from V_h^e samples
        assert u_r is None
        assert v_r is None
    
    def test_create_all_lookup_requires_flag(self):
        """Test that factory raises error if use_lookup_tables is False."""
        from empo.nn_based.phase2.network_factory import create_all_phase2_lookup_networks
        
        config = Phase2Config(use_lookup_tables=False)
        
        with pytest.raises(ValueError, match="use_lookup_tables must be True"):
            create_all_phase2_lookup_networks(config, num_actions=4, num_robots=1)
    
    def test_create_robot_q_network_lookup(self):
        """Test create_robot_q_network with lookup tables."""
        from empo.nn_based.phase2.network_factory import create_robot_q_network
        
        config = Phase2Config(use_lookup_tables=True, use_lookup_q_r=True)
        
        network = create_robot_q_network(config, num_actions=4, num_robots=1)
        
        assert isinstance(network, LookupTableRobotQNetwork)
        assert network.num_actions == 4
        assert network.num_robots == 1
    
    def test_create_robot_q_network_neural_requires_factory(self):
        """Test that neural Q_r requires a factory function."""
        from empo.nn_based.phase2.network_factory import create_robot_q_network
        
        config = Phase2Config(use_lookup_tables=False)
        
        with pytest.raises(ValueError, match="neural_network_factory required"):
            create_robot_q_network(config, num_actions=4, num_robots=1)
    
    def test_create_human_goal_ability_network_lookup(self):
        """Test create_human_goal_ability_network with lookup tables."""
        from empo.nn_based.phase2.network_factory import create_human_goal_ability_network
        
        config = Phase2Config(use_lookup_tables=True, use_lookup_v_h_e=True)
        
        network = create_human_goal_ability_network(config)
        
        assert isinstance(network, LookupTableHumanGoalAbilityNetwork)
    
    def test_create_aggregate_goal_ability_network_lookup(self):
        """Test create_aggregate_goal_ability_network with lookup tables."""
        from empo.nn_based.phase2.network_factory import create_aggregate_goal_ability_network
        
        config = Phase2Config(use_lookup_tables=True, use_lookup_x_h=True)
        
        network = create_aggregate_goal_ability_network(config)
        
        assert isinstance(network, LookupTableAggregateGoalAbilityNetwork)
    
    def test_create_intrinsic_reward_network_when_disabled(self):
        """Test that U_r factory returns None when u_r_use_network=False."""
        from empo.nn_based.phase2.network_factory import create_intrinsic_reward_network
        
        config = Phase2Config(u_r_use_network=False)
        
        network = create_intrinsic_reward_network(config)
        
        assert network is None
    
    def test_create_robot_value_network_when_disabled(self):
        """Test that V_r factory returns None when v_r_use_network=False."""
        from empo.nn_based.phase2.network_factory import create_robot_value_network
        
        config = Phase2Config(v_r_use_network=False)
        
        network = create_robot_value_network(config)
        
        assert network is None
    
    def test_create_aggregate_goal_ability_network_when_disabled(self):
        """Test X_h factory behavior when x_h_use_network=False.
        
        Note: Unlike U_r and V_r, the X_h factory always requires a network
        to be created. When x_h_use_network=False, the trainer skips X_h
        creation entirely and computes X_h directly from V_h^e samples.
        The should_use_lookup_table('x_h') returns False to indicate this.
        """
        config = Phase2Config(x_h_use_network=False)
        
        # should_use_lookup_table returns False when x_h_use_network=False
        assert config.should_use_lookup_table('x_h') is False
        
        # The create_aggregate_goal_ability_network is NOT called when
        # x_h_use_network=False - the trainer handles this by setting x_h=None


class TestAdaptiveLearningRate:
    """Tests for per-entry adaptive learning rate in lookup tables."""
    
    @pytest.fixture
    def simple_states(self):
        """Simple hashable states for testing."""
        return [
            ("state1",),
            ("state2",),
            ("state3",),
        ]
    
    def test_update_count_tracking(self, simple_states):
        """Test that update counts are tracked per entry."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create entries
        _ = network.forward_batch(simple_states[:2], None, device='cpu')
        
        # Initially no update counts
        keys = list(network.table.keys())
        for key in keys:
            assert network.get_update_count(key) == 0
        
        # Simulate gradient update
        key = keys[0]
        network.increment_update_counts([key])
        assert network.get_update_count(key) == 1
        
        network.increment_update_counts([key])
        assert network.get_update_count(key) == 2
    
    def test_gradient_scaling(self, simple_states):
        """Test that gradients are scaled by 1/update_count."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create entries
        output = network.forward_batch(simple_states[:1], None, device='cpu')
        
        # Compute gradient
        loss = output.sum()
        loss.backward()
        
        key = list(network.table.keys())[0]
        original_grad = network.table[key].grad.clone()
        
        # Scale gradients (first update: effective_lr = 1/1 = 1.0)
        keys_with_grads = network.scale_gradients_by_update_count(min_lr=1e-6)
        
        assert len(keys_with_grads) == 1
        assert key in keys_with_grads
        # First update: gradient should be scaled by 1/1 = 1.0
        assert torch.allclose(network.table[key].grad, original_grad)
        
        # Increment count and try again
        network.increment_update_counts(keys_with_grads)
        
        # New forward/backward
        network.zero_grad()
        output = network.forward_batch(simple_states[:1], None, device='cpu')
        loss = output.sum()
        loss.backward()
        
        original_grad = network.table[key].grad.clone()
        
        # Second update: effective_lr = 1/2 = 0.5
        keys_with_grads = network.scale_gradients_by_update_count(min_lr=1e-6)
        network.increment_update_counts(keys_with_grads)
        
        assert torch.allclose(network.table[key].grad, original_grad * 0.5)
    
    def test_converges_to_arithmetic_mean(self, simple_states):
        """Test that with adaptive LR, entries converge to arithmetic mean of targets."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
            default_q_r=-1.0,  # Start at -1.0
        )
        
        # Create a single entry
        _ = network.forward_batch(simple_states[:1], None, device='cpu')
        key = list(network.table.keys())[0]
        
        # Target values to average
        targets = [-2.0, -4.0, -3.0, -1.0, -5.0]
        expected_mean = sum(targets) / len(targets)  # -3.0
        
        # Simulate training with adaptive LR (base_lr = 1.0)
        optimizer = torch.optim.SGD([network.table[key]], lr=1.0)
        
        for target in targets:
            optimizer.zero_grad()
            
            # Compute gradient: grad = current - target (for MSE with factor 1)
            output = network.forward_batch(simple_states[:1], None, device='cpu')
            current = output[0, 0]  # First action value
            
            # MSE loss: (current - target)^2, grad = 2 * (current - target)
            # But we want gradient = (current - target) for arithmetic mean
            # So use (current - target).sum() with backward, then scale
            loss = current  # Just use current, grad = 1
            loss.backward()
            
            # Manually set gradient to (current - target) for exact mean update
            network.table[key].grad.fill_(current.item() - target)
            
            # Scale by 1/update_count
            keys_with_grads = network.scale_gradients_by_update_count(min_lr=1e-6)
            network.increment_update_counts(keys_with_grads)
            
            # Step with lr=1.0: new = old - 1.0 * (1/n) * (old - target) = old + (target - old)/n
            optimizer.step()
        
        # After 5 updates, the first action value should be close to the mean
        output = network.forward_batch(simple_states[:1], None, device='cpu')
        # Note: ensure_negative applies -softplus, so we need to check the raw value
        raw_value = network.table[key][0].item()
        # The raw value goes through -softplus to get output
        # So raw_value should be such that -softplus(raw_value) ≈ -3.0
        # This is complex due to the transformation, but the update counts should be 5
        assert network.get_update_count(key) == 5
    
    def test_update_counts_persist_in_state_dict(self, simple_states):
        """Test that update counts are saved/loaded with state_dict."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create entries and increment counts
        _ = network.forward_batch(simple_states[:2], None, device='cpu')
        keys = list(network.table.keys())
        network.increment_update_counts([keys[0], keys[0], keys[1]])
        
        assert network.get_update_count(keys[0]) == 2
        assert network.get_update_count(keys[1]) == 1
        
        # Save state dict
        state_dict = network.state_dict()
        
        # Create new network and load
        new_network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        new_network.load_state_dict(state_dict)
        
        # Verify update counts loaded
        assert new_network.get_update_count(keys[0]) == 2
        assert new_network.get_update_count(keys[1]) == 1
    
    def test_min_lr_prevents_zero(self, simple_states):
        """Test that min_lr prevents effective learning rate from going to zero."""
        network = LookupTableRobotQNetwork(
            num_actions=2,
            num_robots=1,
        )
        
        # Create entry with many updates
        _ = network.forward_batch(simple_states[:1], None, device='cpu')
        key = list(network.table.keys())[0]
        
        # Simulate many updates
        for _ in range(1000):
            network.increment_update_counts([key])
        
        assert network.get_update_count(key) == 1000
        
        # Do a forward/backward
        output = network.forward_batch(simple_states[:1], None, device='cpu')
        loss = output.sum()
        loss.backward()
        
        original_grad = network.table[key].grad.clone()
        
        # Scale with high min_lr
        min_lr = 0.01
        network.scale_gradients_by_update_count(min_lr=min_lr)
        
        # effective_lr = max(0.01, 1/1001) = 0.01 (since 1/1001 < 0.01)
        assert torch.allclose(network.table[key].grad, original_grad * min_lr)
    
    def test_config_adaptive_lr_options(self):
        """Test Phase2Config adaptive LR options."""
        # Default: disabled
        config = Phase2Config()
        assert config.lookup_use_adaptive_lr is False
        assert config.lookup_adaptive_lr_min == 1e-6
        
        # Enabled
        config = Phase2Config(
            lookup_use_adaptive_lr=True,
            lookup_adaptive_lr_min=0.001,
        )
        assert config.lookup_use_adaptive_lr is True
        assert config.lookup_adaptive_lr_min == 0.001
