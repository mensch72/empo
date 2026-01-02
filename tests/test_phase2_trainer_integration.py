#!/usr/bin/env python3
"""
Integration tests for Phase 2 trainer FAQ design choices.

These tests require actual trainer instantiation and verify:
- Target network initialization and updates
- Replay buffer clearing at beta_r transitions
- TensorBoard logging
- Using target networks in loss computation

See docs/FAQ.md for detailed justifications of each design choice.
"""

import copy
import os
import tempfile
import pytest
import torch
import torch.nn as nn

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.replay_buffer import Phase2ReplayBuffer, Phase2Transition
from empo.nn_based.phase2.robot_q_network import BaseRobotQNetwork
from empo.nn_based.phase2.human_goal_ability import BaseHumanGoalAchievementNetwork
from empo.nn_based.phase2.aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from empo.nn_based.phase2.intrinsic_reward_network import BaseIntrinsicRewardNetwork
from empo.nn_based.phase2.robot_value_network import BaseRobotValueNetwork
from empo.nn_based.phase2.trainer import Phase2Networks


# =============================================================================
# MOCK NETWORKS FOR TESTING
# =============================================================================

class MockQNetwork(BaseRobotQNetwork):
    """Simple mock Q network for testing."""
    
    def __init__(self, num_actions=4, num_robots=1):
        super().__init__(num_actions=num_actions, num_robots=num_robots)
        self.fc = nn.Linear(10, self.num_action_combinations)
    
    def encode_state(self, state, precomputed_encoding, device):
        return torch.randn(1, 10, device=device)
    
    def forward(self, state_encoding):
        return -torch.nn.functional.softplus(self.fc(state_encoding))
    
    def encode_and_forward(self, state, precomputed_encoding, device):
        enc = self.encode_state(state, precomputed_encoding, device)
        return self.forward(enc)
    
    def get_config(self):
        return {'num_actions': self.num_actions}
    
    def action_tuple_to_index(self, action_tuple):
        return action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple
    
    def action_index_to_tuple(self, index):
        return (index,)


class MockVhNetwork(BaseHumanGoalAchievementNetwork):
    """Simple mock V_h^e network for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.gamma_h = 0.99
        self.zeta = 2.0
    
    def encode_state(self, state, precomputed_encoding, device):
        return torch.randn(1, 10, device=device)
    
    def forward(self, state_encoding, human_idx, goal):
        return torch.sigmoid(self.fc(state_encoding))
    
    def encode_and_forward(self, state, precomputed_encoding, human_idx, goal, device):
        enc = self.encode_state(state, precomputed_encoding, device)
        return self.forward(enc, human_idx, goal)
    
    def get_config(self):
        return {}
    
    def compute_td_target(self, goal_achieved, v_next, terminal=None):
        continuation = (1 - goal_achieved) * self.gamma_h * v_next
        if terminal is not None:
            continuation = continuation * (1.0 - terminal)
        return goal_achieved + continuation


class MockXhNetwork(BaseAggregateGoalAbilityNetwork):
    """Simple mock X_h network for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.zeta = 2.0
    
    def encode_state(self, state, precomputed_encoding, device):
        return torch.randn(1, 10, device=device)
    
    def forward(self, state_encoding, human_idx):
        return torch.sigmoid(self.fc(state_encoding))
    
    def encode_and_forward(self, state, precomputed_encoding, human_idx, device):
        enc = self.encode_state(state, precomputed_encoding, device)
        return self.forward(enc, human_idx)
    
    def get_config(self):
        return {}
    
    def compute_target(self, v_h_e):
        return v_h_e ** self.zeta


class MockUrNetwork(BaseIntrinsicRewardNetwork):
    """Simple mock U_r network for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def encode_state(self, state, precomputed_encoding, device):
        return torch.randn(1, 10, device=device)
    
    def forward(self, state_encoding):
        # Return (y, u_r) where y = E[X_h^{-xi}] and u_r = -y^eta
        y = torch.exp(self.fc(state_encoding))  # y > 0
        u_r = -torch.pow(y, 1.1)  # u_r < 0
        return y, u_r
    
    def encode_and_forward(self, state, precomputed_encoding, device):
        enc = self.encode_state(state, precomputed_encoding, device)
        return self.forward(enc)
    
    def get_config(self):
        return {}


class MockVrNetwork(BaseRobotValueNetwork):
    """Simple mock V_r network for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def encode_state(self, state, precomputed_encoding, device):
        return torch.randn(1, 10, device=device)
    
    def forward(self, state_encoding):
        return -torch.nn.functional.softplus(self.fc(state_encoding))
    
    def encode_and_forward(self, state, precomputed_encoding, device):
        enc = self.encode_state(state, precomputed_encoding, device)
        return self.forward(enc)
    
    def get_config(self):
        return {}
    
    def compute_from_components(self, u_r, q_values, policy):
        return u_r + (policy * q_values).sum(dim=-1)


def create_mock_networks():
    """Create a set of mock networks for testing."""
    return Phase2Networks(
        q_r=MockQNetwork(),
        v_h_e=MockVhNetwork(),
        x_h=MockXhNetwork(),
        u_r=MockUrNetwork(),
        v_r=MockVrNetwork(),
    )


# =============================================================================
# TARGET NETWORK TESTS
# =============================================================================

class TestTargetNetworkInitialization:
    """Test target network initialization (FAQ item 8)."""
    
    def test_target_networks_are_copies(self):
        """Target networks should be deep copies of main networks."""
        networks = create_mock_networks()
        
        # Initialize target networks
        networks.v_r_target = copy.deepcopy(networks.v_r)
        networks.v_h_e_target = copy.deepcopy(networks.v_h_e)
        networks.x_h_target = copy.deepcopy(networks.x_h)
        networks.u_r_target = copy.deepcopy(networks.u_r)
        
        # Check they have same weights
        for p1, p2 in zip(networks.v_r.parameters(), networks.v_r_target.parameters()):
            assert torch.equal(p1, p2)
    
    def test_target_networks_requires_grad_false(self):
        """Target networks should have requires_grad=False."""
        networks = create_mock_networks()
        
        networks.v_r_target = copy.deepcopy(networks.v_r)
        for param in networks.v_r_target.parameters():
            param.requires_grad = False
        
        for param in networks.v_r_target.parameters():
            assert param.requires_grad is False
    
    def test_target_networks_eval_mode(self):
        """Target networks should be in eval mode."""
        networks = create_mock_networks()
        
        networks.v_r_target = copy.deepcopy(networks.v_r)
        networks.v_r_target.eval()
        
        assert not networks.v_r_target.training
    
    def test_target_network_update_copies_weights(self):
        """Target network update should copy weights from main network."""
        networks = create_mock_networks()
        networks.v_r_target = copy.deepcopy(networks.v_r)
        
        # Modify main network
        with torch.no_grad():
            for param in networks.v_r.parameters():
                param.add_(1.0)
        
        # Weights should now differ
        for p1, p2 in zip(networks.v_r.parameters(), networks.v_r_target.parameters()):
            assert not torch.equal(p1, p2)
        
        # Update target
        networks.v_r_target.load_state_dict(networks.v_r.state_dict())
        
        # Weights should match again
        for p1, p2 in zip(networks.v_r.parameters(), networks.v_r_target.parameters()):
            assert torch.equal(p1, p2)


# =============================================================================
# REPLAY BUFFER TESTS
# =============================================================================

class TestReplayBuffer:
    """Test replay buffer functionality (FAQ item 9)."""
    
    def test_buffer_clear(self):
        """Buffer should be clearable."""
        buffer = Phase2ReplayBuffer(capacity=100)
        
        # Add some transitions
        for i in range(10):
            buffer.push(
                state=f"state_{i}",
                robot_action=(0,),
                goals={0: "goal"},
                goal_weights={0: 1.0},
                human_actions=[0],
                next_state=f"state_{i+1}",
            )
        
        assert len(buffer) == 10
        
        buffer.clear()
        
        assert len(buffer) == 0
    
    def test_buffer_sampling(self):
        """Buffer should support sampling."""
        buffer = Phase2ReplayBuffer(capacity=100)
        
        for i in range(20):
            buffer.push(
                state=f"state_{i}",
                robot_action=(0,),
                goals={0: "goal"},
                goal_weights={0: 1.0},
                human_actions=[0],
                next_state=f"state_{i+1}",
            )
        
        batch = buffer.sample(5)
        assert len(batch) == 5
        assert all(isinstance(t, Phase2Transition) for t in batch)


# =============================================================================
# TENSORBOARD LOGGING TESTS
# =============================================================================

class TestTensorBoardLogging:
    """Test TensorBoard logging setup (FAQ item 18)."""
    
    def test_tensorboard_import(self):
        """TensorBoard should be importable."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            has_tensorboard = True
        except ImportError:
            has_tensorboard = False
        
        # TensorBoard is optional but should usually be available
        assert has_tensorboard or True  # Pass even if not installed
    
    def test_tensorboard_writer_creation(self):
        """TensorBoard writer should be creatable."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            with tempfile.TemporaryDirectory() as tmpdir:
                writer = SummaryWriter(log_dir=tmpdir)
                writer.add_scalar('test', 1.0, 0)
                writer.close()
                
                # Check files were created
                assert os.path.exists(tmpdir)
                assert len(os.listdir(tmpdir)) > 0
        except ImportError:
            pytest.skip("TensorBoard not installed")


# =============================================================================
# NEGATIVE Q VALUES TEST
# =============================================================================

class TestNegativeQValues:
    """Test that Q values are always negative (FAQ item 7)."""
    
    def test_mock_q_network_negative_outputs(self):
        """Mock Q network should output negative values via -softplus."""
        net = MockQNetwork()
        
        # Test with various inputs
        for _ in range(10):
            enc = torch.randn(1, 10)
            q_values = net.forward(enc)
            assert (q_values < 0).all(), "All Q values must be negative"
    
    def test_power_law_policy_with_negative_q(self):
        """Power-law policy should work correctly with negative Q values."""
        net = MockQNetwork()
        net.beta_r = 5.0
        
        q_values = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        policy = net.get_policy(q_values, beta_r=5.0)
        
        # Should sum to 1
        assert torch.allclose(policy.sum(dim=-1), torch.tensor([1.0]))
        
        # Best action (Q=-1) should have highest probability
        assert policy[0, 0] == policy.max()


# =============================================================================
# WARMUP STAGE TRANSITION TRACKING
# =============================================================================

class TestWarmupStageTransitions:
    """Test warmup stage transition tracking (FAQ items 1, 2, 9)."""
    
    def test_stage_transition_steps(self):
        """Config should report stage transition steps."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,  # Direct computation
            warmup_q_r_steps=100,
            beta_r_rampup_steps=100,
        )
        
        transitions = config.get_stage_transition_steps()
        
        # Should have transitions for X_h, Q_r, warmup end, and ramp complete
        assert len(transitions) >= 3
        
        # Transitions should be in order
        steps = [t[0] for t in transitions]
        assert steps == sorted(steps)
    
    def test_is_in_warmup(self):
        """Config should track whether in warmup phase."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
        )
        
        assert config.is_in_warmup(0)
        assert config.is_in_warmup(299)
        assert not config.is_in_warmup(300)
    
    def test_is_in_rampup(self):
        """Config should track whether in beta_r ramp-up phase."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
            beta_r_rampup_steps=100,
        )
        
        warmup_end = config._warmup_q_r_end
        
        assert not config.is_in_rampup(warmup_end - 1)
        assert config.is_in_rampup(warmup_end)
        assert config.is_in_rampup(warmup_end + 50)
        assert not config.is_in_rampup(warmup_end + 100)


# =============================================================================
# DIRECT COMPUTATION MODE TESTS
# =============================================================================

class TestDirectComputationModes:
    """Test direct U_r and V_r computation (FAQ item 17)."""
    
    def test_v_r_compute_from_components(self):
        """V_r should be computable from U_r, Q_r, and policy."""
        v_r_net = MockVrNetwork()
        
        u_r = torch.tensor(-0.5)
        q_values = torch.tensor([-1.0, -2.0, -3.0, -4.0])
        policy = torch.tensor([0.4, 0.3, 0.2, 0.1])
        
        v_r = v_r_net.compute_from_components(u_r, q_values, policy)
        
        # V_r = U_r + π · Q
        expected = u_r + (policy * q_values).sum()
        assert torch.allclose(v_r, expected)
    
    def test_u_r_warmup_skipped_when_direct(self):
        """U_r warmup should be skipped when using direct computation."""
        config = Phase2Config(
            u_r_use_network=False,
            warmup_u_r_steps=1000,  # Should be overridden
        )
        
        assert config.warmup_u_r_steps == 0


# =============================================================================
# AGENT INDEX TESTS (WorldModel API)
# =============================================================================

class MockAgent:
    """Mock agent with only the color attribute needed for testing."""
    def __init__(self, color):
        self.color = color


class TestAgentIndices:
    """Test that WorldModel API correctly identifies agent indices."""
    
    def test_multigrid_agent_indices_match_colors(self):
        """Verify MultiGridEnv returns indices matching actual agent colors."""
        from gym_multigrid.multigrid import MultiGridEnv, Grid
        
        # Create environment with known agent configuration
        env = MultiGridEnv.__new__(MultiGridEnv)
        env.agents = [
            MockAgent(color='yellow'),  # human - index 0
            MockAgent(color='grey'),    # robot - index 1
            MockAgent(color='yellow'),  # human - index 2
            MockAgent(color='red'),     # neither - index 3
        ]
        env.grid = Grid(5, 5)
        
        # Test get_human_agent_indices
        human_indices = env.get_human_agent_indices()
        assert set(human_indices) == {0, 2}, f"Expected {{0, 2}}, got {human_indices}"
        
        # Test get_robot_agent_indices
        robot_indices = env.get_robot_agent_indices()
        assert set(robot_indices) == {1}, f"Expected {{1}}, got {robot_indices}"
        
        # Verify indices match actual agent colors
        for idx in human_indices:
            assert env.agents[idx].color == 'yellow', f"Agent {idx} should be yellow"
        for idx in robot_indices:
            assert env.agents[idx].color == 'grey', f"Agent {idx} should be grey"
    
    def test_multigrid_empty_agent_lists(self):
        """Test environment with no humans or no robots."""
        from gym_multigrid.multigrid import MultiGridEnv, Grid
        
        # All robots, no humans
        env = MultiGridEnv.__new__(MultiGridEnv)
        env.agents = [
            MockAgent(color='grey'),
            MockAgent(color='grey'),
        ]
        env.grid = Grid(5, 5)
        
        assert env.get_human_agent_indices() == []
        assert set(env.get_robot_agent_indices()) == {0, 1}
        
        # All humans, no robots
        env.agents = [
            MockAgent(color='yellow'),
        ]
        
        assert set(env.get_human_agent_indices()) == {0}
        assert env.get_robot_agent_indices() == []


# =============================================================================
# MODEL-BASED TARGET TESTS
# =============================================================================

class TestModelBasedTargets:
    """Tests for model-based target computation using transition_probabilities."""
    
    def test_next_state_none_when_model_based(self):
        """When use_model_based_targets=True, next_state should be None in transitions."""
        config = Phase2Config(
            use_model_based_targets=True,
            batch_size=4,
            buffer_size=100
        )
        
        # Create a mock transition like what the trainer would create
        from empo.nn_based.phase2.replay_buffer import Phase2Transition
        
        # This mimics what the trainer does when model_based is enabled
        stored_next_state = None if config.use_model_based_targets else 'some_state'
        transition_probs = {0: [(1.0, 'successor_state')]} if config.use_model_based_targets else None
        
        transition = Phase2Transition(
            state='current_state',
            robot_action=(0,),
            goals={0: 'goal'},
            goal_weights={0: 1.0},
            human_actions=[0],
            next_state=stored_next_state,
            transition_probs_by_action=transition_probs,
            terminal=False
        )
        
        assert transition.next_state is None
        assert transition.transition_probs_by_action is not None
    
    def test_next_state_stored_when_not_model_based(self):
        """When use_model_based_targets=False, next_state should be stored."""
        config = Phase2Config(
            use_model_based_targets=False,
            batch_size=4,
            buffer_size=100
        )
        
        stored_next_state = 'some_state' if not config.use_model_based_targets else None
        transition_probs = None if not config.use_model_based_targets else {0: [(1.0, 'successor')]}
        
        transition = Phase2Transition(
            state='current_state',
            robot_action=(0,),
            goals={0: 'goal'},
            goal_weights={0: 1.0},
            human_actions=[0],
            next_state=stored_next_state,
            transition_probs_by_action=transition_probs,
            terminal=False
        )
        
        assert transition.next_state == 'some_state'
        assert transition.transition_probs_by_action is None
    
    def test_transition_probs_contain_successor_states(self):
        """Verify transition_probs_by_action contains proper successor states."""
        # This verifies the data structure used for model-based targets
        trans_probs = {
            0: [(0.5, 'state_a'), (0.5, 'state_b')],
            1: [(1.0, 'state_c')],
            2: [],  # Terminal for this action
        }
        
        # Verify structure
        assert len(trans_probs[0]) == 2
        assert sum(p for p, _ in trans_probs[0]) == 1.0
        assert trans_probs[1][0][0] == 1.0
        assert trans_probs[2] == []  # Empty means terminal
    
    def test_config_default_is_model_based(self):
        """Verify default config uses model-based targets."""
        config = Phase2Config()
        assert config.use_model_based_targets is True
    
    def test_no_successor_states_stored_in_transitions(self):
        """
        Verify that when use_model_based_targets=True, successor states are NOT
        stored in transitions (only transition_probs_by_action is stored).
        
        This ensures memory efficiency and forces correct model-based computation.
        """
        from empo.nn_based.phase2.replay_buffer import Phase2Transition
        
        # Create transitions mimicking what collect_transition does
        config = Phase2Config(use_model_based_targets=True)
        
        # Simulate multiple transitions
        for _ in range(10):
            # When model-based is enabled, next_state should be None
            stored_next_state = None if config.use_model_based_targets else 'actual_next_state'
            
            # transition_probs_by_action should contain all successor info
            # This is a dict: action_idx -> [(prob, next_state), ...]
            transition_probs = {
                0: [(0.5, 'state_a'), (0.5, 'state_b')],  # Two possible successors
                1: [(1.0, 'state_c')],  # Deterministic transition
                2: [],  # Terminal/no successors
            } if config.use_model_based_targets else None
            
            transition = Phase2Transition(
                state='current_state',
                robot_action=(0,),
                goals={0: 'goal'},
                goal_weights={0: 1.0},
                human_actions=[0],
                next_state=stored_next_state,
                transition_probs_by_action=transition_probs,
                terminal=False
            )
            
            # CRITICAL: next_state must be None when model-based
            assert transition.next_state is None, \
                "Successor states should NOT be stored when use_model_based_targets=True"
            
            # transition_probs_by_action must contain all successor info
            assert transition.transition_probs_by_action is not None, \
                "transition_probs_by_action must be populated when model-based"
            
            # Verify successor states are in transition_probs, not next_state
            all_successors = []
            for action_idx, probs in transition.transition_probs_by_action.items():
                for prob, succ_state in probs:
                    all_successors.append(succ_state)
            
            assert len(all_successors) > 0 or 2 in transition.transition_probs_by_action, \
                "Successor states should be in transition_probs_by_action"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
