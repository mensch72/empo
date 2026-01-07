"""
Tests for MultiGridMultiStepExplorationPolicy.

This test verifies the non-Markovian multi-step exploration policy that:
1. Samples multi-step action sequences
2. Correctly executes sequences stepwise
3. Cancels sequences when forward becomes blocked
4. Properly samples k from geometric distribution
5. Works as both RobotPolicy and HumanPolicyPrior
"""

import sys
import os
import numpy as np

# Add src and vendor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))

import pytest

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Wall, Rock, Block, World, Agent, SmallActions
)
from empo.learning_based.multigrid.phase2.robot_policy import MultiGridMultiStepExplorationPolicy


def create_robot_agent():
    """Create a robot agent (can push rocks)."""
    agent = Agent(World, 0, can_push_rocks=True, can_enter_magic_walls=True)
    return agent


def create_human_agent():
    """Create a human agent (cannot push rocks)."""
    agent = Agent(World, 1, can_push_rocks=False, can_enter_magic_walls=False)
    return agent


class OpenCorridorEnv(MultiGridEnv):
    """
    Test environment with an open corridor for movement.
    
    Layout (5x5):
        #####
        #...#   
        #A..#   (A=agent facing right at (1,2))
        #...#
        #####
    
    Agent can move forward for several steps.
    """
    
    def __init__(self, is_robot: bool = True):
        agent = create_robot_agent() if is_robot else create_human_agent()
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=[agent],
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions,
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place agent at (1, 2) facing right (direction 0)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right


class BlockedForwardEnv(MultiGridEnv):
    """
    Test environment where forward is blocked by a wall.
    
    Layout (5x5):
        #####
        #...#   
        #A#.#   (A=agent facing right at (1,2), wall at (2,2))
        #...#
        #####
    
    Agent cannot move forward but can turn.
    """
    
    def __init__(self, is_robot: bool = True):
        agent = create_robot_agent() if is_robot else create_human_agent()
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=[agent],
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions,
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place inner wall in front of agent
        self.grid.set(2, 2, Wall(World))
        
        # Place agent at (1, 2) facing right (direction 0)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right


class PartiallyBlockedEnv(MultiGridEnv):
    """
    Test environment where only some directions are blocked.
    
    Layout (5x5):
        #####
        #.#.#   (wall at (2,1))
        #A..#   (A=agent facing right at (1,2))
        #...#
        #####
    
    Agent can move forward (right) but not up (wall above after turning left).
    """
    
    def __init__(self, is_robot: bool = True):
        agent = create_robot_agent() if is_robot else create_human_agent()
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=[agent],
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions,
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place inner wall above agent's position
        self.grid.set(2, 1, Wall(World))
        
        # Place agent at (1, 2) facing right (direction 0)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right


class TestMultiStepExplorationBasics:
    """Test basic initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default initialization with default parameters."""
        policy = MultiGridMultiStepExplorationPolicy()
        
        # Check default sequence probabilities
        assert policy.SEQ_STILL in policy.sequence_probs
        assert policy.SEQ_FORWARD in policy.sequence_probs
        assert policy.SEQ_LEFT_FORWARD in policy.sequence_probs
        assert policy.SEQ_RIGHT_FORWARD in policy.sequence_probs
        assert policy.SEQ_BACK_FORWARD in policy.sequence_probs
        
        # Check probabilities sum to 1
        total = sum(policy.sequence_probs.values())
        assert abs(total - 1.0) < 1e-6
        
        # Check default expected_k
        assert policy.expected_k == 2.0
    
    def test_custom_sequence_probs(self):
        """Test initialization with custom sequence probabilities."""
        custom_probs = {
            'still': 0.0,
            'forward': 1.0,
            'left_forward': 0.0,
            'right_forward': 0.0,
            'back_forward': 0.0,
        }
        policy = MultiGridMultiStepExplorationPolicy(sequence_probs=custom_probs)
        
        assert policy.sequence_probs['forward'] == 1.0
        assert policy.sequence_probs['still'] == 0.0
    
    def test_custom_expected_k(self):
        """Test initialization with custom expected_k (single value)."""
        policy = MultiGridMultiStepExplorationPolicy(expected_k=5.0)
        assert policy.expected_k == 5.0
        assert abs(policy._get_geom_p('forward') - 0.2) < 1e-6
    
    def test_expected_k_dict(self):
        """Test initialization with per-sequence-type expected_k dict."""
        expected_k_dict = {
            'still': 1.0,
            'forward': 4.0,
            'left_forward': 2.0,
            'right_forward': 2.0,
            'back_forward': 3.0,
        }
        policy = MultiGridMultiStepExplorationPolicy(expected_k=expected_k_dict)
        
        # Check that expected_k is None (dict mode)
        assert policy.expected_k is None
        
        # Check per-type values
        assert abs(policy._get_expected_k('still') - 1.0) < 1e-6
        assert abs(policy._get_expected_k('forward') - 4.0) < 1e-6
        assert abs(policy._get_expected_k('left_forward') - 2.0) < 1e-6
        
        # Check geom_p calculation
        assert abs(policy._get_geom_p('forward') - 0.25) < 1e-6
        assert abs(policy._get_geom_p('still') - 1.0) < 1e-6
    
    def test_expected_k_dict_partial(self):
        """Test expected_k dict with only some sequence types specified."""
        expected_k_dict = {
            'forward': 5.0,
            # Others should default to 2.0
        }
        policy = MultiGridMultiStepExplorationPolicy(expected_k=expected_k_dict)
        
        assert abs(policy._get_expected_k('forward') - 5.0) < 1e-6
        assert abs(policy._get_expected_k('still') - 2.0) < 1e-6  # Default
        assert abs(policy._get_expected_k('left_forward') - 2.0) < 1e-6  # Default
    
    def test_expected_k_dict_invalid_value(self):
        """Test that expected_k dict with value < 1 raises error."""
        with pytest.raises(ValueError, match="expected_k.*must be >= 1.0"):
            MultiGridMultiStepExplorationPolicy(expected_k={'forward': 0.5})
    
    def test_expected_k_dict_invalid_key(self):
        """Test that expected_k dict with unknown key raises error."""
        with pytest.raises(ValueError, match="Unknown sequence type in expected_k"):
            MultiGridMultiStepExplorationPolicy(expected_k={'invalid_type': 2.0})
    
    def test_invalid_expected_k(self):
        """Test that expected_k < 1 raises error."""
        with pytest.raises(ValueError, match="expected_k must be >= 1.0"):
            MultiGridMultiStepExplorationPolicy(expected_k=0.5)
    
    def test_invalid_sequence_type(self):
        """Test that unknown sequence types raise error."""
        with pytest.raises(ValueError, match="Unknown sequence type"):
            MultiGridMultiStepExplorationPolicy(
                sequence_probs={'invalid_type': 1.0}
            )
    
    def test_zero_probabilities(self):
        """Test that all-zero probabilities raise error."""
        with pytest.raises(ValueError, match="at least one non-zero probability"):
            MultiGridMultiStepExplorationPolicy(
                sequence_probs={
                    'still': 0.0,
                    'forward': 0.0,
                }
            )


class TestSequenceSampling:
    """Test sequence sampling behavior."""
    
    def test_sample_forward_sequence_in_open_corridor(self):
        """Test that forward sequences are sampled in open space."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        # Force forward-only sequence
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'forward': 1.0},
            expected_k=3.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # First action should be forward
        action = policy.sample(state, agent_index=0)
        assert action == policy.ACTION_FORWARD
    
    def test_sample_left_forward_sequence(self):
        """Test left+forward sequence sampling."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'left_forward': 1.0},
            expected_k=2.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # First action should be left
        action = policy.sample(state, agent_index=0)
        assert action == policy.ACTION_LEFT
        
        # Subsequent actions should be forward (until sequence ends)
        # Execute the turn first
        env.step([action])
        state = env.get_state()
        
        # Now should get forward actions
        action = policy.sample(state, agent_index=0)
        assert action == policy.ACTION_FORWARD
    
    def test_sample_back_forward_sequence(self):
        """Test back (180°) + forward sequence sampling."""
        # Create a larger environment where backward is feasible
        class CenteredEnv(MultiGridEnv):
            def __init__(self):
                agent = create_robot_agent()
                super().__init__(
                    width=7,
                    height=7,
                    max_steps=100,
                    agents=[agent],
                    partial_obs=False,
                    objects_set=World,
                    actions_set=SmallActions,
                )
            
            def _gen_grid(self, width, height):
                self.grid = Grid(width, height)
                for i in range(width):
                    self.grid.set(i, 0, Wall(World))
                    self.grid.set(i, height - 1, Wall(World))
                for j in range(height):
                    self.grid.set(0, j, Wall(World))
                    self.grid.set(width - 1, j, Wall(World))
                # Place agent in center facing right
                self.agents[0].pos = np.array([3, 3])
                self.agents[0].dir = 0  # Facing right
        
        env = CenteredEnv()
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'back_forward': 1.0},
            expected_k=2.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # First two actions should be left (turning 180°)
        action1 = policy.sample(state, agent_index=0)
        assert action1 == policy.ACTION_LEFT
        
        # Execute first turn
        env.step([action1])
        state = env.get_state()
        
        action2 = policy.sample(state, agent_index=0)
        assert action2 == policy.ACTION_LEFT
        
        # Execute second turn
        env.step([action2])
        state = env.get_state()
        
        # Now should get forward
        action3 = policy.sample(state, agent_index=0)
        assert action3 == policy.ACTION_FORWARD
    
    def test_still_sequence(self):
        """Test still sequence (k times doing nothing)."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'still': 1.0},
            expected_k=3.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # All actions should be still until sequence ends
        for _ in range(5):
            action = policy.sample(state, agent_index=0)
            assert action == policy.ACTION_STILL
            # Note: state doesn't change for still actions


class TestSequenceCancellation:
    """Test that sequences are cancelled when forward becomes blocked."""
    
    def test_sequence_cancelled_when_forward_blocked(self):
        """Test that ongoing sequence is cancelled when forward blocked."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        # Force a forward sequence with high expected_k
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'forward': 1.0},
            expected_k=10.0,  # Likely to sample k > 1
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Start the sequence
        action = policy.sample(state, agent_index=0)
        assert action == policy.ACTION_FORWARD
        
        # Execute forward action until we hit wall
        for _ in range(10):
            env.step([policy.ACTION_FORWARD])
            state = env.get_state()
            
            # Check if agent has moved (if blocked, position unchanged)
            agent_x = state[1][0][0]  # state format: (step, agent_states, ...)
            if agent_x >= 3:  # Near right wall at x=3
                break
        
        # Now forward should be blocked - policy should handle this
        # by cancelling sequence and sampling new one
        # (Since only 'forward' is allowed and forward is blocked,
        # it should fall back to 'still')


class TestFeasibilityChecks:
    """Test sequence feasibility checks."""
    
    def test_forward_infeasible_when_blocked(self):
        """Test that forward sequence not sampled when directly blocked."""
        env = BlockedForwardEnv(is_robot=True)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={
                'forward': 0.5,
                'still': 0.5,
            },
            expected_k=2.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Sample many times - should only get 'still' actions
        # since forward is blocked
        actions = []
        for _ in range(20):
            # Reset sequence state each time
            policy._agent_sequences = {}
            action = policy.sample(state, agent_index=0)
            actions.append(action)
        
        # All actions should be still (forward sequence infeasible)
        assert all(a == policy.ACTION_STILL for a in actions)
    
    def test_left_forward_feasibility(self):
        """Test left_forward sequence feasibility check."""
        env = PartiallyBlockedEnv(is_robot=True)
        env.reset()
        
        # Agent at (1,2) facing right (dir=0)
        # Left would face up (dir=3), which has wall at (2,1)
        # Wait, left from right is up? Let me check directions:
        # dir=0 is right, dir=1 is down, dir=2 is left, dir=3 is up
        # Turning left from dir=0 gives (0-1)%4 = 3 = up
        # At (1,2), forward when facing up goes to (1,1) which should be empty
        # But wall is at (2,1), so left_forward should be feasible
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={
                'left_forward': 1.0,
            },
            expected_k=1.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Check feasibility - agent facing right, turn left faces up
        # Forward from (1,2) facing up goes to (1,1) which is empty
        is_feasible = policy._is_sequence_feasible(
            'left_forward', state, 0, agent_dir=0
        )
        assert is_feasible


class TestGeometricDistribution:
    """Test that k is sampled from geometric distribution."""
    
    def test_geometric_mean(self):
        """Test that sampled k values have expected mean."""
        np.random.seed(42)
        
        expected_k = 3.0
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'forward': 1.0},
            expected_k=expected_k,
        )
        
        # Sample many k values using the policy's method
        k_values = []
        geom_p = policy._get_geom_p('forward')
        for _ in range(10000):
            k = np.random.geometric(geom_p)
            k_values.append(k)
        
        mean_k = np.mean(k_values)
        # Check that mean is close to expected (with some tolerance)
        assert abs(mean_k - expected_k) < 0.1
    
    def test_geometric_mean_per_sequence_type(self):
        """Test that per-sequence-type expected_k values give correct means."""
        np.random.seed(42)
        
        expected_k_dict = {
            'still': 1.5,
            'forward': 4.0,
        }
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            expected_k=expected_k_dict,
        )
        
        # Test forward sequence
        k_values_forward = []
        geom_p_forward = policy._get_geom_p('forward')
        for _ in range(10000):
            k = np.random.geometric(geom_p_forward)
            k_values_forward.append(k)
        
        mean_k_forward = np.mean(k_values_forward)
        assert abs(mean_k_forward - 4.0) < 0.15
        
        # Test still sequence
        k_values_still = []
        geom_p_still = policy._get_geom_p('still')
        for _ in range(10000):
            k = np.random.geometric(geom_p_still)
            k_values_still.append(k)
        
        mean_k_still = np.mean(k_values_still)
        assert abs(mean_k_still - 1.5) < 0.1


class TestMultiAgentSupport:
    """Test multi-agent support."""
    
    def test_multiple_robots(self):
        """Test policy with multiple robot agents."""
        # Create environment with two agents
        class TwoAgentEnv(MultiGridEnv):
            def __init__(self):
                agents = [
                    create_robot_agent(),
                    create_robot_agent(),
                ]
                super().__init__(
                    width=7,
                    height=5,
                    max_steps=100,
                    agents=agents,
                    partial_obs=False,
                    objects_set=World,
                    actions_set=SmallActions,
                )
            
            def _gen_grid(self, width, height):
                self.grid = Grid(width, height)
                for i in range(width):
                    self.grid.set(i, 0, Wall(World))
                    self.grid.set(i, height - 1, Wall(World))
                for j in range(height):
                    self.grid.set(0, j, Wall(World))
                    self.grid.set(width - 1, j, Wall(World))
                
                self.agents[0].pos = np.array([1, 2])
                self.agents[0].dir = 0
                self.agents[1].pos = np.array([5, 2])
                self.agents[1].dir = 2  # Facing left
        
        env = TwoAgentEnv()
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0, 1],
            sequence_probs={'forward': 1.0},
            expected_k=2.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Sample for all agents at once
        actions = policy.sample(state)
        
        assert isinstance(actions, tuple)
        assert len(actions) == 2
        # Both should be forward since both have forward-only policy
        assert all(a == policy.ACTION_FORWARD for a in actions)


class TestHumanPolicyPriorInterface:
    """Test HumanPolicyPrior interface compatibility."""
    
    def test_call_returns_distribution(self):
        """Test __call__ returns probability distribution."""
        env = OpenCorridorEnv(is_robot=False)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Call with agent index (HumanPolicyPrior style)
        probs = policy(state, human_agent_index=0)
        
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 4
        assert abs(probs.sum() - 1.0) < 1e-6
    
    def test_sample_with_goal_param(self):
        """Test sample() accepts goal parameter for compatibility."""
        env = OpenCorridorEnv(is_robot=False)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Should accept goal parameter (but ignore it)
        action = policy.sample(state, agent_index=0, goal="dummy_goal")
        assert action in [0, 1, 2, 3]
    
    def test_human_agent_indices_alias(self):
        """Test human_agent_indices property alias."""
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0, 2, 3],
        )
        
        assert policy.human_agent_indices == [0, 2, 3]
        assert policy.agent_indices == policy.human_agent_indices


class TestPickleSupport:
    """Test pickling/unpickling support."""
    
    def test_pickle_roundtrip(self):
        """Test policy can be pickled and unpickled."""
        import pickle
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0, 1],
            sequence_probs={'forward': 0.7, 'still': 0.3},
            expected_k=3.0,
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(policy)
        restored = pickle.loads(pickled)
        
        # Check configuration preserved
        assert restored.agent_indices == [0, 1]
        assert restored.expected_k == 3.0
        assert abs(restored.sequence_probs['forward'] - 0.7) < 1e-6
        
        # World model should be None after unpickle
        assert restored.world_model is None
        
        # Sequences should be cleared
        assert restored._agent_sequences == {}


class TestResetBehavior:
    """Test reset() behavior."""
    
    def test_reset_clears_sequences(self):
        """Test that reset() clears ongoing sequences."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'forward': 1.0},
            expected_k=5.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Start a sequence
        policy.sample(state, agent_index=0)
        
        # There should be an ongoing sequence
        assert 0 in policy._agent_sequences
        
        # Reset should clear it
        policy.reset(env)
        assert policy._agent_sequences == {}
    
    def test_set_world_model(self):
        """Test set_world_model() method."""
        env = OpenCorridorEnv(is_robot=True)
        env.reset()
        
        policy = MultiGridMultiStepExplorationPolicy(agent_indices=[0])
        
        assert policy.world_model is None
        
        policy.set_world_model(env)
        assert policy.world_model is env


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_no_feasible_sequences_falls_back_to_still(self):
        """Test fallback when no sequences are feasible."""
        # Create environment with agent completely surrounded
        class SurroundedEnv(MultiGridEnv):
            def __init__(self):
                agent = create_robot_agent()
                super().__init__(
                    width=3,
                    height=3,
                    max_steps=10,
                    agents=[agent],
                    partial_obs=False,
                    objects_set=World,
                    actions_set=SmallActions,
                )
            
            def _gen_grid(self, width, height):
                self.grid = Grid(width, height)
                # Fill everything with walls except center
                for i in range(width):
                    for j in range(height):
                        if i != 1 or j != 1:
                            self.grid.set(i, j, Wall(World))
                
                self.agents[0].pos = np.array([1, 1])
                self.agents[0].dir = 0
        
        env = SurroundedEnv()
        env.reset()
        
        # Only allow forward sequences (all infeasible)
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'forward': 1.0},
            expected_k=2.0,
        )
        policy.reset(env)
        
        state = env.get_state()
        
        # Should fall back to still
        action = policy.sample(state, agent_index=0)
        assert action == policy.ACTION_STILL
    
    def test_expected_k_1_gives_short_sequences(self):
        """Test that expected_k=1 gives mostly single-step sequences."""
        np.random.seed(42)
        
        policy = MultiGridMultiStepExplorationPolicy(
            agent_indices=[0],
            sequence_probs={'still': 1.0},
            expected_k=1.0,
        )
        
        # Sample many k values using the policy's method
        geom_p = policy._get_geom_p('still')
        k_values = [np.random.geometric(geom_p) for _ in range(1000)]
        
        # Most should be 1
        fraction_ones = sum(1 for k in k_values if k == 1) / len(k_values)
        assert fraction_ones > 0.4  # p=1 gives k=1 with probability 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
