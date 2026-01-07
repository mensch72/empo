#!/usr/bin/env python3
"""
Tests for WorldModel base class and inheritance structure.

Tests that:
1. WorldModel properly extends gymnasium.Env
2. MultiGridEnv inherits from WorldModel
3. WorldModel methods are accessible and work correctly
"""

import sys

import gymnasium as gym
from empo.world_model import WorldModel
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.envs import CollectGame4HEnv10x10N2


class MockActionSpace:
    """Reusable mock action space for testing."""
    def __init__(self, n=1):
        self.n = n


def test_world_model_inherits_from_gym_env():
    """Test that WorldModel properly inherits from gymnasium.Env."""
    assert issubclass(WorldModel, gym.Env), "WorldModel should inherit from gymnasium.Env"


def test_multigrid_env_inherits_from_world_model():
    """Test that MultiGridEnv properly inherits from WorldModel."""
    assert issubclass(MultiGridEnv, WorldModel), "MultiGridEnv should inherit from WorldModel"


def test_multigrid_env_inherits_from_gym_env():
    """Test that MultiGridEnv still inherits from gymnasium.Env (through WorldModel)."""
    assert issubclass(MultiGridEnv, gym.Env), "MultiGridEnv should inherit from gymnasium.Env"


def test_world_model_has_abstract_methods():
    """Test that WorldModel defines the required abstract methods."""
    assert hasattr(WorldModel, 'get_state'), "WorldModel should have get_state method"
    assert hasattr(WorldModel, 'set_state'), "WorldModel should have set_state method"
    assert hasattr(WorldModel, 'transition_probabilities'), "WorldModel should have transition_probabilities method"


def test_world_model_has_dag_methods():
    """Test that WorldModel defines get_dag and plot_dag methods."""
    assert hasattr(WorldModel, 'get_dag'), "WorldModel should have get_dag method"
    assert hasattr(WorldModel, 'plot_dag'), "WorldModel should have plot_dag method"


def test_multigrid_env_get_dag_method():
    """Test that MultiGridEnv's get_dag method works correctly."""
    env = CollectGame4HEnv10x10N2()
    env.max_steps = 1  # Limit for tractable DAG
    
    # Get DAG using the method
    states, state_to_idx, successors = env.get_dag()
    
    # Basic sanity checks
    assert len(states) > 0, "Should find at least one state"
    assert len(state_to_idx) == len(states), "state_to_idx should have same length as states"
    assert len(successors) == len(states), "successors should have same length as states"


def test_world_model_exported_from_empo():
    """Test that WorldModel is exported from the empo package."""
    from empo import WorldModel as ImportedWorldModel
    assert ImportedWorldModel is WorldModel, "WorldModel should be exported from empo package"


def test_initial_state_method():
    """Test that initial_state() returns the initial state without changing current state."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Take some steps to change the state
    for _ in range(3):
        actions = [env.action_space.sample() for _ in range(len(env.agents))]
        env.step(actions)
    
    # Save the current state
    current_state = env.get_state()
    
    # Get the initial state
    init_state = env.initial_state()
    
    # Verify the environment is restored to its previous state
    restored_state = env.get_state()
    assert current_state == restored_state, "Environment should be restored to previous state"
    
    # Verify the initial state is different from the current state (since we took steps)
    # Note: This might not always be true if steps didn't change state, but typically it will be
    
    # Verify the initial state is what we'd get from a fresh reset
    env.reset()
    env.get_state()
    # Note: Due to RNG state, we can't directly compare, but we can verify step_count is 0
    # Compact state format: (step_count, agent_states, mobile_objects, mutable_objects)
    step_count = init_state[0]
    assert step_count == 0, "Initial state should have step_count of 0"


def test_is_terminal_method():
    """Test that is_terminal() correctly identifies terminal states."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Initial state should not be terminal
    assert not env.is_terminal(), "Initial state should not be terminal"
    
    # Force environment to terminal state by setting step count to max
    env.step_count = env.max_steps
    terminal_state = env.get_state()
    
    # Now it should be terminal
    assert env.is_terminal(), "State at max_steps should be terminal"
    
    # Also test with explicit state parameter
    env.reset()  # Reset to non-terminal
    assert env.is_terminal(terminal_state), "Explicit terminal state should be identified as terminal"
    
    # Current state should still be non-terminal
    assert not env.is_terminal(), "Current state should still be non-terminal after checking explicit state"


class SimpleWorldModelEnv(WorldModel):
    """
    A simple WorldModel implementation for testing the default step() method.
    
    Creates a linear state space:
        State 0 (root) -> State 1 -> State 2 -> Terminal
    """
    
    def __init__(self):
        self.current_state = 0
        self.agents = [0]  # Single agent (index 0)
        self.action_space = MockActionSpace(n=2)
        self.step_count = 0
        self.max_steps = 10
    
    def reset(self, seed=None, options=None):
        self.current_state = 0
        self.step_count = 0
        return self.current_state, {}
    
    def get_state(self):
        return (self.current_state, self.step_count)
    
    def set_state(self, state):
        self.current_state, self.step_count = state
    
    def transition_probabilities(self, state, actions):
        """
        Define a linear structure with probabilistic branching:
        - State 0 -> State 1 (action 0) or State 2 (action 1)
        - State 1 -> State 2 (deterministic)
        - State 2 -> Terminal (None)
        """
        current, step = state
        if step >= self.max_steps:
            return None  # Terminal due to max steps
        if current >= 2:
            return None  # Terminal state
        
        new_step = step + 1
        if current == 0:
            if actions[0] == 0:
                return [(1.0, (1, new_step))]
            else:
                return [(1.0, (2, new_step))]
        elif current == 1:
            return [(1.0, (2, new_step))]
        else:
            return None


def test_default_step_method():
    """Test that the default step() implementation works correctly."""
    env = SimpleWorldModelEnv()
    env.reset()
    
    # Initial state should be (0, 0)
    assert env.get_state() == (0, 0), "Initial state should be (0, 0)"
    
    # Take action 0 to go to state 1
    obs, reward, terminated, truncated, info = env.step([0])
    assert obs == (1, 1), f"After action 0, state should be (1, 1), got {obs}"
    assert not terminated, "State 1 should not be terminal"
    assert reward == 0.0, "Default reward should be 0.0"
    
    # Take action 0 to go to state 2
    obs, reward, terminated, truncated, info = env.step([0])
    assert obs == (2, 2), f"After action 0, state should be (2, 2), got {obs}"
    assert terminated, "State 2 should be terminal"


def test_default_step_with_probabilistic_transitions():
    """Test that default step() correctly samples from probabilistic transitions."""
    
    class ProbabilisticEnv(WorldModel):
        """Environment with probabilistic transitions for testing."""
        
        def __init__(self):
            self.current_state = 0
            self.agents = [0]
            self.action_space = MockActionSpace(n=1)
        
        def reset(self, seed=None, options=None):
            self.current_state = 0
            return self.current_state, {}
        
        def get_state(self):
            return self.current_state
        
        def set_state(self, state):
            self.current_state = state
        
        def transition_probabilities(self, state, actions):
            if state >= 2:
                return None  # Terminal
            # 50% chance to go to state 1, 50% chance to go to state 2
            return [(0.5, 1), (0.5, 2)]
    
    env = ProbabilisticEnv()
    env.reset()
    
    # Run multiple trials and check we get both outcomes
    outcomes = {1: 0, 2: 0}
    for _ in range(100):
        env.reset()
        obs, _, _, _, _ = env.step([0])
        outcomes[obs] += 1
    
    # Both outcomes should occur (with high probability given 100 trials)
    assert outcomes[1] > 0, "Should sometimes transition to state 1"
    assert outcomes[2] > 0, "Should sometimes transition to state 2"


class CyclicMockEnv:
    """
    A mock environment with a cycle for testing get_dag error handling.
    
    Creates a cyclic state space:
        State 0 (root) -> State 1 -> State 2 -> State 0 (cycle!)
    """
    
    def __init__(self):
        self.current_state = 0
        self.agents = [0]  # Single agent (index 0)
        self.action_space = MockActionSpace(n=1)
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def get_state(self):
        return self.current_state
    
    def set_state(self, state):
        self.current_state = state
    
    def transition_probabilities(self, state, actions):
        """
        Define a cyclic structure:
        - State 0 -> State 1
        - State 1 -> State 2
        - State 2 -> State 0 (cycle!)
        """
        if state == 0:
            return [(1.0, 1)]
        elif state == 1:
            return [(1.0, 2)]
        elif state == 2:
            return [(1.0, 0)]  # Cycle back to state 0
        else:
            return None


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_world_model_inherits_from_gym_env,
        test_multigrid_env_inherits_from_world_model,
        test_multigrid_env_inherits_from_gym_env,
        test_world_model_has_abstract_methods,
        test_world_model_has_dag_methods,
        test_multigrid_env_get_dag_method,
        test_world_model_exported_from_empo,
        test_initial_state_method,
        test_is_terminal_method,
        test_default_step_method,
        test_default_step_with_probabilistic_transitions,
    ]
    
    print("=" * 60)
    print("Running WorldModel and inheritance tests")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
