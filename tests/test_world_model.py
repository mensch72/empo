#!/usr/bin/env python3
"""
Tests for WorldModel base class and inheritance structure.

Tests that:
1. WorldModel properly extends gymnasium.Env
2. MultiGridEnv inherits from WorldModel
3. WorldModel methods are accessible and work correctly
4. Backward compatibility with env_utils functions is maintained
"""

import sys
from pathlib import Path

# Setup path to import empo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

import gymnasium as gym
from empo.world_model import WorldModel
from empo import env_utils
from gym_multigrid.multigrid import MultiGridEnv
from gym_multigrid.envs import CollectGame4HEnv10x10N2


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


def test_backward_compatible_get_dag_function():
    """Test that the env_utils.get_dag function still works."""
    env = CollectGame4HEnv10x10N2()
    env.max_steps = 1  # Limit for tractable DAG
    
    # Get DAG using the function
    states, state_to_idx, successors = env_utils.get_dag(env)
    
    # Basic sanity checks
    assert len(states) > 0, "Should find at least one state"
    assert len(state_to_idx) == len(states), "state_to_idx should have same length as states"
    assert len(successors) == len(states), "successors should have same length as states"


def test_world_model_exported_from_empo():
    """Test that WorldModel is exported from the empo package."""
    from empo import WorldModel as ImportedWorldModel
    assert ImportedWorldModel is WorldModel, "WorldModel should be exported from empo package"


class CyclicMockEnv:
    """
    A mock environment with a cycle for testing get_dag error handling.
    
    Creates a cyclic state space:
        State 0 (root) -> State 1 -> State 2 -> State 0 (cycle!)
    """
    
    def __init__(self):
        self.current_state = 0
        self.agents = [None]  # Single agent
        self.action_space = type('MockActionSpace', (), {'n': 1})()
    
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


def test_get_dag_raises_on_cyclic_env():
    """Test that get_dag raises ValueError for environments with cycles."""
    import pytest
    
    cyclic_env = CyclicMockEnv()
    
    # The get_dag function should raise ValueError when it detects a cycle
    with pytest.raises(ValueError, match="Environment contains cycles"):
        env_utils.get_dag(cyclic_env)


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_world_model_inherits_from_gym_env,
        test_multigrid_env_inherits_from_world_model,
        test_multigrid_env_inherits_from_gym_env,
        test_world_model_has_abstract_methods,
        test_world_model_has_dag_methods,
        test_multigrid_env_get_dag_method,
        test_backward_compatible_get_dag_function,
        test_world_model_exported_from_empo,
        test_get_dag_raises_on_cyclic_env,
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
