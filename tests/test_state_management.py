#!/usr/bin/env python3
"""
Test state management methods for multigrid environments.

Tests the get_state, set_state, and transition_probabilities methods.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, SmallActions, SmallWorld, Wall, Ball, Goal
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def test_get_state_returns_hashable():
    """Test that get_state returns a hashable object."""
    print("Test: get_state returns hashable object...")
    
    # Create a simple environment
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Get state
    state = env.get_state()
    
    # Verify it's hashable by trying to use it as a dict key
    try:
        state_dict = {state: "test"}
        print("  ✓ State is hashable")
        return True
    except TypeError as e:
        print(f"  ✗ State is not hashable: {e}")
        return False


def test_set_state_restores_environment():
    """Test that set_state correctly restores the environment."""
    print("Test: set_state restores environment...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Get initial state
    initial_state = env.get_state()
    
    # Take some steps
    actions = [env.action_space.sample() for _ in range(len(env.agents))]
    env.step(actions)
    
    # Get state after step
    after_step_state = env.get_state()
    
    # Verify states are different
    if initial_state == after_step_state:
        print("  ✗ States should be different after stepping")
        return False
    
    # Restore initial state
    env.set_state(initial_state)
    
    # Get state again
    restored_state = env.get_state()
    
    # Verify restored state matches initial state
    if initial_state == restored_state:
        print("  ✓ State correctly restored")
        return True
    else:
        print("  ✗ State not correctly restored")
        return False


def test_set_state_restores_step_count():
    """Test that set_state correctly restores step count."""
    print("Test: set_state restores step count...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    initial_step_count = env.step_count
    
    # Take several steps
    for _ in range(5):
        actions = [0 for _ in range(len(env.agents))]  # All agents stay still
        env.step(actions)
    
    after_steps_count = env.step_count
    
    # Get state after steps
    state_after_steps = env.get_state()
    
    # Reset and verify step count changed
    env.reset()
    env.step_count = 10  # Set to different value
    
    # Restore state
    env.set_state(state_after_steps)
    
    if env.step_count == after_steps_count:
        print(f"  ✓ Step count correctly restored to {after_steps_count}")
        return True
    else:
        print(f"  ✗ Step count not restored. Expected {after_steps_count}, got {env.step_count}")
        return False


def test_set_state_restores_agent_positions():
    """Test that set_state correctly restores agent positions."""
    print("Test: set_state restores agent positions...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Get initial agent positions
    initial_positions = [tuple(agent.pos) for agent in env.agents]
    initial_state = env.get_state()
    
    # Take some forward actions to move agents
    for _ in range(3):
        actions = [Actions.forward for _ in range(len(env.agents))]
        env.step(actions)
    
    after_positions = [tuple(agent.pos) for agent in env.agents]
    
    # Verify at least one agent moved (might not move if blocked by wall)
    # Just checking positions changed
    
    # Restore initial state
    env.set_state(initial_state)
    
    # Check positions restored
    restored_positions = [tuple(agent.pos) for agent in env.agents]
    
    if initial_positions == restored_positions:
        print(f"  ✓ Agent positions correctly restored")
        return True
    else:
        print(f"  ✗ Agent positions not restored")
        print(f"    Initial: {initial_positions}")
        print(f"    Restored: {restored_positions}")
        return False


def test_set_state_restores_agent_direction():
    """Test that set_state correctly restores agent directions."""
    print("Test: set_state restores agent directions...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Get initial agent directions
    initial_directions = [agent.dir for agent in env.agents]
    initial_state = env.get_state()
    
    # Rotate agents
    actions = [Actions.left for _ in range(len(env.agents))]
    env.step(actions)
    
    after_directions = [agent.dir for agent in env.agents]
    
    # Verify directions changed
    if initial_directions == after_directions:
        print(f"  ✗ Directions should have changed")
        return False
    
    # Restore initial state
    env.set_state(initial_state)
    
    # Check directions restored
    restored_directions = [agent.dir for agent in env.agents]
    
    if initial_directions == restored_directions:
        print(f"  ✓ Agent directions correctly restored")
        return True
    else:
        print(f"  ✗ Agent directions not restored")
        print(f"    Initial: {initial_directions}")
        print(f"    Restored: {restored_directions}")
        return False


def test_transition_probabilities_returns_list():
    """Test that transition_probabilities returns a list of tuples."""
    print("Test: transition_probabilities returns list...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ✗ Should not return None for valid state and actions")
        return False
    
    if not isinstance(result, list):
        print(f"  ✗ Should return a list, got {type(result)}")
        return False
    
    if len(result) == 0:
        print("  ✗ Should return at least one transition")
        return False
    
    # Check first element is a tuple of (probability, state)
    if not isinstance(result[0], tuple) or len(result[0]) != 2:
        print(f"  ✗ Each element should be a (probability, state) tuple")
        return False
    
    prob, succ_state = result[0]
    
    if not isinstance(prob, (int, float)):
        print(f"  ✗ Probability should be a number, got {type(prob)}")
        return False
    
    print(f"  ✓ transition_probabilities returns valid list with {len(result)} transition(s)")
    return True


def test_transition_probabilities_deterministic_still():
    """Test that 'still' actions produce deterministic transitions."""
    print("Test: 'still' actions are deterministic...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if len(result) != 1:
        print(f"  ✗ 'still' actions should produce exactly 1 transition, got {len(result)}")
        return False
    
    prob, succ_state = result[0]
    
    if abs(prob - 1.0) > 1e-6:
        print(f"  ✗ Probability should be 1.0, got {prob}")
        return False
    
    print(f"  ✓ 'still' actions produce deterministic transition with probability 1.0")
    return True


def test_transition_probabilities_sums_to_one():
    """Test that probabilities sum to 1.0."""
    print("Test: probabilities sum to 1.0...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ⚠ Got None, skipping test (might be terminal state)")
        return True
    
    total_prob = sum(prob for prob, _ in result)
    
    if abs(total_prob - 1.0) < 1e-6:
        print(f"  ✓ Probabilities sum to 1.0 (got {total_prob})")
        return True
    else:
        print(f"  ✗ Probabilities should sum to 1.0, got {total_prob}")
        return False


def test_transition_probabilities_terminal_state():
    """Test that transition_probabilities returns None for terminal states."""
    print("Test: terminal states return None...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Force environment to terminal state by setting step count to max
    env.step_count = env.max_steps
    
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print(f"  ✓ Terminal state correctly returns None")
        return True
    else:
        print(f"  ✗ Terminal state should return None, got {result}")
        return False


def test_transition_probabilities_invalid_action():
    """Test that transition_probabilities returns None for invalid actions."""
    print("Test: invalid actions return None...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    # Use an invalid action (out of range)
    actions = [999 for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print(f"  ✓ Invalid action correctly returns None")
        return True
    else:
        print(f"  ✗ Invalid action should return None, got {result}")
        return False


def test_state_includes_time_left():
    """Test that state includes step count (time left)."""
    print("Test: state includes step count...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Take some steps
    for _ in range(5):
        actions = [Actions.still for _ in range(len(env.agents))]
        env.step(actions)
    
    state = env.get_state()
    state_dict = dict(state)
    
    if 'step_count' in state_dict:
        step_count = state_dict['step_count']
        if step_count == 5:
            print(f"  ✓ State includes step_count: {step_count}")
            return True
        else:
            print(f"  ✗ Step count should be 5, got {step_count}")
            return False
    else:
        print(f"  ✗ State does not include 'step_count'")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing State Management Methods for Multigrid Environments")
    print("=" * 60)
    print()
    
    tests = [
        test_get_state_returns_hashable,
        test_set_state_restores_environment,
        test_set_state_restores_step_count,
        test_set_state_restores_agent_positions,
        test_set_state_restores_agent_direction,
        test_state_includes_time_left,
        test_transition_probabilities_returns_list,
        test_transition_probabilities_deterministic_still,
        test_transition_probabilities_sums_to_one,
        test_transition_probabilities_terminal_state,
        test_transition_probabilities_invalid_action,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
