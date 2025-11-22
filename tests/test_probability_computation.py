#!/usr/bin/env python3
"""
Advanced tests for exact probability computation in transition_probabilities.

Tests that probabilities are computed exactly (not sampled) and that
multiple permutations leading to the same state are properly aggregated.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, Ball
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def test_probabilities_sum_exactly_to_one():
    """Test that probabilities sum to exactly 1.0 (not approximately)."""
    print("Test: probabilities sum to exactly 1.0...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # Try different action combinations
    test_cases = [
        [Actions.still, Actions.still, Actions.still],  # All still
        [Actions.left, Actions.right, Actions.forward],  # Different actions
        [Actions.forward, Actions.forward, Actions.forward],  # Same actions
    ]
    
    for actions in test_cases:
        result = env.transition_probabilities(state, actions)
        if result is None:
            continue
        
        total_prob = sum(prob for prob, _ in result)
        
        # Check for EXACT equality, not just approximate
        if total_prob == 1.0:
            print(f"  ✓ Actions {actions}: probabilities sum to exactly 1.0")
        else:
            print(f"  ✗ Actions {actions}: probabilities sum to {total_prob}, not exactly 1.0")
            return False
    
    return True


def test_each_state_appears_once():
    """Test that each unique successor state appears exactly once in the result."""
    print("Test: each successor state appears exactly once...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ⚠ Got None, skipping test")
        return True
    
    # Check that all states are unique
    states = [s for _, s in result]
    if len(states) == len(set(states)):
        print(f"  ✓ All {len(states)} successor states are unique")
        return True
    else:
        print(f"  ✗ Found duplicate states: {len(states)} total, {len(set(states))} unique")
        return False


def test_deterministic_when_one_agent():
    """Test that transitions are deterministic when only one agent acts."""
    print("Test: deterministic with one agent acting...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    # Only first agent acts
    actions = [Actions.forward, Actions.still, Actions.still]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ⚠ Got None, skipping test")
        return True
    
    if len(result) == 1 and abs(result[0][0] - 1.0) < 1e-10:
        print(f"  ✓ One agent acting produces deterministic transition (prob={result[0][0]})")
        return True
    else:
        print(f"  ✗ One agent should be deterministic, got {len(result)} states")
        for prob, _ in result:
            print(f"    - prob: {prob}")
        return False


def test_rotation_actions_deterministic():
    """Test that rotation actions (left/right) are always deterministic."""
    print("Test: rotation actions are deterministic...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # Test various rotation combinations
    test_cases = [
        [Actions.left, Actions.left, Actions.left],
        [Actions.right, Actions.right, Actions.right],
        [Actions.left, Actions.right, Actions.still],
    ]
    
    for actions in test_cases:
        result = env.transition_probabilities(state, actions)
        if result is None:
            continue
        
        if len(result) == 1 and abs(result[0][0] - 1.0) < 1e-10:
            print(f"  ✓ Rotation actions {actions} are deterministic")
        else:
            print(f"  ✗ Rotation actions {actions} should be deterministic, got {len(result)} states")
            return False
    
    return True


def test_probabilistic_when_agents_interact():
    """Test that transitions can be probabilistic when multiple agents interact."""
    print("Test: can be probabilistic when agents interact...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Try to find a scenario where multiple agents act
    state = env.get_state()
    
    # Multiple agents moving forward
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ⚠ Got None (terminal or invalid), skipping test")
        return True
    
    # With 3 agents all acting, we might get multiple outcomes depending on their positions
    # This depends on the specific configuration, so we just check the result is valid
    print(f"  ✓ Got {len(result)} possible successor state(s)")
    
    # Verify probabilities are computed correctly
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-10:
        print(f"  ✓ Probabilities sum correctly to 1.0")
        return True
    else:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False


def test_probability_values_are_rational():
    """Test that probability values are rational (form k/n! for some integers k, n)."""
    print("Test: probability values are rational multiples of 1/n!...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ⚠ Got None, skipping test")
        return True
    
    # With n=3 agents, n! = 6, so all probabilities should be multiples of 1/6
    # That is, probability * 6 should be close to an integer
    n_agents = len(env.agents)
    factorial_n = 1
    for i in range(1, n_agents + 1):
        factorial_n *= i
    
    all_rational = True
    for prob, _ in result:
        # Check if prob * n! is close to an integer
        scaled = prob * factorial_n
        nearest_int = round(scaled)
        if abs(scaled - nearest_int) < 1e-9:
            print(f"  ✓ Probability {prob} = {nearest_int}/{factorial_n}")
        else:
            print(f"  ✗ Probability {prob} is not a multiple of 1/{factorial_n}")
            all_rational = False
    
    return all_rational


def test_state_consistency_across_calls():
    """Test that calling transition_probabilities doesn't modify the environment state."""
    print("Test: transition_probabilities doesn't modify environment...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state_before = env.get_state()
    actions = [Actions.forward, Actions.left, Actions.right]
    
    # Call transition_probabilities
    result = env.transition_probabilities(state_before, actions)
    
    # Get state after
    state_after = env.get_state()
    
    if state_before == state_after:
        print("  ✓ Environment state unchanged after transition_probabilities call")
        return True
    else:
        print("  ✗ Environment state was modified")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Exact Probability Computation")
    print("=" * 70)
    print()
    
    tests = [
        test_probabilities_sum_exactly_to_one,
        test_each_state_appears_once,
        test_deterministic_when_one_agent,
        test_rotation_actions_deterministic,
        test_probabilistic_when_agents_interact,
        test_probability_values_are_rational,
        test_state_consistency_across_calls,
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
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
