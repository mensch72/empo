#!/usr/bin/env python3
"""
Advanced tests for exact probability computation in transition_probabilities.

Tests that probabilities are computed exactly (not sampled) and that
multiple permutations leading to the same state are properly aggregated.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, Ball
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def test_probabilities_sum_exactly_to_one():
    """Test that probabilities sum to exactly 1.0 (not approximately)."""
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
        assert total_prob == 1.0, f"Actions {actions}: probabilities sum to {total_prob}, not exactly 1.0"


def test_each_state_appears_once():
    """Test that each unique successor state appears exactly once in the result."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        pytest.skip("Got None result, skipping test")
    
    # Check that all states are unique
    states = [s for _, s in result]
    assert len(states) == len(set(states)), f"Found duplicate states: {len(states)} total, {len(set(states))} unique"


def test_deterministic_when_one_agent():
    """Test that transitions are deterministic when only one agent acts."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    # Only first agent acts
    actions = [Actions.forward, Actions.still, Actions.still]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        pytest.skip("Got None result, skipping test")
    
    assert len(result) == 1, f"One agent should be deterministic, got {len(result)} states"
    assert abs(result[0][0] - 1.0) < 1e-10, f"Probability should be 1.0, got {result[0][0]}"


def test_rotation_actions_deterministic():
    """Test that rotation actions (left/right) are always deterministic."""
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
        
        assert len(result) == 1, f"Rotation actions {actions} should be deterministic, got {len(result)} states"
        assert abs(result[0][0] - 1.0) < 1e-10, f"Rotation probability should be 1.0, got {result[0][0]}"


def test_probabilistic_when_agents_interact():
    """Test that transitions can be probabilistic when multiple agents interact."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Try to find a scenario where multiple agents act
    state = env.get_state()
    
    # Multiple agents moving forward
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        pytest.skip("Got None (terminal or invalid), skipping test")
    
    # Verify probabilities are computed correctly
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-10, f"Probabilities sum to {total_prob}, not 1.0"


def test_probability_values_are_rational():
    """Test that probability values are rational (form k/n! for some integers k, n)."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        pytest.skip("Got None, skipping test")
    
    # With n=3 agents, n! = 6, so all probabilities should be multiples of 1/6
    # That is, probability * 6 should be close to an integer
    n_agents = len(env.agents)
    factorial_n = 1
    for i in range(1, n_agents + 1):
        factorial_n *= i
    
    for prob, _ in result:
        # Check if prob * n! is close to an integer
        scaled = prob * factorial_n
        nearest_int = round(scaled)
        assert abs(scaled - nearest_int) < 1e-9, f"Probability {prob} is not a multiple of 1/{factorial_n}"


def test_state_consistency_across_calls():
    """Test that calling transition_probabilities doesn't modify the environment state."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state_before = env.get_state()
    actions = [Actions.forward, Actions.left, Actions.right]
    
    # Call transition_probabilities
    result = env.transition_probabilities(state_before, actions)
    
    # Get state after
    state_after = env.get_state()
    
    assert state_before == state_after, "Environment state was modified by transition_probabilities"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
