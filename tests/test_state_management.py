#!/usr/bin/env python3
"""
Test state management methods for multigrid environments.

Tests the get_state, set_state, and transition_probabilities methods.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, SmallActions, SmallWorld, Wall, Ball, Goal
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def test_get_state_returns_hashable():
    """Test that get_state returns a hashable object."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    state = env.get_state()
    # Verify it's hashable by trying to use it as a dict key
    state_dict = {state: "test"}  # This will raise TypeError if not hashable


def test_set_state_restores_environment():
    """Test that set_state correctly restores the environment."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    initial_state = env.get_state()
    
    actions = [env.action_space.sample() for _ in range(len(env.agents))]
    env.step(actions)
    after_step_state = env.get_state()
    
    assert initial_state != after_step_state, "States should be different after stepping"
    
    env.set_state(initial_state)
    restored_state = env.get_state()
    assert initial_state == restored_state, "State not correctly restored"


def test_set_state_restores_step_count():
    """Test that set_state correctly restores step count."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    for _ in range(5):
        actions = [0 for _ in range(len(env.agents))]
        env.step(actions)
    
    after_steps_count = env.step_count
    state_after_steps = env.get_state()
    
    env.reset()
    env.step_count = 10
    env.set_state(state_after_steps)
    
    assert env.step_count == after_steps_count, f"Step count not restored. Expected {after_steps_count}, got {env.step_count}"


def test_set_state_restores_agent_positions():
    """Test that set_state correctly restores agent positions."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    initial_positions = [tuple(agent.pos) for agent in env.agents]
    initial_state = env.get_state()
    
    for _ in range(3):
        actions = [Actions.forward for _ in range(len(env.agents))]
        env.step(actions)
    
    env.set_state(initial_state)
    restored_positions = [tuple(agent.pos) for agent in env.agents]
    
    assert initial_positions == restored_positions, f"Agent positions not restored. Initial: {initial_positions}, Restored: {restored_positions}"


def test_set_state_restores_agent_direction():
    """Test that set_state correctly restores agent directions."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    initial_directions = [agent.dir for agent in env.agents]
    initial_state = env.get_state()
    
    actions = [Actions.left for _ in range(len(env.agents))]
    env.step(actions)
    after_directions = [agent.dir for agent in env.agents]
    
    assert initial_directions != after_directions, "Directions should have changed"
    
    env.set_state(initial_state)
    restored_directions = [agent.dir for agent in env.agents]
    
    assert initial_directions == restored_directions, f"Agent directions not restored. Initial: {initial_directions}, Restored: {restored_directions}"


def test_transition_probabilities_returns_list():
    """Test that transition_probabilities returns a list of tuples."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    assert result is not None, "Should not return None for valid state and actions"
    assert isinstance(result, list), f"Should return a list, got {type(result)}"
    assert len(result) > 0, "Should return at least one transition"
    assert isinstance(result[0], tuple) and len(result[0]) == 2, "Each element should be a (probability, state) tuple"
    
    prob, succ_state = result[0]
    assert isinstance(prob, (int, float)), f"Probability should be a number, got {type(prob)}"


def test_transition_probabilities_deterministic_still():
    """Test that 'still' actions produce deterministic transitions."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    assert len(result) == 1, f"'still' actions should produce exactly 1 transition, got {len(result)}"
    prob, succ_state = result[0]
    assert abs(prob - 1.0) < 1e-6, f"Probability should be 1.0, got {prob}"


def test_transition_probabilities_sums_to_one():
    """Test that probabilities sum to 1.0."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        pytest.skip("Got None, might be terminal state")
    
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-6, f"Probabilities should sum to 1.0, got {total_prob}"


def test_transition_probabilities_terminal_state():
    """Test that transition_probabilities returns None for terminal states."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    env.step_count = env.max_steps
    state = env.get_state()
    actions = [Actions.still for _ in range(len(env.agents))]
    
    result = env.transition_probabilities(state, actions)
    assert result is None, f"Terminal state should return None, got {result}"


def test_transition_probabilities_invalid_action():
    """Test that transition_probabilities handles invalid actions."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [999 for _ in range(len(env.agents))]
    
    with pytest.raises(ValueError):
        env.transition_probabilities(state, actions)


def test_state_includes_time_left():
    """Test that state includes step count (time left)."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    for _ in range(5):
        actions = [Actions.still for _ in range(len(env.agents))]
        env.step(actions)
    
    state = env.get_state()
    step_count = state[0]
    assert step_count == 5, f"Step count should be 5, got {step_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
