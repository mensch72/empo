#!/usr/bin/env python3
"""
Tests for WorldModel step duration extension.

Tests that:
1. Default transition_durations() returns [1.0, ...] for any transition list
2. Default terminal_duration() returns 1.0
3. Subclass can override transition_durations() and terminal_duration()
4. Existing environments (MultiGridEnv) are unaffected by the new methods
"""

import sys

import gymnasium as gym
sys.modules['gym'] = gym

from empo.world_model import WorldModel
from gym_multigrid.envs import CollectGame4HEnv10x10N2


# =============================================================================
# Concrete WorldModel subclass for testing duration defaults
# =============================================================================

class SimpleWorldModel(WorldModel):
    """Minimal concrete WorldModel for testing duration methods."""

    def __init__(self):
        super().__init__()
        self.agents = [type('Agent', (), {})()]  # dummy agent
        self.action_space = type('Space', (), {'n': 2})()

    def get_state(self):
        return (0,)

    def set_state(self, state):
        pass

    def transition_probabilities(self, state, actions):
        if state == (999,):
            return None  # terminal
        return [(0.5, (1,)), (0.5, (2,))]


# =============================================================================
# Subclass that overrides durations
# =============================================================================

class VariableDurationWorldModel(WorldModel):
    """WorldModel with non-uniform transition durations for testing."""

    def __init__(self):
        super().__init__()
        self.agents = [type('Agent', (), {})()]
        self.action_space = type('Space', (), {'n': 2})()

    def get_state(self):
        return (0,)

    def set_state(self, state):
        pass

    def transition_probabilities(self, state, actions):
        if state == (999,):
            return None
        return [(0.7, (1,)), (0.3, (2,))]

    def transition_durations(self, state, actions, transitions):
        """Return variable durations depending on successor state."""
        durations = []
        for _prob, succ_state in transitions:
            # Farther states take longer
            durations.append(float(succ_state[0]))
        return durations

    def terminal_duration(self, state):
        """Terminal states have duration 0.5."""
        return 0.5


# =============================================================================
# Tests for default behavior
# =============================================================================

def test_default_transition_durations_returns_ones():
    """Default transition_durations returns [1.0] for each transition outcome."""
    env = SimpleWorldModel()
    state = (0,)
    actions = [0]
    transitions = env.transition_probabilities(state, actions)
    durations = env.transition_durations(state, actions, transitions)
    assert durations == [1.0, 1.0]


def test_default_transition_durations_length_matches():
    """Default durations list has same length as transitions list."""
    env = SimpleWorldModel()
    state = (0,)
    actions = [0]
    transitions = env.transition_probabilities(state, actions)
    durations = env.transition_durations(state, actions, transitions)
    assert len(durations) == len(transitions)


def test_default_transition_durations_empty():
    """Default durations for empty transition list returns empty list."""
    env = SimpleWorldModel()
    durations = env.transition_durations((0,), [0], [])
    assert durations == []


def test_default_terminal_duration():
    """Default terminal_duration returns 1.0."""
    env = SimpleWorldModel()
    assert env.terminal_duration((999,)) == 1.0


# =============================================================================
# Tests for subclass override
# =============================================================================

def test_override_transition_durations():
    """Subclass can override transition_durations to return non-uniform values."""
    env = VariableDurationWorldModel()
    state = (0,)
    actions = [0]
    transitions = env.transition_probabilities(state, actions)
    durations = env.transition_durations(state, actions, transitions)
    # Successor states are (1,) and (2,), so durations are 1.0 and 2.0
    assert durations == [1.0, 2.0]


def test_override_terminal_duration():
    """Subclass can override terminal_duration."""
    env = VariableDurationWorldModel()
    assert env.terminal_duration((999,)) == 0.5


# =============================================================================
# Tests for existing environments (backward compatibility)
# =============================================================================

def test_multigrid_default_transition_durations():
    """MultiGridEnv inherits default transition_durations (unit duration)."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    state = env.get_state()
    actions = [0] * len(env.agents)
    transitions = env.transition_probabilities(state, actions)
    if transitions is not None:
        durations = env.transition_durations(state, actions, transitions)
        assert len(durations) == len(transitions)
        assert all(d == 1.0 for d in durations)


def test_multigrid_default_terminal_duration():
    """MultiGridEnv inherits default terminal_duration (1.0)."""
    env = CollectGame4HEnv10x10N2()
    env.reset()
    state = env.get_state()
    assert env.terminal_duration(state) == 1.0


def test_world_model_has_duration_methods():
    """WorldModel class defines the new duration methods."""
    assert hasattr(WorldModel, 'transition_durations')
    assert hasattr(WorldModel, 'terminal_duration')
