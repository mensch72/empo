#!/usr/bin/env python3
"""
Tests for HierarchicalWorldModel and LevelMapper ABCs.

Tests that:
1. HierarchicalWorldModel construction, validation, and property access
2. LevelMapper is an abstract base class with required methods
3. Concrete LevelMapper subclass works correctly
4. Exports are available from empo package
"""

import sys
import numpy as np
import pytest

import gymnasium as gym
sys.modules['gym'] = gym

from empo.world_model import WorldModel
from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.level_mapper import LevelMapper


# =============================================================================
# Simple WorldModel subclasses for testing
# =============================================================================

class SimpleCoarseModel(WorldModel):
    """Minimal coarse-level world model with 3 macro-states."""

    def __init__(self):
        super().__init__()
        self.agents = [type('Agent', (), {})()]
        self.action_space = type('Space', (), {'n': 2})()
        self._state = (0,)

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def transition_probabilities(self, state, actions):
        if state == (2,):
            return None  # terminal
        return [(1.0, (state[0] + 1,))]


class SimpleFineModel(WorldModel):
    """Minimal fine-level world model with 9 micro-states."""

    def __init__(self):
        super().__init__()
        self.agents = [type('Agent', (), {})()]
        self.action_space = type('Space', (), {'n': 4})()
        self._state = (0, 0)

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def transition_probabilities(self, state, actions):
        if state[0] >= 2 and state[1] >= 2:
            return None
        return [(1.0, (state[0] + 1, state[1]))]


# =============================================================================
# Concrete LevelMapper for testing
# =============================================================================

class SimpleLevelMapper(LevelMapper):
    """Maps fine states (x, y) to coarse states (x // 3,).
    
    Assumes fine states are tuples where the first element is a spatial position.
    The coarse state groups positions into macro-cells of size 3.
    """

    def super_state(self, fine_state):
        return (fine_state[0] // 3,)

    def super_agent(self, fine_agent_index):
        return fine_agent_index  # identity mapping

    def is_feasible(self, coarse_action_profile, fine_state, fine_action_profile):
        return True  # all actions feasible

    def is_abort(self, coarse_action_profile, fine_state, fine_action_profile):
        return fine_action_profile == (0,)  # action 0 is abort

    def return_control(self, coarse_action_profile, fine_state, fine_action_profile, fine_successor_state):
        # Return control when fine state changes macro-cell
        return self.super_state(fine_state) != self.super_state(fine_successor_state)


# =============================================================================
# Tests for HierarchicalWorldModel construction
# =============================================================================

def test_construct_two_level():
    """Two-level hierarchy constructs successfully."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    hierarchy = HierarchicalWorldModel(levels=[coarse, fine], mappers=[mapper])
    assert hierarchy.num_levels == 2
    assert hierarchy.coarsest() is coarse
    assert hierarchy.finest() is fine


def test_construct_three_level():
    """Three-level hierarchy constructs successfully."""
    l0 = SimpleCoarseModel()
    l1 = SimpleFineModel()
    l2 = SimpleFineModel()
    m01 = SimpleLevelMapper(l0, l1)
    m12 = SimpleLevelMapper(l1, l2)
    hierarchy = HierarchicalWorldModel(levels=[l0, l1, l2], mappers=[m01, m12])
    assert hierarchy.num_levels == 3
    assert hierarchy.coarsest() is l0
    assert hierarchy.finest() is l2


def test_construct_validates_level_count():
    """Construction fails with fewer than 2 levels."""
    with pytest.raises(ValueError, match="at least 2 levels"):
        HierarchicalWorldModel(levels=[SimpleCoarseModel()], mappers=[])


def test_construct_validates_mapper_count():
    """Construction fails with wrong number of mappers."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    with pytest.raises(ValueError, match="mappers"):
        HierarchicalWorldModel(levels=[coarse, fine], mappers=[mapper, mapper])


def test_levels_list():
    """levels attribute stores the world models in order."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    hierarchy = HierarchicalWorldModel(levels=[coarse, fine], mappers=[mapper])
    assert hierarchy.levels == [coarse, fine]


def test_mappers_list():
    """mappers attribute stores the level mappers in order."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    hierarchy = HierarchicalWorldModel(levels=[coarse, fine], mappers=[mapper])
    assert hierarchy.mappers == [mapper]


# =============================================================================
# Tests for LevelMapper ABC
# =============================================================================

def test_level_mapper_is_abstract():
    """LevelMapper cannot be instantiated directly."""
    with pytest.raises(TypeError):
        LevelMapper(SimpleCoarseModel(), SimpleFineModel())


def test_level_mapper_stores_models():
    """LevelMapper stores references to coarse and fine models."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    assert mapper.coarse_model is coarse
    assert mapper.fine_model is fine


def test_level_mapper_super_state():
    """super_state maps fine state to coarse state."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    assert mapper.super_state((0, 0)) == (0,)
    assert mapper.super_state((2, 1)) == (0,)
    assert mapper.super_state((3, 0)) == (1,)
    assert mapper.super_state((6, 2)) == (2,)


def test_level_mapper_super_agent():
    """super_agent maps fine agent to coarse agent (identity in this case)."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    assert mapper.super_agent(0) == 0
    assert mapper.super_agent(1) == 1


def test_level_mapper_is_feasible():
    """is_feasible checks action compatibility."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    assert mapper.is_feasible((1,), (0, 0), (1,)) is True


def test_level_mapper_is_abort():
    """is_abort detects abort actions."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    assert mapper.is_abort((1,), (0, 0), (0,)) is True
    assert mapper.is_abort((1,), (0, 0), (1,)) is False


def test_level_mapper_return_control():
    """return_control detects when control should return to coarse level."""
    coarse = SimpleCoarseModel()
    fine = SimpleFineModel()
    mapper = SimpleLevelMapper(coarse, fine)
    # Same macro-cell: no return
    assert mapper.return_control((1,), (0, 0), (1,), (1, 0)) is False
    # Different macro-cell: return control
    assert mapper.return_control((1,), (2, 0), (1,), (3, 0)) is True


# =============================================================================
# Tests for exports
# =============================================================================

def test_hierarchy_exported_from_empo():
    """HierarchicalWorldModel and LevelMapper are exported from empo package."""
    from empo import HierarchicalWorldModel as Imported1
    from empo import LevelMapper as Imported2
    assert Imported1 is HierarchicalWorldModel
    assert Imported2 is LevelMapper


def test_hierarchy_exported_from_subpackage():
    """HierarchicalWorldModel and LevelMapper are exported from empo.hierarchical."""
    from empo.hierarchical import HierarchicalWorldModel as Imported1
    from empo.hierarchical import LevelMapper as Imported2
    assert Imported1 is HierarchicalWorldModel
    assert Imported2 is LevelMapper
