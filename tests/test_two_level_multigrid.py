"""Tests for TwoLevelMultigrid — end-to-end two-level hierarchy.

Covers: construction, property access, level/mapper consistency,
super_state consistency across mapper and macro_env, and hierarchy
invariants.
"""

import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.macro_grid_env import MacroGridEnv, MACRO_PASS, macro_walk
from empo.hierarchical.multigrid_level_mapper import MultiGridLevelMapper
from empo.hierarchical.two_level_multigrid import TwoLevelMultigrid


# ── helpers ──────────────────────────────────────────────────────────

def _make_rock_gateway():
    env = MultiGridEnv(
        config_file='multigrid_worlds/copilot_challenges/rock_gateway.yaml',
    )
    env.reset()
    return env


def _make_door_test():
    env = MultiGridEnv(
        config_file='multigrid_worlds/obstacles/door_test.yaml',
    )
    env.reset()
    return env


# ── TestConstruction ─────────────────────────────────────────────────

class TestConstruction:

    def test_is_hierarchical_world_model(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert isinstance(h, HierarchicalWorldModel)

    def test_has_two_levels(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert h.num_levels == 2

    def test_coarsest_is_macro_grid_env(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert isinstance(h.coarsest(), MacroGridEnv)

    def test_finest_is_multi_grid_env(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert isinstance(h.finest(), MultiGridEnv)

    def test_mapper_is_multigrid_level_mapper(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert isinstance(h.mapper, MultiGridLevelMapper)

    def test_seed_deterministic(self):
        micro = _make_rock_gateway()
        h1 = TwoLevelMultigrid(micro, seed=42)
        h2 = TwoLevelMultigrid(micro, seed=42)
        assert h1.macro_env.get_state() == h2.macro_env.get_state()

    def test_door_test_env_works(self):
        h = TwoLevelMultigrid(_make_door_test(), seed=42)
        assert h.num_levels == 2
        assert h.macro_env.num_cells > 0


# ── TestProperties ───────────────────────────────────────────────────

class TestProperties:

    def test_macro_env_property(self):
        micro = _make_rock_gateway()
        h = TwoLevelMultigrid(micro, seed=42)
        assert h.macro_env is h.levels[0]

    def test_micro_env_property(self):
        micro = _make_rock_gateway()
        h = TwoLevelMultigrid(micro, seed=42)
        assert h.micro_env is micro

    def test_mapper_property(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert h.mapper is h.mappers[0]


# ── TestLevelMapperConsistency ───────────────────────────────────────

class TestLevelMapperConsistency:

    def test_mapper_coarse_is_macro(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        assert h.mapper.coarse_model is h.macro_env

    def test_mapper_fine_is_micro(self):
        micro = _make_rock_gateway()
        h = TwoLevelMultigrid(micro, seed=42)
        assert h.mapper.fine_model is micro


# ── TestSuperStateConsistency ────────────────────────────────────────

class TestSuperStateConsistency:

    def test_initial_super_state_matches_macro_state(self):
        micro = _make_rock_gateway()
        h = TwoLevelMultigrid(micro, seed=42)
        micro_state = micro.get_state()
        mapped = h.mapper.super_state(micro_state)
        initial = h.macro_env.get_state()
        assert mapped == initial

    def test_super_state_preserves_remaining_time(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        assert macro_state[0] == h.micro_env.max_steps - micro_state[0]

    def test_super_state_after_micro_step(self):
        """super_state is consistent after a micro-level transition."""
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        micro_state = h.micro_env.get_state()
        # Apply a micro-level transition (PASS for all agents)
        actions = [0] * len(h.micro_env.agents)
        transitions = h.micro_env.transition_probabilities(
            micro_state, actions,
        )
        if transitions is None:
            pytest.skip("No transitions available")
        _, next_micro = transitions[0]
        next_macro = h.mapper.super_state(next_micro)
        # Remaining time should decrease by 1 (step count incremented)
        assert next_macro[0] == macro_state_from_initial_time(h) - 1


def macro_state_from_initial_time(h):
    """Helper: initial remaining time."""
    return h.micro_env.max_steps


# ── TestMacroTransitionsFromHierarchy ────────────────────────────────

class TestMacroTransitions:

    def test_macro_pass_from_hierarchy(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        macro_state = h.macro_env.get_state()
        t = h.macro_env.transition_probabilities(
            macro_state, [MACRO_PASS, MACRO_PASS],
        )
        assert t is not None
        assert len(t) == 1

    def test_macro_walk_from_hierarchy(self):
        h = TwoLevelMultigrid(_make_rock_gateway(), seed=42)
        macro_state = h.macro_env.get_state()
        cell = macro_state[2][0][0]
        # Walk to first open adjacent cell
        for adj in h.macro_env.partition.adjacency.get(cell, frozenset()):
            if h.macro_env.passage_open(macro_state, cell, adj):
                t = h.macro_env.transition_probabilities(
                    macro_state, [macro_walk(adj), MACRO_PASS],
                )
                assert t[0][1][2][0][0] == adj
                return
        pytest.skip("No open adjacent cell")


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_exported_from_hierarchical_package(self):
        from empo.hierarchical import TwoLevelMultigrid as Cls
        assert Cls is TwoLevelMultigrid
