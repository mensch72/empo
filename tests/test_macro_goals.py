"""Tests for macro-level goals and goal generator.

Covers: MacroCellGoal, MacroProximityGoal, MacroGoalGenerator —
goal achievement, hashing, equality, immutability, generator output.
"""

import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical.macro_grid_env import MacroGridEnv
from empo.hierarchical.macro_goals import (
    MacroCellGoal,
    MacroProximityGoal,
    MacroGoalGenerator,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_rock_gateway() -> MultiGridEnv:
    env = MultiGridEnv(
        config_file='multigrid_worlds/copilot_challenges/rock_gateway.yaml',
    )
    env.reset()
    return env


def _make_door_test() -> MultiGridEnv:
    env = MultiGridEnv(
        config_file='multigrid_worlds/obstacles/door_test.yaml',
    )
    env.reset()
    return env


def _macro_env(micro=None, seed=42) -> MacroGridEnv:
    if micro is None:
        micro = _make_rock_gateway()
    return MacroGridEnv(micro, seed=seed)


# ── TestMacroCellGoal ────────────────────────────────────────────────

class TestMacroCellGoal:

    def test_achieved_when_agent_in_target_cell(self):
        macro = _macro_env()
        state = macro.get_state()
        agent_cell = state[2][0][0]
        goal = MacroCellGoal(macro, agent_index=0, target_cell=agent_cell)
        assert goal.is_achieved(state) == 1

    def test_not_achieved_when_agent_elsewhere(self):
        macro = _macro_env()
        state = macro.get_state()
        agent_cell = state[2][0][0]
        other_cell = (agent_cell + 1) % macro.num_cells
        goal = MacroCellGoal(macro, agent_index=0, target_cell=other_cell)
        assert goal.is_achieved(state) == 0

    def test_hash_deterministic(self):
        macro = _macro_env()
        g1 = MacroCellGoal(macro, 0, 2)
        g2 = MacroCellGoal(macro, 0, 2)
        assert hash(g1) == hash(g2)

    def test_hash_differs_for_different_targets(self):
        macro = _macro_env()
        g1 = MacroCellGoal(macro, 0, 0)
        g2 = MacroCellGoal(macro, 0, 1)
        # Hash collisions are allowed; assert set/dict can distinguish them.
        assert len({g1, g2}) == 2

    def test_equality(self):
        macro = _macro_env()
        g1 = MacroCellGoal(macro, 0, 2)
        g2 = MacroCellGoal(macro, 0, 2)
        assert g1 == g2

    def test_inequality_different_target(self):
        macro = _macro_env()
        g1 = MacroCellGoal(macro, 0, 0)
        g2 = MacroCellGoal(macro, 0, 1)
        assert g1 != g2

    def test_inequality_different_agent(self):
        macro = _macro_env()
        if len(macro.agents) < 2:
            pytest.skip("Need at least 2 agents")
        g1 = MacroCellGoal(macro, 0, 0)
        g2 = MacroCellGoal(macro, 1, 0)
        assert g1 != g2

    def test_immutable_after_creation(self):
        macro = _macro_env()
        goal = MacroCellGoal(macro, 0, 0)
        with pytest.raises(AttributeError):
            goal.target_cell = 1

    def test_usable_as_dict_key(self):
        macro = _macro_env()
        g1 = MacroCellGoal(macro, 0, 0)
        g2 = MacroCellGoal(macro, 0, 0)
        d = {g1: 42}
        assert d[g2] == 42

    def test_repr(self):
        macro = _macro_env()
        goal = MacroCellGoal(macro, 0, 3)
        r = repr(goal)
        assert "MacroCellGoal" in r
        assert "3" in r

    def test_index_attribute(self):
        macro = _macro_env()
        goal = MacroCellGoal(macro, 0, 0, index=5)
        assert goal.index == 5


# ── TestMacroProximityGoal ───────────────────────────────────────────

class TestMacroProximityGoal:

    def _two_agent_macro(self):
        macro = _macro_env()
        if len(macro.agents) < 2:
            pytest.skip("Need at least 2 agents")
        return macro

    def test_same_cell_achieved_when_together(self):
        macro = self._two_agent_macro()
        state = macro.get_state()
        cell0 = state[2][0][0]
        cell1 = state[2][1][0]
        goal = MacroProximityGoal(macro, 0, 1, same_cell=True)
        if cell0 == cell1:
            assert goal.is_achieved(state) == 1
        else:
            assert goal.is_achieved(state) == 0

    def test_same_cell_not_achieved_when_apart(self):
        macro = self._two_agent_macro()
        state = macro.get_state()
        cell0 = state[2][0][0]
        cell1 = state[2][1][0]
        goal = MacroProximityGoal(macro, 0, 1, same_cell=True)
        if cell0 != cell1:
            assert goal.is_achieved(state) == 0

    def test_diff_cell_achieved_when_apart(self):
        macro = self._two_agent_macro()
        state = macro.get_state()
        cell0 = state[2][0][0]
        cell1 = state[2][1][0]
        goal = MacroProximityGoal(macro, 0, 1, same_cell=False)
        if cell0 != cell1:
            assert goal.is_achieved(state) == 1
        else:
            assert goal.is_achieved(state) == 0

    def test_same_and_diff_are_complementary(self):
        macro = self._two_agent_macro()
        state = macro.get_state()
        g_same = MacroProximityGoal(macro, 0, 1, same_cell=True)
        g_diff = MacroProximityGoal(macro, 0, 1, same_cell=False)
        assert g_same.is_achieved(state) + g_diff.is_achieved(state) == 1

    def test_hash_deterministic(self):
        macro = self._two_agent_macro()
        g1 = MacroProximityGoal(macro, 0, 1, True)
        g2 = MacroProximityGoal(macro, 0, 1, True)
        assert hash(g1) == hash(g2)

    def test_hash_differs_for_same_vs_diff(self):
        macro = self._two_agent_macro()
        g_same = MacroProximityGoal(macro, 0, 1, True)
        g_diff = MacroProximityGoal(macro, 0, 1, False)
        # Hash collisions are allowed; assert set/dict can distinguish them.
        assert len({g_same, g_diff}) == 2

    def test_equality(self):
        macro = self._two_agent_macro()
        g1 = MacroProximityGoal(macro, 0, 1, True)
        g2 = MacroProximityGoal(macro, 0, 1, True)
        assert g1 == g2

    def test_inequality_different_mode(self):
        macro = self._two_agent_macro()
        g_same = MacroProximityGoal(macro, 0, 1, True)
        g_diff = MacroProximityGoal(macro, 0, 1, False)
        assert g_same != g_diff

    def test_inequality_with_cell_goal(self):
        macro = self._two_agent_macro()
        g_prox = MacroProximityGoal(macro, 0, 1, True)
        g_cell = MacroCellGoal(macro, 0, 0)
        assert g_prox != g_cell

    def test_immutable_after_creation(self):
        macro = self._two_agent_macro()
        goal = MacroProximityGoal(macro, 0, 1, True)
        with pytest.raises(AttributeError):
            goal.same_cell = False

    def test_usable_as_dict_key(self):
        macro = self._two_agent_macro()
        g1 = MacroProximityGoal(macro, 0, 1, True)
        g2 = MacroProximityGoal(macro, 0, 1, True)
        d = {g1: 99}
        assert d[g2] == 99

    def test_repr(self):
        macro = self._two_agent_macro()
        goal = MacroProximityGoal(macro, 0, 1, True)
        assert "MacroProximityGoal" in repr(goal)


# ── TestMacroGoalGenerator ───────────────────────────────────────────

class TestMacroGoalGenerator:

    def test_generates_cell_goals(self):
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = list(gen.generate(state, 0))
        cell_goals = [g for g, w in goals if isinstance(g, MacroCellGoal)]
        assert len(cell_goals) == macro.num_cells

    def test_cell_goals_cover_all_cells(self):
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = list(gen.generate(state, 0))
        cells = {g.target_cell for g, w in goals if isinstance(g, MacroCellGoal)}
        assert cells == set(range(macro.num_cells))

    def test_generates_proximity_goals_for_multi_agent(self):
        macro = _macro_env()
        if len(macro.agents) < 2:
            pytest.skip("Need at least 2 agents")
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = list(gen.generate(state, 0))
        prox_goals = [g for g, w in goals if isinstance(g, MacroProximityGoal)]
        # 2 per other agent (same + diff)
        expected = 2 * (len(macro.agents) - 1)
        assert len(prox_goals) == expected

    def test_all_weights_are_one(self):
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        for _, weight in gen.generate(state, 0):
            assert weight == 1.0

    def test_total_goal_count(self):
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = list(gen.generate(state, 0))
        expected = macro.num_cells + 2 * (len(macro.agents) - 1)
        assert len(goals) == expected

    def test_goals_unique(self):
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = [g for g, w in gen.generate(state, 0)]
        assert len(goals) == len(set(goals))

    def test_different_agent_index_different_goals(self):
        macro = _macro_env()
        if len(macro.agents) < 2:
            pytest.skip("Need at least 2 agents")
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals_0 = set(g for g, w in gen.generate(state, 0))
        goals_1 = set(g for g, w in gen.generate(state, 1))
        # Cell goals have different agent_index → different hash/eq
        assert goals_0 != goals_1

    def test_single_agent_no_proximity_goals(self):
        """With 1 agent, generator should only produce cell goals."""
        micro = _make_door_test()
        macro = MacroGridEnv(micro, seed=42)
        if len(macro.agents) != 1:
            pytest.skip("door_test must have exactly 1 agent for this test")
        gen = MacroGoalGenerator(macro)
        state = macro.get_state()
        goals = list(gen.generate(state, 0))
        assert all(isinstance(g, MacroCellGoal) for g, _ in goals)
        assert len(goals) == macro.num_cells


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_exported_from_hierarchical_package(self):
        from empo.hierarchical import (
            MacroCellGoal, MacroProximityGoal, MacroGoalGenerator,
        )
        assert MacroCellGoal is not None
        assert MacroProximityGoal is not None
        assert MacroGoalGenerator is not None
