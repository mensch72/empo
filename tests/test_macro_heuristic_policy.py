"""Tests for MacroHeuristicPolicy — heuristic human policy at macro level.

Covers: Boltzmann action distributions conditioned on MacroCellGoal /
MacroProximityGoal, marginal distributions, shortest-path distances,
normalization, edge cases, and API compatibility.
"""

import numpy as np
import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical.macro_grid_env import MacroGridEnv, MACRO_PASS, macro_walk
from empo.hierarchical.macro_goals import (
    MacroCellGoal,
    MacroProximityGoal,
    MacroGoalGenerator,
)
from empo.hierarchical.macro_heuristic_policy import MacroHeuristicPolicy


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


def _policy(macro=None, beta=5.0):
    if macro is None:
        macro = _macro_env()
    gen = MacroGoalGenerator(macro)
    return MacroHeuristicPolicy(
        macro,
        human_agent_indices=list(macro.human_agent_indices),
        possible_goal_generator=gen,
        beta=beta,
    )


# ── TestBasicProperties ──────────────────────────────────────────────

class TestBasicProperties:

    def test_returns_numpy_array(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert isinstance(dist, np.ndarray)

    def test_correct_shape(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert dist.shape == (macro.action_space.n,)

    def test_sums_to_one(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_non_negative(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert (dist >= 0).all()

    def test_pass_action_always_has_probability(self):
        """PASS is always available, so should always have nonzero prob."""
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert dist[MACRO_PASS] > 0


# ── TestCellGoalConditioning ─────────────────────────────────────────

class TestCellGoalConditioning:

    def test_conditioned_sums_to_one(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        goal = MacroCellGoal(macro, 0, 0)
        dist = pol(state, 0, goal)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_goal_in_current_cell_favours_pass(self):
        """When already at the target cell, PASS should be the best action."""
        macro = _macro_env()
        pol = _policy(macro, beta=10.0)
        state = macro.get_state()
        agent_cell = state[2][0][0]
        goal = MacroCellGoal(macro, 0, agent_cell)
        dist = pol(state, 0, goal)
        # PASS should have highest or near-highest probability
        assert dist[MACRO_PASS] >= dist.max() - 1e-6

    def test_goal_in_adjacent_cell_favours_walk(self):
        """Walking to the target cell should be strongly preferred."""
        macro = _macro_env()
        pol = _policy(macro, beta=10.0)
        state = macro.get_state()
        agent_cell = state[2][0][0]
        available = macro.available_actions(state, 0)
        # Only consider WALK actions to cells with an open passage.
        walk_actions = [
            a for a in available
            if a != MACRO_PASS
            and macro.passage_open(state, agent_cell, a - 1)
        ]
        if not walk_actions:
            pytest.skip("No adjacent cells with open passage")
        target_cell = walk_actions[0] - 1  # WALK(j) = j+1
        goal = MacroCellGoal(macro, 0, target_cell)
        dist = pol(state, 0, goal)
        # WALK(target_cell) should have highest probability
        walk_action = macro_walk(target_cell)
        assert dist[walk_action] > dist[MACRO_PASS]

    def test_unreachable_goal_gives_valid_distribution(self):
        """Even for unreachable target cells, should return valid dist."""
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        # Try a cell that might be unreachable (disconnected passage)
        goal = MacroCellGoal(macro, 0, macro.num_cells - 1)
        dist = pol(state, 0, goal)
        assert abs(dist.sum() - 1.0) < 1e-9
        assert (dist >= 0).all()


# ── TestProximityGoalConditioning ────────────────────────────────────

class TestProximityGoalConditioning:

    def _two_agent_macro(self):
        macro = _macro_env()
        if len(macro.agents) < 2:
            pytest.skip("Need at least 2 agents")
        return macro

    def test_same_cell_conditioned_sums_to_one(self):
        macro = self._two_agent_macro()
        pol = _policy(macro)
        state = macro.get_state()
        goal = MacroProximityGoal(macro, 0, 1, same_cell=True)
        dist = pol(state, 0, goal)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_diff_cell_conditioned_sums_to_one(self):
        macro = self._two_agent_macro()
        pol = _policy(macro)
        state = macro.get_state()
        goal = MacroProximityGoal(macro, 0, 1, same_cell=False)
        dist = pol(state, 0, goal)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_same_cell_already_together_favours_pass(self):
        """When agents are already together, approaching should not be preferred."""
        macro = self._two_agent_macro()
        state = macro.get_state()
        cell0 = state[2][0][0]
        cell1 = state[2][1][0]
        if cell0 != cell1:
            pytest.skip("Agents not in same cell")
        pol = _policy(macro, beta=10.0)
        goal = MacroProximityGoal(macro, 0, 1, same_cell=True)
        dist = pol(state, 0, goal)
        # Already in same cell — PASS should be preferred
        assert dist[MACRO_PASS] >= dist.max() - 1e-6


# ── TestMarginalDistribution ─────────────────────────────────────────

class TestMarginalDistribution:

    def test_marginal_sums_to_one(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_marginal_non_negative(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert (dist >= 0).all()

    def test_marginal_shape(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert dist.shape == (macro.action_space.n,)

    def test_marginal_is_average_of_conditionals(self):
        """Marginal should be weighted average over all goals."""
        macro = _macro_env()
        gen = MacroGoalGenerator(macro)
        pol = _policy(macro)
        state = macro.get_state()

        # Manual averaging
        total = np.zeros(macro.action_space.n)
        total_w = 0.0
        for goal, weight in gen.generate(state, 0):
            total += weight * pol(state, 0, goal)
            total_w += weight
        expected = total / total_w
        expected /= expected.sum()

        actual = pol(state, 0)
        np.testing.assert_allclose(actual, expected, atol=1e-10)


# ── TestBoltzmannBehavior ────────────────────────────────────────────

class TestBoltzmannBehavior:

    def test_high_beta_is_more_deterministic(self):
        """Higher beta should produce more peaked distributions."""
        macro = _macro_env()
        state = macro.get_state()
        agent_cell = state[2][0][0]
        available = macro.available_actions(state, 0)
        walk_actions = [a for a in available if a != MACRO_PASS]
        if not walk_actions:
            pytest.skip("No walk actions")
        target_cell = walk_actions[0] - 1
        goal = MacroCellGoal(macro, 0, target_cell)

        pol_low = _policy(macro, beta=1.0)
        pol_high = _policy(macro, beta=20.0)

        dist_low = pol_low(state, 0, goal)
        dist_high = pol_high(state, 0, goal)

        # Higher beta → higher max probability
        assert dist_high.max() >= dist_low.max()

    def test_zero_beta_is_uniform_over_available(self):
        """beta=0 should produce uniform over available actions."""
        macro = _macro_env()
        pol = _policy(macro, beta=0.0)
        state = macro.get_state()
        goal = MacroCellGoal(macro, 0, 0)
        dist = pol(state, 0, goal)
        available = macro.available_actions(state, 0)
        # All available actions should have equal probability
        probs = [dist[a] for a in available]
        np.testing.assert_allclose(probs, probs[0], atol=1e-10)
        # Unavailable actions should have zero probability
        for a in range(macro.action_space.n):
            if a not in available:
                assert dist[a] < 1e-12


# ── TestShortestPath ─────────────────────────────────────────────────

class TestShortestPath:

    def test_distance_to_self_is_zero(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        d = pol._shortest_path_distance(0, 0, state)
        assert d == 0.0

    def test_adjacent_cells_have_finite_distance(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        partition = macro.partition
        for cell_i in range(macro.num_cells):
            for cell_j in partition.adjacency.get(cell_i, frozenset()):
                if macro.passage_open(state, cell_i, cell_j):
                    d = pol._shortest_path_distance(cell_i, cell_j, state)
                    assert d < float('inf')
                    assert d > 0

    def test_unreachable_returns_inf(self):
        """If passage is closed, non-adjacent cells → inf distance."""
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        # Make all passages closed
        remaining, _, agents, objects = state
        closed_flags = tuple(False for _ in macro.adj_pairs)
        closed_state = (remaining, closed_flags, agents, objects)
        if macro.num_cells >= 2:
            d = pol._shortest_path_distance(0, macro.num_cells - 1, closed_state)
            assert d == float('inf')


# ── TestSampleMethod ─────────────────────────────────────────────────

class TestSampleMethod:

    def test_sample_returns_int(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        action = pol.sample(state, 0)
        assert isinstance(action, int)

    def test_sample_returns_valid_action(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        for _ in range(20):
            action = pol.sample(state, 0)
            assert 0 <= action < macro.action_space.n

    def test_sample_with_goal(self):
        macro = _macro_env()
        pol = _policy(macro)
        state = macro.get_state()
        goal = MacroCellGoal(macro, 0, 0)
        action = pol.sample(state, 0, goal)
        assert isinstance(action, int)
        assert 0 <= action < macro.action_space.n


# ── TestDoorTestEnv ──────────────────────────────────────────────────

class TestDoorTestEnv:
    """Test on a different environment to ensure generality."""

    def test_policy_on_door_test(self):
        micro = _make_door_test()
        macro = MacroGridEnv(micro, seed=42)
        pol = _policy(macro)
        state = macro.get_state()
        dist = pol(state, 0)
        assert abs(dist.sum() - 1.0) < 1e-9

    def test_conditioned_on_door_test(self):
        micro = _make_door_test()
        macro = MacroGridEnv(micro, seed=42)
        pol = _policy(macro)
        state = macro.get_state()
        goal = MacroCellGoal(macro, 0, 0)
        dist = pol(state, 0, goal)
        assert abs(dist.sum() - 1.0) < 1e-9


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_exported_from_hierarchical_package(self):
        from empo.hierarchical import MacroHeuristicPolicy
        assert MacroHeuristicPolicy is not None


# ── TestProfileDistribution ─────────────────────────────────────────

class TestProfileDistribution:
    """Tests that MacroHeuristicPolicy inherits profile_distribution from base class."""

    def test_has_profile_distribution(self):
        pol = _policy()
        assert hasattr(pol, 'profile_distribution')
        assert callable(pol.profile_distribution)

    def test_has_profile_distribution_with_fixed_goal(self):
        pol = _policy()
        assert hasattr(pol, 'profile_distribution_with_fixed_goal')
        assert callable(pol.profile_distribution_with_fixed_goal)

    def test_profile_distribution_returns_list(self):
        pol = _policy()
        macro = pol.world_model
        state = macro.get_state()
        dist = pol.profile_distribution(state)
        assert isinstance(dist, list)
        assert len(dist) > 0

    def test_profile_distribution_probabilities_sum_to_one(self):
        pol = _policy()
        macro = pol.world_model
        state = macro.get_state()
        dist = pol.profile_distribution(state)
        total = sum(p for p, _ in dist)
        assert abs(total - 1.0) < 1e-9

    def test_profile_distribution_entries_are_tuples(self):
        pol = _policy()
        macro = pol.world_model
        state = macro.get_state()
        dist = pol.profile_distribution(state)
        for entry in dist:
            prob, profile = entry
            assert isinstance(prob, float)
            assert isinstance(profile, list)

    def test_profile_distribution_with_fixed_goal_sums_to_one(self):
        pol = _policy()
        macro = pol.world_model
        state = macro.get_state()
        human_idx = macro.human_agent_indices[0]
        gen = MacroGoalGenerator(macro)
        goals = list(gen.generate(state, human_idx))
        if not goals:
            pytest.skip("No goals for this state")
        goal, _ = goals[0]
        dist = pol.profile_distribution_with_fixed_goal(state, human_idx, goal)
        total = sum(p for p, _ in dist)
        assert abs(total - 1.0) < 1e-9

    def test_profile_distribution_with_fixed_goal_differs_from_marginal(self):
        """Goal-conditioned profile should differ from marginal."""
        pol = _policy()
        macro = pol.world_model
        state = macro.get_state()
        human_idx = macro.human_agent_indices[0]
        gen = MacroGoalGenerator(macro)
        goals = list(gen.generate(state, human_idx))
        if len(goals) < 2:
            pytest.skip("Need ≥2 goals to compare")

        # Verify that at least two goals produce meaningfully different
        # conditioned distributions; otherwise the marginal can match
        # the conditioned one (e.g., only one available action).
        dists = []
        for g, _ in goals:
            d = pol.profile_distribution_with_fixed_goal(state, human_idx, g)
            dists.append({tuple(p): prob for prob, p in d})
        goals_differ = any(
            any(
                abs(dists[0].get(k, 0) - dists[j].get(k, 0)) > 1e-9
                for k in set(dists[0]) | set(dists[j])
            )
            for j in range(1, len(dists))
        )
        if not goals_differ:
            pytest.skip("All goals produce identical conditioned distributions")

        marginal = pol.profile_distribution(state)

        # Convert to dict for comparison
        marginal_dict = {tuple(p): prob for prob, p in marginal}

        # The marginal (average over goals) should differ from at least one
        # goal-conditioned distribution.  (It can coincide with one of them
        # when symmetric pairs cancel out, but not all of them — we already
        # checked that at least two conditioned distributions differ.)
        differs_from_any = any(
            any(
                abs(marginal_dict.get(k, 0) - d.get(k, 0)) > 1e-9
                for k in set(marginal_dict) | set(d)
            )
            for d in dists
        )
        assert differs_from_any, "Marginal should differ from at least one conditioned distribution"
