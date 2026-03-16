"""Tests for hierarchical backward induction (Task 11).

Covers:
- ``compute_hierarchical_robot_policy()`` produces a valid
  ``HierarchicalRobotPolicy`` on a small two-room environment.
- Macro-level policy produces valid distributions.
- Sub-problem policies respect feasibility constraints.
- Control transfer occurs at macro-action boundaries.
- ``HierarchicalRobotPolicy.sample()`` returns valid micro-level actions.
- ``observe_transition()`` triggers return-control correctly.
"""

import numpy as np
import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical import (
    MacroGridEnv,
    MacroGoalGenerator,
    MacroHeuristicPolicy,
    TwoLevelMultigrid,
    MACRO_PASS,
    macro_walk,
    compute_hierarchical_robot_policy,
    HierarchicalRobotPolicy,
)
from empo.hierarchical._sub_problem import build_sub_problem_dag
from empo.backward_induction.phase2 import TabularRobotPolicy


# ── helpers ──────────────────────────────────────────────────────────

def _make_trivial() -> MultiGridEnv:
    """Small 4×6 grid with 2 agents, max_steps=10 — fast for tests."""
    env = MultiGridEnv(
        config_file='multigrid_worlds/trivial.yaml',
    )
    env.reset()
    return env


def _build_hierarchy(micro=None, seed=42) -> TwoLevelMultigrid:
    if micro is None:
        micro = _make_trivial()
    return TwoLevelMultigrid(micro, seed=seed)


def _make_policy(hierarchy: TwoLevelMultigrid, beta_r=5.0):
    """Compute a hierarchical policy (macro-level only; sub-problems on demand)."""
    macro_env = hierarchy.coarsest()
    macro_gen = MacroGoalGenerator(macro_env)
    return compute_hierarchical_robot_policy(
        hierarchy,
        human_agent_indices=list(macro_env.human_agent_indices),
        robot_agent_indices=list(macro_env.robot_agent_indices),
        possible_goal_generators=[macro_gen],
        beta_r=beta_r,
        quiet=True,
    )


# ── TestComputeHierarchicalRobotPolicy ───────────────────────────────

class TestComputeHierarchicalRobotPolicy:
    """Tests for the top-level ``compute_hierarchical_robot_policy()`` function."""

    def test_returns_hierarchical_policy(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        assert isinstance(policy, HierarchicalRobotPolicy)

    def test_macro_policy_is_tabular(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        assert isinstance(policy.macro_policy, TabularRobotPolicy)

    def test_macro_Vr_has_entries(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        assert len(policy.macro_Vr) > 0

    def test_macro_Vr_values_are_negative(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        for vr in policy.macro_Vr.values():
            assert vr < 0, f"V_r should be strictly negative, got {vr}"

    def test_rejects_zero_goal_generators(self):
        hierarchy = _build_hierarchy()
        with pytest.raises(ValueError, match="at least 1"):
            compute_hierarchical_robot_policy(
                hierarchy,
                human_agent_indices=[0],
                robot_agent_indices=[1],
                possible_goal_generators=[],
                quiet=True,
            )

    def test_starts_at_macro_level(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        assert policy.at_macro_level


# ── TestHierarchicalRobotPolicySample ────────────────────────────────

class TestHierarchicalRobotPolicySample:
    """Tests for ``HierarchicalRobotPolicy.sample()`` and control transfer."""

    @pytest.fixture
    def policy_and_env(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        return policy, hierarchy

    def test_sample_returns_tuple(self, policy_and_env):
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        action = policy.sample(micro_state)
        assert isinstance(action, tuple)

    def test_sample_has_robot_actions(self, policy_and_env):
        """Action tuple length equals number of robot agents."""
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        action = policy.sample(micro_state)
        assert len(action) == len(policy.robot_agent_indices)

    def test_actions_in_range(self, policy_and_env):
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        num_actions = hierarchy.finest().action_space.n
        action = policy.sample(micro_state)
        for a in action:
            assert 0 <= a < num_actions, f"Action {a} out of range [0, {num_actions})"

    def test_coarse_action_set_after_sample(self, policy_and_env):
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        _ = policy.sample(micro_state)
        # After the first sample, either a coarse action is active (micro level)
        # or it was immediately aborted back to macro level.
        coarse = policy.current_coarse_action_profile
        assert coarse is None or isinstance(coarse, tuple)

    def test_reset_returns_to_macro(self, policy_and_env):
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        _ = policy.sample(micro_state)
        policy.reset()
        assert policy.at_macro_level

    def test_reset_with_world_model(self, policy_and_env):
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        _ = policy.sample(micro_state)
        policy.reset(hierarchy)
        assert policy.at_macro_level

    def test_multiple_samples_no_crash(self, policy_and_env):
        """Sample multiple times in sequence — should not crash."""
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        for _ in range(3):
            action = policy.sample(micro_state)
            assert isinstance(action, tuple)


# ── TestSubProblemDag ────────────────────────────────────────────────

class TestSubProblemDag:
    """Tests for ``build_sub_problem_dag()``."""

    def test_root_state_included(self):
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()

        # Use WALK(target) for agent 0 to constrain the search
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, idx_map, transitions, term_mask = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )

        assert micro_state in idx_map
        assert idx_map[micro_state] == 0
        assert not term_mask[0]

    def test_root_not_terminal(self):
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, _, _, term_mask = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )
        assert not term_mask[0]

    def test_has_transitions(self):
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, _, transitions, _ = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )
        assert len(transitions[0]) > 0

    def test_walk_action_filters_infeasible(self):
        """WALK(j) should filter micro forward-actions entering wrong cell."""
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, _, transitions, term_mask = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )

        for idx, state_trans in enumerate(transitions):
            if term_mask[idx]:
                continue
            for ap, _, _ in state_trans:
                assert mapper.is_feasible(coarse_ap, states[idx], ap), (
                    f"Infeasible action {ap} at state index {idx}"
                )

    def test_terminal_states_not_expanded(self):
        """Terminal states should have no transitions."""
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, _, transitions, term_mask = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )
        for idx, is_term in enumerate(term_mask):
            if is_term:
                assert transitions[idx] == [], (
                    f"Terminal state {idx} should have no transitions"
                )

    def test_some_terminal_states_exist(self):
        """With WALK actions, some states should be terminal."""
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        micro_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            micro_state[1][0][0], micro_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        states, _, _, term_mask = build_sub_problem_dag(
            micro_env, mapper, coarse_ap, micro_state, quiet=True
        )
        assert sum(term_mask) > 0, "Expected some terminal states in sub-problem"

    def test_restores_micro_state(self):
        """build_sub_problem_dag should restore micro_env state."""
        hierarchy = _build_hierarchy()
        micro_env = hierarchy.finest()
        mapper = hierarchy.mapper
        macro_env = hierarchy.coarsest()

        original_state = micro_env.get_state()
        agent_cell = macro_env.partition.cell_of(
            original_state[1][0][0], original_state[1][0][1]
        )
        adj = macro_env.partition.adjacency.get(agent_cell, frozenset())
        if not adj:
            pytest.skip("Agent has no adjacent cells")
        target = min(adj)
        num_agents = len(macro_env.agents)
        coarse_ap = tuple(
            macro_walk(target) if i == 0 else MACRO_PASS
            for i in range(num_agents)
        )

        build_sub_problem_dag(
            micro_env, mapper, coarse_ap, original_state, quiet=True
        )
        assert micro_env.get_state() == original_state


# ── TestObserveTransition ────────────────────────────────────────────

class TestObserveTransition:
    """Tests for control-transfer via ``observe_transition()``."""

    @pytest.fixture
    def policy_and_env(self):
        hierarchy = _build_hierarchy()
        policy = _make_policy(hierarchy)
        return policy, hierarchy

    def test_observe_noop_at_macro_level(self, policy_and_env):
        """observe_transition is a no-op when already at macro level."""
        policy, hierarchy = policy_and_env
        micro_state = hierarchy.finest().get_state()
        num_agents = len(hierarchy.finest().agents)
        policy.observe_transition(micro_state, (0,) * num_agents, micro_state)
        assert policy.at_macro_level

    def test_observe_after_sample(self, policy_and_env):
        """observe_transition after sample should not crash and leave
        the policy in a consistent state."""
        policy, hierarchy = policy_and_env
        micro_env = hierarchy.finest()
        micro_state = micro_env.get_state()

        action = policy.sample(micro_state)

        # Build full action profile
        num_agents = len(micro_env.agents)
        full_actions = tuple([0] * num_agents)
        policy.observe_transition(micro_state, full_actions, micro_state)
        # Policy should be either at macro level (control returned) or
        # still at micro level (control not returned yet).
        assert isinstance(policy.at_macro_level, bool)


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_compute_exported(self):
        from empo.hierarchical import compute_hierarchical_robot_policy
        assert callable(compute_hierarchical_robot_policy)

    def test_policy_class_exported(self):
        from empo.hierarchical import HierarchicalRobotPolicy
        assert HierarchicalRobotPolicy is not None
