"""Tests for MultiGridLevelMapper — level mapping between M^0 and M^1.

Covers: super_state, super_agent, is_feasible, is_abort, return_control.
"""

import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical.macro_grid_env import MACRO_PASS, macro_walk
from empo.hierarchical.multigrid_level_mapper import MultiGridLevelMapper
from empo.hierarchical.two_level_multigrid import TwoLevelMultigrid


# ── helpers ──────────────────────────────────────────────────────────

def _make_hierarchy():
    """Build a TwoLevelMultigrid from rock_gateway."""
    env = MultiGridEnv(
        config_file='multigrid_worlds/copilot_challenges/rock_gateway.yaml',
    )
    env.reset()
    return TwoLevelMultigrid(env, seed=42)


# ── TestSuperState ───────────────────────────────────────────────────

class TestSuperState:

    def test_returns_hashable_tuple(self):
        h = _make_hierarchy()
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        assert isinstance(hash(macro_state), int)

    def test_remaining_time_from_step_count(self):
        h = _make_hierarchy()
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        # In new encoding, micro_state[0] is already remaining_steps!
        expected = micro_state[0]
        assert macro_state[0] == expected

    def test_agent_cell_matches_micro_position(self):
        h = _make_hierarchy()
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        for i, (micro_a, macro_a) in enumerate(
            zip(micro_state[1], macro_state[2])
        ):
            if micro_a[0] is not None:
                expected = h.macro_env.partition.cell_of(
                    micro_a[0], micro_a[1],
                )
                assert macro_a[0] == expected

    def test_passage_flags_are_bools(self):
        h = _make_hierarchy()
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        for flag in macro_state[1]:
            assert isinstance(flag, bool)

    def test_consistent_with_macro_initial_state(self):
        h = _make_hierarchy()
        micro_state = h.micro_env.get_state()
        macro_state = h.mapper.super_state(micro_state)
        initial_macro = h.macro_env.get_state()
        assert macro_state == initial_macro


# ── TestSuperAgent ───────────────────────────────────────────────────

class TestSuperAgent:

    def test_identity_mapping(self):
        h = _make_hierarchy()
        for i in range(len(h.micro_env.agents)):
            assert h.mapper.super_agent(i) == i


# ── TestIsFeasible ───────────────────────────────────────────────────

class TestIsFeasible:

    def test_pass_always_feasible(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert h.mapper.is_feasible(
            (MACRO_PASS, MACRO_PASS), micro, (0, 0),
        )

    def test_still_is_feasible_with_walk(self):
        """Still (action 0) is compatible with WALK — it just aborts."""
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert h.mapper.is_feasible(
            (macro_walk(1), MACRO_PASS), micro, (0, 0),
        )

    def test_turn_is_feasible_with_walk(self):
        """Turning actions (left=1, right=2) are always feasible with WALK."""
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Action 1 = left turn in standard MultiGrid action space
        assert h.mapper.is_feasible(
            (macro_walk(1), MACRO_PASS), micro, (1, 0),
        )

    def test_forward_within_cell_feasible(self):
        """Forward that stays within the same cell is feasible."""
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Agent 0 at (2,2) facing right (dir=0), forward→(3,2) is cell 0
        assert h.mapper.is_feasible(
            (macro_walk(1), MACRO_PASS), micro, (3, 0),  # forward=3
        )

    def test_forward_to_target_cell_feasible(self):
        """Forward into the target macro-cell is feasible."""
        h = _make_hierarchy()
        # Place agent at border of cell 0 facing toward cell 1
        micro = h.micro_env.get_state()
        # Agent 0 at (5,1) facing right (dir=0), forward→(6,1) is cell 1
        agents = list(micro[1])
        agents[0] = (5, 1, 0, False, True, False, None, None, None)
        modified = (micro[0], tuple(agents), micro[2], micro[3])
        assert h.mapper.is_feasible(
            (macro_walk(1), MACRO_PASS), modified, (3, 0),
        )

    def test_forward_to_wrong_cell_infeasible(self):
        """Forward into a cell other than the target is infeasible."""
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Agent 0 at (5,1) facing right → goes to cell 1
        # But WALK target is cell 2 (the rock cell)
        agents = list(micro[1])
        agents[0] = (5, 1, 0, False, True, False, None, None, None)
        modified = (micro[0], tuple(agents), micro[2], micro[3])
        assert not h.mapper.is_feasible(
            (macro_walk(2), MACRO_PASS), modified, (3, 0),
        )


# ── TestIsAbort ──────────────────────────────────────────────────────

class TestIsAbort:

    def test_still_with_walk_is_abort(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert h.mapper.is_abort(
            (macro_walk(1), MACRO_PASS), micro, (0, 0),
        )

    def test_still_with_pass_is_not_abort(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert not h.mapper.is_abort(
            (MACRO_PASS, MACRO_PASS), micro, (0, 0),
        )

    def test_forward_with_walk_is_not_abort(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert not h.mapper.is_abort(
            (macro_walk(1), MACRO_PASS), micro, (3, 0),
        )

    def test_second_agent_abort(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert h.mapper.is_abort(
            (MACRO_PASS, macro_walk(3)), micro, (3, 0),  # agent 1 does still
        )


# ── TestReturnControl ────────────────────────────────────────────────

class TestReturnControl:

    def test_abort_returns_control(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Still + WALK → abort → return control
        assert h.mapper.return_control(
            (macro_walk(1), MACRO_PASS), micro, (0, 0), micro,
        )

    def test_reaching_target_returns_control(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Agent 0 was in cell 0, now in cell 1 (at position 6,2)
        agents = list(micro[1])
        agents[0] = (6, 2, 0, False, True, False, None, None, None)
        successor = (micro[0], tuple(agents), micro[2], micro[3])
        assert h.mapper.return_control(
            (macro_walk(1), MACRO_PASS), micro, (3, 0), successor,
        )

    def test_not_reaching_target_keeps_control(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        # Agent 0 is still in cell 0 after forward action
        assert not h.mapper.return_control(
            (macro_walk(1), MACRO_PASS), micro, (3, 0), micro,
        )

    def test_pass_does_not_trigger_return(self):
        h = _make_hierarchy()
        micro = h.micro_env.get_state()
        assert not h.mapper.return_control(
            (MACRO_PASS, MACRO_PASS), micro, (0, 0), micro,
        )


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_exported_from_hierarchical_package(self):
        from empo.hierarchical import MultiGridLevelMapper as Cls
        assert Cls is MultiGridLevelMapper
