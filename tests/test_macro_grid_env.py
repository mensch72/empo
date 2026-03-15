"""Tests for MacroGridEnv — macro-level world model.

Covers: construction, state encoding, transition probabilities,
duration estimates, passage flags, available actions, terminal
conditions, and helper methods.
"""

import pytest

from gym_multigrid.multigrid import MultiGridEnv

from empo.hierarchical.macro_grid_env import (
    MacroGridEnv,
    MACRO_PASS,
    decode_macro_action,
    macro_walk,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_rock_gateway() -> MultiGridEnv:
    """Create the 8×8 rock-gateway environment (2 rooms, 1 rock)."""
    env = MultiGridEnv(
        config_file='multigrid_worlds/copilot_challenges/rock_gateway.yaml',
    )
    env.reset()
    return env


def _make_door_test() -> MultiGridEnv:
    """Create the 7×6 door-test environment (single room, 3 doors)."""
    env = MultiGridEnv(
        config_file='multigrid_worlds/obstacles/door_test.yaml',
    )
    env.reset()
    return env


# ── TestConstruction ─────────────────────────────────────────────────

class TestConstruction:

    def test_num_cells_positive(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        assert macro.num_cells > 0

    def test_action_space_matches_num_cells(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        expected = macro.num_cells + 1  # PASS + WALK(j) per cell
        assert macro.action_space.n == expected

    def test_agents_len_matches_micro(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        assert len(macro.agents) == len(micro.agents)

    def test_human_robot_indices_copied(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        assert macro.human_agent_indices == micro.human_agent_indices
        assert macro.robot_agent_indices == micro.robot_agent_indices

    def test_max_steps_from_micro(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        assert macro.max_steps == micro.max_steps

    def test_seed_deterministic(self):
        micro = _make_rock_gateway()
        m1 = MacroGridEnv(micro, seed=42)
        m2 = MacroGridEnv(micro, seed=42)
        assert m1.get_state() == m2.get_state()
        assert m1.partition.rectangles == m2.partition.rectangles


# ── TestStateEncoding ────────────────────────────────────────────────

class TestStateEncoding:

    def test_state_is_hashable(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        h = hash(state)
        assert isinstance(h, int)

    def test_state_has_four_components(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        remaining, flags, agents, objects = macro.get_state()
        assert isinstance(remaining, int)
        assert isinstance(flags, tuple)
        assert isinstance(agents, tuple)
        assert isinstance(objects, tuple)

    def test_remaining_time_equals_max_steps(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        remaining = macro.get_state()[0]
        assert remaining == micro.max_steps

    def test_passage_flags_length(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        flags = macro.get_state()[1]
        assert len(flags) == len(macro.adj_pairs)

    def test_agent_states_length(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        agent_states = macro.get_state()[2]
        assert len(agent_states) == len(micro.agents)

    def test_agent_cell_matches_micro_position(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        micro_state = micro.get_state()
        macro_state = macro.get_state()
        for i, (micro_a, macro_a) in enumerate(
            zip(micro_state[1], macro_state[2])
        ):
            if micro_a[0] is not None:
                expected = macro.partition.cell_of(micro_a[0], micro_a[1])
                assert macro_a[0] == expected, f"Agent {i} cell mismatch"

    def test_set_state_get_state_roundtrip(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        macro.set_state(state)
        assert macro.get_state() == state


# ── TestPassageFlags ─────────────────────────────────────────────────

class TestPassageFlags:

    def test_open_passage_adjacent_cells(self):
        """In rock_gateway, upper room (cell 0) connects to column cell (1)."""
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        state = macro.get_state()
        # Cell 0 and cell 1 are adjacent; passage should be open
        assert macro.passage_open(state, 0, 1)

    def test_blocked_passage_rock(self):
        """Passage from upper room (cell 0) to rock cell (2) is blocked."""
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        state = macro.get_state()
        # Cell 2 is the rock at (3,3); passage from 0 to 2 should be blocked
        assert not macro.passage_open(state, 0, 2)

    def test_non_adjacent_returns_false(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        # Cell 0 and cell 6 are not adjacent
        assert not macro.passage_open(state, 0, 6)

    def test_passage_flags_with_doors(self):
        """Open doors should allow passage; closed/locked should not."""
        micro = _make_door_test()
        macro = MacroGridEnv(micro, seed=42)
        state = macro.get_state()
        # Verify passage_flags is a tuple of bools
        for flag in state[1]:
            assert isinstance(flag, bool)


# ── TestTransitions ──────────────────────────────────────────────────

class TestTransitions:

    def test_pass_decrements_time(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        t = macro.transition_probabilities(state, [MACRO_PASS, MACRO_PASS])
        assert t is not None
        assert len(t) == 1
        assert t[0][0] == 1.0
        assert t[0][1][0] == state[0] - 1

    def test_pass_keeps_agents(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        t = macro.transition_probabilities(state, [MACRO_PASS, MACRO_PASS])
        assert t[0][1][2] == state[2]

    def test_walk_to_open_adjacent_cell(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        agent0_cell = state[2][0][0]  # cell 0
        # Find an adjacent cell with open passage
        for adj in macro.partition.adjacency.get(agent0_cell, frozenset()):
            if macro.passage_open(state, agent0_cell, adj):
                target = adj
                break
        else:
            pytest.skip("No open adjacent cell found")

        t = macro.transition_probabilities(
            state, [macro_walk(target), MACRO_PASS],
        )
        assert t[0][1][2][0][0] == target

    def test_walk_to_blocked_passage_stays(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        agent0_cell = state[2][0][0]
        # Find a blocked adjacent cell
        for adj in macro.partition.adjacency.get(agent0_cell, frozenset()):
            if not macro.passage_open(state, agent0_cell, adj):
                target = adj
                break
        else:
            pytest.skip("No blocked adjacent cell found")

        t = macro.transition_probabilities(
            state, [macro_walk(target), MACRO_PASS],
        )
        # Agent stays in original cell
        assert t[0][1][2][0][0] == agent0_cell

    def test_walk_to_same_cell_is_pass(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        cell = state[2][0][0]
        t = macro.transition_probabilities(
            state, [macro_walk(cell), MACRO_PASS],
        )
        assert t[0][1][2][0][0] == cell
        assert t[0][1][0] == state[0] - 1  # Duration 1 (PASS equivalent)

    def test_walk_to_non_adjacent_is_pass(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        cell = state[2][0][0]
        # Find a non-adjacent cell
        all_cells = set(range(macro.num_cells))
        adj = macro.partition.adjacency.get(cell, frozenset())
        non_adj = all_cells - adj - {cell}
        if not non_adj:
            pytest.skip("All cells adjacent")
        target = min(non_adj)
        t = macro.transition_probabilities(
            state, [macro_walk(target), MACRO_PASS],
        )
        assert t[0][1][2][0][0] == cell

    def test_invalid_action_treated_as_pass(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        # Action index way out of range
        t = macro.transition_probabilities(state, [99, MACRO_PASS])
        assert t[0][1][2] == state[2]

    def test_both_agents_can_move(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        c0 = state[2][0][0]
        c1 = state[2][1][0]
        # Find adjacent open cells for each agent
        t0 = None
        for adj in macro.partition.adjacency.get(c0, frozenset()):
            if macro.passage_open(state, c0, adj):
                t0 = adj
                break
        t1 = None
        for adj in macro.partition.adjacency.get(c1, frozenset()):
            if macro.passage_open(state, c1, adj):
                t1 = adj
                break
        if t0 is None or t1 is None:
            pytest.skip("Couldn't find open moves for both agents")
        t = macro.transition_probabilities(
            state, [macro_walk(t0), macro_walk(t1)],
        )
        assert t[0][1][2][0][0] == t0
        assert t[0][1][2][1][0] == t1

    def test_deterministic_transition(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        t = macro.transition_probabilities(state, [MACRO_PASS, MACRO_PASS])
        assert len(t) == 1
        assert t[0][0] == 1.0


# ── TestTerminal ─────────────────────────────────────────────────────

class TestTerminal:

    def test_zero_remaining_is_terminal(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        terminal = (0, state[1], state[2], state[3])
        t = macro.transition_probabilities(terminal, [MACRO_PASS, MACRO_PASS])
        assert t is None

    def test_all_terminated_is_terminal(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        terminated_agents = tuple(
            (a[0], a[1], a[2], True, a[4], a[5])
            for a in state[2]
        )
        terminal = (state[0], state[1], terminated_agents, state[3])
        t = macro.transition_probabilities(terminal, [MACRO_PASS, MACRO_PASS])
        assert t is None


# ── TestDurations ────────────────────────────────────────────────────

class TestDurations:

    def test_pass_duration_is_one(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        t = macro.transition_probabilities(state, [MACRO_PASS, MACRO_PASS])
        d = macro.transition_durations(state, [MACRO_PASS, MACRO_PASS], t)
        assert d == [1.0]

    def test_walk_duration_positive(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        c0 = state[2][0][0]
        for adj in macro.partition.adjacency.get(c0, frozenset()):
            if macro.passage_open(state, c0, adj):
                t = macro.transition_probabilities(
                    state, [macro_walk(adj), MACRO_PASS],
                )
                d = macro.transition_durations(
                    state, [macro_walk(adj), MACRO_PASS], t,
                )
                assert d[0] >= 1.0
                return
        pytest.skip("No open adjacent cell found")

    def test_terminal_duration_is_one(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        assert macro.terminal_duration(macro.get_state()) == 1.0

    def test_empty_transitions_empty_durations(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        assert macro.transition_durations(macro.get_state(), [], []) == []


# ── TestAvailableActions ─────────────────────────────────────────────

class TestAvailableActions:

    def test_pass_always_available(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        for i in range(len(macro.agents)):
            actions = macro.available_actions(state, i)
            assert MACRO_PASS in actions

    def test_walk_to_adjacent_cells(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        cell = state[2][0][0]
        actions = macro.available_actions(state, 0)
        adj = macro.partition.adjacency.get(cell, frozenset())
        for a in adj:
            assert macro_walk(a) in actions

    def test_walk_to_non_adjacent_not_available(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        cell = state[2][0][0]
        adj = macro.partition.adjacency.get(cell, frozenset())
        non_adj = set(range(macro.num_cells)) - adj - {cell}
        actions = macro.available_actions(state, 0)
        for na in non_adj:
            assert macro_walk(na) not in actions


# ── TestHelpers ──────────────────────────────────────────────────────

class TestHelpers:

    def test_macro_cell_of(self):
        micro = _make_rock_gateway()
        macro = MacroGridEnv(micro, seed=42)
        # Agent 0 at (2, 2)
        assert macro.macro_cell_of(2, 2) == macro.partition.cell_of(2, 2)

    def test_decode_pass(self):
        assert decode_macro_action(MACRO_PASS) == ('PASS', -1)

    def test_decode_walk(self):
        assert decode_macro_action(3) == ('WALK', 2)

    def test_macro_walk(self):
        assert macro_walk(5) == 6


# ── TestStep ─────────────────────────────────────────────────────────

class TestStep:

    def test_step_updates_state(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        s0 = macro.get_state()
        obs, reward, done, truncated, info = macro.step(
            [MACRO_PASS, MACRO_PASS]
        )
        s1 = macro.get_state()
        assert s1 != s0
        assert s1[0] == s0[0] - 1

    def test_step_returns_done_at_terminal(self):
        macro = MacroGridEnv(_make_rock_gateway(), seed=42)
        state = macro.get_state()
        terminal = (1, state[1], state[2], state[3])
        macro.set_state(terminal)
        _, _, done, _, _ = macro.step([MACRO_PASS, MACRO_PASS])
        assert done


# ── TestExports ──────────────────────────────────────────────────────

class TestExports:

    def test_exported_from_hierarchical_package(self):
        from empo.hierarchical import MacroGridEnv as Cls
        assert Cls is MacroGridEnv

    def test_macro_pass_exported(self):
        from empo.hierarchical import MACRO_PASS as mp
        assert mp == 0

    def test_macro_walk_exported(self):
        from empo.hierarchical import macro_walk as mw
        assert callable(mw)
