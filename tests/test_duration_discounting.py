#!/usr/bin/env python3
"""
Tests for duration-aware discounting in Phase 1 and Phase 2 backward induction.

Tests that:
1. Default durations (all 1.0) produce identical results to non-duration-aware code
2. Non-uniform durations produce different results from uniform durations
3. Longer durations lead to more discounting (lower V values)
4. Zero-rho (gamma=1.0) skips duration computation entirely
"""

import sys
import numpy as np
import pytest

import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, SmallActions, SmallWorld
)
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import (
    compute_human_policy_prior,
    compute_robot_policy,
)


# =============================================================================
# Simple Environment for duration tests (3x3 grid, 1 human + 1 robot)
# =============================================================================

class DurationTestEnv(MultiGridEnv):
    """3x3 grid environment with configurable transition durations."""

    def __init__(self, durations_map=None):
        """
        Args:
            durations_map: Optional dict mapping (state_tuple, action_tuple) -> list of
                float durations, one per transition outcome. Keys are (state, tuple(actions))
                where state is from get_state() and actions is the joint action profile.
                If None, uses default durations (all 1.0).
        """
        agents = [
            Agent(SmallWorld, 0, view_size=3),  # Human
            Agent(SmallWorld, 1, view_size=3),  # Robot
        ]
        super().__init__(
            grid_size=3,
            max_steps=2,
            agents=agents,
            agent_view_size=3,
            actions_set=SmallActions,
        )
        self._durations_map = durations_map

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.horz_wall(SmallWorld, 0, 0)
        self.grid.horz_wall(SmallWorld, 0, height-1)
        self.grid.vert_wall(SmallWorld, 0, 0)
        self.grid.vert_wall(SmallWorld, width-1, 0)
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0
        self.agents[1].pos = np.array([1, 1])
        self.agents[1].dir = 0

    @property
    def human_agent_indices(self):
        return [0]

    @property
    def robot_agent_indices(self):
        return [1]

    def transition_durations(self, state, actions, transitions):
        if self._durations_map is not None:
            key = (state, tuple(actions))
            if key in self._durations_map:
                return self._durations_map[key]
        return [1.0] * len(transitions)


# =============================================================================
# Goal definitions (reused from test_robot_policy_backward_induction.py)
# =============================================================================

class ReachRectGoal(PossibleGoal):
    """Goal: human agent is within a rectangle."""

    def __init__(self, env, human_agent_index, rect):
        self.env = env
        self.human_agent_index = human_agent_index
        self.rect = rect
        self._freeze()

    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[self.human_agent_index]
        agent_x, agent_y = agent_state[0], agent_state[1]
        x_min, y_min, x_max, y_max = self.rect
        if x_min <= agent_x <= x_max and y_min <= agent_y <= y_max:
            return 1
        return 0

    def __hash__(self):
        return hash((self.human_agent_index, self.rect))

    def __eq__(self, other):
        return (isinstance(other, ReachRectGoal) and
                self.human_agent_index == other.human_agent_index and
                self.rect == other.rect)


class WholeMapGoalGenerator(PossibleGoalGenerator):
    """Goal generator covering the whole walkable area."""

    def __init__(self, world_model, grid_size: int):
        super().__init__(world_model)
        self.grid_size = grid_size

    def generate(self, state, human_agent_index: int):
        whole_map = ReachRectGoal(
            self.env, human_agent_index,
            (1, 1, self.grid_size - 2, self.grid_size - 2)
        )
        yield (whole_map, 1.0)


# =============================================================================
# Phase 1 Duration Tests
# =============================================================================

def test_phase1_gamma_one_skips_durations():
    """With gamma_h=1.0, duration computation is skipped (rho=0)."""
    
    class NoDurationEnv(DurationTestEnv):
        """Env that raises if transition_durations is ever called."""
        def transition_durations(self, state, actions, transitions):
            raise AssertionError("transition_durations should not be called when gamma_h=1.0")
    
    env = NoDurationEnv()
    goal_gen = WholeMapGoalGenerator(env, 3)
    # This should run without calling transition_durations at all
    policy = compute_human_policy_prior(
        env, human_agent_indices=[0],
        possible_goal_generator=goal_gen,
        beta_h=10.0, gamma_h=1.0, quiet=True,
    )
    # Verify policy exists and has valid probabilities
    env.reset()
    state = env.get_state()
    for goal, _ in goal_gen.generate(state, 0):
        probs = policy(state, 0, goal)
        assert probs is not None
        assert abs(sum(probs) - 1.0) < 1e-6


def test_phase1_default_durations_match_uniform():
    """Default durations (all 1.0) produce same results with gamma_h < 1."""
    # With default durations and gamma_h=0.9:
    # e^{-rho*1} = gamma_h = 0.9
    # So per-transition discount = 0.9, same as the constant gamma_h
    env_default = DurationTestEnv()  # default durations (all 1.0)

    class ExplicitUnitEnv(DurationTestEnv):
        """Env that explicitly returns [1.0] durations."""
        def transition_durations(self, state, actions, transitions):
            return [1.0] * len(transitions)

    env_explicit = ExplicitUnitEnv()

    goal_gen_d = WholeMapGoalGenerator(env_default, 3)
    goal_gen_e = WholeMapGoalGenerator(env_explicit, 3)

    _, Vh_default = compute_human_policy_prior(
        env_default, human_agent_indices=[0],
        possible_goal_generator=goal_gen_d,
        beta_h=10.0, gamma_h=0.9, quiet=True,
        return_Vh=True,
    )
    _, Vh_explicit = compute_human_policy_prior(
        env_explicit, human_agent_indices=[0],
        possible_goal_generator=goal_gen_e,
        beta_h=10.0, gamma_h=0.9, quiet=True,
        return_Vh=True,
    )

    # Compare Vh values at all states
    for state in Vh_default:
        assert state in Vh_explicit, f"State {state} missing from explicit-unit run"
        for agent_idx in Vh_default[state]:
            for goal in Vh_default[state][agent_idx]:
                v_d = Vh_default[state][agent_idx][goal]
                v_e = Vh_explicit[state][agent_idx].get(goal, 0)
                assert abs(float(v_d) - float(v_e)) < 1e-4, \
                    f"Vh mismatch at state agent={agent_idx} goal={goal}: default={v_d}, explicit={v_e}"


def test_phase1_larger_durations_reduce_values():
    """Longer durations lead to more discounting (lower V values) when gamma_h < 1."""
    env_fast = DurationTestEnv()  # default: all durations = 1.0

    # Create env with longer durations (all 3.0)
    class SlowDurationEnv(DurationTestEnv):
        def transition_durations(self, state, actions, transitions):
            return [3.0] * len(transitions)

    env_slow = SlowDurationEnv()

    goal_gen_fast = WholeMapGoalGenerator(env_fast, 3)
    goal_gen_slow = WholeMapGoalGenerator(env_slow, 3)

    gamma_h = 0.9  # rho_h = -ln(0.9) ≈ 0.105

    _, Vh_fast = compute_human_policy_prior(
        env_fast, human_agent_indices=[0],
        possible_goal_generator=goal_gen_fast,
        beta_h=10.0, gamma_h=gamma_h, quiet=True,
        return_Vh=True,
    )
    _, Vh_slow = compute_human_policy_prior(
        env_slow, human_agent_indices=[0],
        possible_goal_generator=goal_gen_slow,
        beta_h=10.0, gamma_h=gamma_h, quiet=True,
        return_Vh=True,
    )

    # Compare V values at the initial state
    env_fast.reset()
    init_state = env_fast.get_state()

    fast_vals = Vh_fast.get(init_state, {}).get(0, {})
    slow_vals = Vh_slow.get(init_state, {}).get(0, {})

    # Longer durations mean more discounting, so V values should be lower (or equal)
    for goal in fast_vals:
        if goal in slow_vals:
            assert slow_vals[goal] <= fast_vals[goal] + 1e-6, \
                f"Slow env V ({slow_vals[goal]}) should be <= fast env V ({fast_vals[goal]})"


# =============================================================================
# Phase 2 Duration Tests
# =============================================================================

def test_phase2_gamma_r_one_skips_durations():
    """With gamma_r=1.0 and gamma_h=1.0, duration computation is skipped."""
    env = DurationTestEnv()
    goal_gen = WholeMapGoalGenerator(env, 3)
    policy_prior = compute_human_policy_prior(
        env, human_agent_indices=[0],
        possible_goal_generator=goal_gen,
        beta_h=10.0, gamma_h=1.0, quiet=True,
    )
    robot_policy = compute_robot_policy(
        env, human_agent_indices=[0], robot_agent_indices=[1],
        possible_goal_generator=goal_gen,
        human_policy_prior=policy_prior,
        beta_r=5.0, gamma_h=1.0, gamma_r=1.0, quiet=True,
    )
    # Verify policy exists
    assert robot_policy is not None
    env.reset()
    state = env.get_state()
    action = robot_policy.sample(state)
    assert action is not None


def test_phase2_default_durations_runs():
    """Phase 2 with default durations and gamma < 1 runs without error."""
    env = DurationTestEnv()
    goal_gen = WholeMapGoalGenerator(env, 3)
    policy_prior = compute_human_policy_prior(
        env, human_agent_indices=[0],
        possible_goal_generator=goal_gen,
        beta_h=10.0, gamma_h=0.95, quiet=True,
    )
    robot_policy = compute_robot_policy(
        env, human_agent_indices=[0], robot_agent_indices=[1],
        possible_goal_generator=goal_gen,
        human_policy_prior=policy_prior,
        beta_r=5.0, gamma_h=0.95, gamma_r=0.95, quiet=True,
    )
    assert robot_policy is not None


def test_phase2_longer_durations_produce_different_values():
    """Phase 2 with non-uniform durations produces different robot values."""
    env_default = DurationTestEnv()

    class LongDurationEnv(DurationTestEnv):
        def transition_durations(self, state, actions, transitions):
            return [5.0] * len(transitions)

    env_long = LongDurationEnv()

    gamma_h, gamma_r = 0.95, 0.95

    goal_gen_d = WholeMapGoalGenerator(env_default, 3)
    goal_gen_l = WholeMapGoalGenerator(env_long, 3)

    pp_d = compute_human_policy_prior(
        env_default, [0], goal_gen_d, beta_h=10.0, gamma_h=gamma_h, quiet=True)
    rp_d, Vr_d, _Vh_d = compute_robot_policy(
        env_default, [0], [1], goal_gen_d, pp_d,
        beta_r=5.0, gamma_h=gamma_h, gamma_r=gamma_r, quiet=True,
        return_values=True)

    pp_l = compute_human_policy_prior(
        env_long, [0], goal_gen_l, beta_h=10.0, gamma_h=gamma_h, quiet=True)
    rp_l, Vr_l, _Vh_l = compute_robot_policy(
        env_long, [0], [1], goal_gen_l, pp_l,
        beta_r=5.0, gamma_h=gamma_h, gamma_r=gamma_r, quiet=True,
        return_values=True)

    # Both should be valid (not NaN, not inf)
    # Vr_d and Vr_l are dicts mapping state -> Vr value
    env_default.reset()
    init_state = env_default.get_state()

    assert init_state in Vr_d
    assert init_state in Vr_l
    assert np.isfinite(Vr_d[init_state])
    assert np.isfinite(Vr_l[init_state])

    # Non-uniform durations (5.0) should produce different Vr from default (1.0)
    assert Vr_d[init_state] != Vr_l[init_state], \
        f"Expected different Vr values but got same: default={Vr_d[init_state]}, long={Vr_l[init_state]}"
