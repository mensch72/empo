"""
Tests for the BushWorld world model, goals, human policy, loader, rendering,
and the learning-based Phase 2 computation.

Run with::

    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
        python -m pytest tests/test_bushworld.py -v
"""

import math

import numpy as np
import pytest

from empo.bushworld import (
    Actions,
    BushWorld,
    BushWorldConfigGoalGenerator,
    ReachCellGoal,
    ReachRectangleGoal,
    ShortestPathHumanPolicyPrior,
    all_cell_goal_coords,
    load_bushworld,
    parse_bushworld_map,
)
from empo.bushworld.learning import (
    Phase2Params,
    compute_tabular_phase2,
    enumerate_reachable_states,
    phase2_local_update,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_corridor(max_steps=4, B=1):
    """1x5 corridor: robot at (2,0), humans at (0,0) and (4,0), all density 1."""
    return BushWorld(
        width=5,
        height=1,
        num_robots=1,
        num_humans=2,
        max_steps=max_steps,
        B=B,
        robot_positions=[(2, 0)],
        human_positions=[(0, 0), (4, 0)],
        initial_densities=[[1, 1, 1, 1, 1]],
    )


# --------------------------------------------------------------------------- #
# State management
# --------------------------------------------------------------------------- #
def test_get_set_state_roundtrip():
    env = make_corridor()
    s0 = env.get_state()
    # State must be hashable.
    hash(s0)
    # Mutate then restore.
    actions = [Actions.east, Actions.pass_, Actions.pass_]
    transitions = env.transition_probabilities(s0, actions)
    _, s1 = transitions[0]
    env.set_state(s1)
    assert env.get_state() == s1
    env.set_state(s0)
    assert env.get_state() == s0


def test_state_structure():
    env = make_corridor()
    step_count, positions, densities = env.get_state()
    assert step_count == 0
    # Robots first, then humans.
    assert positions[0] == (2, 0)
    assert positions[1] == (0, 0)
    assert positions[2] == (4, 0)
    assert len(densities) == env.width * env.height


# --------------------------------------------------------------------------- #
# Transition probabilities
# --------------------------------------------------------------------------- #
def test_transition_probabilities_sum_to_one():
    env = make_corridor(B=2)
    state = env.get_state()
    num_actions = env.action_space.n
    rng = np.random.default_rng(0)
    for _ in range(50):
        actions = [int(rng.integers(0, num_actions)) for _ in range(env.num_players)]
        transitions = env.transition_probabilities(state, actions)
        total = sum(p for p, _ in transitions)
        assert math.isclose(total, 1.0, rel_tol=1e-9), (actions, total)
        # All successors hashable and valid length.
        for _p, succ in transitions:
            hash(succ)


def test_no_two_players_share_a_cell():
    env = make_corridor(B=1)
    state = env.get_state()
    num_actions = env.action_space.n
    import itertools

    for actions in itertools.product(range(num_actions), repeat=env.num_players):
        for _p, succ in env.transition_probabilities(state, list(actions)):
            positions = succ[1]
            assert len(set(positions)) == len(positions), (actions, positions)


def test_robot_move_decrements_target_density():
    # Robot at (2,0) moves east into (3,0) which has density 1 -> becomes 0.
    env = make_corridor(B=2)
    env.set_state((0, ((2, 0), (0, 0), (4, 0)), (1, 1, 1, 1, 1)))
    state = env.get_state()
    transitions = env.transition_probabilities(state, [Actions.east, Actions.pass_, Actions.pass_])
    # Human passes are deterministic, robot move deterministic -> single outcome.
    assert len(transitions) == 1
    _p, succ = transitions[0]
    densities = succ[2]
    assert densities[env.cell_index(3, 0)] == 0
    assert succ[1][0] == (3, 0)


def test_robot_pass_decrements_current_density():
    env = make_corridor(B=2)
    env.set_state((0, ((2, 0), (0, 0), (4, 0)), (1, 1, 2, 1, 1)))
    state = env.get_state()
    _p, succ = env.transition_probabilities(
        state, [Actions.pass_, Actions.pass_, Actions.pass_]
    )[0]
    assert succ[2][env.cell_index(2, 0)] == 1  # 2 -> 1


def test_human_move_is_stochastic_in_bush():
    # Human at (0,0) wants to move east into (1,0) with density 1, B=2 ->
    # success prob 1 - 1/2 = 0.5.
    env = make_corridor(B=2)
    env.set_state((0, ((4, 0), (0, 0), (2, 0)), (0, 1, 0, 0, 0)))
    state = env.get_state()
    # robot pass, human0 east, human1 pass
    transitions = env.transition_probabilities(
        state, [Actions.pass_, Actions.east, Actions.pass_]
    )
    probs = sorted(p for p, _ in transitions)
    assert len(transitions) == 2
    assert math.isclose(probs[0], 0.5, rel_tol=1e-9)
    assert math.isclose(probs[1], 0.5, rel_tol=1e-9)


def test_humans_do_not_change_density():
    env = make_corridor(B=2)
    env.set_state((0, ((4, 0), (0, 0), (2, 0)), (0, 0, 0, 0, 0)))
    state = env.get_state()
    for _p, succ in env.transition_probabilities(
        state, [Actions.pass_, Actions.east, Actions.pass_]
    ):
        assert succ[2] == state[2]


def test_off_grid_robot_move_is_noop_without_density_change():
    env = make_corridor(B=2)
    # Robot at (0,0) tries to move west (off grid).
    env.set_state((0, ((0, 0), (2, 0), (4, 0)), (1, 1, 1, 1, 1)))
    state = env.get_state()
    transitions = env.transition_probabilities(
        state, [Actions.west, Actions.pass_, Actions.pass_]
    )
    assert len(transitions) == 1
    _p, succ = transitions[0]
    assert succ[1][0] == (0, 0)
    assert succ[2] == state[2]  # no density change for blocked move


def test_conflict_resolution_first_in_line_wins():
    # Two robots both move into the same empty cell; only the lower-id one enters.
    env = BushWorld(
        width=3,
        height=1,
        num_robots=2,
        num_humans=0,
        max_steps=4,
        B=1,
        robot_positions=[(0, 0), (2, 0)],
        initial_densities=[[0, 0, 0]],
    )
    state = env.get_state()
    _p, succ = env.transition_probabilities(state, [Actions.east, Actions.west])[0]
    positions = succ[1]
    # Lower-id robot (0) gets (1,0); robot 1 stays at (2,0).
    assert positions[0] == (1, 0)
    assert positions[1] == (2, 0)


# --------------------------------------------------------------------------- #
# Goals
# --------------------------------------------------------------------------- #
def test_cell_goal_is_achieved():
    env = make_corridor()
    goal = ReachCellGoal(env, human_agent_index=1, target_pos=(0, 0))
    assert goal.is_achieved(env.get_state()) == 1
    goal2 = ReachCellGoal(env, human_agent_index=1, target_pos=(3, 0))
    assert goal2.is_achieved(env.get_state()) == 0


def test_rectangle_goal_is_achieved():
    env = make_corridor()
    goal = ReachRectangleGoal(env, human_agent_index=2, target_rect=(3, 0, 4, 0))
    assert goal.is_achieved(env.get_state()) == 1  # human2 at (4,0)
    goal2 = ReachRectangleGoal(env, human_agent_index=2, target_rect=(0, 0, 1, 0))
    assert goal2.is_achieved(env.get_state()) == 0


def test_goals_are_hashable_and_equal():
    env = make_corridor()
    g1 = ReachCellGoal(env, 1, (2, 0))
    g2 = ReachCellGoal(env, 1, (2, 0))
    g3 = ReachCellGoal(env, 1, (3, 0))
    assert g1 == g2 and hash(g1) == hash(g2)
    assert g1 != g3
    assert len({g1, g2, g3}) == 2


def test_goal_generator_yields_weighted_goals():
    env = make_corridor()
    coords = all_cell_goal_coords(env.width, env.height)
    gen = BushWorldConfigGoalGenerator(env, coords)
    goals = list(gen.generate(env.get_state(), human_agent_index=1))
    assert len(goals) == env.width * env.height
    total_weight = sum(w for _g, w in goals)
    assert math.isclose(total_weight, 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Human policy prior
# --------------------------------------------------------------------------- #
def test_shortest_path_policy_moves_toward_goal():
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    # Human 1 at (0,0), goal at (4,0): should prefer moving east.
    goal = ReachCellGoal(env, 1, (4, 0))
    dist = hpp(env.get_state(), human_agent_index=1, possible_goal=goal)
    assert dist.shape == (env.action_space.n,)
    assert math.isclose(dist.sum(), 1.0, rel_tol=1e-9)
    assert int(np.argmax(dist)) == int(Actions.east)


def test_policy_distribution_independent_product():
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    profiles = hpp.profile_distribution(env.get_state())
    total = sum(p for p, _ in profiles)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def test_parse_bushworld_map():
    text = "\n".join(["Hu . Ro . Hu"])
    width, height, densities, robot_positions, human_positions = parse_bushworld_map(
        text, fill_density=1
    )
    assert width == 5
    assert height == 1
    assert len(robot_positions) == 1
    assert len(human_positions) == 2
    # Player cells carry the fill density.
    assert densities[0][0] == 1  # human
    assert densities[0][2] == 1  # robot


def test_load_example_yaml():
    env = load_bushworld("bushworld_worlds/two_humans_one_robot.yaml")
    assert env.num_robots == 1
    assert env.num_humans == 2
    state = env.get_state()
    assert sum(p for p, _ in env.transition_probabilities(
        state, [Actions.pass_] * env.num_players)) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Learning-based Phase 2
# --------------------------------------------------------------------------- #
def test_tabular_learner_matches_backward_induction():
    from empo.backward_induction.phase2 import compute_robot_policy

    env = make_corridor(max_steps=4, B=1)
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    gen = env.possible_goal_generator
    params = Phase2Params(beta_r=4.0, gamma_h=0.9, gamma_r=0.9)

    bi = compute_robot_policy(
        env, list(env.human_agent_indices), list(env.robot_agent_indices),
        gen, hpp, beta_r=params.beta_r, gamma_h=params.gamma_h, gamma_r=params.gamma_r,
        zeta=params.zeta, xi=params.xi, eta=params.eta,
        level_fct=lambda s: s[0], quiet=True,
    )
    pol, _tables, hist = compute_tabular_phase2(env, hpp, params, quiet=True)
    assert hist["iterations"] >= 1

    states = [s for s in enumerate_reachable_states(env, hpp) if not env.is_terminal(s)]
    max_diff = 0.0
    for s in states:
        d_bi, d_lt = bi(s), pol(s)
        for k in set(d_bi) | set(d_lt):
            max_diff = max(max_diff, abs(d_bi.get(k, 0.0) - d_lt.get(k, 0.0)))
    # Backward induction uses float16 caches internally, so allow a small tolerance.
    assert max_diff < 1e-3, max_diff


def test_tabular_policy_save_load(tmp_path):
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    params = Phase2Params(beta_r=3.0)
    pol, _tables, _hist = compute_tabular_phase2(env, hpp, params, quiet=True)
    path = tmp_path / "policy.pkl"
    pol.save(str(path))
    from empo.bushworld.learning import LearnedTabularRobotPolicy

    loaded = LearnedTabularRobotPolicy.load(str(path), env)
    s = env.initial_state()
    assert pol(s) == loaded(s)


def test_phase2_local_update_produces_valid_policy():
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    gen = env.possible_goal_generator
    params = Phase2Params()
    robot_profiles = [(a,) for a in range(env.action_space.n)]

    def vr_fn(succ):
        return params.terminal_Vr

    def vhe_fn(succ, agent_index, goal):
        return 0.0

    qr, pi_r, vr, vhe, ur = phase2_local_update(
        env, hpp, gen, env.initial_state(), params,
        vr_fn, vhe_fn, robot_profiles, list(env.human_agent_indices),
    )
    assert np.all(qr < 0.0)  # Q_r always negative
    assert math.isclose(float(pi_r.sum()), 1.0, rel_tol=1e-9)
    assert ur < 0.0  # U_r always negative


@pytest.mark.parametrize("method", ["tabular"])
def test_train_dispatcher_checkpoint_resume(tmp_path, method):
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    params = Phase2Params(beta_r=3.0)
    from empo.bushworld.learning import train_bushworld_phase2

    cp = tmp_path / "ckpt.pkl"
    pol1, h1 = train_bushworld_phase2(
        env, hpp, params, method=method, checkpoint_path=str(cp), max_iterations=2
    )
    assert cp.exists()
    pol2, h2 = train_bushworld_phase2(
        env, hpp, params, method=method, checkpoint_path=str(cp), max_iterations=20
    )
    # Resumed run should converge.
    assert h2["final_delta"] < 1e-6


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
