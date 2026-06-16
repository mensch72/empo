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
from empo.backward_induction.phase2 import compute_robot_policy
from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.bushworld.phase2 import (
    BushWorldRobotPolicy,
    create_phase2_networks,
    train_bushworld_phase2,
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


def _reachable_states(env, max_profiles=50):
    """Breadth-first enumeration of reachable states from the initial state."""
    import itertools

    seen = set()
    frontier = [env.initial_state()]
    profiles = list(
        itertools.product(range(env.action_space.n), repeat=env.num_players)
    )[:max_profiles]
    while frontier:
        s = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        if env.is_terminal(s):
            continue
        for pr in profiles:
            for p, ns in env.transition_probabilities(s, list(pr)):
                if p > 0 and ns not in seen:
                    frontier.append(ns)
    return seen


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
# Learning-based Phase 2 (shared infrastructure, mirroring multigrid)
# --------------------------------------------------------------------------- #
def _bi_policy(env, hpp, beta_r=4.0, gamma=0.9):
    gen = env.possible_goal_generator
    return compute_robot_policy(
        env,
        list(env.human_agent_indices),
        list(env.robot_agent_indices),
        gen,
        hpp,
        beta_r=beta_r,
        gamma_h=gamma,
        gamma_r=gamma,
        level_fct=lambda s: s[0],
        quiet=True,
    )


def _lookup_config(beta_r=4.0, gamma=0.9, num_training_steps=400):
    return Phase2Config(
        use_lookup_tables=True,
        use_lookup_q_r=True,
        use_lookup_v_h_e=True,
        use_lookup_x_h=True,
        use_lookup_u_r=True,
        use_lookup_v_r=True,
        u_r_use_network=True,
        v_r_use_network=True,
        lookup_use_adaptive_lr=True,
        gamma_r=gamma,
        gamma_h=gamma,
        beta_r=beta_r,
        warmup_v_h_e_steps=30,
        warmup_x_h_steps=30,
        warmup_u_r_steps=30,
        warmup_q_r_steps=30,
        beta_r_rampup_steps=30,
        num_training_steps=num_training_steps,
        steps_per_episode=4,
        buffer_size=500,
        batch_size=32,
        use_count_based_curiosity=True,
        epsilon_r_start=0.6,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=200,
    )


def _neural_config(beta_r=4.0, gamma=0.9, num_training_steps=60):
    return Phase2Config(
        use_lookup_tables=False,
        use_encoders=True,
        u_r_use_network=False,
        v_r_use_network=False,
        x_h_use_network=True,
        gamma_r=gamma,
        gamma_h=gamma,
        beta_r=beta_r,
        hidden_dim=32,
        goal_feature_dim=16,
        warmup_v_h_e_steps=10,
        warmup_x_h_steps=10,
        warmup_u_r_steps=0,
        warmup_q_r_steps=10,
        beta_r_rampup_steps=10,
        num_training_steps=num_training_steps,
        steps_per_episode=4,
        buffer_size=200,
        batch_size=16,
        epsilon_r_start=0.5,
        epsilon_r_end=0.1,
        epsilon_r_decay_steps=40,
    )


def _train(env, hpp, config, tensorboard_dir):
    gen = env.possible_goal_generator
    return train_bushworld_phase2(
        env,
        list(env.human_agent_indices),
        list(env.robot_agent_indices),
        hpp,
        gen.get_sampler(),
        config=config,
        verbose=False,
        tensorboard_dir=str(tensorboard_dir),
    )


def test_backward_induction_valid_policy():
    env = make_corridor(max_steps=4, B=1)
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    bi = _bi_policy(env, hpp, beta_r=4.0, gamma=0.9)

    for s in _reachable_states(env):
        if env.is_terminal(s):
            continue
        dist = bi(s)
        assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-3)
        assert all(p >= 0.0 for p in dist.values())


def test_create_phase2_networks_lookup_and_neural():
    env = make_corridor()
    num_actions = env.action_space.n

    lookup_nets = create_phase2_networks(
        env, _lookup_config(), env.num_robots, num_actions, device="cpu"
    )
    neural_nets = create_phase2_networks(
        env, _neural_config(), env.num_robots, num_actions, device="cpu"
    )
    for nets in (lookup_nets, neural_nets):
        for name in ("q_r", "v_h_e", "x_h"):
            assert getattr(nets, name) is not None
    # u_r / v_r are networks only when the config requests them.
    assert lookup_nets.u_r is not None
    assert lookup_nets.v_r is not None

    # q_r forward produces one (negative) value per joint action.
    q = neural_nets.q_r.forward(env.initial_state(), env, "cpu")
    assert q.numel() == num_actions ** env.num_robots
    assert bool((q < 0).all())


def test_lookup_phase2_trains_and_valid_policy(tmp_path):
    env = make_corridor(max_steps=4, B=1)
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)

    q_r, _nets, _hist, _trainer = _train(
        env, hpp, _lookup_config(beta_r=4.0, gamma=0.9), tmp_path / "tb"
    )

    # The shared (sampling-based) infrastructure should yield a valid robot
    # policy at every reachable non-terminal state: probabilities sum to one
    # and every Q_r value is negative.
    states = [s for s in _reachable_states(env) if not env.is_terminal(s)]
    assert states
    for s in states:
        q = q_r.forward(s, env, "cpu")
        pi = q_r.get_policy(q, beta_r=4.0).detach().numpy().ravel()
        assert math.isclose(float(pi.sum()), 1.0, rel_tol=1e-6)
        assert bool((q.detach().numpy() < 0).all())


def test_neural_phase2_trains_and_valid_policy(tmp_path):
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)

    q_r, nets, _hist, _trainer = _train(
        env, hpp, _neural_config(), tmp_path / "tb"
    )
    assert q_r.__class__.__name__ == "BushWorldRobotQNetwork"

    q = q_r.forward(env.initial_state(), env, "cpu")
    pi = q_r.get_policy(q, beta_r=4.0).detach().numpy().ravel()
    assert math.isclose(float(pi.sum()), 1.0, rel_tol=1e-6)
    assert bool((q.detach().numpy() < 0).all())


def test_robot_policy_save_load(tmp_path):
    env = make_corridor()
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    _q_r, _nets, _hist, trainer = _train(
        env, hpp, _neural_config(), tmp_path / "tb"
    )

    path = tmp_path / "policy.pt"
    trainer.save_policy(str(path))
    assert path.exists()

    pol = BushWorldRobotPolicy(path=str(path))
    pol.reset(env)
    action = pol.sample(env.initial_state())
    assert isinstance(action, tuple)
    assert len(action) == env.num_robots
    assert all(0 <= a < env.action_space.n for a in action)



if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
