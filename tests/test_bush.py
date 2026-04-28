"""
Tests for robot-tramplable bushes.
"""

import numpy as np

from gym_multigrid.multigrid import Agent, Bush, Grid, MultiGridEnv, Wall, World


class BushTestEnv(MultiGridEnv):
    """Small corridor with a bush between a robot and open floor."""

    def __init__(self, robot=True, trample_probability=1.0):
        agent = Agent(
            World,
            5 if robot else 4,
            can_push_rocks=robot,
            can_enter_magic_walls=robot,
        )
        super().__init__(
            width=5,
            height=5,
            max_steps=10,
            agents=[agent],
            partial_obs=False,
            objects_set=World,
            trample_probability=trample_probability,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))

        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0
        self.grid.set(1, 2, self.agents[0])
        self.grid.set(2, 2, Bush(World, trample_probability=self.trample_probability))


class BushHumanPassageEnv(MultiGridEnv):
    """Robot tramples a bush, then a human can pass through the cleared cell."""

    def __init__(self):
        robot = Agent(World, 5, can_push_rocks=True, can_enter_magic_walls=True)
        human = Agent(World, 4, can_push_rocks=False, can_enter_magic_walls=False)
        super().__init__(
            width=5,
            height=5,
            max_steps=10,
            agents=[robot, human],
            partial_obs=False,
            objects_set=World,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        for x in range(width):
            self.grid.set(x, 0, Wall(World))
            self.grid.set(x, height - 1, Wall(World))
        for y in range(height):
            self.grid.set(0, y, Wall(World))
            self.grid.set(width - 1, y, Wall(World))

        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0
        self.grid.set(1, 2, self.agents[0])

        self.agents[1].pos = np.array([2, 3])
        self.agents[1].dir = 3
        self.grid.set(2, 3, self.agents[1])

        self.grid.set(2, 2, Bush(World))


def test_bush_creation():
    bush = Bush(World)

    assert bush.type == 'bush'
    assert bush.color == 'green'
    assert bush.trampled is False
    assert bush.trample_probability == 1.0
    assert not bush.can_overlap()
    assert not bush.can_pickup()


def test_bush_creation_with_probability():
    bush = Bush(World, trample_probability=0.5)
    assert bush.trample_probability == 0.5


def test_human_cannot_enter_untrampled_bush():
    env = BushTestEnv(robot=False)
    env.reset()

    obs, rewards, done, info = env.step([3])

    assert np.array_equal(env.agents[0].pos, [1, 2])
    bush = env.grid.get(2, 2)
    assert bush is not None
    assert bush.type == 'bush'
    assert bush.trampled is False


def test_robot_tramples_bush_and_state_tracks_it():
    env = BushTestEnv(robot=True)
    env.reset()

    assert env.can_forward(env.get_state(), 0) is True
    obs, rewards, done, info = env.step([3])

    assert np.array_equal(env.agents[0].pos, [2, 2])
    assert env.grid.get(2, 2) is env.agents[0]
    assert ('bush', 2, 2, True) in env.get_state()[3]


def test_human_can_pass_after_robot_tramples_bush():
    env = BushHumanPassageEnv()
    env.reset()

    env.step([3, 0])  # Robot tramples bush, human waits.
    env.step([3, 3])  # Robot leaves; human's target was occupied at step start.
    env.step([0, 3])  # Human can now enter the cleared cell.

    assert np.array_equal(env.agents[1].pos, [2, 2])
    assert ('bush', 2, 2, True) in env.get_state()[3]


def test_set_state_restores_untrampled_bush():
    env = BushTestEnv(robot=True)
    env.reset()
    initial_state = env.get_state()

    env.step([3])
    assert ('bush', 2, 2, True) in env.get_state()[3]

    env.set_state(initial_state)
    bush = env.grid.get(2, 2)
    assert bush is not None
    assert bush.type == 'bush'
    assert bush.trampled is False


def test_probabilistic_bush_transition_probabilities():
    """transition_probabilities returns 2 outcomes when trample_probability < 1."""
    env = BushTestEnv(robot=True, trample_probability=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])  # forward action

    assert outcomes is not None
    assert len(outcomes) == 2, f"Expected 2 outcomes, got {len(outcomes)}"
    probs = sorted([p for p, _ in outcomes], reverse=True)
    assert abs(probs[0] - 0.6) < 1e-9, f"Expected succeed prob 0.6, got {probs[0]}"
    assert abs(probs[1] - 0.4) < 1e-9, f"Expected fail prob 0.4, got {probs[1]}"


def test_probabilistic_bush_succeed_outcome_moves_agent():
    """When trampling succeeds, agent moves into the cell and bush is trampled."""
    env = BushTestEnv(robot=True, trample_probability=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    # Find the succeed outcome (agent at [2,2])
    succeed_state = None
    for prob, s in outcomes:
        # Restore state, set to candidate, check agent pos
        env.set_state(s)
        if np.array_equal(env.agents[0].pos, [2, 2]):
            succeed_state = (prob, s)
            break

    assert succeed_state is not None, "No succeed outcome found"
    assert abs(succeed_state[0] - 0.6) < 1e-9
    # Bush should be recorded as trampled in successor state
    assert ('bush', 2, 2, True) in succeed_state[1][3]


def test_probabilistic_bush_fail_outcome_keeps_agent():
    """When trampling fails, agent stays put and bush remains."""
    env = BushTestEnv(robot=True, trample_probability=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    # Find the fail outcome (agent still at [1,2])
    fail_state = None
    for prob, s in outcomes:
        env.set_state(s)
        if np.array_equal(env.agents[0].pos, [1, 2]):
            fail_state = (prob, s)
            break

    assert fail_state is not None, "No fail outcome found"
    assert abs(fail_state[0] - 0.4) < 1e-9
    # Bush should still be untrampled
    assert ('bush', 2, 2, False) in fail_state[1][3]


def test_deterministic_bush_has_single_outcome():
    """With trample_probability=1.0, transition_probabilities returns a single outcome."""
    env = BushTestEnv(robot=True, trample_probability=1.0)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    assert outcomes is not None
    assert len(outcomes) == 1
    assert abs(outcomes[0][0] - 1.0) < 1e-9


def test_step_consistent_with_transition_probabilities_for_probabilistic_bush():
    """step() and transition_probabilities() produce consistent distributions."""
    env = BushTestEnv(robot=True, trample_probability=0.5)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])
    outcome_states = {s for _, s in outcomes}

    # Run many steps and verify every resulting state is a valid outcome
    n_trials = 200
    for _ in range(n_trials):
        env.set_state(state)
        env.step([3])
        resulting_state = env.get_state()
        assert resulting_state in outcome_states, (
            "step() produced a state not present in transition_probabilities() outcomes"
        )


def test_probabilistic_bush_via_config():
    """trample_probability from MultiGridEnv constructor applies to map-parsed bushes."""
    env = MultiGridEnv(
        map="We We We We\nWe Ae Bu We\nWe We We We",
        trample_probability=0.5,
        objects_set=World,
        partial_obs=False,
    )
    env.reset()
    # Find a bush in the grid
    found_bush = None
    for y in range(env.grid.height):
        for x in range(env.grid.width):
            cell = env.grid.get(x, y)
            if cell is not None and cell.type == 'bush':
                found_bush = cell
                break
        if found_bush:
            break
    assert found_bush is not None, "No bush found in the grid"
    assert found_bush.trample_probability == 0.5
