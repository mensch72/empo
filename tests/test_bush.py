"""
Tests for bush semantics.

Semantics
---------
* **Robots** (can_push_rocks / can_enter_magic_walls / grey) always **trample** the
  bush with certainty: the bush is permanently removed from the grid and the cell
  becomes empty.  ``enter_bush_success_prob`` does not affect robots.

* **Human** agents (all others) **enter** the bush without trampling it.  The agent
  occupies the cell alongside the bush (like terrain); when the agent leaves the
  cell the bush is restored.  Success is stochastic with probability
  ``enter_bush_success_prob`` (default 1.0, i.e. always succeeds by default).

Key invariants tested here
--------------------------
* After a robot enters a bush  -> bush is gone (trampled=True).
* After a human enters a bush  -> bush is still there (trampled=False), agent at (2,2).
* When a human leaves the bush cell -> bush is still there (restored to grid).
* ``can_forward()`` returns True for both agent types when facing a bush.
* ``transition_probabilities()`` returns a single outcome for robots and for humans
  with ``enter_bush_success_prob==1.0``.
* ``transition_probabilities()`` returns two outcomes for humans with
  ``enter_bush_success_prob<1.0``: succeed (agent on bush, bush intact) and fail (agent
  stays, bush intact).
* ``set_state()`` correctly restores a state where a human is on a bush.
"""

import numpy as np

from gym_multigrid.multigrid import Agent, Bush, Grid, MultiGridEnv, Wall, World


class BushTestEnv(MultiGridEnv):
    """Small 5x5 corridor: agent at (1,2) facing right, bush at (2,2).

    Use ``robot=True`` (default) for a grey robot that always tramples;
    use ``robot=False`` for a yellow human whose success depends on
    ``enter_bush_success_prob``.
    """

    def __init__(self, robot=True, enter_bush_success_prob=1.0):
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
            enter_bush_success_prob=enter_bush_success_prob,
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
        self.grid.set(2, 2, Bush(World, enter_bush_success_prob=self.enter_bush_success_prob))


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


# ---------------------------------------------------------------------------
# Basic creation
# ---------------------------------------------------------------------------

def test_bush_creation():
    bush = Bush(World)

    assert bush.type == 'bush'
    assert bush.color == 'green'
    assert bush.trampled is False
    assert bush.enter_bush_success_prob == 1.0
    assert not bush.can_overlap()
    assert not bush.can_pickup()


def test_bush_creation_with_probability():
    bush = Bush(World, enter_bush_success_prob=0.5)
    assert bush.enter_bush_success_prob == 0.5


# ---------------------------------------------------------------------------
# can_forward: both robots and humans should report forward=True for a bush
# ---------------------------------------------------------------------------

def test_robot_can_forward_into_bush():
    env = BushTestEnv(robot=True)
    env.reset()
    assert env.can_forward(env.get_state(), 0) is True


def test_human_can_forward_into_bush():
    """can_forward returns True for human agents facing a bush."""
    env = BushTestEnv(robot=False)
    env.reset()
    assert env.can_forward(env.get_state(), 0) is True


# ---------------------------------------------------------------------------
# Robot trampling: bush is permanently removed
# ---------------------------------------------------------------------------

def test_robot_tramples_bush_and_state_tracks_it():
    """Robots always trample with certainty; bush is removed from grid."""
    env = BushTestEnv(robot=True)
    env.reset()

    obs, rewards, done, info = env.step([3])

    assert np.array_equal(env.agents[0].pos, [2, 2])
    assert env.grid.get(2, 2) is env.agents[0]
    assert ('bush', 2, 2, True) in env.get_state()[3]


def test_robot_always_tramples_regardless_of_enter_bush_success_prob():
    """Robots trample with certainty even when enter_bush_success_prob is set low."""
    env = BushTestEnv(robot=True, enter_bush_success_prob=0.1)
    env.reset()

    obs, rewards, done, info = env.step([3])

    assert np.array_equal(env.agents[0].pos, [2, 2])
    assert ('bush', 2, 2, True) in env.get_state()[3]


def test_robot_transition_is_always_deterministic_single_outcome():
    """Robots always produce a single (p=1.0) outcome regardless of enter_bush_success_prob."""
    env = BushTestEnv(robot=True, enter_bush_success_prob=0.3)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    assert outcomes is not None
    assert len(outcomes) == 1, (
        f"Expected 1 deterministic outcome for robot, got {len(outcomes)}"
    )
    assert abs(outcomes[0][0] - 1.0) < 1e-9


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


def test_human_can_pass_after_robot_tramples_bush():
    """After a robot tramples a bush, the cell is empty and a human can pass freely."""
    env = BushHumanPassageEnv()
    env.reset()

    env.step([3, 0])  # Robot tramples bush, human waits.
    env.step([3, 3])  # Robot leaves; human's target was occupied at step start.
    env.step([0, 3])  # Human can now enter the cleared cell.

    assert np.array_equal(env.agents[1].pos, [2, 2])
    assert ('bush', 2, 2, True) in env.get_state()[3]


# ---------------------------------------------------------------------------
# Human entry: bush stays intact
# ---------------------------------------------------------------------------

def test_human_enters_bush_with_certainty_when_probability_is_1():
    """With enter_bush_success_prob==1.0 a human always enters, but the bush stays."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=1.0)
    env.reset()

    obs, rewards, done, info = env.step([3])

    # Agent moved to (2,2)
    assert np.array_equal(env.agents[0].pos, [2, 2])
    # Bush is still intact (untrampled) -- stored in terrain_grid
    assert ('bush', 2, 2, False) in env.get_state()[3]
    # Grid shows agent; bush is in terrain_grid
    assert env.grid.get(2, 2) is env.agents[0]
    terrain_cell = env.terrain_grid.get(2, 2)
    assert terrain_cell is not None
    assert terrain_cell.type == 'bush'


def test_human_cannot_enter_bush_when_probability_is_zero():
    """When enter_bush_success_prob==0.0 a human can never enter the bush."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=0.0)
    env.reset()

    obs, rewards, done, info = env.step([3])

    assert np.array_equal(env.agents[0].pos, [1, 2])
    bush = env.grid.get(2, 2)
    assert bush is not None
    assert bush.type == 'bush'
    assert bush.trampled is False


def test_human_exits_bush_and_bush_is_restored():
    """After a human enters a bush and then leaves, the bush is still there."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=1.0)
    env.reset()

    env.step([3])  # Human enters bush at (2,2)
    assert np.array_equal(env.agents[0].pos, [2, 2])

    env.step([3])  # Human moves forward to (3,2), leaving the bush
    assert np.array_equal(env.agents[0].pos, [3, 2])

    # Bush should be restored at (2,2)
    bush = env.grid.get(2, 2)
    assert bush is not None
    assert bush.type == 'bush'
    assert bush.trampled is False


def test_human_state_after_entering_bush():
    """get_state() reports the bush as untrampled (False) when a human is on it."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=1.0)
    env.reset()

    env.step([3])

    state = env.get_state()
    agent_x, agent_y = state[1][0][0], state[1][0][1]
    assert agent_x == 2 and agent_y == 2, "Agent should be on the bush cell"
    # Bush should still be there, untrampled
    assert ('bush', 2, 2, False) in state[3]


def test_set_state_with_human_on_bush():
    """set_state() correctly restores a state where a human is on a bush."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=1.0)
    env.reset()

    env.step([3])  # Human enters bush
    on_bush_state = env.get_state()

    # Restore to a different state, then restore the on-bush state
    env.step([0])  # Turn left (so internal state changes)
    env.set_state(on_bush_state)

    # Agent should be at (2,2), bush should be in terrain_grid
    assert np.array_equal(env.agents[0].pos, [2, 2])
    assert env.terrain_grid.get(2, 2) is not None
    assert env.terrain_grid.get(2, 2).type == 'bush'
    # Advance one step to make the human leave; bush should be restored to the grid
    env.step([3])  # Move forward to (3,2)
    bush = env.grid.get(2, 2)
    assert bush is not None
    assert bush.type == 'bush'


# ---------------------------------------------------------------------------
# Stochastic human entry
# ---------------------------------------------------------------------------

def test_human_probabilistic_bush_transition_probabilities():
    """For humans with enter_bush_success_prob < 1, transition_probabilities returns 2 outcomes."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    assert outcomes is not None
    assert len(outcomes) == 2, f"Expected 2 outcomes for human, got {len(outcomes)}"
    probs = sorted([p for p, _ in outcomes], reverse=True)
    assert abs(probs[0] - 0.6) < 1e-9, f"Expected succeed prob 0.6, got {probs[0]}"
    assert abs(probs[1] - 0.4) < 1e-9, f"Expected fail prob 0.4, got {probs[1]}"


def test_human_probabilistic_bush_succeed_outcome_moves_agent_bush_intact():
    """When human entry succeeds, agent moves to bush cell and bush remains (untrampled)."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    succeed_state = None
    for prob, s in outcomes:
        env.set_state(s)
        if np.array_equal(env.agents[0].pos, [2, 2]):
            succeed_state = (prob, s)
            break

    assert succeed_state is not None, "No succeed outcome found"
    assert abs(succeed_state[0] - 0.6) < 1e-9
    # Bush must still be UNTRAMPLED -- human entered, did not trample
    assert ('bush', 2, 2, False) in succeed_state[1][3]


def test_human_probabilistic_bush_fail_outcome_keeps_agent():
    """When human entry fails, agent stays put and bush remains."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=0.6)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    fail_state = None
    for prob, s in outcomes:
        env.set_state(s)
        if np.array_equal(env.agents[0].pos, [1, 2]):
            fail_state = (prob, s)
            break

    assert fail_state is not None, "No fail outcome found"
    assert abs(fail_state[0] - 0.4) < 1e-9
    # Bush is still untrampled
    assert ('bush', 2, 2, False) in fail_state[1][3]


def test_human_deterministic_bush_single_outcome():
    """With enter_bush_success_prob=1.0, a human produces a single deterministic outcome."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=1.0)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])

    assert outcomes is not None
    assert len(outcomes) == 1
    assert abs(outcomes[0][0] - 1.0) < 1e-9


def test_step_consistent_with_transition_probabilities_for_human_probabilistic_bush():
    """step() and transition_probabilities() produce consistent distributions for humans."""
    env = BushTestEnv(robot=False, enter_bush_success_prob=0.5)
    env.reset()

    state = env.get_state()
    outcomes = env.transition_probabilities(state, [3])
    outcome_states = {s for _, s in outcomes}

    n_trials = 200
    for _ in range(n_trials):
        env.set_state(state)
        env.step([3])
        resulting_state = env.get_state()
        assert resulting_state in outcome_states, (
            "step() produced a state not present in transition_probabilities() outcomes"
        )


# ---------------------------------------------------------------------------
# Config / map-parsed bushes
# ---------------------------------------------------------------------------

def test_probabilistic_bush_via_config():
    """enter_bush_success_prob from MultiGridEnv constructor applies to map-parsed bushes."""
    env = MultiGridEnv(
        map="We We We We\nWe Ae Bu We\nWe We We We",
        enter_bush_success_prob=0.5,
        objects_set=World,
        partial_obs=False,
    )
    env.reset()
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
    assert found_bush.enter_bush_success_prob == 0.5
