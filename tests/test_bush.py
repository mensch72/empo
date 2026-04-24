"""
Tests for robot-tramplable bushes.
"""

import numpy as np

from gym_multigrid.multigrid import Agent, Bush, Grid, MultiGridEnv, Wall, World


class BushTestEnv(MultiGridEnv):
    """Small corridor with a bush between a robot and open floor."""

    def __init__(self, robot=True):
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
        self.grid.set(2, 2, Bush(World))


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
    assert not bush.can_overlap()
    assert not bush.can_pickup()


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
