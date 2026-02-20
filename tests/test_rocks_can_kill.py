"""
Test rocks_can_kill feature - agents with can_push_rocks=True can push rocks
onto agents with can_push_rocks=False to terminate them.
"""

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Rock, Wall, World


class RocksCanKillTestEnv(MultiGridEnv):
    """Test environment for rocks_can_kill feature."""

    def __init__(self, rocks_can_kill=True):
        # Create agents: agent 0 can push rocks, agent 1 cannot
        agent_pusher = Agent(World, 0, can_push_rocks=True)
        agent_victim = Agent(World, 1, can_push_rocks=False)
        self.agents = [agent_pusher, agent_victim]
        super().__init__(
            width=8,
            height=5,
            max_steps=100,
            agents=self.agents,
            partial_obs=False,
            objects_set=World,
            rocks_can_kill=rocks_can_kill
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Walls around perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))

        # Agent 0 (pusher) at (1,2) facing right
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0
        self.grid.set(1, 2, self.agents[0])

        # Rock at (2,2)
        rock = Rock(World)
        self.grid.set(2, 2, rock)

        # Agent 1 (victim) at (3,2) facing right
        self.agents[1].pos = np.array([3, 2])
        self.agents[1].dir = 0
        self.grid.set(3, 2, self.agents[1])


def test_rock_kills_agent_when_enabled():
    """Test that pushing a rock onto a killable agent terminates them."""
    env = RocksCanKillTestEnv(rocks_can_kill=True)
    env.reset()

    # Verify initial state
    assert not env.agents[1].terminated
    assert np.array_equal(env.agents[0].pos, [1, 2])
    assert np.array_equal(env.agents[1].pos, [3, 2])

    # Agent 0 pushes rock forward (action 3 = forward)
    actions = [3, 0]  # agent 0 forward, agent 1 still
    env.step(actions)

    # Agent 1 should be terminated
    assert env.agents[1].terminated == True
    # Agent 0 should have moved to (2,2)
    assert np.array_equal(env.agents[0].pos, [2, 2])
    # Rock should be at (3,2) where the victim was
    cell = env.grid.get(3, 2)
    assert cell is not None and cell.type == 'rock'


def test_rock_does_not_kill_when_disabled():
    """Test that rocks_can_kill=False prevents killing."""
    env = RocksCanKillTestEnv(rocks_can_kill=False)
    env.reset()

    # Store initial positions
    pusher_pos = env.agents[0].pos.copy()
    victim_pos = env.agents[1].pos.copy()

    actions = [3, 0]  # agent 0 tries to push
    env.step(actions)

    # Push should have failed - agent 1 blocks
    assert env.agents[1].terminated == False
    # Agent 0 should not have moved
    assert np.array_equal(env.agents[0].pos, pusher_pos)
    # Agent 1 should still be at original position
    assert np.array_equal(env.agents[1].pos, victim_pos)


def test_cannot_kill_agent_with_can_push_rocks():
    """Test that agents with can_push_rocks=True cannot be killed by rocks."""

    class BothCanPushEnv(MultiGridEnv):
        """Environment where both agents can push rocks."""

        def __init__(self):
            agent0 = Agent(World, 0, can_push_rocks=True)
            agent1 = Agent(World, 1, can_push_rocks=True)  # Also can push
            self.agents = [agent0, agent1]
            super().__init__(
                width=8,
                height=5,
                max_steps=100,
                agents=self.agents,
                partial_obs=False,
                objects_set=World,
                rocks_can_kill=True  # Feature enabled
            )

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            for i in range(width):
                self.grid.set(i, 0, Wall(World))
                self.grid.set(i, height - 1, Wall(World))
            for j in range(height):
                self.grid.set(0, j, Wall(World))
                self.grid.set(width - 1, j, Wall(World))

            self.agents[0].pos = np.array([1, 2])
            self.agents[0].dir = 0
            self.grid.set(1, 2, self.agents[0])

            rock = Rock(World)
            self.grid.set(2, 2, rock)

            self.agents[1].pos = np.array([3, 2])
            self.agents[1].dir = 0
            self.grid.set(3, 2, self.agents[1])

    env = BothCanPushEnv()
    env.reset()

    pusher_pos = env.agents[0].pos.copy()

    actions = [3, 0]
    env.step(actions)

    # Push should have failed - can't kill agent with can_push_rocks=True
    assert env.agents[1].terminated == False
    assert np.array_equal(env.agents[0].pos, pusher_pos)


def test_agent_without_can_push_rocks_cannot_kill():
    """Test that agents without can_push_rocks cannot kill even with rocks_can_kill enabled."""

    class NoPushEnv(MultiGridEnv):
        """Environment where pusher cannot push rocks."""

        def __init__(self):
            agent0 = Agent(World, 0, can_push_rocks=False)  # Cannot push
            agent1 = Agent(World, 1, can_push_rocks=False)
            self.agents = [agent0, agent1]
            super().__init__(
                width=8,
                height=5,
                max_steps=100,
                agents=self.agents,
                partial_obs=False,
                objects_set=World,
                rocks_can_kill=True  # Feature enabled but agent can't push
            )

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            for i in range(width):
                self.grid.set(i, 0, Wall(World))
                self.grid.set(i, height - 1, Wall(World))
            for j in range(height):
                self.grid.set(0, j, Wall(World))
                self.grid.set(width - 1, j, Wall(World))

            self.agents[0].pos = np.array([1, 2])
            self.agents[0].dir = 0
            self.grid.set(1, 2, self.agents[0])

            rock = Rock(World)
            self.grid.set(2, 2, rock)

            self.agents[1].pos = np.array([3, 2])
            self.agents[1].dir = 0
            self.grid.set(3, 2, self.agents[1])

    env = NoPushEnv()
    env.reset()

    pusher_pos = env.agents[0].pos.copy()

    actions = [3, 0]
    env.step(actions)

    # Agent 0 cannot push rocks, so nothing happens
    assert env.agents[1].terminated == False
    assert np.array_equal(env.agents[0].pos, pusher_pos)


def test_config_loading_rocks_can_kill():
    """Test that rocks_can_kill can be loaded from config dict."""
    env = MultiGridEnv(
        width=5,
        height=5,
        config={'map': 'We We We We We\nWe .. .. .. We\nWe .. .. .. We\nWe .. .. .. We\nWe We We We We',
                'rocks_can_kill': True}
    )
    assert env.rocks_can_kill == True

    env2 = MultiGridEnv(
        width=5,
        height=5,
        config={'map': 'We We We We We\nWe .. .. .. We\nWe .. .. .. We\nWe .. .. .. We\nWe We We We We',
                'rocks_can_kill': False}
    )
    assert env2.rocks_can_kill == False


def test_terminated_agent_removed_from_grid():
    """Test that the terminated agent is removed from the grid and rock takes its place."""
    env = RocksCanKillTestEnv(rocks_can_kill=True)
    env.reset()

    victim_pos = tuple(env.agents[1].pos)

    # Before push, victim should be on grid
    cell_before = env.grid.get(*victim_pos)
    assert cell_before is not None
    assert cell_before.type == 'agent'

    actions = [3, 0]
    env.step(actions)

    # After push, rock should be where victim was
    cell_after = env.grid.get(*victim_pos)
    assert cell_after is not None
    assert cell_after.type == 'rock'

    # Victim should be terminated
    assert env.agents[1].terminated == True


def test_terminated_agent_cannot_act():
    """Test that a terminated agent cannot perform actions."""
    env = RocksCanKillTestEnv(rocks_can_kill=True)
    env.reset()

    # Kill agent 1
    actions = [3, 0]
    env.step(actions)
    assert env.agents[1].terminated == True

    # Store position after termination
    victim_pos = env.agents[1].pos.copy()
    victim_dir = env.agents[1].dir

    # Try to make the terminated agent act
    actions = [0, 3]  # agent 0 still, agent 1 tries to move forward
    env.step(actions)

    # Terminated agent should not have moved
    assert np.array_equal(env.agents[1].pos, victim_pos)
    assert env.agents[1].dir == victim_dir
