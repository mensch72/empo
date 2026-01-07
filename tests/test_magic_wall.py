"""
Test magic wall functionality.
"""

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, MagicWall, Wall, World


class SimpleMagicWallEnv(MultiGridEnv):
    """Simple test environment with magic walls."""
    
    def __init__(self, num_agents=2, agent_can_enter=None):
        """
        Args:
            num_agents: Number of agents
            agent_can_enter: List of bools indicating which agents can enter magic walls
        """
        if agent_can_enter is None:
            agent_can_enter = [False] * num_agents
        
        self.agents = [Agent(World, i, can_enter_magic_walls=agent_can_enter[i]) 
                      for i in range(num_agents)]
        super().__init__(
            width=10,
            height=10,
            max_steps=100,
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height-1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width-1, j, Wall(World))
        
        # Place agents
        for idx, agent in enumerate(self.agents):
            agent.pos = np.array([2, 2 + idx * 2])
            agent.dir = 0  # facing right
            self.grid.set(2, 2 + idx * 2, agent)


def test_magic_wall_creation():
    """Test that magic walls can be created and placed."""
    env = SimpleMagicWallEnv(num_agents=1)
    
    # Create a magic wall with magic side = right (0), entry probability = 0.7
    magic_wall = MagicWall(World, magic_side=0, entry_probability=0.7)
    env.grid.set(5, 5, magic_wall)
    
    # Verify it exists
    cell = env.grid.get(5, 5)
    assert cell is not None
    assert cell.type == 'magicwall'
    assert cell.color == 'grey'
    assert cell.magic_side == 0
    assert cell.entry_probability == 0.7
    assert not cell.see_behind()
    assert not cell.can_overlap()  # Magic walls block normal movement


def test_magic_wall_different_sides():
    """Test magic walls with different magic sides."""
    env = SimpleMagicWallEnv(num_agents=1)
    
    # Create magic walls for each direction
    mw_right = MagicWall(World, magic_side=0, entry_probability=1.0)
    mw_down = MagicWall(World, magic_side=1, entry_probability=1.0)
    mw_left = MagicWall(World, magic_side=2, entry_probability=1.0)
    mw_up = MagicWall(World, magic_side=3, entry_probability=1.0)
    
    env.grid.set(3, 3, mw_right)
    env.grid.set(4, 3, mw_down)
    env.grid.set(5, 3, mw_left)
    env.grid.set(6, 3, mw_up)
    
    # Verify all directions
    assert env.grid.get(3, 3).magic_side == 0
    assert env.grid.get(4, 3).magic_side == 1
    assert env.grid.get(5, 3).magic_side == 2
    assert env.grid.get(6, 3).magic_side == 3


def test_agent_without_magic_wall_permission():
    """Test that agents without permission cannot enter magic walls."""
    env = SimpleMagicWallEnv(num_agents=1, agent_can_enter=[False])
    
    # Place magic wall in front of agent
    # Agent at (2, 2) facing right (dir=0)
    # Magic wall at (3, 2) with magic side = left (2), entry probability = 1.0
    # Agent approaches from left, which is the magic side
    magic_wall = MagicWall(World, magic_side=2, entry_probability=1.0)
    env.grid.set(3, 2, magic_wall)
    
    # Agent tries to move forward (should fail)
    actions = [3]  # forward
    obs, rewards, done, info = env.step(actions)
    
    # Agent should still be at original position
    assert tuple(env.agents[0].pos) == (2, 2)
    assert env.grid.get(3, 2).type == 'magicwall'


def test_agent_with_magic_wall_permission_certain_entry():
    """Test that agents with permission can enter magic walls (100% probability)."""
    env = SimpleMagicWallEnv(num_agents=1, agent_can_enter=[True])
    
    # Place magic wall in front of agent
    # Agent at (2, 2) facing right (dir=0)
    # Magic wall at (3, 2) with magic side = left (2), entry probability = 1.0
    magic_wall = MagicWall(World, magic_side=2, entry_probability=1.0)
    env.grid.set(3, 2, magic_wall)
    
    # Agent tries to move forward (should succeed with 100% probability)
    actions = [3]  # forward
    obs, rewards, done, info = env.step(actions)
    
    # Agent should now be on the magic wall cell
    assert tuple(env.agents[0].pos) == (3, 2)


def test_agent_with_magic_wall_permission_probabilistic_entry():
    """Test that magic wall entry is probabilistic."""
    env = SimpleMagicWallEnv(num_agents=1, agent_can_enter=[True])
    
    # Place magic wall in front of agent
    # Agent at (2, 2) facing right (dir=0)
    # Magic wall at (3, 2) with magic side = left (2), entry probability = 0.5
    magic_wall = MagicWall(World, magic_side=2, entry_probability=0.5)
    env.grid.set(3, 2, magic_wall)
    
    # Run multiple trials to verify probabilistic behavior
    successes = 0
    failures = 0
    num_trials = 100
    
    for _ in range(num_trials):
        # Reset environment
        env.reset()
        env.agents[0].pos = np.array([2, 2])
        env.agents[0].dir = 0
        env.grid.set(2, 2, env.agents[0])
        env.grid.set(3, 2, MagicWall(World, magic_side=2, entry_probability=0.5))
        
        # Agent tries to move forward
        actions = [3]  # forward
        obs, rewards, done, info = env.step(actions)
        
        # Check if agent moved
        if tuple(env.agents[0].pos) == (3, 2):
            successes += 1
        else:
            failures += 1
    
    # With probability 0.5, we expect roughly half successes and half failures
    # Allow some variance (at least 30% and at most 70% for 100 trials)
    success_rate = successes / num_trials
    assert 0.3 < success_rate < 0.7, f"Success rate {success_rate} not close to 0.5"


def test_magic_wall_wrong_direction():
    """Test that agents cannot enter from non-magic sides."""
    env = SimpleMagicWallEnv(num_agents=1, agent_can_enter=[True])
    
    # Place magic wall with magic side = left (2)
    # Agent approaches from right side (which is NOT the magic side)
    magic_wall = MagicWall(World, magic_side=2, entry_probability=1.0)
    env.grid.set(3, 2, magic_wall)
    
    # Position agent to the right of the magic wall, facing left
    env.agents[0].pos = np.array([4, 2])
    env.agents[0].dir = 2  # facing left
    env.grid.set(4, 2, env.agents[0])
    
    # Agent tries to move forward (should fail - approaching from wrong side)
    actions = [3]  # forward
    obs, rewards, done, info = env.step(actions)
    
    # Agent should still be at original position
    assert tuple(env.agents[0].pos) == (4, 2)


def test_agent_can_step_off_magic_wall():
    """Test that agents can step off magic walls as if they were empty cells."""
    env = SimpleMagicWallEnv(num_agents=1, agent_can_enter=[True])
    
    # Place agent on a magic wall
    magic_wall = MagicWall(World, magic_side=2, entry_probability=1.0)
    env.grid.set(3, 2, magic_wall)
    env.agents[0].pos = np.array([3, 2])
    env.agents[0].dir = 0  # facing right
    
    # Agent should be able to step forward off the magic wall
    actions = [3]  # forward
    obs, rewards, done, info = env.step(actions)
    
    # Agent should have moved off the magic wall
    assert tuple(env.agents[0].pos) == (4, 2)


def test_magic_wall_encoding():
    """Test that magic walls encode correctly."""
    env = SimpleMagicWallEnv(num_agents=1)
    
    # Create a magic wall
    magic_wall = MagicWall(World, magic_side=1, entry_probability=0.8)
    
    # Test encoding
    encoding = magic_wall.encode(World)
    assert encoding[0] == World.OBJECT_TO_IDX['magicwall']
    assert encoding[1] == World.COLOR_TO_IDX['grey']
    assert encoding[2] == 1  # magic_side
    assert encoding[3] == int(0.8 * 255)  # entry_probability scaled


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
