"""
Tests for can_forward() method and capability-aware exploration policies.

This test verifies that:
1. Robot agents (grey) can push rocks and enter magic walls
2. Human agents (purple/blue) cannot push rocks and cannot enter magic walls
3. Both agent types can push blocks
4. The exploration policies respect these capability differences
"""

import sys
import os
import numpy as np

# Add src and vendor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))

import pytest

from gym_multigrid.multigrid import MultiGridEnv, Grid, Wall, Rock, Block, MagicWall, World, Agent


def create_robot_agent():
    """Create a robot agent (can push rocks, can enter magic walls)."""
    agent = Agent(World, 0, can_push_rocks=True, can_enter_magic_walls=True)
    return agent


def create_human_agent():
    """Create a human agent (cannot push rocks, cannot enter magic walls)."""
    agent = Agent(World, 1, can_push_rocks=False, can_enter_magic_walls=False)
    return agent


class RockPushTestEnv(MultiGridEnv):
    """
    Test environment with a rock in front of an agent.
    
    Layout:
        ...
        .AR.   (A=agent facing right, R=rock, .=empty)
        ...
    
    Robot (can_push_rocks=True) should be able to move forward.
    Human (can_push_rocks=False) should NOT be able to move forward.
    """
    
    def __init__(self, is_robot: bool = True):
        """
        Args:
            is_robot: If True, create robot agent (can push rocks).
                     If False, create human agent (cannot push rocks).
        """
        self._is_robot = is_robot
        agent = create_robot_agent() if is_robot else create_human_agent()
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
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place agent at (1, 2) facing right (direction 0)
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right
        
        # Place rock at (2, 2) - directly in front of agent
        self.grid.set(2, 2, Rock(World))
        
        # Empty space at (3, 2) for rock to be pushed into


class BlockPushTestEnv(MultiGridEnv):
    """
    Test environment with a block in front of an agent.
    
    Layout:
        ...
        .AB.   (A=agent facing right, B=block, .=empty)
        ...
    
    Both robot and human should be able to push blocks.
    """
    
    def __init__(self, is_robot: bool = True):
        self._is_robot = is_robot
        agent = create_robot_agent() if is_robot else create_human_agent()
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
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place agent at (1, 2) facing right
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right
        
        # Place block at (2, 2) - directly in front of agent
        self.grid.set(2, 2, Block(World))


class MagicWallTestEnv(MultiGridEnv):
    """
    Test environment with a magic wall in front of an agent.
    
    Layout:
        ...
        .AM.   (A=agent facing right, M=magic wall, .=empty)
        ...
    
    Robot (can_enter_magic_walls=True) should be able to attempt entry.
    Human (can_enter_magic_walls=False) should NOT be able to attempt entry.
    """
    
    def __init__(self, is_robot: bool = True):
        self._is_robot = is_robot
        agent = create_robot_agent() if is_robot else create_human_agent()
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
        
        # Create walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))
        
        # Place agent at (1, 2) facing right
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # Facing right
        
        # Place active magic wall at (2, 2) - directly in front of agent
        # magic_side=4 means all sides are magic (can be entered from any direction)
        # entry_probability=1.0 means guaranteed entry for authorized agents
        self.grid.set(2, 2, MagicWall(World, magic_side=4, entry_probability=1.0))


class TestCanForward:
    """Test the can_forward() method for different agent capabilities."""
    
    def test_robot_can_push_rock(self):
        """Robot (can_push_rocks=True) should be able to push rocks."""
        env = RockPushTestEnv(is_robot=True)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is True, "Robot should be able to push rock and move forward"
    
    def test_human_cannot_push_rock(self):
        """Human (can_push_rocks=False) should NOT be able to push rocks."""
        env = RockPushTestEnv(is_robot=False)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is False, "Human should NOT be able to push rock"
    
    def test_robot_can_push_block(self):
        """Robot should be able to push blocks."""
        env = BlockPushTestEnv(is_robot=True)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is True, "Robot should be able to push block"
    
    def test_human_can_push_block(self):
        """Human should also be able to push blocks (no restriction)."""
        env = BlockPushTestEnv(is_robot=False)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is True, "Human should be able to push block"
    
    def test_robot_can_enter_magic_wall(self):
        """Robot (can_enter_magic_walls=True) should be able to attempt magic wall entry."""
        env = MagicWallTestEnv(is_robot=True)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is True, "Robot should be able to attempt magic wall entry"
    
    def test_human_cannot_enter_magic_wall(self):
        """Human (can_enter_magic_walls=False) should NOT be able to enter magic wall."""
        env = MagicWallTestEnv(is_robot=False)
        env.reset()
        
        state = env.get_state()
        can_forward = env.can_forward(state, agent_index=0)
        
        assert can_forward is False, "Human should NOT be able to enter magic wall"


class TestExplorationPolicies:
    """Test that exploration policies respect agent capabilities."""
    
    def test_robot_exploration_allows_rock_push(self):
        """Robot exploration policy should allow forward when facing pushable rock."""
        from empo.nn_based.multigrid.phase2.robot_policy import MultiGridRobotExplorationPolicy
        
        env = RockPushTestEnv(is_robot=True)
        env.reset()
        
        exploration = MultiGridRobotExplorationPolicy(
            action_probs=[0.0, 0.0, 0.0, 1.0],  # Always try forward
            robot_agent_indices=[0]
        )
        exploration.reset(env)
        
        state = env.get_state()
        
        # Sample many times - should always get forward (action 3)
        actions = [exploration.sample(state)[0] for _ in range(100)]
        
        # Should get forward action (3) every time
        assert all(a == 3 for a in actions), "Robot should always get forward when facing rock"
    
    def test_human_exploration_blocks_rock_push(self):
        """Human exploration policy should block forward when facing rock."""
        from empo.human_policy_prior import MultiGridHumanExplorationPolicy
        
        env = RockPushTestEnv(is_robot=False)
        env.reset()
        
        exploration = MultiGridHumanExplorationPolicy(
            world_model=env,
            human_agent_indices=[0],
            action_probs=[0.0, 0.0, 0.0, 1.0]  # Would always try forward if allowed
        )
        
        state = env.get_state()
        
        # Sample many times - should never get forward (action 3)
        actions = [exploration.sample(state, human_agent_index=0) for _ in range(100)]
        
        # Should never get forward action (3) since human can't push rocks
        assert all(a != 3 for a in actions), "Human should never get forward when facing rock"
    
    def test_human_exploration_allows_block_push(self):
        """Human exploration policy should allow forward when facing pushable block."""
        from empo.human_policy_prior import MultiGridHumanExplorationPolicy
        
        env = BlockPushTestEnv(is_robot=False)
        env.reset()
        
        exploration = MultiGridHumanExplorationPolicy(
            world_model=env,
            human_agent_indices=[0],
            action_probs=[0.0, 0.0, 0.0, 1.0]  # Always try forward
        )
        
        state = env.get_state()
        
        # Sample many times - should always get forward (action 3)
        actions = [exploration.sample(state, human_agent_index=0) for _ in range(100)]
        
        # Should get forward action (3) every time since blocks can be pushed by anyone
        assert all(a == 3 for a in actions), "Human should always get forward when facing block"
    
    def test_human_exploration_blocks_magic_wall(self):
        """Human exploration policy should block forward when facing magic wall."""
        from empo.human_policy_prior import MultiGridHumanExplorationPolicy
        
        env = MagicWallTestEnv(is_robot=False)
        env.reset()
        
        exploration = MultiGridHumanExplorationPolicy(
            world_model=env,
            human_agent_indices=[0],
            action_probs=[0.0, 0.0, 0.0, 1.0]  # Would always try forward if allowed
        )
        
        state = env.get_state()
        
        # Sample many times - should never get forward (action 3)
        actions = [exploration.sample(state, human_agent_index=0) for _ in range(100)]
        
        # Should never get forward action (3) since human can't enter magic walls
        assert all(a != 3 for a in actions), "Human should never get forward when facing magic wall"
    
    def test_robot_exploration_allows_magic_wall(self):
        """Robot exploration policy should allow forward when facing magic wall."""
        from empo.nn_based.multigrid.phase2.robot_policy import MultiGridRobotExplorationPolicy
        
        env = MagicWallTestEnv(is_robot=True)
        env.reset()
        
        exploration = MultiGridRobotExplorationPolicy(
            action_probs=[0.0, 0.0, 0.0, 1.0],  # Always try forward
            robot_agent_indices=[0]
        )
        exploration.reset(env)
        
        state = env.get_state()
        
        # Sample many times - should always get forward (action 3)
        actions = [exploration.sample(state)[0] for _ in range(100)]
        
        # Should get forward action (3) every time
        assert all(a == 3 for a in actions), "Robot should always get forward when facing magic wall"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
