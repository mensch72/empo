#!/usr/bin/env python3
"""
One or Three Chambers Environment

A multigrid environment with a specific layout featuring:
- Multiple chambers created by internal walls
- Human agents in the upper-center area
- Robot agents in the middle
- A rock and block as interactive objects

The layout is based on the ASCII map specification:
```
WWWWWWWWWWWWWWWWWWWWWWWWWWWW
W        WHHHHHHHW         W
W        WHHHHHHHW         W
W        WWWWHWWWW         W
W        W  ROR  W         W
W        W WWWWW WWW       W
W          W  WW   W       W
W        WWW  W  B   W     W
W        W      W WWWWWWWWWW
W        W      W          W
W        W      W          W
W        W      W          W
W        W      W          W
W        W      W          W
W        W      W          W
WWWWWWWWWWWWWWWWWWWWWWWWWWWW

legend:
W wall
H human
R robot
O rock
B block
```
"""

import sys
import os

# Add vendor/multigrid to path for gym_multigrid imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'vendor', 'multigrid'))

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Wall, Block, Rock, World, SmallActions


class OneOrThreeChambersEnv(MultiGridEnv):
    """
    A multi-chamber environment with humans and robots.
    
    Layout (28 columns x 16 rows) - exact layout from ASCII map:
    - Walls around the perimeter
    - Three main chambers separated by internal walls
    - Left chamber: large empty open space
    - Center chamber: contains human agents (top) and robot agents with rock (middle)
    - Right chamber: contains a block
    
    Agents:
    - 15 human agents (red): 7 in row 1, 7 in row 2, 1 in row 3
    - 2 robot agents (green): row 4, columns 12 and 14
    
    Objects:
    - 1 rock between the robots (column 13, row 4)
    - 1 block in the right chamber (column 17, row 7)
    """
    
    def __init__(self, num_humans=15, num_robots=2):
        """
        Initialize the One or Three Chambers environment.
        
        Args:
            num_humans: Number of human agents (default: 15, based on layout)
            num_robots: Number of robot agents (default: 2)
        """
        self.num_humans = num_humans
        self.num_robots = num_robots
        
        # Create agents: humans (red) and robots (green)
        self.agents = []
        for i in range(num_humans):
            # Human agents get color index 0 (red)
            self.agents.append(Agent(World, 0))
        for i in range(num_robots):
            # Robot agents get color index 1 (green)
            self.agents.append(Agent(World, 1))
        
        super().__init__(
            width=28,
            height=16,
            max_steps=1000,
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        """Generate the grid layout exactly as specified in the ASCII map."""
        self.grid = Grid(width, height)
        
        # === ROW 0: All walls ===
        for x in range(28):
            self.grid.set(x, 0, Wall(World))
        
        # === ROW 1: W@0, W@9, H@10-16, W@17, W@27 ===
        self.grid.set(0, 1, Wall(World))
        self.grid.set(9, 1, Wall(World))
        # H@10, H@11, H@12, H@13, H@14, H@15, H@16 - placed later
        self.grid.set(17, 1, Wall(World))
        self.grid.set(27, 1, Wall(World))
        
        # === ROW 2: W@0, W@9, H@10-16, W@17, W@27 ===
        self.grid.set(0, 2, Wall(World))
        self.grid.set(9, 2, Wall(World))
        # H@10, H@11, H@12, H@13, H@14, H@15, H@16 - placed later
        self.grid.set(17, 2, Wall(World))
        self.grid.set(27, 2, Wall(World))
        
        # === ROW 3: W@0, W@9, W@10, W@11, W@12, H@13, W@14, W@15, W@16, W@17, W@27 ===
        self.grid.set(0, 3, Wall(World))
        self.grid.set(9, 3, Wall(World))
        self.grid.set(10, 3, Wall(World))
        self.grid.set(11, 3, Wall(World))
        self.grid.set(12, 3, Wall(World))
        # H@13 - placed later
        self.grid.set(14, 3, Wall(World))
        self.grid.set(15, 3, Wall(World))
        self.grid.set(16, 3, Wall(World))
        self.grid.set(17, 3, Wall(World))
        self.grid.set(27, 3, Wall(World))
        
        # === ROW 4: W@0, W@9, R@12, O@13, R@14, W@17, W@27 ===
        self.grid.set(0, 4, Wall(World))
        self.grid.set(9, 4, Wall(World))
        # R@12 - placed later
        # O@13 - placed later
        # R@14 - placed later
        self.grid.set(17, 4, Wall(World))
        self.grid.set(27, 4, Wall(World))
        
        # === ROW 5: W@0, W@9, W@11, W@12, W@13, W@14, W@15, W@17, W@18, W@19, W@27 ===
        self.grid.set(0, 5, Wall(World))
        self.grid.set(9, 5, Wall(World))
        self.grid.set(11, 5, Wall(World))
        self.grid.set(12, 5, Wall(World))
        self.grid.set(13, 5, Wall(World))
        self.grid.set(14, 5, Wall(World))
        self.grid.set(15, 5, Wall(World))
        self.grid.set(17, 5, Wall(World))
        self.grid.set(18, 5, Wall(World))
        self.grid.set(19, 5, Wall(World))
        self.grid.set(27, 5, Wall(World))
        
        # === ROW 6: W@0, W@11, W@14, W@15, W@19, W@27 ===
        self.grid.set(0, 6, Wall(World))
        self.grid.set(11, 6, Wall(World))
        self.grid.set(14, 6, Wall(World))
        self.grid.set(15, 6, Wall(World))
        self.grid.set(19, 6, Wall(World))
        self.grid.set(27, 6, Wall(World))
        
        # === ROW 7: W@0, W@9, W@10, W@11, W@14, B@17, W@21, W@27 ===
        self.grid.set(0, 7, Wall(World))
        self.grid.set(9, 7, Wall(World))
        self.grid.set(10, 7, Wall(World))
        self.grid.set(11, 7, Wall(World))
        self.grid.set(14, 7, Wall(World))
        # B@17 - placed later
        self.grid.set(21, 7, Wall(World))
        self.grid.set(27, 7, Wall(World))
        
        # === ROW 8: W@0, W@9, W@16, W@18, W@19, W@20, W@21, W@22, W@23, W@24, W@25, W@26, W@27 ===
        self.grid.set(0, 8, Wall(World))
        self.grid.set(9, 8, Wall(World))
        self.grid.set(16, 8, Wall(World))
        for x in range(18, 28):
            self.grid.set(x, 8, Wall(World))
        
        # === ROW 9-14: W@0, W@9, W@16, W@27 ===
        for y in range(9, 15):
            self.grid.set(0, y, Wall(World))
            self.grid.set(9, y, Wall(World))
            self.grid.set(16, y, Wall(World))
            self.grid.set(27, y, Wall(World))
        
        # === ROW 15: All walls ===
        for x in range(28):
            self.grid.set(x, 15, Wall(World))
        
        # === PLACE HUMAN AGENTS ===
        # Row 1: H@10, H@11, H@12, H@13, H@14, H@15, H@16 (7 humans)
        # Row 2: H@10, H@11, H@12, H@13, H@14, H@15, H@16 (7 humans)
        # Row 3: H@13 (1 human)
        human_positions = [
            # Row 1 (7 humans)
            (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1),
            # Row 2 (7 humans)
            (10, 2), (11, 2), (12, 2), (13, 2), (14, 2), (15, 2), (16, 2),
            # Row 3 (1 human)
            (13, 3),
        ]
        
        for i, (x, y) in enumerate(human_positions[:self.num_humans]):
            agent = self.agents[i]
            agent.pos = np.array([x, y])
            agent.dir = 0  # facing right
            self.grid.set(x, y, agent)
        
        # === PLACE ROBOT AGENTS ===
        # Row 4: R@12, R@14 (2 robots)
        robot_positions = [
            (12, 4), (14, 4)
        ]
        
        for i, (x, y) in enumerate(robot_positions[:self.num_robots]):
            agent_idx = self.num_humans + i
            agent = self.agents[agent_idx]
            agent.pos = np.array([x, y])
            agent.dir = 0  # facing right
            self.grid.set(x, y, agent)
        
        # === PLACE ROCK ===
        # O@13 at row 4
        rock = Rock(World, pushable_by=None)  # Pushable by all agents
        self.grid.set(13, 4, rock)
        
        # === PLACE BLOCK ===
        # B@17 at row 7
        block = Block(World)
        self.grid.set(17, 7, block)
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        return super().step(actions)
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
            Initial observations
        """
        return super().reset()


# Map-based implementation using the new map parameter
# The map uses two-character codes:
# - We: grey wall
# - ..: empty cell
# - Ay: yellow agent (human)
# - Ae: grey agent (robot)
# - Ro: rock
# - Bl: block

ONE_OR_THREE_CHAMBERS_MAP = """
WeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWe
We................WeAyAyAyAyAyAyAyWe..................We
We................WeAyAyAyAyAyAyAyWe..................We
We................WeWeWeWeAyWeWeWeWe..................We
We................We....AeRoAe....We..................We
We................We..WeWeWeWeWe..WeWeWe..............We
We....................We....WeWe......We..............We
We................WeWeWe....We....Bl......We..........We
We................We............We..WeWeWeWeWeWeWeWeWeWe
We................We............We....................We
We................We............We....................We
We................We............We....................We
We................We............We....................We
We................We............We....................We
We................We............We....................We
WeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWeWe
"""

SMALL_ONE_OR_THREE_CHAMBERS_MAP = """
We We We We We .. .. .. .. .. 
We .. We Ay We .. .. .. .. ..
We .. We Ay We We We .. .. ..
We Ae Ro .. .. .. We .. .. ..
We .. We We We .. We We We ..
We .. .. .. We .. .. .. We We
We .. .. We .. .. Bl .. .. We
We .. .. .. We We .. We We We
We We We We .. We We We .. ..
"""

class OneOrThreeChambersMapEnv(MultiGridEnv):
    """
    A multi-chamber environment with humans and robots, implemented using the map parameter.
    
    This is an alternative implementation of OneOrThreeChambersEnv that uses the
    new map specification format instead of manually placing objects.
    
    Layout (28 columns x 16 rows) - exact layout from ASCII map:
    - Walls around the perimeter
    - Three main chambers separated by internal walls
    - Left chamber: large empty open space
    - Center chamber: contains human agents (top) and robot agents with rock (middle)
    - Right chamber: contains a block
    
    Agents:
    - 15 human agents (red): 7 in row 1, 7 in row 2, 1 in row 3
    - 2 robot agents (green): row 4, columns 12 and 14
    
    Objects:
    - 1 rock between the robots (column 13, row 4)
    - 1 block in the right chamber (column 17, row 7)
    """
    
    def __init__(self):
        """
        Initialize the One or Three Chambers environment using the map parameter.
        """
        super().__init__(
            map=ONE_OR_THREE_CHAMBERS_MAP,
            max_steps=1000,
            partial_obs=False,
            objects_set=World
        )
        
        # Track counts for compatibility with the original implementation
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')

class SmallOneOrThreeChambersMapEnv(MultiGridEnv):
    """
    A smaller version of the multi-chamber environment with humans and robots, implemented using the map parameter.
    """
    
    def __init__(self):
        """
        Initialize the One or Three Chambers environment using the map parameter.
        """
        super().__init__(
            map=SMALL_ONE_OR_THREE_CHAMBERS_MAP,
            max_steps=1000,
            partial_obs=False,
            objects_set=World
        )
        
        # Track counts for compatibility with the original implementation
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')

# Small environment for DAG computation and backward induction
# This uses SMALL_ONE_OR_THREE_CHAMBERS_MAP with 1 robot (grey) and 2 humans (yellow)
# Target cell for reward is (7, 3)


class SmallOneOrTwoChambersMapEnv(MultiGridEnv):
    """
    A small multi-chamber environment for DAG computation and backward induction.
    
    This is a simplified environment designed to have a tractable state space
    for computing the full DAG and performing backward induction.
    
    Layout (10 columns x 10 rows):
    - Walls creating chamber structure
    - 2 human agents (yellow) in upper left area
    - 1 robot agent (grey) in left side
    - 1 rock and 1 block as obstacles
    
    The environment uses a 9-step timeout to keep the state space finite.
    Each agent has 4 actions: still, left, right, forward (4^3 = 64 combinations).
    
    Target cell for reward: (7, 3) - the robot receives reward 1 when reaching this cell.
    """
    
    def __init__(self):
        """
        Initialize the Small One or Two Chambers environment.
        """
        super().__init__(
            map=SMALL_ONE_OR_THREE_CHAMBERS_MAP,
            max_steps=9,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        
        # Track counts for compatibility
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')


if __name__ == "__main__":
    # Simple test to verify the environment works
    env = OneOrThreeChambersEnv()
    obs = env.reset()
    print(f"Environment created successfully!")
    print(f"Grid size: {env.width} x {env.height}")
    print(f"Number of agents: {len(env.agents)}")
    print(f"  - Humans: {env.num_humans}")
    print(f"  - Robots: {env.num_robots}")
    
    # Test a few random steps
    for step in range(5):
        actions = [env.action_space.sample() for _ in range(len(env.agents))]
        obs, rewards, done, info = env.step(actions)
        print(f"Step {step + 1}: done={done}")
    
    print("Test completed successfully!")
    
    # Also test the map-based implementation
    print("\n--- Testing Map-based Implementation ---")
    env_map = OneOrThreeChambersMapEnv()
    obs = env_map.reset()
    print(f"Map-based environment created successfully!")
    print(f"Grid size: {env_map.width} x {env_map.height}")
    print(f"Number of agents: {len(env_map.agents)}")
    print(f"  - Humans (red): {env_map.num_humans}")
    print(f"  - Robots (green): {env_map.num_robots}")
    
    # Test a few random steps
    for step in range(5):
        actions = [env_map.action_space.sample() for _ in range(len(env_map.agents))]
        obs, rewards, done, info = env_map.step(actions)
        print(f"Step {step + 1}: done={done}")
    
    print("Map-based test completed successfully!")
