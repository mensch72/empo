#!/usr/bin/env python3
"""
Example script demonstrating magic wall cell type.

This script creates a 12x12 multigrid environment with magic walls
and agents, simulates their movements, and saves the result as an animation.

Magic walls can only be entered from a specific direction (their "magic side")
with a configurable probability. Only agents with can_enter_magic_walls=True
can attempt to enter them.
"""

import os

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Wall, World, MagicWall


class MagicWallDemoEnv(MultiGridEnv):
    """Demo environment showing agents navigating magic walls."""
    
    def __init__(self, num_agents=10, num_magic_walls=30):
        self.num_magic_walls = num_magic_walls
        # World only has 6 colors (indices 0-5), so we cycle through them
        # All but one agent can enter magic walls (9 out of 10)
        self.agents = []
        for i in range(num_agents):
            can_enter = (i < num_agents - 1)  # All but last agent can enter magic walls
            self.agents.append(Agent(World, i % 6, can_enter_magic_walls=can_enter))
        
        super().__init__(
            width=16,
            height=16,
            max_steps=200,
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
        
        # Place magic walls randomly
        magic_wall_positions = []
        for _ in range(self.num_magic_walls):
            while True:
                x = self._rand_int(2, width-2)
                y = self._rand_int(2, height-2)
                if self.grid.get(x, y) is None and (x, y) not in magic_wall_positions:
                    magic_wall_positions.append((x, y))
                    break
            
            # Create magic wall with random magic side, entry probability, and solidify probability
            magic_side = self._rand_int(0, 5)  # 0=right, 1=down, 2=left, 3=up, 4=all
            entry_prob = 0.5 + 0.5 * np.random.random()  # 50% to 100% for better visibility
            solidify_prob = 0.1 + 0.2 * np.random.random()  # 10% to 30% chance to solidify on failed entry
            magic_wall = MagicWall(World, magic_side=magic_side, 
                                   entry_probability=entry_prob, 
                                   solidify_probability=solidify_prob, color='grey')
            self.grid.set(x, y, magic_wall)
        
        # Place agents randomly in empty cells
        for agent in self.agents:
            while True:
                x = self._rand_int(1, width-1)
                y = self._rand_int(1, height-1)
                cell = self.grid.get(x, y)
                if cell is None:
                    agent.pos = np.array([x, y])
                    agent.dir = self._rand_int(0, 4)
                    agent.init_dir = agent.dir
                    agent.init_pos = agent.pos.copy()
                    self.grid.set(x, y, agent)
                    break


def create_animation(output_path='magic_wall_animation.mp4', num_steps=50, fps=5):
    """Create and save an animation showing agents navigating magic walls."""
    
    print("Creating magic wall animation...")
    print(f"  Grid size: 16x16")
    print(f"  Number of agents: 10 (9 can enter magic walls, 1 cannot)")
    print(f"  Number of magic walls: 30")
    print(f"  Entry probabilities: 50% to 100% (random)")
    print(f"  Solidify probabilities: 10% to 30% (random)")
    print()
    
    # Create environment
    env = MagicWallDemoEnv(num_agents=10, num_magic_walls=30)
    env.reset()
    
    # Print agent capabilities
    print("Agent capabilities:")
    for i, agent in enumerate(env.agents):
        status = "CAN" if agent.can_enter_magic_walls else "CANNOT"
        print(f"  Agent {i} (color {agent.color}): {status} enter magic walls")
    print()
    
    # Start video recording using MultiGridEnv's built-in method
    env.start_video_recording()
    
    # Initial state
    env.render(mode='rgb_array')
    
    # Simulate steps with agents moving forward randomly
    print(f"Simulating {num_steps} steps...")
    for step in range(num_steps):
        # Most agents move forward, some turn
        actions = []
        for i in range(len(env.agents)):
            # 70% forward, 15% left, 15% right
            rand = np.random.random()
            if rand < 0.7:
                actions.append(3)  # forward
            elif rand < 0.85:
                actions.append(1)  # left
            else:
                actions.append(2)  # right
        
        obs, rewards, done, info = env.step(actions)
        env.render(mode='rgb_array')  # Frame automatically captured
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps} complete")
    
    # Save video using MultiGridEnv's built-in method
    env.save_video(output_path, fps=fps)


def main():
    """Main function to run the animation example."""
    print("=" * 70)
    print("Magic Wall Animation Example")
    print("=" * 70)
    print()
    print("This example demonstrates:")
    print("  - Creating a 16x16 multigrid environment with 30 magic walls")
    print("  - 10 agents navigating (9 can enter magic walls, 1 cannot)")
    print("  - Magic walls that can be entered from specific directions or all directions")
    print("  - Five magic_side options: right(0), down(1), left(2), up(3), all(4)")
    print("  - Probabilistic entry success (50% to 100%)")
    print("  - Magic walls that may solidify into normal walls on failed entry")
    print("  - Blue dashed lines indicate which side(s) can be entered")
    print("  - All-sides walls have blue lines on all four edges")
    print("  - Magenta flash indicates successful entry")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'magic_wall_animation.mp4')
    
    # Create animation
    create_animation(output_path, num_steps=50)
    
    print()
    print("=" * 70)
    print("Done! You can view the animation at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
