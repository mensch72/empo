#!/usr/bin/env python3
"""
Example script demonstrating unsteady ground cell type.

This script creates a 10x10 multigrid environment with 10 unsteady ground cells
and 10 agents, simulates their movements, and saves the result as an animation.
"""

import os

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Wall, World, UnsteadyGround


class UnsteadyGroundDemoEnv(MultiGridEnv):
    """Demo environment showing agents navigating unsteady ground."""
    
    def __init__(self, num_agents=10, num_unsteady_cells=10):
        self.num_unsteady_cells = num_unsteady_cells
        # World only has 6 colors (indices 0-5), so we cycle through them
        self.agents = [Agent(World, i % 6) for i in range(num_agents)]
        super().__init__(
            width=10,
            height=10,
            max_steps=200,
            agents=self.agents,
            partial_obs=False,
            objects_set=World,
            stumble_probability=0.7
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
        
        # Place unsteady ground cells randomly
        unsteady_positions = []
        for _ in range(self.num_unsteady_cells):
            while True:
                x = self._rand_int(1, width-1)
                y = self._rand_int(1, height-1)
                if self.grid.get(x, y) is None and (x, y) not in unsteady_positions:
                    unsteady_positions.append((x, y))
                    break
            
            # Create unsteady ground with 50% stumble probability
            unsteady = UnsteadyGround(World, stumble_probability=0.5, color='brown')
            self.grid.set(x, y, unsteady)
            # Also store in terrain grid so it persists under agents
            self.terrain_grid.set(x, y, unsteady)
        
        # Place agents randomly in empty cells
        for agent in self.agents:
            while True:
                x = self._rand_int(1, width-1)
                y = self._rand_int(1, height-1)
                cell = self.grid.get(x, y)
                if cell is None or cell.type == 'unsteadyground':
                    agent.pos = np.array([x, y])
                    agent.dir = self._rand_int(0, 4)
                    agent.init_dir = agent.dir
                    agent.init_pos = agent.pos.copy()
                    # Set on_unsteady_ground flag if the agent is placed on unsteady ground
                    agent.on_unsteady_ground = (cell is not None and cell.type == 'unsteadyground')
                    # If placing on unsteady ground, also store it in terrain_grid
                    if agent.on_unsteady_ground:
                        self.terrain_grid.set(x, y, cell)
                    self.grid.set(x, y, agent)
                    break


def create_animation(output_path='unsteady_ground_animation.mp4', num_steps=50, fps=5):
    """Create and save an animation showing agents navigating unsteady ground."""
    
    print("Creating unsteady ground animation...")
    print(f"  Grid size: 10x10")
    print(f"  Number of agents: 10")
    print(f"  Number of unsteady cells: 10")
    print(f"  Stumble probability: 70%")
    print()
    
    # Create environment
    env = UnsteadyGroundDemoEnv(num_agents=10, num_unsteady_cells=10)
    env.reset()
    
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
                actions.append(0)  # left
            else:
                actions.append(1)  # right
        
        obs, rewards, done, info = env.step(actions)
        env.render(mode='rgb_array')  # Frame automatically captured
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{num_steps} complete")
    
    # Save video using MultiGridEnv's built-in method
    env.save_video(output_path, fps=fps)


def main():
    """Main function to run the animation example."""
    print("=" * 70)
    print("Unsteady Ground Animation Example")
    print("=" * 70)
    print()
    print("This example demonstrates:")
    print("  - Creating a 10x10 multigrid environment with unsteady ground cells")
    print("  - 10 agents navigating the environment")
    print("  - Agents stumbling when moving forward on unsteady ground")
    print("  - Conflict resolution for agents competing for the same cell")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'unsteady_ground_animation.mp4')
    
    # Create animation
    create_animation(output_path, num_steps=50)
    
    print()
    print("=" * 70)
    print("Done! You can view the animation at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
