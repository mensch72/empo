#!/usr/bin/env python3
"""
Example script demonstrating consecutive blocks and rocks being pushed.

This script creates a multigrid environment with blocks and rocks, simulates
agents pushing them consecutively, and saves the result as an MP4 animation.
"""

import os

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Block, Rock, Wall, World


class BlockRockDemoEnv(MultiGridEnv):
    """Demo environment showing blocks and rocks being pushed."""
    
    def __init__(self, num_agents=1):
        # Agent with can_push_rocks=True so it can push rocks
        self.agents = [Agent(World, i, can_push_rocks=True) for i in range(num_agents)]
        super().__init__(
            width=12,
            height=8,
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
        
        # Place agent at starting position
        agent = self.agents[0]
        agent.pos = np.array([2, 3])
        agent.dir = 0  # facing right
        self.grid.set(2, 3, agent)
        
        # Place consecutive blocks in a row
        for i in range(3):
            block = Block(World)
            self.grid.set(3 + i, 3, block)
        
        # Place consecutive rocks in another row
        for i in range(3):
            rock = Rock(World)  # Pushable by agents with can_push_rocks=True
            self.grid.set(3 + i, 5, rock)


def create_animation(output_path='blocks_rocks_animation.mp4', fps=2):
    """Create and save an animation showing blocks and rocks being pushed."""
    
    print("Creating blocks and rocks pushing animation...")
    
    # Create environment
    env = BlockRockDemoEnv(num_agents=1)
    env.reset()
    
    # Start video recording using MultiGridEnv's built-in method
    env.start_video_recording()
    
    # Initial state
    env.render(mode='rgb_array')
    
    # Agent pushes blocks (3 consecutive blocks)
    print("Pushing blocks...")
    for i in range(4):
        actions = [3]  # forward action
        obs, rewards, done, info = env.step(actions)
        env.render(mode='rgb_array')  # Frame automatically captured
        print(f"  Step {i+1}: Agent at {env.agents[0].pos}")
    
    # Move agent to second row (turn down, move down twice, turn right)
    print("Moving to rocks row...")
    
    # Turn down
    actions = [1]  # turn right (to face down)
    env.step(actions)
    env.render(mode='rgb_array')
    
    # Move down
    actions = [3]  # forward
    env.step(actions)
    env.render(mode='rgb_array')
    
    # Move down again
    actions = [3]  # forward
    env.step(actions)
    env.render(mode='rgb_array')
    
    # Turn right to face rocks
    actions = [1]  # turn right
    env.step(actions)
    env.render(mode='rgb_array')
    
    # Agent pushes rocks (3 consecutive rocks)
    print("Pushing rocks...")
    for i in range(4):
        actions = [3]  # forward action
        obs, rewards, done, info = env.step(actions)
        env.render(mode='rgb_array')  # Frame automatically captured
        print(f"  Step {i+1}: Agent at {env.agents[0].pos}")
    
    # Save video using MultiGridEnv's built-in method
    env.save_video(output_path, fps=fps)


def main():
    """Main function to run the animation example."""
    print("=" * 70)
    print("Blocks and Rocks Pushing Animation Example")
    print("=" * 70)
    print()
    print("This example demonstrates:")
    print("  - Creating a multigrid environment with blocks and rocks")
    print("  - Agent pushing 3 consecutive blocks")
    print("  - Agent pushing 3 consecutive rocks")
    print("  - Saving the simulation as an MP4 animation")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'blocks_rocks_animation.mp4')
    
    # Create animation
    create_animation(output_path)
    
    print()
    print("=" * 70)
    print("Done! You can view the animation at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
