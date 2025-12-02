#!/usr/bin/env python3
"""
Example script demonstrating consecutive blocks and rocks being pushed.

This script creates a multigrid environment with blocks and rocks, simulates
agents pushing them consecutively, and saves the result as an MP4 animation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    # Get the full grid rendering
    img = env.render(mode='rgb_array', highlight=False)
    return img


def create_animation(output_path='blocks_rocks_animation.mp4'):
    """Create and save an animation showing blocks and rocks being pushed."""
    
    print("Creating blocks and rocks pushing animation...")
    
    # Create environment
    env = BlockRockDemoEnv(num_agents=1)
    env.reset()
    
    # Collect frames
    frames = []
    
    # Initial state
    frames.append(render_grid_to_array(env))
    
    # Agent pushes blocks (3 consecutive blocks)
    print("Pushing blocks...")
    for i in range(4):
        actions = [3]  # forward action
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        print(f"  Step {i+1}: Agent at {env.agents[0].pos}")
    
    # Move agent to second row (turn down, move down twice, turn right)
    print("Moving to rocks row...")
    
    # Turn down
    actions = [1]  # turn right (to face down)
    env.step(actions)
    frames.append(render_grid_to_array(env))
    
    # Move down
    actions = [3]  # forward
    env.step(actions)
    frames.append(render_grid_to_array(env))
    
    # Move down again
    actions = [3]  # forward
    env.step(actions)
    frames.append(render_grid_to_array(env))
    
    # Turn right to face rocks
    actions = [1]  # turn right
    env.step(actions)
    frames.append(render_grid_to_array(env))
    
    # Agent pushes rocks (3 consecutive rocks)
    print("Pushing rocks...")
    for i in range(4):
        actions = [3]  # forward action
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        print(f"  Step {i+1}: Agent at {env.agents[0].pos}")
    
    # Create animation
    print(f"Saving animation to {output_path}...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Blocks and Rocks Pushing Demo - Frame {frame_idx+1}/{len(frames)}', 
                     fontsize=14, fontweight='bold')
        return [im]
    
    # Create animation with slower frame rate for better visualization
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=500,  # 500ms between frames (2 fps)
        blit=True,
        repeat=True
    )
    
    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=2, bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"✓ Animation saved successfully to {output_path}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Duration: ~{len(frames)/2:.1f} seconds")
    except Exception as e:
        print(f"✗ Error saving animation: {e}")
        print("  Note: FFmpeg is required to save MP4 files.")
        print("  Install with: apt-get install ffmpeg  (on Ubuntu)")
        print("               brew install ffmpeg     (on macOS)")
        
        # Try saving as GIF as fallback
        print("\nTrying to save as GIF instead...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=2)
            print(f"✓ Animation saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"✗ Error saving GIF: {e2}")
    
    plt.close()


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
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
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
