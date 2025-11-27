#!/usr/bin/env python3
"""
One or Three Chambers - Random Play Video Example

This script creates a video of 1000 steps of random play in the
OneOrThreeChambersEnv environment.

The environment layout is based on the ASCII map specification with:
- 15 human agents (yellow)
- 2 robot agents (grey)
- 1 rock (pushable)
- 1 block

Usage:
    python examples/one_or_three_chambers_random_play.py

Output:
    outputs/one_or_three_chambers_random_play.mp4
"""

import sys
import os

# Add vendor/multigrid and src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs.one_or_three_chambers import OneOrThreeChambersMapEnv


def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    img = env.render(mode='rgb_array', highlight=False)
    return img


def create_random_play_video(output_path='one_or_three_chambers_random_play.mp4', num_steps=1000):
    """
    Create and save a video of random play in the OneOrThreeChambersEnv.
    
    Args:
        output_path: Path to save the output video
        num_steps: Number of steps to simulate (default: 1000)
    """
    
    print("=" * 70)
    print("One or Three Chambers - Random Play Video")
    print("=" * 70)
    print()
    print("Environment Layout:")
    print("  - 15 human agents (yellow) in upper center chamber")
    print("  - 2 robot agents (grey) in middle")
    print("  - 1 rock between robots")
    print("  - 1 block in right chamber")
    print()
    print("Three chambers created by walls:")
    print("  - Left chamber: open space")
    print("  - Center chamber: humans, robots, rock")
    print("  - Right chamber: block")
    print()
    
    # Create environment
    env = OneOrThreeChambersMapEnv()
    env.reset()
    
    print(f"Grid size: {env.width} x {env.height}")
    print(f"Number of agents: {len(env.agents)}")
    print(f"  - Humans: {env.num_humans}")
    print(f"  - Robots: {env.num_robots}")
    print()
    
    # Collect frames
    frames = []
    
    # Initial state
    frames.append(render_grid_to_array(env))
    
    print(f"Simulating {num_steps} steps of random play...")
    for step in range(num_steps):
        # Random actions for all agents
        actions = [env.action_space.sample() for _ in range(len(env.agents))]
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_steps} complete")
        
        if done:
            print(f"  Episode terminated at step {step + 1}")
            break
    
    print(f"\nCollected {len(frames)} frames")
    
    # Create animation
    print(f"Saving video to {output_path}...")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'One or Three Chambers - Random Play - Step {frame_idx}/{len(frames)-1}', 
                     fontsize=12, fontweight='bold')
        return [im]
    
    # Create animation - faster frame rate for 1000 frames
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=50,  # 50ms between frames (20 fps)
        blit=True,
        repeat=True
    )
    
    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=20, bitrate=2400)
        anim.save(output_path, writer=writer)
        print(f"✓ Video saved successfully to {output_path}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Duration: ~{len(frames)/20:.1f} seconds")
    except Exception as e:
        print(f"✗ Error saving video: {e}")
        print("  Note: FFmpeg is required to save MP4 files.")
        print("  Install with: apt-get install ffmpeg  (on Ubuntu)")
        print("               brew install ffmpeg     (on macOS)")
        
        # Try saving as GIF as fallback
        print("\nTrying to save as GIF instead...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            # Subsample frames for GIF (to reduce file size)
            subsample = 5
            subsampled_frames = frames[::subsample]
            
            fig2, ax2 = plt.subplots(figsize=(14, 8))
            ax2.set_aspect('equal')
            ax2.axis('off')
            im2 = ax2.imshow(subsampled_frames[0])
            
            def update2(frame_idx):
                im2.set_array(subsampled_frames[frame_idx])
                ax2.set_title(f'One or Three Chambers - Random Play', fontsize=12, fontweight='bold')
                return [im2]
            
            anim2 = animation.FuncAnimation(
                fig2, update2, frames=len(subsampled_frames),
                interval=100, blit=True, repeat=True
            )
            anim2.save(gif_path, writer='pillow', fps=10)
            print(f"✓ Animation saved as GIF to {gif_path}")
            print(f"  (Subsampled to {len(subsampled_frames)} frames)")
            plt.close(fig2)
        except Exception as e2:
            print(f"✗ Error saving GIF: {e2}")
    
    plt.close(fig)


def main():
    """Main function to run the random play video example."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'one_or_three_chambers_random_play.mp4')
    
    # Create video with 1000 steps
    create_random_play_video(output_path, num_steps=1000)
    
    print()
    print("=" * 70)
    print("Done! You can view the video at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
