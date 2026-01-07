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

import numpy as np

from multigrid_worlds.one_or_three_chambers import OneOrThreeChambersMapEnv


def create_random_play_video(output_path='one_or_three_chambers_random_play.mp4', num_steps=1000, fps=20):
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
    
    # Start video recording using MultiGridEnv's built-in method
    env.start_video_recording()
    
    # Initial state
    env.render(mode='rgb_array')
    
    print(f"Simulating {num_steps} steps of random play...")
    for step in range(num_steps):
        # Random actions for all agents
        actions = [env.action_space.sample() for _ in range(len(env.agents))]
        obs, rewards, done, info = env.step(actions)
        env.render(mode='rgb_array')  # Frame automatically captured
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_steps} complete")
        
        if done:
            print(f"  Episode terminated at step {step + 1}")
            break
    
    # Save video using MultiGridEnv's built-in method
    env.save_video(output_path, fps=fps)


# Configuration for quick mode vs full mode
NUM_STEPS_FULL = 1000   # Full mode: 1000 steps
NUM_STEPS_QUICK = 50    # Quick mode: 50 steps


def main(quick_mode=False):
    """Main function to run the random play video example."""
    num_steps = NUM_STEPS_QUICK if quick_mode else NUM_STEPS_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print(f"[{mode_str}]")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'one_or_three_chambers_random_play.mp4')
    
    # Create video with specified number of steps
    create_random_play_video(output_path, num_steps=num_steps)
    
    print()
    print("=" * 70)
    print("Done! You can view the video at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='One or Three Chambers Random Play')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer steps')
    args = parser.parse_args()
    main(quick_mode=args.quick)
