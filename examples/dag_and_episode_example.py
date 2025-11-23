#!/usr/bin/env python3
"""
Example demonstrating DAG computation and episode visualization.

This script creates a small 4x4 multigrid environment with:
- 2 agents placed in opposite corners
- One agent that can push rocks, one that cannot
- 2 rocks and 2 blocks in the center
- 10-step timeout

It then:
1. Computes and plots the DAG structure of all reachable states
2. Saves one sample episode as a GIF animation
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Block, Rock, Wall, World
from empo.env_utils import get_dag, plot_dag


class SmallDAGEnv(MultiGridEnv):
    """Small 4x4 environment for DAG visualization."""
    
    def __init__(self):
        # Create 2 agents - agent 1 can push rocks, agent 0 cannot
        self.agents = [Agent(World, 0), Agent(World, 1)]
        
        super().__init__(
            width=4,
            height=4,
            max_steps=10,  # 10-step timeout
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
        
        # Place agents in opposite corners
        # Agent 0 in bottom-left corner (cannot push rocks)
        agent0 = self.agents[0]
        agent0.pos = np.array([1, 1])
        agent0.dir = 0  # facing right
        self.grid.set(1, 1, agent0)
        
        # Agent 1 in top-right corner (can push rocks)
        agent1 = self.agents[1]
        agent1.pos = np.array([2, 2])
        agent1.dir = 2  # facing left
        self.grid.set(2, 2, agent1)
        
        # Place 2 blocks in the center/wall area
        block1 = Block(World)
        self.grid.set(1, 2, block1)
        
        block2 = Block(World)
        self.grid.set(2, 1, block2)
        
        # Place 2 rocks on the wall boundaries (replacing walls)
        # Rock pushable only by agent 1 (who can push rocks)
        rock1 = Rock(World, pushable_by=1)
        self.grid.set(2, 3, rock1)  # Top boundary
        
        # Another rock pushable only by agent 1
        rock2 = Rock(World, pushable_by=1)
        self.grid.set(3, 2, rock2)  # Right boundary


def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    img = env.render(mode='rgb_array', highlight=False)
    return img


def create_sample_episode_gif(env, output_path='sample_episode.gif'):
    """Create and save a GIF of one sample episode."""
    print(f"\nCreating sample episode GIF...")
    
    # Reset environment
    env.reset()
    
    # Collect frames
    frames = []
    frames.append(render_grid_to_array(env))
    
    # Run a sample episode with random actions
    done = False
    step_count = 0
    max_steps = 10
    
    print(f"Running sample episode (max {max_steps} steps)...")
    while not done and step_count < max_steps:
        # Random actions for both agents
        actions = [env.action_space.sample() for _ in env.agents]
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        step_count += 1
        print(f"  Step {step_count}: actions={actions}, done={done}")
    
    # Create animation
    print(f"Saving GIF to {output_path}...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Sample Episode - Step {frame_idx}/{len(frames)-1}', 
                     fontsize=12, fontweight='bold')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=500,  # 500ms between frames
        blit=True,
        repeat=True
    )
    
    # Save as GIF
    try:
        anim.save(output_path, writer='pillow', fps=2)
        print(f"✓ Sample episode GIF saved to {output_path}")
        print(f"  Total frames: {len(frames)}")
    except Exception as e:
        print(f"✗ Error saving GIF: {e}")
    
    plt.close()


def compute_and_plot_dag(env, output_path='dag_plot.pdf'):
    """Compute and plot the DAG structure."""
    print("\nComputing DAG structure...")
    print("  This may take a moment for complex environments...")
    
    try:
        # Get DAG structure
        states, state_to_idx, successors = get_dag(env)
        
        print(f"✓ DAG computed successfully!")
        print(f"  Total reachable states: {len(states)}")
        print(f"  Root state index: 0")
        
        # Count terminal states
        terminal_count = sum(1 for succ_list in successors if len(succ_list) == 0)
        print(f"  Terminal states: {terminal_count}")
        
        # Count edges
        total_edges = sum(len(succ_list) for succ_list in successors)
        print(f"  Total transitions: {total_edges}")
        
        # Plot DAG
        print(f"\nPlotting DAG to {output_path}...")
        
        # Create simple labels showing state index
        state_labels = {state: f"S{idx}" for idx, state in enumerate(states)}
        
        # Plot with PDF format
        plot_dag(
            states, 
            state_to_idx, 
            successors, 
            output_file=output_path.replace('.pdf', ''),
            format='pdf',
            state_labels=state_labels,
            rankdir='TB'  # Top to bottom layout
        )
        
        print(f"✓ DAG plot saved to {output_path}")
        
        return states, state_to_idx, successors
        
    except Exception as e:
        print(f"✗ Error computing/plotting DAG: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """Main function to run the DAG and episode example."""
    print("=" * 70)
    print("DAG Computation and Episode Visualization Example")
    print("=" * 70)
    print()
    print("Environment Setup:")
    print("  - Grid size: 4x4")
    print("  - Agents: 2 (in opposite corners)")
    print("  - Agent 0: Bottom-left corner (cannot push rocks)")
    print("  - Agent 1: Top-right corner (can push rocks)")
    print("  - Objects: 2 blocks + 2 rocks in center area")
    print("  - Timeout: 10 steps")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = SmallDAGEnv()
    env.reset()
    print("✓ Environment created")
    
    # Compute and plot DAG
    dag_output = os.path.join(output_dir, 'dag_plot.pdf')
    states, state_to_idx, successors = compute_and_plot_dag(env, dag_output)
    
    # Create sample episode GIF
    gif_output = os.path.join(output_dir, 'sample_episode.gif')
    create_sample_episode_gif(env, gif_output)
    
    # Summary
    print()
    print("=" * 70)
    print("Done! Generated files:")
    print(f"  DAG plot (PDF): {os.path.abspath(dag_output)}")
    print(f"  Sample episode (GIF): {os.path.abspath(gif_output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
