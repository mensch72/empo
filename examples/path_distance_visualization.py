#!/usr/bin/env python3
"""
Path Distance Visualization Example.

This script demonstrates the PathDistanceCalculator by visualizing the effective
path-based distances from an agent to all possible target cells on a grid.

The visualization shows:
- Grid layout with walls, agents, and objects
- Color-coded overlay showing effective distance from the first human agent to each cell
- Comparison between path-based distance (with passing costs) and simple Manhattan distance
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.learning_based import PathDistanceCalculator


# ============================================================================
# Environment Definitions
# ============================================================================

# Simple 7x7 grid with some obstacles
OBSTACLE_MAP = '''
We We We We We We We We We
We .. .. .. We .. .. .. We
We .. Ay .. We .. .. .. We
We .. .. .. .. .. Bl .. We
We We We .. We We We .. We
We .. .. .. .. .. .. .. We
We .. Ro .. .. Ay .. .. We
We .. .. .. .. .. .. Ae We
We We We We We We We We We
'''

class ObstacleEnv(MultiGridEnv):
    """Environment with walls, blocks, and rocks for testing path distances."""
    
    def __init__(self, max_steps: int = 50):
        super().__init__(
            map=OBSTACLE_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )


def compute_distance_grid(env, calc, agent_pos, use_path_based=True):
    """
    Compute effective distance from agent_pos to all cells in the grid.
    
    Args:
        env: The environment
        calc: PathDistanceCalculator instance
        agent_pos: (x, y) position of the agent
        use_path_based: If True, use path-based cost; if False, use Manhattan distance
    
    Returns:
        2D numpy array with distances (inf for unreachable cells)
    """
    distances = np.full((env.height, env.width), np.inf)
    
    for y in range(env.height):
        for x in range(env.width):
            target = (x, y)
            
            if use_path_based:
                # Use path-based distance with passing costs
                cost = calc.compute_path_cost(agent_pos, target, env)
            else:
                # Use simple Manhattan distance
                path = calc.get_shortest_path(agent_pos, target)
                if path is not None:
                    cost = len(path) - 1  # Path length (excluding source)
                else:
                    cost = np.inf
            
            distances[y, x] = cost
    
    return distances


def render_grid_base(env, ax):
    """Render the basic grid layout (walls, objects, agents)."""
    grid_width = env.width
    grid_height = env.height
    
    # Draw grid lines
    for x in range(grid_width + 1):
        ax.axvline(x, color='lightgray', linewidth=0.5)
    for y in range(grid_height + 1):
        ax.axhline(y, color='lightgray', linewidth=0.5)
    
    # Draw cells
    for y in range(grid_height):
        for x in range(grid_width):
            cell = env.grid.get(x, y)
            
            if cell is not None:
                cell_type = getattr(cell, 'type', None)
                
                if cell_type == 'wall':
                    rect = plt.Rectangle((x, y), 1, 1, facecolor='gray', edgecolor='black')
                    ax.add_patch(rect)
                elif cell_type == 'magicwall':
                    rect = plt.Rectangle((x, y), 1, 1, facecolor='purple', edgecolor='black', alpha=0.7)
                    ax.add_patch(rect)
                elif cell_type == 'block':
                    rect = plt.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, facecolor='brown', edgecolor='black')
                    ax.add_patch(rect)
                elif cell_type == 'rock':
                    circle = plt.Circle((x + 0.5, y + 0.5), 0.35, facecolor='darkgray', edgecolor='black')
                    ax.add_patch(circle)
                elif cell_type == 'door':
                    is_open = getattr(cell, 'is_open', False)
                    color = 'green' if is_open else 'red'
                    rect = plt.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, facecolor=color, edgecolor='black', alpha=0.7)
                    ax.add_patch(rect)
    
    # Draw agents
    for i, agent in enumerate(env.agents):
        if agent.pos is not None:
            ax_pos, ay_pos = agent.pos
            color = agent.color if hasattr(agent, 'color') else 'blue'
            
            # Map color names to matplotlib colors
            color_map = {
                'yellow': 'gold',
                'grey': 'silver',
                'red': 'red',
                'blue': 'blue',
                'green': 'green',
            }
            mpl_color = color_map.get(color, color)
            
            # Draw agent as a triangle pointing in its direction
            circle = plt.Circle((ax_pos + 0.5, ay_pos + 0.5), 0.3, 
                               facecolor=mpl_color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            # Add agent index label
            ax.text(ax_pos + 0.5, ay_pos + 0.5, str(i), 
                   ha='center', va='center', fontsize=10, fontweight='bold')


def visualize_distances(env, calc, agent_idx=0, save_path=None):
    """
    Create a visualization comparing path-based and Manhattan distances.
    
    Args:
        env: The environment
        calc: PathDistanceCalculator instance
        agent_idx: Index of the agent to compute distances from
        save_path: Optional path to save the figure
    """
    # Get agent position
    agent = env.agents[agent_idx]
    agent_pos = (int(agent.pos[0]), int(agent.pos[1]))
    
    # Compute distances
    path_distances = compute_distance_grid(env, calc, agent_pos, use_path_based=True)
    manhattan_distances = compute_distance_grid(env, calc, agent_pos, use_path_based=False)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Find reasonable max value for color scaling (ignoring inf)
    finite_path = path_distances[np.isfinite(path_distances)]
    finite_manhattan = manhattan_distances[np.isfinite(manhattan_distances)]
    
    if len(finite_path) > 0:
        vmax_path = np.max(finite_path)
    else:
        vmax_path = 1
    
    if len(finite_manhattan) > 0:
        vmax_manhattan = np.max(finite_manhattan)
    else:
        vmax_manhattan = 1
    
    # Use common scale for comparison
    vmax = max(vmax_path, vmax_manhattan)
    
    for ax_idx, (ax, distances, title) in enumerate([
        (axes[0], path_distances, 'Path-Based Distance (with passing costs)'),
        (axes[1], manhattan_distances, 'Manhattan Distance (path length only)')
    ]):
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match grid coordinates
        
        # Draw distance heatmap
        cmap = plt.cm.RdYlGn_r  # Red = far, Green = close
        norm = Normalize(vmin=0, vmax=vmax)
        
        for y in range(env.height):
            for x in range(env.width):
                dist = distances[y, x]
                
                if np.isfinite(dist):
                    color = cmap(norm(dist))
                    rect = plt.Rectangle((x, y), 1, 1, facecolor=color, alpha=0.6)
                    ax.add_patch(rect)
                    
                    # Add distance value text
                    if dist < 100:
                        text_color = 'black' if dist < vmax * 0.6 else 'white'
                        ax.text(x + 0.5, y + 0.5, f'{dist:.0f}', 
                               ha='center', va='center', fontsize=8, color=text_color)
        
        # Render grid elements on top
        render_grid_base(env, ax)
        
        # Mark agent position with a star
        ax.plot(agent_pos[0] + 0.5, agent_pos[1] + 0.5, 'w*', markersize=20, 
               markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Distance / Cost', fontsize=10)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='black', label='Wall'),
        mpatches.Patch(facecolor='brown', edgecolor='black', label='Block (cost=2)'),
        mpatches.Circle((0, 0), 0.1, facecolor='darkgray', edgecolor='black', label='Rock (cost=50)'),
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Human Agent'),
        mpatches.Patch(facecolor='silver', edgecolor='black', label='Robot Agent'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9)
    
    plt.suptitle(f'Effective Distance from Agent {agent_idx} at {agent_pos}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def main():
    print("=" * 70)
    print("Path Distance Visualization")
    print("Comparing path-based costs vs simple Manhattan distance")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment with obstacles...")
    env = ObstacleEnv(max_steps=50)
    env.reset()
    
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of agents: {len(env.agents)}")
    
    # Show agent positions
    for i, agent in enumerate(env.agents):
        agent_type = "Human" if agent.color == 'yellow' else "Robot"
        print(f"  Agent {i} ({agent_type}): pos={tuple(agent.pos)}")
    
    print()
    
    # Create PathDistanceCalculator
    print("Initializing PathDistanceCalculator...")
    calc = PathDistanceCalculator(env)
    print(f"  Precomputed paths for {sum(len(v) for v in calc._shortest_paths.values())} cell pairs")
    print()
    
    # Show passing costs
    print("Passing costs:")
    for key, cost in sorted(calc.passing_costs.items()):
        if cost != float('inf'):
            print(f"  {key}: {cost}")
        else:
            print(f"  {key}: impassable")
    print()
    
    # Find first human agent
    human_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
            break
    
    if human_idx is None:
        human_idx = 0
    
    # Create visualization
    print(f"Creating distance visualization from Agent {human_idx}...")
    save_path = os.path.join(output_dir, 'path_distance_visualization.png')
    fig = visualize_distances(env, calc, agent_idx=human_idx, save_path=save_path)
    
    print()
    print("Visualization complete!")
    print()
    
    # Show some example distances
    agent_pos = (int(env.agents[human_idx].pos[0]), int(env.agents[human_idx].pos[1]))
    print(f"Example distances from agent at {agent_pos}:")
    
    # Pick a few target cells
    targets = [(1, 1), (7, 1), (3, 4), (7, 7)]
    for target in targets:
        path = calc.get_shortest_path(agent_pos, target)
        if path is not None:
            path_cost = calc.compute_path_cost(agent_pos, target, env)
            manhattan = len(path) - 1
            print(f"  To {target}: path_cost={path_cost:.0f}, manhattan={manhattan}, path={path}")
        else:
            print(f"  To {target}: unreachable (wall)")
    
    plt.show()


if __name__ == '__main__':
    main()
