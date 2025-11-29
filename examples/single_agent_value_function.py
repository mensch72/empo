#!/usr/bin/env python3
"""
Single Agent Value Function Example

This script creates a simple 7x7 gridworld with a single agent in the center,
computes human policy priors using backward induction for 5 time steps,
and displays the value function for all 25 possible goal states (empty cells)
as a 2D heatmap.

It loops over different beta values (1, 2, 4, 8, 16) and compares parallel
vs sequential computation to verify correctness.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, Tuple

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, Grid, World
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior

# Define the simple 7x7 environment map
SINGLE_AGENT_MAP = """
We We We We We We We
We .. .. .. Un Un We
We .. .. .. Un Un We
We .. .. Ay Un Un We
We .. .. .. Un Un We
We .. .. .. Un Un We
We We We We We We We
"""


class SingleAgentEnv(MultiGridEnv):
    """Simple 7x7 environment with a single agent in the center."""
    
    def __init__(self, max_steps=5):
        super().__init__(
            map=SINGLE_AGENT_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World
        )


class ReachCellGoal(PossibleGoal):
    """A goal where the agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = target_pos
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the agent is at the target position, 0 otherwise."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            pos_x, pos_y = agent_state[0], agent_state[1]
            if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell({self.target_pos[0]},{self.target_pos[1]})"
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                self.target_pos == other.target_pos)


class GridGoalGenerator(PossibleGoalGenerator):
    """Generates goals for all empty cells in the 5x5 interior of the grid."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
        
        # The 5x5 interior cells (excluding walls)
        # x ranges from 1 to 5, y ranges from 1 to 5
        self.empty_cells = []
        for y in range(1, 6):
            for x in range(1, 6):
                self.empty_cells.append((x, y))
        
        print(f"Goal cells: {len(self.empty_cells)} cells")
        self._goals_cache = {}
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        """Yields all possible goals for the agent with equal probability weights."""
        if human_agent_index not in self._goals_cache:
            goals = [ReachCellGoal(self.world_model, human_agent_index, pos) 
                    for pos in self.empty_cells]
            self._goals_cache[human_agent_index] = goals
        
        weight = 1.0 / len(self.empty_cells)
        for goal in self._goals_cache[human_agent_index]:
            yield goal, weight


def compute_for_beta(world_model, human_agent_indices, goal_generator, beta, parallel=True):
    """Compute V-values for a given beta, return the root state's value grid."""
    _, V_values = compute_human_policy_prior(
        world_model=world_model,
        human_agent_indices=human_agent_indices,
        possible_goal_generator=goal_generator,
        beta=beta,
        parallel=parallel,
        level_fct=lambda state: state[0],
        return_V_values=True
    )
    
    # Find root state (step_count=0)
    root_state = None
    for state in V_values.keys():
        if state[0] == 0:
            root_state = state
            break
    
    if root_state is None or root_state not in V_values or 0 not in V_values[root_state]:
        return None, None
    
    # Extract value grid
    value_grid = np.zeros((5, 5))
    goal_values = V_values[root_state][0]
    
    for goal, value in goal_values.items():
        target_x, target_y = goal.target_pos
        grid_x = target_x - 1
        grid_y = target_y - 1
        value_grid[grid_y, grid_x] = value
    
    return value_grid, goal_values


def main():
    print("=" * 70)
    print("Single Agent Value Function Example")
    print("Comparing Parallel vs Sequential for Multiple Beta Values")
    print("=" * 70)
    print()
    
    # Create environment with 5 time steps
    max_steps = 5
    print(f"Creating environment with max_steps={max_steps}...")
    world_model = SingleAgentEnv(max_steps=max_steps)
    world_model.reset()
    
    print(f"Grid size: {world_model.width} x {world_model.height}")
    print(f"Number of agents: {len(world_model.agents)}")
    print(f"Agent starting position: {world_model.agents[0].pos}")
    print()
    
    # The single agent is a human agent (index 0)
    human_agent_indices = [0]
    
    # Create goal generator
    print("Creating goal generator...")
    goal_generator = GridGoalGenerator(world_model)
    print()
    
    # Beta values to test
    beta_values = [1, 2, 4, 8, 16]
    
    # Store results for plotting
    all_value_grids = {}
    all_times = {'parallel': {}, 'sequential': {}}
    all_passed = True
    
    for beta in beta_values:
        print("=" * 70)
        print(f"Beta = {beta}")
        print("=" * 70)
        
        # Compute parallel
        print(f"\nComputing PARALLEL (beta={beta})...")
        start_time = time.time()
        value_grid_parallel, _ = compute_for_beta(
            world_model, human_agent_indices, goal_generator, beta, parallel=True
        )
        parallel_time = time.time() - start_time
        all_times['parallel'][beta] = parallel_time
        print(f"  Parallel took {parallel_time:.2f}s")
        
        # Compute sequential
        print(f"Computing SEQUENTIAL (beta={beta})...")
        start_time = time.time()
        value_grid_sequential, _ = compute_for_beta(
            world_model, human_agent_indices, goal_generator, beta, parallel=False
        )
        sequential_time = time.time() - start_time
        all_times['sequential'][beta] = sequential_time
        print(f"  Sequential took {sequential_time:.2f}s")
        
        # Compare results
        if value_grid_parallel is None or value_grid_sequential is None:
            print(f"  ✗ FAIL: Could not compute values for beta={beta}")
            all_passed = False
            continue
        
        max_diff = np.max(np.abs(value_grid_parallel - value_grid_sequential))
        if max_diff < 1e-10:
            print(f"  ✓ PASS: Parallel and sequential results are identical (max_diff={max_diff:.2e})")
        else:
            print(f"  ✗ FAIL: Results differ! max_diff={max_diff:.2e}")
            all_passed = False
        
        # Store for plotting
        all_value_grids[beta] = value_grid_parallel
        
        # Print value grid
        print(f"\n  Value Function Grid (beta={beta}):")
        print("       ", end="")
        for x in range(1, 6):
            print(f"  x={x} ", end="")
        print()
        for y in range(1, 6):
            print(f"  y={y}  ", end="")
            for x in range(1, 6):
                print(f" {value_grid_parallel[y-1, x-1]:.3f}", end="")
            print()
        
        # Speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nTiming comparison:")
    print(f"{'Beta':<8} {'Parallel':<12} {'Sequential':<12} {'Speedup':<10}")
    print("-" * 42)
    for beta in beta_values:
        p_time = all_times['parallel'].get(beta, 0)
        s_time = all_times['sequential'].get(beta, 0)
        speedup = s_time / p_time if p_time > 0 else 0
        print(f"{beta:<8} {p_time:<12.3f} {s_time:<12.3f} {speedup:<10.2f}x")
    
    if all_passed:
        print("\n✓ All beta values: Parallel and sequential results match!")
    else:
        print("\n✗ Some tests failed!")
    
    # Create multi-panel heatmap visualization
    fig, axes = plt.subplots(1, len(beta_values), figsize=(4*len(beta_values), 5))
    
    for idx, beta in enumerate(beta_values):
        ax = axes[idx]
        value_grid = all_value_grids.get(beta)
        
        if value_grid is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'β = {beta}')
            continue
        
        # Plot the heatmap
        im = ax.imshow(value_grid, cmap='RdYlGn', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels([str(i+1) for i in range(5)])
        ax.set_yticklabels([str(i+1) for i in range(5)])
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # Get min/max for text color threshold
        vmin, vmax = value_grid.min(), value_grid.max()
        vmid = (vmin + vmax) / 2
        
        # Add value text in each cell
        for y in range(5):
            for x in range(5):
                val = value_grid[y, x]
                text_color = 'white' if val < vmid else 'black'
                ax.text(x, y, f'{val:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
        
        # Mark agent starting position
        ax.plot(2, 2, 'b*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        
        ax.set_title(f'β = {beta}', fontsize=12, fontweight='bold')
        ax.set_xlabel('x')
        if idx == 0:
            ax.set_ylabel('y')
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('V(s₀, goal)', rotation=270, labelpad=15)
    
    fig.suptitle(f'Value Function for Different β Values\n(Single Agent, {max_steps} Time Steps, γ=1)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Save the figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'single_agent_value_function_beta_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
