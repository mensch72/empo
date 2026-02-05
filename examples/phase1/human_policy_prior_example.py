#!/usr/bin/env python3
"""
Example script demonstrating compute_human_policy_prior() function.

This script creates a small gridworld environment and computes human policy priors
using backward induction. Each initially empty cell is treated as a possible goal.
Includes line profiling of backward_induction.py, possible_goal.py, and multigrid.py.
"""

import os
import time
import numpy as np
from typing import Iterator, Tuple

# Add line profiling support
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    print("Warning: line_profiler not available. Install with: pip install line_profiler")
    LINE_PROFILER_AVAILABLE = False

LINE_PROFILER_AVAILABLE = False

from multigrid_worlds.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior


class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = np.array(target_pos)
        self._hash = hash((self.human_agent_index, target_pos[0], target_pos[1]))
        super()._freeze()  # Make immutable
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position, 0 otherwise."""
        # Extract agent states from the compact state format
        # State format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Check if the specific human agent is at the target position
        # agent_states format: (pos_x, pos_y, dir, terminated, started, paused, carrying_type, carrying_color)
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            pos_x, pos_y = agent_state[0], agent_state[1]
            if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell(agent_{self.human_agent_index}_to_{self.target_pos[0]},{self.target_pos[1]})"
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                np.array_equal(self.target_pos, other.target_pos))


class EmptyCellGoalGenerator(PossibleGoalGenerator):
    """Generates goals for all initially empty cells in the environment."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
        
        # Find all potentially reachable cells (not walls)
        self.empty_cells = []
        
        for x in range(world_model.width):
            for y in range(world_model.height):
                cell = world_model.grid.get(x, y)
                # Cell is reachable if it's not a wall
                if cell is None:
                    # Truly empty cell
                    self.empty_cells.append((x, y))
                elif hasattr(cell, 'type') and cell.type != 'wall':
                    # Cell contains non-wall object (agent, block, etc.) - still a valid goal
                    self.empty_cells.append((x, y))
                elif hasattr(cell, 'color') and cell.color in ['yellow', 'grey']:
                    # Cell contains an agent - valid goal position
                    self.empty_cells.append((x, y))
        
        print(f"Found {len(self.empty_cells)} reachable cells: {self.empty_cells[:10]}...")
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        """Yields all possible goals for the specific human agent with equal probability weights."""
        total_goals = len(self.empty_cells)
        if total_goals == 0:
            return
        
        weight = 1.0 / total_goals  # Equal probability for each goal
        
        for pos in self.empty_cells:
            goal = ReachCellGoal(self.world_model, human_agent_index, pos)
            yield goal, weight


def setup_line_profiler():
    """Setup line profiler for key modules."""
    if not LINE_PROFILER_AVAILABLE:
        return None
    
    profiler = LineProfiler()
    
    # Import modules to profile
    import empo.backward_induction as backward_induction_module
    
    # Import multigrid
    import gym_multigrid.multigrid as multigrid_module
    
    # Add functions to profile from backward_induction.py
    if hasattr(backward_induction_module, 'compute_human_policy_prior'):
        profiler.add_function(backward_induction_module.compute_human_policy_prior)
    
    # Add methods to profile from possible_goal.py
    # Note: Individual goal methods will be profiled when called
    
    # Add key methods from multigrid.py
    if hasattr(multigrid_module, 'MultiGridEnv'):
        profiler.add_function(multigrid_module.MultiGridEnv.get_state)
        profiler.add_function(multigrid_module.MultiGridEnv.set_state)
        profiler.add_function(multigrid_module.MultiGridEnv.transition_probabilities)
        profiler.add_function(multigrid_module.MultiGridEnv.get_dag)
        if hasattr(multigrid_module.MultiGridEnv, 'step'):
            profiler.add_function(multigrid_module.MultiGridEnv.step)
    
    return profiler


def save_profiler_results(profiler, output_path):
    """Save line profiler results to a text file."""
    if profiler is None:
        return
    
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Capture profiler output
    import io
    from contextlib import redirect_stdout
    
    output = io.StringIO()
    with redirect_stdout(output):
        profiler.print_stats()
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("Line Profiling Results for Human Policy Prior Computation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("Profiled modules:\n")
        f.write("- empo.backward_induction\n")
        f.write("- empo.possible_goal\n")
        f.write("- gym_multigrid.multigrid\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write(output.getvalue())
    
    print(f"Line profiling results saved to: {output_path}")


# Configuration for quick mode vs full mode
MAX_STEPS_FULL = 3        # Full mode (still relatively small)
MAX_STEPS_QUICK = 2       # Quick mode (minimal)


def main(quick_mode=False):
    """Main function to demonstrate compute_human_policy_prior()."""
    max_steps = MAX_STEPS_QUICK if quick_mode else MAX_STEPS_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print("=" * 70)
    print("Human Policy Prior Computation Example with Line Profiling")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    
    # Setup line profiler
    profiler = setup_line_profiler()
    if profiler:
        print("✓ Line profiler setup complete")
    else:
        print("✗ Line profiler not available")
    print()
    
    # Create environment
    print("Creating SmallOneOrThreeChambersMapEnv...")
    world_model = SmallOneOrThreeChambersMapEnv()
    world_model.max_steps = max_steps
    world_model.reset()
    
    print(f"Environment created successfully!")
    print(f"Grid size: {world_model.width} x {world_model.height}")
    print(f"Number of agents: {len(world_model.agents)}")
    print(f"Max steps: {world_model.max_steps}")
    print()
    
    # Identify human agents
    human_agent_indices = []
    for i, agent in enumerate(world_model.agents):
        if agent.color == 'yellow':  # Human agents are yellow
            human_agent_indices.append(i)
            print(f"Human agent {i}: pos={tuple(agent.pos)}")
    
    print(f"Found {len(human_agent_indices)} human agents")
    print()
    
    # Create goal generator
    print("Creating EmptyCellGoalGenerator...")
    goal_generator = EmptyCellGoalGenerator(world_model)
    print()
        
    # Compute human policy prior
    print("Computing human policy prior using backward induction...")
    print("This may take a while for large state spaces...")
    
    try:
        t0 = time.time()
        
        if profiler:
            # Run with line profiler
            profiler.enable()
        
        human_policy_prior = compute_human_policy_prior(
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            possible_goal_generator=goal_generator,
            parallel=False, #True,
            level_fct=lambda state: state[0]  # Use step_count for fast level computation
            # Using default values for believed_others_policy, beta=1, gamma=1
        )
        
        if profiler:
            profiler.disable()
            
        print(f"(took {time.time()-t0} seconds)")
        
        print("✓ Human policy prior computed successfully!")
        print(f"Policy prior type: {type(human_policy_prior)}")
        print(f"Human agent indices: {human_policy_prior.human_agent_indices}")
        print()
        
        # Test the policy prior
        print("Testing policy prior for initial state...")
        world_model.reset()
        initial_state = world_model.get_state()

        if len(human_agent_indices) > 0:
            first_human_idx = human_agent_indices[0]
            
            # Test with first goal
            first_goal = None
            for goal, weight in goal_generator.generate(initial_state, first_human_idx):
                first_goal = goal
                break
            
            if first_goal is not None:
                action_probs = human_policy_prior(initial_state, first_human_idx, first_goal)
                print(f"Action probabilities for human {first_human_idx} with goal {first_goal}:")
                for action_idx, prob in enumerate(action_probs):
                    action_name = world_model.actions.available[action_idx] if action_idx < len(world_model.actions.available) else f"action_{action_idx}"
                    print(f"  {action_name}: {prob:.4f}")
                print()
                
                # Test sampling
                sampled_action = human_policy_prior.sample(initial_state, first_human_idx, first_goal)
                sampled_action_name = world_model.actions.available[sampled_action] if sampled_action < len(world_model.actions.available) else f"action_{sampled_action}"
                print(f"Sampled action: {sampled_action_name}")
        
        print()
        print("=" * 70)
        print("Example completed successfully!")
        
        # Save profiling results
        if profiler:
            output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
            profile_output_path = os.path.join(output_dir, 'human_policy_prior_line_profile.txt')
            save_profiler_results(profiler, profile_output_path)
        
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Error computing human policy prior: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Human Policy Prior Example')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer time steps')
    args = parser.parse_args()
    main(quick_mode=args.quick)