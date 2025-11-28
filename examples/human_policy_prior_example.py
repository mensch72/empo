#!/usr/bin/env python3
"""
Example script demonstrating compute_human_policy_prior() function.

This script creates a small gridworld environment and computes human policy priors
using backward induction. Each initially empty cell is treated as a possible goal.
"""

import sys
import os
import numpy as np
from typing import Iterator, Tuple

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior


class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = np.array(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position, 0 otherwise."""
        # Extract agent states from the compact state format
        # State format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Check if the specific human agent is at the target position
        # agent_states format: (pos_x, pos_y, dir, terminated, started, paused, on_unsteady, carrying_type, carrying_color)
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            pos_x, pos_y = agent_state[0], agent_state[1]
            if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell(agent_{self.human_agent_index}_to_{self.target_pos[0]},{self.target_pos[1]})"
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
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


def main():
    """Main function to demonstrate compute_human_policy_prior()."""
    print("=" * 70)
    print("Human Policy Prior Computation Example")
    print("=" * 70)
    print()
    
    # Create environment
    print("Creating SmallOneOrThreeChambersMapEnv...")
    env = SmallOneOrThreeChambersMapEnv()
    env.max_steps = 3  # Set to small value for quick testing
    env.reset()
    
    print(f"Environment created successfully!")
    print(f"Grid size: {env.width} x {env.height}")
    print(f"Number of agents: {len(env.agents)}")
    print(f"Max steps: {env.max_steps}")
    print()
    
    # Identify human agents
    human_agent_indices = []
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':  # Human agents are yellow
            human_agent_indices.append(i)
            print(f"Human agent {i}: pos={tuple(agent.pos)}")
    
    print(f"Found {len(human_agent_indices)} human agents")
    print()
    
    # Create goal generator
    print("Creating EmptyCellGoalGenerator...")
    goal_generator = EmptyCellGoalGenerator(env)
    print()
    
    # Test goal generation for initial state
    print("Testing goal generation for initial state...")
    initial_state = env.get_state()
    goal_count = 0
    for goal, weight in goal_generator.generate(initial_state, human_agent_indices[0]):
        if goal_count < 5:  # Show first 5 goals
            print(f"  Goal {goal_count + 1}: {goal} (weight: {weight:.4f})")
        goal_count += 1
    print(f"  ... and {goal_count - 5} more goals")
    print()
    
    # Compute human policy prior
    print("Computing human policy prior using backward induction...")
    print("This may take a while for large state spaces...")
    
    try:
        human_policy_prior = compute_human_policy_prior(
            world_model=env,
            human_agent_indices=human_agent_indices,
            possible_goal_generator=goal_generator
            # Using default values for believed_others_policy, beta=1, gamma=1
        )
        
        print("✓ Human policy prior computed successfully!")
        print(f"Policy prior type: {type(human_policy_prior)}")
        print(f"Human agent indices: {human_policy_prior.human_agent_indices}")
        print()
        
        # Test the policy prior
        print("Testing policy prior for initial state...")
        initial_state = env.get_state()
        
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
                    action_name = env.actions.available[action_idx] if action_idx < len(env.actions.available) else f"action_{action_idx}"
                    print(f"  {action_name}: {prob:.4f}")
                print()
                
                # Test sampling
                sampled_action = human_policy_prior.sample(initial_state, first_human_idx, first_goal)
                sampled_action_name = env.actions.available[sampled_action] if sampled_action < len(env.actions.available) else f"action_{sampled_action}"
                print(f"Sampled action: {sampled_action_name}")
        
        print()
        print("=" * 70)
        print("Example completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Error computing human policy prior: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()