#!/usr/bin/env python3
"""
Debug script to trace exact state transitions and understand value function computation.
"""

import sys
import os
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior

# Define the environment map with unsteady ground on the right
SINGLE_AGENT_MAP = """
We We We We We We We
We .. .. .. Un Un We
We .. .. .. Un Un We
We .. .. Ay Un Un We
We .. .. .. Un Un We
We .. .. .. Un Un We
We We We We We We We
"""

# Direction names
DIR_NAMES = {0: 'E', 1: 'S', 2: 'W', 3: 'N'}
ACTION_NAMES = ['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']


class SingleAgentEnv(MultiGridEnv):
    def __init__(self, max_steps=5):
        super().__init__(
            map=SINGLE_AGENT_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World
        )


class ReachCellGoal(PossibleGoal):
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = target_pos
    
    def is_achieved(self, state) -> int:
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


class SingleGoalGenerator:
    """Generator for a single specific goal."""
    def __init__(self, world_model, target_pos):
        self.world_model = world_model
        self.target_pos = target_pos
        self._goal = ReachCellGoal(world_model, 0, target_pos)
    
    def generate(self, state, human_agent_index):
        yield self._goal, 1.0


def format_state(state):
    """Format state for readable output."""
    step_count, agent_states, mobile_objects, mutable_objects = state
    agent = agent_states[0]
    x, y, dir_idx = agent[0], agent[1], agent[2]
    return f"t={step_count}, pos=({x},{y}), dir={DIR_NAMES[dir_idx]}"


def main():
    print("=" * 70)
    print("Debug: Tracing Value Function for Goal (5,4)")
    print("=" * 70)
    
    # Create environment
    max_steps = 5
    world_model = SingleAgentEnv(max_steps=max_steps)
    world_model.reset()
    
    print(f"\nEnvironment: 7x7 grid, max_steps={max_steps}")
    print(f"Agent starts at (3,3) facing {DIR_NAMES[world_model.agents[0].dir]}")
    print(f"Goal: reach cell (5,4)")
    print(f"Unsteady ground at x=4,5 for y=1-5 (stumble_prob=0.5)")
    print()
    
    # Target goal
    target_pos = (5, 4)
    goal_generator = SingleGoalGenerator(world_model, target_pos)
    
    # Compute with very high beta for near-deterministic behavior (now numerically stable)
    beta = 1e15
    print(f"Computing with beta={beta} (near-deterministic)...")
    
    _, V_values = compute_human_policy_prior(
        world_model=world_model,
        human_agent_indices=[0],
        possible_goal_generator=goal_generator,
        beta=beta,
        parallel=False,
        return_V_values=True
    )
    
    # Get DAG for transition analysis
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True)
    
    print(f"\nTotal states in DAG: {len(states)}")
    
    # Find root state
    root_state = None
    for state in states:
        if state[0] == 0:
            root_state = state
            break
    
    if root_state is None:
        print("ERROR: Root state not found!")
        return
    
    root_idx = state_to_idx[root_state]
    print(f"\nRoot state: {format_state(root_state)}")
    
    # Get the goal object
    goal = goal_generator._goal
    
    # Print V-value for root state
    if root_state in V_values and 0 in V_values[root_state]:
        root_V = V_values[root_state][0].get(goal, 0)
        print(f"V(root, goal=(5,4)) = {root_V}")
    else:
        print("V-value for root not found!")
        return
    
    print("\n" + "=" * 70)
    print("Tracing optimal path and transitions from root state")
    print("=" * 70)
    
    # Trace through the DAG
    def trace_state(state, depth=0, prob_so_far=1.0, visited=None):
        if visited is None:
            visited = set()
        
        indent = "  " * depth
        state_idx = state_to_idx[state]
        
        if state_idx in visited:
            print(f"{indent}(already visited)")
            return
        visited.add(state_idx)
        
        # Get V-value
        V = 0
        if state in V_values and 0 in V_values[state]:
            V = V_values[state][0].get(goal, 0)
        
        # Check if goal achieved
        achieved = goal.is_achieved(state)
        
        print(f"{indent}{format_state(state)} | V={V:.6f} | prob_so_far={prob_so_far:.6f} | achieved={achieved}")
        
        if achieved:
            return
        
        # Check if terminal (no transitions)
        if not transitions[state_idx]:
            print(f"{indent}  -> TERMINAL (no more steps)")
            return
        
        # Find the best action (highest Q-value)
        # For deterministic analysis, we need to find which action the agent takes
        
        # Get all transitions from this state
        trans_list = transitions[state_idx]
        
        # Group by action
        num_actions = 8
        action_transitions = {}
        for action_profile_tuple, probs, next_indices in trans_list:
            action = action_profile_tuple[0]  # Single agent
            if action not in action_transitions:
                action_transitions[action] = []
            action_transitions[action].append((probs, next_indices))
        
        # For each action, compute expected V
        action_EVs = {}
        for action, trans in action_transitions.items():
            EV = 0
            for probs, next_indices in trans:
                for i, next_idx in enumerate(next_indices):
                    next_state = states[next_idx]
                    next_V = 0
                    if next_state in V_values and 0 in V_values[next_state]:
                        next_V = V_values[next_state][0].get(goal, 0)
                    EV += probs[i] * next_V
            action_EVs[action] = EV
        
        # Find best action
        best_action = max(action_EVs.keys(), key=lambda a: action_EVs[a])
        
        print(f"{indent}  Actions and expected V-values:")
        for action in sorted(action_EVs.keys()):
            marker = " <-- BEST" if action == best_action else ""
            print(f"{indent}    {ACTION_NAMES[action]}: E[V]={action_EVs[action]:.6f}{marker}")
        
        # Show transitions for best action
        print(f"{indent}  Transitions for {ACTION_NAMES[best_action]}:")
        for probs, next_indices in action_transitions[best_action]:
            for i, next_idx in enumerate(next_indices):
                next_state = states[next_idx]
                next_V = 0
                if next_state in V_values and 0 in V_values[next_state]:
                    next_V = V_values[next_state][0].get(goal, 0)
                print(f"{indent}    p={probs[i]:.4f} -> {format_state(next_state)} | V={next_V:.6f}")
                
                # Recurse only for significant probabilities
                if probs[i] >= 0.1 and depth < 6:
                    trace_state(next_state, depth + 1, prob_so_far * probs[i], visited.copy())
    
    trace_state(root_state)
    
    print("\n" + "=" * 70)
    print("Summary: Computing V(root) from first-step transitions")
    print("=" * 70)
    
    # More detailed breakdown of the root state
    root_trans = transitions[root_idx]
    
    # Group by action
    action_details = {}
    for action_profile_tuple, probs, next_indices in root_trans:
        action = action_profile_tuple[0]
        if action not in action_details:
            action_details[action] = {'probs': [], 'next_states': [], 'next_Vs': []}
        
        for i, next_idx in enumerate(next_indices):
            next_state = states[next_idx]
            next_V = 0
            if next_state in V_values and 0 in V_values[next_state]:
                next_V = V_values[next_state][0].get(goal, 0)
            action_details[action]['probs'].append(probs[i])
            action_details[action]['next_states'].append(next_state)
            action_details[action]['next_Vs'].append(next_V)
    
    print("\nDetailed action analysis from root state:")
    for action in sorted(action_details.keys()):
        details = action_details[action]
        EV = sum(p * v for p, v in zip(details['probs'], details['next_Vs']))
        print(f"\n{ACTION_NAMES[action]}: E[V] = {EV:.6f}")
        for p, s, v in zip(details['probs'], details['next_states'], details['next_Vs']):
            print(f"  p={p:.4f}: {format_state(s)} -> V={v:.6f}")


if __name__ == "__main__":
    main()
