#!/usr/bin/env python3
"""
Bellman Backward Induction Example

This script demonstrates backward induction on the SmallOneOrTwoChambersMapEnv:
1. Computes the DAG of all reachable states
2. Performs backward induction from terminal states to the initial state
3. Uses the Bellman equation to compute value functions and optimal policies
4. Assumes humans move randomly and the robot follows an optimal policy
5. Produces a movie of an episode showing optimal robot play vs random humans

The reward function gives +1 when the robot is in cell (7, 3), 0 otherwise.
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

from envs.one_or_three_chambers import SmallOneOrTwoChambersMapEnv
from empo.env_utils import get_dag


# Target cell where the robot receives a reward of 1
TARGET_CELL = (7, 3)


def get_robot_position_from_state(env, state):
    """
    Extract the robot's position from a state tuple.
    
    Args:
        env: The environment instance
        state: A state tuple as returned by get_state()
        
    Returns:
        tuple: (x, y) position of the robot, or None if not found
    """
    # Set the environment to this state to easily extract information
    env.set_state(state)
    
    # Find the robot agent (green, index 1)
    for agent in env.agents:
        if agent.color == 'green':
            return tuple(agent.pos)
    return None


def compute_reward(env, state):
    """
    Compute the immediate reward for a given state.
    
    The reward is 1 if the robot is in the target cell, 0 otherwise.
    
    Args:
        env: The environment instance
        state: A state tuple
        
    Returns:
        float: The immediate reward
    """
    robot_pos = get_robot_position_from_state(env, state)
    if robot_pos == TARGET_CELL:
        return 1.0
    return 0.0


def compute_value_functions_and_policy(env, states, state_to_idx, successors):
    """
    Compute value functions and optimal policy using backward induction.
    
    Uses the Bellman equation for a Markov Decision Process where:
    - The robot is the decision-maker (player)
    - Humans move randomly (uniformly over their actions)
    - The robot aims to maximize expected reward
    
    The value function satisfies:
        V(s) = R(s) + γ * max_a [ Σ_{s'} P(s'|s,a) * V(s') ]
    
    where:
    - R(s) is the immediate reward at state s
    - γ is the discount factor (we use 1.0 for undiscounted)
    - a is the robot's action
    - P(s'|s,a) accounts for both the robot's action and random human actions
    
    Args:
        env: The environment instance
        states: List of states in topological order (from get_dag)
        state_to_idx: Dictionary mapping states to indices
        successors: List of successor indices for each state
        
    Returns:
        tuple: (value_function, optimal_policy)
            - value_function: dict mapping state index to value
            - optimal_policy: dict mapping state index to optimal robot action
    """
    num_states = len(states)
    num_actions = env.action_space.n
    num_agents = len(env.agents)
    
    # Find the robot agent index (the green one)
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'green':
            robot_idx = i
            break
    
    if robot_idx is None:
        raise ValueError("No robot (green) agent found in environment")
    
    human_indices = [i for i in range(num_agents) if i != robot_idx]
    num_humans = len(human_indices)
    
    print(f"Robot agent index: {robot_idx}")
    print(f"Human agent indices: {human_indices}")
    
    # Initialize value function (will be computed backwards)
    value_function = {}
    optimal_policy = {}
    
    # Initialize all terminal states with their immediate reward
    for state_idx in range(num_states):
        if len(successors[state_idx]) == 0:
            # Terminal state - value is just the immediate reward
            value_function[state_idx] = compute_reward(env, states[state_idx])
            optimal_policy[state_idx] = env.actions.still
    
    print(f"\nInitialized {len(value_function)} terminal states")
    
    # Backward induction: process states from last to first
    print("\nPerforming backward induction...")
    processed_count = 0
    
    for state_idx in reversed(range(num_states)):
        if state_idx in value_function:
            # Already processed (terminal state)
            processed_count += 1
            continue
        
        state = states[state_idx]
        immediate_reward = compute_reward(env, state)
        
        # Compute expected value for each robot action
        best_value = float('-inf')
        best_action = env.actions.still
        
        for robot_action in range(num_actions):
            # Compute expected value under random human actions
            expected_value = 0.0
            total_prob = 0.0
            
            # Enumerate all possible human action combinations
            human_action_combinations = list(product(range(num_actions), repeat=num_humans))
            
            for human_actions in human_action_combinations:
                # Build full action vector
                actions = [0] * num_agents
                actions[robot_idx] = robot_action
                for h_idx, h_action in zip(human_indices, human_actions):
                    actions[h_idx] = h_action
                
                # Get transition probabilities
                env.set_state(state)
                transitions = env.transition_probabilities(state, actions)
                
                if transitions is None:
                    # Terminal state or invalid action
                    continue
                
                # Each human action combination has equal probability
                human_action_prob = 1.0 / len(human_action_combinations)
                
                for prob, successor_state in transitions:
                    if successor_state in state_to_idx:
                        succ_idx = state_to_idx[successor_state]
                        if succ_idx in value_function:
                            expected_value += human_action_prob * prob * value_function[succ_idx]
                            total_prob += human_action_prob * prob
            
            if total_prob > 0:
                # Normalize if needed (should sum to ~1)
                expected_value = expected_value / total_prob if total_prob < 0.99 else expected_value
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = robot_action
        
        # Store value and optimal policy
        if best_value == float('-inf'):
            # No valid transitions found, set to immediate reward only
            value_function[state_idx] = immediate_reward
            optimal_policy[state_idx] = env.actions.still
        else:
            value_function[state_idx] = immediate_reward + best_value
            optimal_policy[state_idx] = best_action
        
        processed_count += 1
        if processed_count % 1000 == 0:
            print(f"  Processed {processed_count}/{num_states} states...")
    
    print(f"  Processed {processed_count}/{num_states} states - Done!")
    
    return value_function, optimal_policy


def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    img = env.render(mode='rgb_array', highlight=False)
    return img


def run_episode_with_optimal_policy(env, states, state_to_idx, optimal_policy, max_steps=None):
    """
    Run an episode where the robot follows the optimal policy and humans act randomly.
    
    Args:
        env: The environment instance
        states: List of states from get_dag
        state_to_idx: Dictionary mapping states to indices
        optimal_policy: Dictionary mapping state indices to optimal robot actions
        max_steps: Maximum number of steps (default: env.max_steps)
        
    Returns:
        list: List of frames (numpy arrays) for the episode
    """
    if max_steps is None:
        max_steps = env.max_steps
    
    # Reset environment
    env.reset()
    
    # Find robot and human indices
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'green':
            robot_idx = i
            break
    
    human_indices = [i for i in range(len(env.agents)) if i != robot_idx]
    
    # Collect frames
    frames = []
    frames.append(render_grid_to_array(env))
    
    print("\nRunning episode with optimal policy...")
    for step in range(max_steps):
        # Get current state
        current_state = env.get_state()
        
        # Check if we know this state
        if current_state in state_to_idx:
            state_idx = state_to_idx[current_state]
            robot_action = optimal_policy.get(state_idx, env.actions.still)
        else:
            # Unknown state, use still action
            robot_action = env.actions.still
        
        # Random actions for humans
        actions = [0] * len(env.agents)
        actions[robot_idx] = robot_action
        for h_idx in human_indices:
            actions[h_idx] = env.action_space.sample()
        
        # Take step
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        
        # Check if robot reached target
        robot_pos = tuple(env.agents[robot_idx].pos)
        action_name = env.actions.available[robot_action]
        print(f"  Step {step + 1}: robot_action={action_name}, robot_pos={robot_pos}, done={done}")
        
        if robot_pos == TARGET_CELL:
            print(f"  Robot reached target cell {TARGET_CELL}!")
        
        if done:
            print(f"  Episode ended at step {step + 1}")
            break
    
    return frames


def create_episode_movie(frames, output_path):
    """
    Create and save a movie from episode frames.
    
    Args:
        frames: List of numpy arrays (frames)
        output_path: Path to save the movie
    """
    print(f"\nCreating movie with {len(frames)} frames...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Optimal Robot Policy vs Random Humans - Step {frame_idx}/{len(frames)-1}\n'
                     f'Target: {TARGET_CELL}', fontsize=11, fontweight='bold')
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
    
    # Try saving as MP4 first, fallback to GIF
    try:
        writer = animation.FFMpegWriter(fps=2, bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"✓ Movie saved to {output_path}")
    except Exception as e:
        print(f"Note: Could not save as MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=2)
            print(f"✓ Movie saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"✗ Error saving movie: {e2}")
    
    plt.close()


def main():
    """Main function to run the backward induction example."""
    print("=" * 70)
    print("Bellman Backward Induction Example")
    print("=" * 70)
    print()
    print("This script demonstrates backward induction on a small gridworld:")
    print(f"  - Target cell: {TARGET_CELL} (robot receives reward=1 here)")
    print("  - Robot follows optimal policy computed via Bellman equation")
    print("  - Humans move randomly")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of agents: {len(env.agents)}")
    print(f"  Max steps: {env.max_steps}")
    
    # Print agent positions
    for i, agent in enumerate(env.agents):
        print(f"  Agent {i} ({agent.color}): pos={tuple(agent.pos)}")
    print()
    
    # Compute DAG
    print("Computing DAG structure...")
    print("  This may take a while for large state spaces...")
    states, state_to_idx, successors = get_dag(env)
    
    print(f"  Total reachable states: {len(states)}")
    terminal_count = sum(1 for succ_list in successors if len(succ_list) == 0)
    print(f"  Terminal states: {terminal_count}")
    total_edges = sum(len(succ_list) for succ_list in successors)
    print(f"  Total transitions: {total_edges}")
    print()
    
    # Compute value functions and optimal policy
    print("Computing value functions and optimal policy...")
    value_function, optimal_policy = compute_value_functions_and_policy(
        env, states, state_to_idx, successors
    )
    
    # Print initial state value
    env.reset()
    initial_state = env.get_state()
    if initial_state in state_to_idx:
        initial_idx = state_to_idx[initial_state]
        print(f"\nInitial state value: {value_function[initial_idx]:.4f}")
        print(f"Optimal initial action: {env.actions.available[optimal_policy[initial_idx]]}")
    print()
    
    # Run episode with optimal policy
    frames = run_episode_with_optimal_policy(env, states, state_to_idx, optimal_policy)
    
    # Create movie
    movie_path = os.path.join(output_dir, 'bellman_optimal_policy.mp4')
    create_episode_movie(frames, movie_path)
    
    # Summary
    print()
    print("=" * 70)
    print("Done! Generated files:")
    print(f"  Movie: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
