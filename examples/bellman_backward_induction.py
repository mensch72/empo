#!/usr/bin/env python3
"""
Bellman Backward Induction Example

This script demonstrates backward induction on the SmallOneOrTwoChambersMapEnv:
1. Computes the DAG of all reachable states
2. Performs backward induction from terminal states to the initial state
3. Uses the Bellman equation to compute value functions and optimal policies
4. Assumes humans move randomly and the robot follows an optimal policy
5. Produces a movie of an episode showing optimal robot play vs random humans

The reward function gives +1 when the robot is in cell (3, 7), 0 otherwise.

Usage:
    python bellman_backward_induction.py           # Full run (8 time steps)
    python bellman_backward_induction.py --quick   # Quick test (4 time steps)
    python bellman_backward_induction.py --profile # Enable line-by-line profiling

Run with --profile to enable line-by-line profiling of multigrid.py.
"""

import sys
import os
import time
import argparse

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
from tqdm import tqdm

from envs.one_or_three_chambers import SmallOneOrTwoChambersMapEnv

# Line profiler global
line_profiler = None

# Configuration constants
MAX_STEPS_FULL = 8        # Full run: 8 time steps
MAX_STEPS_QUICK = 4       # Quick test: 4 time steps for faster DAG computation


def setup_line_profiler():
    """Set up line profiler for all methods in multigrid.py."""
    global line_profiler
    try:
        from line_profiler import LineProfiler
        line_profiler = LineProfiler()
        
        # Import the multigrid module
        import gym_multigrid.multigrid as multigrid_module
        
        # Get the multigrid.py file path - only profile functions defined in this file
        multigrid_path = os.path.abspath(multigrid_module.__file__)
        print(f"Profiling methods from: {multigrid_path}")
        
        # Get all classes defined in multigrid.py (not imported ones)
        classes_to_profile = []
        for name in dir(multigrid_module):
            obj = getattr(multigrid_module, name)
            if isinstance(obj, type):
                # Check if the class is actually defined in multigrid.py
                try:
                    if hasattr(obj, '__module__') and 'multigrid' in obj.__module__:
                        classes_to_profile.append(obj)
                except:
                    pass
        
        print(f"Found {len(classes_to_profile)} classes to profile")
        
        methods_added = 0
        for cls in classes_to_profile:
            for attr_name in dir(cls):
                # Skip most dunder methods but keep important ones
                if attr_name.startswith('__') and attr_name not in ['__init__', '__call__', '__getitem__', '__setitem__', '__len__']:
                    continue
                try:
                    attr = getattr(cls, attr_name)
                    if callable(attr):
                        # Check if this function is defined in multigrid.py
                        func = None
                        if hasattr(attr, '__func__'):
                            func = attr.__func__
                        elif hasattr(attr, '__code__'):
                            func = attr
                        
                        if func and hasattr(func, '__code__'):
                            code_file = os.path.abspath(func.__code__.co_filename)
                            if code_file == multigrid_path:
                                line_profiler.add_function(func)
                                methods_added += 1
                except (AttributeError, TypeError):
                    pass
        
        # Also add module-level functions from multigrid.py
        for name in dir(multigrid_module):
            obj = getattr(multigrid_module, name)
            if callable(obj) and hasattr(obj, '__code__'):
                code_file = os.path.abspath(obj.__code__.co_filename)
                if code_file == multigrid_path:
                    try:
                        line_profiler.add_function(obj)
                        methods_added += 1
                    except:
                        pass
        
        print(f"Added {methods_added} methods to line profiler")
        return True
    except ImportError:
        print("Error: line_profiler not installed. Run 'pip install line_profiler'")
        return False
    except Exception as e:
        print(f"Error setting up line profiler: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_line_profiler_stats():
    """Print the line profiler statistics."""
    global line_profiler
    if line_profiler is None:
        return
    
    print("\n" + "=" * 80)
    print("LINE-BY-LINE PROFILING RESULTS FOR multigrid.py")
    print("=" * 80)
    
    # Print stats to stdout
    line_profiler.print_stats()
    
    # Also save to file
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, 'line_profile_multigrid.txt')
    
    with open(profile_path, 'w') as f:
        line_profiler.print_stats(stream=f)
    print(f"\nLine profile saved to: {os.path.abspath(profile_path)}")


# Target cell where the robot receives a reward of 1
# In the map, this is position (3, 7) - column 3, row 7:
# We We We We We .. .. .. .. ..    <- row 0
# We .. We .. We .. .. .. .. ..    <- row 1 (removed upper human)
# We .. We Ay We We We .. .. ..    <- row 2
# We Ae Ro .. .. .. We .. .. ..    <- row 3
# We .. We We We .. We We We ..    <- row 4
# We .. .. .. We .. .. .. We We    <- row 5
# We .. .. We .. .. Bl .. .. We    <- row 6
# We .. .. TT We We .. We We We    <- row 7 (target at column 3)
# We We We We .. We We We .. ..    <- row 8
TARGET_CELL = (3, 7)


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
    
    # Find the robot agent (grey)
    for agent in env.agents:
        if agent.color == 'grey':
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


def compute_value_functions_and_policy(env, states, state_to_idx, successors, transitions=None):
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
        transitions: Optional cached transitions from get_dag(return_probabilities=True).
            Each element is a list of (action, probs, succ_indices) tuples.
        
    Returns:
        tuple: (value_function, optimal_policy)
            - value_function: dict mapping state index to value
            - optimal_policy: dict mapping state index to optimal robot action
    """
    num_states = len(states)
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    
    print(f"Number of actions per agent: {num_actions}")
    print(f"Total action combinations: {num_actions}^{num_agents} = {num_actions ** num_agents}")
    
    # Find the robot agent (the grey one)
    robot_idx = None
    robot_agent = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'grey':
            robot_idx = i
            robot_agent = agent
            break
    
    if robot_idx is None:
        raise ValueError("No robot (grey) agent found in environment")
    
    # Convert TARGET_CELL to numpy array to match agent.pos type
    target_pos = np.array(TARGET_CELL)
    
    human_indices = [i for i in range(num_agents) if i != robot_idx]
    num_humans = len(human_indices)
    num_human_actions = num_actions ** num_humans
    
    print(f"Robot agent index: {robot_idx}")
    print(f"Human agent indices: {human_indices}")
    print(f"Using cached transitions: {transitions is not None}")
    
    # Initialize value function (will be computed backwards)
    value_function = {}
    optimal_policy = {}
    
    # Initialize all terminal states with their immediate reward
    for state_idx in range(num_states):
        if len(successors[state_idx]) == 0:
            # Terminal state - value is just the immediate reward
            # Inline reward computation: 1 if robot at target, else 0
            env.set_state(states[state_idx])
            reward = 1.0 if np.array_equal(robot_agent.pos, target_pos) else 0.0
            value_function[state_idx] = reward
            optimal_policy[state_idx] = env.actions.still
    
    print(f"\nInitialized {len(value_function)} terminal states")
    
    # Pre-compute action set
    action_set = list(range(num_actions))
    
    # Backward induction: process states from last to first
    print("\nPerforming backward induction...")
    
    for state_idx in tqdm(reversed(range(num_states)), total=num_states, desc="Backward induction", unit="states"):
        if state_idx in value_function:
            # Already processed (terminal state)
            continue
        
        state = states[state_idx]
        # Inline reward computation: 1 if robot at target, else 0
        env.set_state(state)
        immediate_reward = 1.0 if np.array_equal(robot_agent.pos, target_pos) else 0.0
        
        # Compute expected value for each robot action
        best_value = float('-inf')
        best_action = env.actions.still
        
        if transitions is not None:
            # Use cached transitions - group by robot action
            robot_action_values = {}
            robot_action_counts = {}
            
            for action, probs, succ_indices in transitions[state_idx]:
                robot_action = action[robot_idx]
                if robot_action not in robot_action_values:
                    robot_action_values[robot_action] = 0.0
                    robot_action_counts[robot_action] = 0
                
                # Sum over transition probabilities
                for prob, succ_idx in zip(probs, succ_indices):
                    if succ_idx in value_function:
                        robot_action_values[robot_action] += prob * value_function[succ_idx]
                robot_action_counts[robot_action] += 1
            
            # Find best action (average over human actions)
            for robot_action in robot_action_values:
                if robot_action_counts[robot_action] > 0:
                    expected_value = robot_action_values[robot_action] / num_human_actions
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = robot_action
        else:
            # Compute transitions on-the-fly (slow path)
            all_action_combinations = list(product(action_set, repeat=num_agents))
            
            # Group action combinations by robot action
            robot_action_groups = {}
            for actions in all_action_combinations:
                r_action = actions[robot_idx]
                if r_action not in robot_action_groups:
                    robot_action_groups[r_action] = []
                robot_action_groups[r_action].append(actions)
            
            for robot_action in action_set:
                if robot_action not in robot_action_groups:
                    continue
                action_combos = robot_action_groups[robot_action]
                
                expected_value = 0.0
                valid_transitions = 0
                
                for actions in action_combos:
                    env.set_state(state)
                    trans_result = env.transition_probabilities(state, list(actions))
                    
                    if trans_result is None:
                        continue
                    
                    for prob, successor_state in trans_result:
                        if successor_state in state_to_idx:
                            succ_idx = state_to_idx[successor_state]
                            if succ_idx in value_function:
                                expected_value += prob * value_function[succ_idx]
                                valid_transitions += 1
                
                if valid_transitions > 0:
                    expected_value = expected_value / num_human_actions
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = robot_action
        
        # Store value and optimal policy
        if best_value == float('-inf'):
            value_function[state_idx] = immediate_reward
            optimal_policy[state_idx] = env.actions.still
        else:
            value_function[state_idx] = immediate_reward + best_value
            optimal_policy[state_idx] = best_action
    
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
        if agent.color == 'grey':
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


def main(enable_profiling=False, quick_mode=False):
    """Main function to run the backward induction example."""
    global line_profiler
    
    # Determine configuration based on mode
    max_steps = MAX_STEPS_QUICK if quick_mode else MAX_STEPS_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    # Set up line profiler if requested
    if enable_profiling:
        print("Setting up line profiler...")
        if not setup_line_profiler():
            print("Continuing without profiling...")
            enable_profiling = False
    
    print("=" * 70)
    print("Bellman Backward Induction Example")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    print("This script demonstrates backward induction on a small gridworld:")
    print(f"  - Target cell: {TARGET_CELL} (robot receives reward=1 here)")
    print("  - Robot follows optimal policy computed via Bellman equation")
    print("  - Humans move randomly")
    print(f"  - Max steps: {max_steps}")
    if enable_profiling:
        print("  - LINE PROFILING ENABLED")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = SmallOneOrTwoChambersMapEnv()
    env.max_steps = max_steps  # Override max_steps for quick mode
    env.reset()
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of agents: {len(env.agents)}")
    print(f"  Max steps: {env.max_steps}")
    
    # Print agent positions
    for i, agent in enumerate(env.agents):
        print(f"  Agent {i} ({agent.color}): pos={tuple(agent.pos)}")
    print()
    
    # Enable profiler before DAG computation
    if enable_profiling and line_profiler is not None:
        print("Enabling line profiler...")
        line_profiler.enable()
    
    # Compute DAG with cached transitions
    print("Computing DAG structure with transitions...")
    print("  This may take a while for large state spaces...")
    dag_start = time.time()
    states, state_to_idx, successors, transitions = env.get_dag(return_probabilities=True)
    dag_time = time.time() - dag_start
    
    print(f"  Total reachable states: {len(states)}")
    terminal_count = sum(1 for succ_list in successors if len(succ_list) == 0)
    print(f"  Terminal states: {terminal_count}")
    total_edges = sum(len(succ_list) for succ_list in successors)
    print(f"  Total transitions: {total_edges}")
    total_action_trans = sum(len(t) for t in transitions)
    print(f"  Total action-transitions: {total_action_trans}")
    print(f"  DAG computation time: {dag_time:.1f}s")
    print()
    
    # Compute value functions and optimal policy using cached transitions
    print("Computing value functions and optimal policy...")
    value_function, optimal_policy = compute_value_functions_and_policy(
        env, states, state_to_idx, successors, transitions
    )
    
    # Disable profiler 
    if enable_profiling and line_profiler is not None:
        line_profiler.disable()
    
    # Print initial state value
    env.reset()
    initial_state = env.get_state()
    if initial_state in state_to_idx:
        initial_idx = state_to_idx[initial_state]
        print(f"\nInitial state value: {value_function[initial_idx]:.4f}")
        print(f"Optimal initial action: {env.actions.available[optimal_policy[initial_idx]]}")
    print()
    
    # Print profiling results
    if enable_profiling:
        print_line_profiler_stats()
    
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
    parser = argparse.ArgumentParser(description='Bellman backward induction example')
    parser.add_argument('--profile', action='store_true', 
                        help='Enable line-by-line profiling of multigrid.py')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with reduced time steps')
    args = parser.parse_args()
    main(enable_profiling=args.profile, quick_mode=args.quick)
