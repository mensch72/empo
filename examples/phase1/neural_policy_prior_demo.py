#!/usr/bin/env python3
"""
Neural Network Policy Prior Demo with Value Function Visualization.

This script demonstrates the neural network-based policy prior on a minimal
5x5 empty grid environment with:
- 2 human agents (yellow)
- 1 robot agent (grey)
- 10 time steps
- All 25 cells as possible goals

The demo uses the nn_based module to:
1. Create the environment
2. Train neural networks (h_Q) to learn human policies for all 25 goal cells
3. Produce a movie of 10 rollouts where:
   - Humans follow their learned goal-specific Boltzmann policies
   - Robot uses random policy
   - First human's actual goal marked with a star
   - All cells color-coded by the first human's value function for alternative goals
"""

import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Tuple, Dict, List, Any, Optional

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.world_specific_helpers.multigrid import ReachCellGoal, MultiGridGoalSampler, RandomPolicy, render_goal_overlay
from empo.learning_based.multigrid import (
    MultiGridQNetwork as QNetwork,
    MultiGridNeuralHumanPolicyPrior as NeuralHumanPolicyPrior,
    train_multigrid_neural_policy_prior as train_neural_policy_prior,
    NUM_OBJECT_TYPE_CHANNELS,
    OBJECT_TYPE_TO_CHANNEL,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
)


# ============================================================================
# Environment Definition: 5x5 Empty Grid with Walls
# ============================================================================

EMPTY_5X5_MAP = """
We We We We We We We
We .. .. .. .. .. We
We .. .. We Ay .. We
We .. Ay Ae Ro .. We
We .. .. .. .. Bl We
We .. .. .. .. .. We
We We We We We We We
"""
MAX_STEPS = 20  # Used for normalization in state_to_grid_tensor

class Empty5x5Env(MultiGridEnv):
    """
    A minimal 5x5 empty grid environment with 2 humans and 1 robot.
    
    Grid layout (7x7 total with walls):
        - Interior: 5x5 empty cells
        - Agents: 2 yellow (humans) at corners, 1 grey (robot) in center
    """
    
    def __init__(self, max_steps: int = MAX_STEPS):
        super().__init__(
            map=EMPTY_5X5_MAP,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        self.num_humans = sum(1 for a in self.agents if a.color == 'yellow')
        self.num_robots = sum(1 for a in self.agents if a.color == 'grey')


# ============================================================================
# Helper functions for the demo
# ============================================================================

def state_to_grid_tensor(
    state, 
    grid_width: int, 
    grid_height: int,
    num_agents: int,
    num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
    device: str = 'cpu',
    world_model: Any = None,
    human_agent_indices: Optional[List[int]] = None,
    query_agent_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a multigrid state to tensor representation for the neural network.
    
    Uses the same channel structure as StateEncoder:
    - num_object_types: explicit object type channels
    - 3: "other" object channels (overlappable, immobile, mobile)
    - 1: per-color agent channel (backward compatibility mode)
    - 1: query agent channel
    """
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    # Channel structure (matching StateEncoder with num_agents_per_color=None)
    num_other_object_channels = 3
    num_color_channels = 1  # Backward compatibility: single channel for all agents
    num_channels = num_object_types + num_other_object_channels + num_color_channels + 1
    
    # Channel indices
    other_overlappable_idx = num_object_types
    other_immobile_idx = num_object_types + 1
    other_mobile_idx = num_object_types + 2
    color_channels_start = num_object_types + num_other_object_channels
    query_agent_channel_idx = color_channels_start + num_color_channels
    
    grid_tensor = torch.zeros(1, num_channels, grid_height, grid_width, device=device)
    
    # 1. Encode object-type channels from the persistent world grid
    if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
        for y in range(grid_height):
            for x in range(grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is not None:
                        if cell_type in OBJECT_TYPE_TO_CHANNEL:
                            channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                            if channel_idx < num_object_types:
                                grid_tensor[0, channel_idx, y, x] = 1.0
                        else:
                            # Object type not in explicit channels - use "other" channels
                            if cell_type in OVERLAPPABLE_OBJECTS:
                                grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                                grid_tensor[0, other_mobile_idx, y, x] = 1.0
                            else:
                                grid_tensor[0, other_immobile_idx, y, x] = 1.0
    
    # 2. Encode all agent positions in single color channel (backward compatibility)
    for i, agent_state in enumerate(agent_states):
        x, y = int(agent_state[0]), int(agent_state[1])
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid_tensor[0, color_channels_start, y, x] = 1.0
    
    # 3. Encode query agent channel
    if query_agent_index is not None and query_agent_index < len(agent_states):
        agent_state = agent_states[query_agent_index]
        x, y = int(agent_state[0]), int(agent_state[1])
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid_tensor[0, query_agent_channel_idx, y, x] = 1.0
    
    # Normalize step count
    max_steps = MAX_STEPS
    step_tensor = torch.tensor([[step_count / max_steps]], device=device, dtype=torch.float32)
    
    return grid_tensor, step_tensor


def get_agent_tensors(
    state,
    human_idx: int,
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract agent position, direction, and index tensors from state.
    
    The AgentEncoder handles clamping internally for indices beyond its capacity.
    """
    _, agent_states, _, _ = state
    agent_state = agent_states[human_idx]
    
    position = torch.tensor([[
        agent_state[0] / grid_width,
        agent_state[1] / grid_height
    ]], device=device, dtype=torch.float32)
    
    direction = torch.zeros(1, 4, device=device)
    dir_idx = int(agent_state[2]) % 4
    direction[0, dir_idx] = 1.0
    
    agent_idx_tensor = torch.tensor([human_idx], device=device)
    
    return position, direction, agent_idx_tensor


def get_goal_tensor(
    goal_pos: Tuple[int, int],
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convert goal position to normalized tensor.
    
    The tensor has 4 values [x1, y1, x2, y2] to support both point goals
    and rectangular region goals. For point goals, x1=x2 and y1=y2.
    """
    return torch.tensor([[
        goal_pos[0] / grid_width,
        goal_pos[1] / grid_height,
        goal_pos[0] / grid_width,  # Same as x1 for point goals
        goal_pos[1] / grid_height  # Same as y1 for point goals
    ]], device=device, dtype=torch.float32)


def compute_value_for_goals(
    q_network: QNetwork,
    state,
    human_idx: int,
    goal_cells: List[Tuple[int, int]],
    grid_width: int,
    grid_height: int,
    num_agents: int,
    beta: float = 5.0,
    device: str = 'cpu',
    world_model: Any = None,
    human_agent_indices: Optional[List[int]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Compute V-value for each goal at the current state using the Q-network.
    
    V(s, g) = Σ_a π(a|s,g) * Q(s,a,g) where π = softmax(β*Q)
    
    If the agent is already at the goal, V(s,g) = 1.0 (goal achieved).
    """
    values = {}
    
    # Get agent position
    _, agent_states, _, _ = state
    agent_pos = agent_states[human_idx]
    agent_x, agent_y = int(agent_pos[0]), int(agent_pos[1])
    
    # Create simple goal object for encode_and_forward
    class SimpleGoal:
        def __init__(self, pos):
            self.target_pos = pos
    
    with torch.no_grad():
        for goal_pos in goal_cells:
            # If agent is already at this goal, value is 1.0
            if agent_x == goal_pos[0] and agent_y == goal_pos[1]:
                values[goal_pos] = 1.0
                continue
            
            goal = SimpleGoal(goal_pos)
            q_values = q_network.forward(
                state, world_model, human_idx, goal, device
            )
            
            # V = E_π[Q] where π = softmax(β*Q)
            policy = F.softmax(beta * q_values, dim=1)
            v_value = (policy * q_values).sum().item()
            values[goal_pos] = v_value
    
    return values


# ============================================================================
# Rendering with Value Function Overlay
# ============================================================================

def render_with_value_overlay(
    env: MultiGridEnv,
    value_dict: Dict[Tuple[int, int], float],
    actual_goal: Tuple[int, int],
    tile_size: int = 32
) -> np.ndarray:
    """
    Render the environment with value function overlay and goal rectangle.
    
    Uses render_goal_overlay from empo.multigrid for dashed blue rectangle
    boundaries and agent-to-goal connection lines.
    """
    img = env.render(mode='rgb_array', highlight=False)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Normalize values for colormap
    values = list(value_dict.values())
    if len(values) > 0 and max(values) > min(values):
        norm = Normalize(vmin=min(values), vmax=max(values))
    else:
        norm = Normalize(vmin=0, vmax=1)
    
    cmap = plt.cm.RdYlGn
    
    # Overlay value colors on each cell
    for (x, y), val in value_dict.items():
        px = x * tile_size + tile_size // 2
        py = y * tile_size + tile_size // 2
        
        color = cmap(norm(val))
        circle = plt.Circle((px, py), tile_size * 0.35, color=color, alpha=0.5)
        ax.add_patch(circle)
        
        ax.text(px, py, f'{val:.2f}', ha='center', va='center', 
                fontsize=7, fontweight='bold', color='black')
    
    # Mark first human and render goal with dashed rectangle and connection line
    step_count, agent_states, _, _ = env.get_state()
    if len(agent_states) > 0 and actual_goal:
        first_human_pos = agent_states[0]
        agent_pos = (float(first_human_pos[0]), float(first_human_pos[1]))
        
        # Point goal represented as (x, y, x, y)
        goal = (actual_goal[0], actual_goal[1], actual_goal[0], actual_goal[1])
        
        render_goal_overlay(
            ax=ax,
            goal=goal,
            agent_pos=agent_pos,
            agent_idx=0,
            tile_size=tile_size,
            goal_color=(0.0, 0.4, 1.0, 0.7),  # Blue, semi-transparent
            line_width=2.5,
            inset=0.08
        )
        
        hx = int(first_human_pos[0]) * tile_size + tile_size // 2
        hy = int(first_human_pos[1]) * tile_size + tile_size // 2
        # Add "H1" label
        ax.text(hx, hy - tile_size * 0.3, 'H1', ha='center', va='center',
                fontsize=9, fontweight='bold', color='cyan',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = buf[:, :, :3]
    
    plt.close(fig)
    return buf


# ============================================================================
# Rollout with Learned Policies
# ============================================================================

def run_rollout_with_learned_policies(
    env: MultiGridEnv,
    neural_prior: NeuralHumanPolicyPrior,
    goal_cells: List[Tuple[int, int]],
    human_goals: Dict[int, ReachCellGoal],  # ReachCellGoal for each human
    human_agent_indices: List[int],
    robot_index: int,
    first_human_idx: int,  # which human's value function to visualize
    beta: float = 5.0,
    device: str = 'cpu'
) -> List[np.ndarray]:
    """
    Run a single rollout where:
    - Each human follows their learned goal-specific policy using neural_prior.sample()
    - Robot uses RandomPolicy
    - Visualization shows first human's value function
    """
    env.reset()
    frames = []
    
    grid_width = env.width
    grid_height = env.height
    num_agents = len(env.agents)
    
    # Get first human's goal position for visualization
    first_human_goal = human_goals[first_human_idx]
    first_human_goal_pos = first_human_goal.target_pos
    
    # Robot uses RandomPolicy
    robot_policy = RandomPolicy()
    
    # Get Q-network for value visualization
    q_network = neural_prior.q_network
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Compute value function for first human across all goals (for visualization)
        value_dict = compute_value_for_goals(
            q_network, state, first_human_idx, goal_cells,
            grid_width, grid_height, num_agents, beta, device,
            world_model=env, human_agent_indices=human_agent_indices
        )
        
        # Render with overlay (showing first human's actual goal)
        frame = render_with_value_overlay(env, value_dict, first_human_goal_pos)
        frames.append(frame)
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(num_agents):
            if agent_idx in human_agent_indices:
                # Human uses learned policy via neural_prior.sample()
                goal = human_goals[agent_idx]
                action = neural_prior.sample(state, agent_idx, goal)
            else:
                # Robot uses random policy
                action = robot_policy.sample()
            actions.append(action)
        
        # Take step
        _, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Final frame
    state = env.get_state()
    value_dict = compute_value_for_goals(
        q_network, state, first_human_idx, goal_cells,
        grid_width, grid_height, num_agents, beta, device,
        world_model=env, human_agent_indices=human_agent_indices
    )
    frame = render_with_value_overlay(env, value_dict, first_human_goal_pos)
    frames.append(frame)
    
    return frames


# ============================================================================
# Movie Creation
# ============================================================================

# Configuration for quick mode vs full mode
N_ROLLOUTS = 50  # Number of rollouts to visualize (full mode)
N_ROLLOUTS_QUICK = 3  # Quick test mode
N_EPISODES_FULL = 5000  # Training episodes for full mode
N_EPISODES_QUICK = 100  # Training episodes for quick test

def create_multi_rollout_movie(
    all_rollout_frames: List[List[np.ndarray]],
    goal_positions: List[Tuple[int, int]],
    output_path: str,
    n_rollouts: int
):
    """
    Create a movie with multiple rollouts.
    
    Args:
        all_rollout_frames: List of frame lists, one per rollout
        goal_positions: List of goal positions for each rollout
        output_path: Path to save the movie
        n_rollouts: Total number of rollouts (for title display)
    """
    print(f"Creating movie with {len(all_rollout_frames)} rollouts...")
    
    frames = []
    rollout_info = []
    
    for rollout_idx, (rollout_frames, goal_pos) in enumerate(zip(all_rollout_frames, goal_positions)):
        for frame_idx, frame in enumerate(rollout_frames):
            frames.append(frame)
            rollout_info.append((rollout_idx, frame_idx, goal_pos))
    
    if len(frames) == 0:
        print("No frames to create movie!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        rollout_idx, step_idx, goal_pos = rollout_info[frame_idx]
        im.set_array(frames[frame_idx])
        title.set_text(f'Rollout {rollout_idx + 1}/{n_rollouts} | Step {step_idx} | '
                      f'Human 0 Goal: ({goal_pos[0]}, {goal_pos[1]})\n'
                      f'★ = actual goal | Colors = NN V-values for alternative goals\n'
                      f'Humans: learned Boltzmann policy | Robot: random policy')
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=500, blit=True, repeat=True
    )
    
    try:
        writer = animation.FFMpegWriter(fps=2, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✓ Movie saved to {output_path}")
    except Exception as e:
        print(f"Could not save MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=2)
            print(f"✓ Movie saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"Error saving movie: {e2}")
    
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main(quick_mode=False):
    n_rollouts = N_ROLLOUTS_QUICK if quick_mode else N_ROLLOUTS
    n_episodes = N_EPISODES_QUICK if quick_mode else N_EPISODES_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print("=" * 70)
    print("Neural Network Policy Prior Demo: 5x5 Grid")
    print(f"  [{mode_str}]")
    print("Using nn_based module for learning human policies")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    max_steps = MAX_STEPS
    print(f"Creating 5x5 empty grid environment (max_steps={max_steps})...")
    env = Empty5x5Env(max_steps=max_steps)
    env.reset()
    
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of agents: {len(env.agents)}")
    print(f"  Max steps: {env.max_steps}")
    
    # Identify agents
    human_agent_indices = []
    robot_index = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_agent_indices.append(i)
            print(f"  Human agent {i}: pos={tuple(agent.pos)}")
        elif agent.color == 'grey':
            robot_index = i
            print(f"  Robot agent {i}: pos={tuple(agent.pos)}")
    
    print()
    
    # Get all 25 goal cells
    goal_cells = []
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            cell = env.grid.get(x, y)
            if cell is None or not hasattr(cell, 'type') or cell.type != 'wall':
                goal_cells.append((x, y))
    
    print(f"Training neural network for all {len(goal_cells)} goal cells...")
    print(f"  Training episodes: {n_episodes}")
    print("  Humans learn goal-specific Boltzmann policies")
    print("  Using batch learning with replay buffer")
    print()
    
    # Create goal sampler using MultiGridGoalSampler from empo.multigrid
    # All goals are bounding boxes (x1, y1, x2, y2) - point goals are represented as (x, y, x, y)
    # The sampler uses weight-proportional sampling based on goal area
    goal_sampler = MultiGridGoalSampler(env)
    
    # Train neural network using the module's function
    # Hyperparameters tuned for this demo (5x5 grid, 25 goal cells)
    device = 'cpu'
    beta = 1000.0  # Higher temperature for more deterministic policies
    
    t0 = time.time()
    neural_prior = train_neural_policy_prior(
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=n_episodes,
        steps_per_episode=env.max_steps,  # Match env's max_steps (10)
        beta=beta,
        gamma=0.99,  # Mild discounting to encourage earlier goal reaching
        learning_rate=1e-3,
        batch_size=128,
        replay_buffer_size=10000,
        updates_per_episode=4,
        train_phi_network=False,  # We only need Q-network for this demo
        epsilon=0.3,
        exploration_policy=[0.06,0.19,0.19,0.56],  # Biased exploration: prefer forward, rarely still
        device=device,
        use_path_based_shaping=True,
        verbose=True
    )
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    print()
    
    # Select n_rollouts random goal cells for first human's rollouts
    print(f"Selecting {n_rollouts} random goal cells for first human's rollouts...")
    random.seed(42)
    selected_goals = random.sample(goal_cells, min(n_rollouts, len(goal_cells)))
    print(f"  Selected goals for human 0: {selected_goals}")
    print()
    
    # Run n_rollouts rollouts with visualization
    print(f"Running {n_rollouts} rollouts:")
    print("  - Humans follow learned goal-specific policies using neural_prior.sample()")
    print("  - Robot uses RandomPolicy")
    print("  - Visualization shows human 0's value function")
    print()
    
    all_frames = []
    first_human_idx = human_agent_indices[0]
    
    for i, goal_pos in enumerate(selected_goals):
        # Assign goals: first human gets the selected goal, second human gets random goal
        # Create ReachCellGoal objects for each human
        human_goals = {first_human_idx: ReachCellGoal(env, first_human_idx, goal_pos)}
        for h_idx in human_agent_indices:
            if h_idx != first_human_idx:
                other_goal_pos = random.choice(goal_cells)
                human_goals[h_idx] = ReachCellGoal(env, h_idx, other_goal_pos)
        
        human_goals[first_human_idx]
        other_goals_str = {h: human_goals[h].target_pos for h in human_agent_indices if h != first_human_idx}
        print(f"  Rollout {i + 1}/{n_rollouts}: Human 0 goal = {goal_pos}, Others = {other_goals_str}")
        
        frames = run_rollout_with_learned_policies(
            env=env,
            neural_prior=neural_prior,
            goal_cells=goal_cells,
            human_goals=human_goals,
            human_agent_indices=human_agent_indices,
            robot_index=robot_index,
            first_human_idx=first_human_idx,
            beta=beta,
            device=device
        )
        all_frames.append(frames)
        print(f"    Captured {len(frames)} frames")
    
    print()
    
    # Create movie
    movie_path = os.path.join(output_dir, 'neural_policy_prior_demo.mp4')
    create_multi_rollout_movie(all_frames, selected_goals, movie_path, n_rollouts=n_rollouts)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Neural Policy Prior Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes and rollouts')
    args = parser.parse_args()
    main(quick_mode=args.quick)
