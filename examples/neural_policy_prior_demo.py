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

import sys
import os
import time
import random

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from typing import Iterator, Tuple, Dict, List, Any, Optional

from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Wall, World, SmallActions
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler
from empo.nn_based import (
    StateEncoder, AgentEncoder, GoalEncoder,
    QNetwork, PolicyPriorNetwork,
    train_neural_policy_prior,
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
# Goal Definitions
# ============================================================================

class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = tuple(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            pos_x, pos_y = agent_state[0], agent_state[1]
            if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell({self.target_pos[0]},{self.target_pos[1]})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and 
                self.target_pos == other.target_pos)


class ReachCellGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that samples uniformly from a list of target cell positions.
    
    This sampler creates ReachCellGoal instances for randomly selected cells.
    """
    
    def __init__(self, world_model, goal_cells: List[Tuple[int, int]]):
        """
        Initialize the goal sampler.
        
        Args:
            world_model: The environment.
            goal_cells: List of (x, y) tuples representing possible goal positions.
        """
        super().__init__(world_model)
        self.goal_cells = goal_cells
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """
        Sample a random goal cell uniformly.
        
        Args:
            state: Current world state.
            human_agent_index: Index of the human agent.
        
        Returns:
            Tuple of (ReachCellGoal, weight=1.0).
        """
        target_pos = random.choice(self.goal_cells)
        goal = ReachCellGoal(self.world_model, human_agent_index, target_pos)
        return goal, 1.0


# ============================================================================
# Helper functions for the demo
# ============================================================================

def state_to_grid_tensor(
    state, 
    grid_width: int, 
    grid_height: int,
    num_agents: int,
    num_object_types: int = 8,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a multigrid state to tensor representation for the neural network.
    """
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    num_channels = num_object_types + num_agents
    grid_tensor = torch.zeros(1, num_channels, grid_height, grid_width, device=device)
    
    # Encode agent positions
    for i, agent_state in enumerate(agent_states):
        if i < num_agents:
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                channel_idx = num_object_types + i
                grid_tensor[0, channel_idx, y, x] = 1.0
    
    # Normalize step count (max_steps should match environment setting)
    max_steps = MAX_STEPS  # Must match env.max_steps for proper normalization
    step_tensor = torch.tensor([[step_count / max_steps]], device=device, dtype=torch.float32)
    
    return grid_tensor, step_tensor


def get_agent_tensors(
    state,
    human_idx: int,
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract agent position, direction, and index tensors from state."""
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
    device: str = 'cpu'
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
    
    grid_tensor, step_tensor = state_to_grid_tensor(
        state, grid_width, grid_height, num_agents, device=device
    )
    position, direction, agent_idx_t = get_agent_tensors(
        state, human_idx, grid_width, grid_height, device
    )
    
    with torch.no_grad():
        for goal_pos in goal_cells:
            # If agent is already at this goal, value is 1.0
            if agent_x == goal_pos[0] and agent_y == goal_pos[1]:
                values[goal_pos] = 1.0
                continue
            
            goal_coords = get_goal_tensor(goal_pos, grid_width, grid_height, device)
            
            q_values = q_network(
                grid_tensor, step_tensor,
                position, direction, agent_idx_t,
                goal_coords
            )
            
            # V = E_π[Q] where π = softmax(β*Q)
            policy = F.softmax(beta * q_values, dim=1)
            v_value = (policy * q_values).sum().item()
            values[goal_pos] = v_value
    
    return values


def get_boltzmann_action(
    q_network: QNetwork,
    state,
    human_idx: int,
    goal_pos: Tuple[int, int],
    grid_width: int,
    grid_height: int,
    num_agents: int,
    beta: float = 5.0,
    device: str = 'cpu'
) -> int:
    """
    Sample an action from the learned Boltzmann policy.
    """
    grid_tensor, step_tensor = state_to_grid_tensor(
        state, grid_width, grid_height, num_agents, device=device
    )
    position, direction, agent_idx_t = get_agent_tensors(
        state, human_idx, grid_width, grid_height, device
    )
    goal_coords = get_goal_tensor(goal_pos, grid_width, grid_height, device)
    
    with torch.no_grad():
        q_values = q_network(
            grid_tensor, step_tensor,
            position, direction, agent_idx_t,
            goal_coords
        ) # shape: (1, num_actions)
        if beta == float('inf'):
            # Greedy action
            action = torch.argmax(q_values, dim=1).item()
        else:
            q_values -= torch.max(q_values, dim=1, keepdim=True)  # For numerical stability
            policy = F.softmax(beta * q_values, dim=1)
            action = torch.multinomial(policy, 1).item()
    
    return action


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
    Render the environment with value function overlay.
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
    
    # Mark actual goal with a star
    if actual_goal:
        gx = actual_goal[0] * tile_size + tile_size // 2
        gy = actual_goal[1] * tile_size + tile_size // 2
        ax.plot(gx, gy, marker='*', markersize=20, color='blue', 
                markeredgecolor='white', markeredgewidth=2)
    
    # Mark first human (agent index 0) with a cyan border/ring
    # This helps identify which human's value function is being visualized
    step_count, agent_states, _, _ = env.get_state()
    if len(agent_states) > 0:
        first_human_pos = agent_states[0]
        hx = int(first_human_pos[0]) * tile_size + tile_size // 2
        hy = int(first_human_pos[1]) * tile_size + tile_size // 2
        # Draw a cyan ring around the first human
        ring = plt.Circle((hx, hy), tile_size * 0.45, fill=False, 
                          color='cyan', linewidth=3)
        ax.add_patch(ring)
        # Also add "H1" label
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
    q_network: QNetwork,
    goal_cells: List[Tuple[int, int]],
    human_goals: Dict[int, Tuple[int, int]],  # goal for each human
    human_agent_indices: List[int],
    robot_index: int,
    first_human_idx: int,  # which human's value function to visualize
    beta: float = 5.0,
    device: str = 'cpu'
) -> List[np.ndarray]:
    """
    Run a single rollout where:
    - Each human follows their learned goal-specific Boltzmann policy
    - Robot uses random policy
    - Visualization shows first human's value function
    """
    env.reset()
    frames = []
    
    grid_width = env.width
    grid_height = env.height
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    
    first_human_goal = human_goals[first_human_idx]
    
    for step in range(env.max_steps):
        state = env.get_state()
        
        # Compute value function for first human across all goals
        value_dict = compute_value_for_goals(
            q_network, state, first_human_idx, goal_cells,
            grid_width, grid_height, num_agents, beta, device
        )
        
        # Render with overlay (showing first human's actual goal)
        frame = render_with_value_overlay(env, value_dict, first_human_goal)
        frames.append(frame)
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(num_agents):
            if agent_idx in human_agent_indices:
                # Human uses learned Boltzmann policy
                goal_pos = human_goals[agent_idx]
                action = get_boltzmann_action(
                    q_network, state, agent_idx, goal_pos,
                    grid_width, grid_height, num_agents, beta, device
                )
            else:
                # Robot uses random policy
                action = random.randint(0, num_actions - 1)
            actions.append(action)
        
        # Take step
        _, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Final frame
    state = env.get_state()
    value_dict = compute_value_for_goals(
        q_network, state, first_human_idx, goal_cells,
        grid_width, grid_height, num_agents, beta, device
    )
    frame = render_with_value_overlay(env, value_dict, first_human_goal)
    frames.append(frame)
    
    return frames


# ============================================================================
# Movie Creation
# ============================================================================

N_ROLLOUTS = 50  # Number of rollouts to visualize

def create_multi_rollout_movie(
    all_rollout_frames: List[List[np.ndarray]],
    goal_positions: List[Tuple[int, int]],
    output_path: str
):
    """Create a movie with multiple rollouts."""
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
        title.set_text(f'Rollout {rollout_idx + 1}/{N_ROLLOUTS} | Step {step_idx} | '
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

def main():
    print("=" * 70)
    print("Neural Network Policy Prior Demo: 5x5 Grid")
    print("Using nn_based module for learning human policies")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
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
    print("  Humans learn goal-specific Boltzmann policies")
    print("  Using batch learning with replay buffer")
    print()
    
    # Create goal sampler for the training function
    goal_sampler = ReachCellGoalSampler(env, goal_cells)
    
    # Train neural network using the module's function
    # Hyperparameters tuned for this demo (5x5 grid, 25 goal cells)
    device = 'cpu'
    beta = np.inf  # Higher temperature for more deterministic policies
    
    t0 = time.time()
    neural_prior = train_neural_policy_prior(
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=5000,
        steps_per_episode=env.max_steps,  # Match env's max_steps (10)
        beta=beta,
        gamma=0.99,  # some incentive to reach goals quickly
        learning_rate=1e-3,
        batch_size=64,
        replay_buffer_size=10000,
        updates_per_episode=4,
        train_phi_network=False,  # We only need Q-network for this demo
        epsilon=0.3,
        device=device,
        verbose=True
    )
    # Extract the Q-network from the trained policy prior
    q_network = neural_prior.q_network
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    print()
    
    # Select N_ROLLOUTS random goal cells for first human's rollouts
    print(f"Selecting {N_ROLLOUTS} random goal cells for first human's rollouts...")
    random.seed(42)
    selected_goals = random.sample(goal_cells, min(N_ROLLOUTS, len(goal_cells)))
    print(f"  Selected goals for human 0: {selected_goals}")
    print()
    
    # Run N_ROLLOUTS rollouts with visualization
    print(f"Running {N_ROLLOUTS} rollouts:")
    print("  - Humans follow learned goal-specific Boltzmann policies")
    print("  - Robot uses random policy")
    print("  - Visualization shows human 0's value function")
    print()
    
    all_frames = []
    first_human_idx = human_agent_indices[0]
    
    for i, goal_pos in enumerate(selected_goals):
        # Assign goals: first human gets the selected goal, second human gets random goal
        human_goals = {first_human_idx: goal_pos}
        for h_idx in human_agent_indices:
            if h_idx != first_human_idx:
                human_goals[h_idx] = random.choice(goal_cells)
        
        print(f"  Rollout {i + 1}/{N_ROLLOUTS}: Human 0 goal = {goal_pos}, Human 1 goal = {human_goals.get(human_agent_indices[1], 'N/A')}")
        
        frames = run_rollout_with_learned_policies(
            env=env,
            q_network=q_network,
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
    create_multi_rollout_movie(all_frames, selected_goals, movie_path)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
