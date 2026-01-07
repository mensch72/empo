#!/usr/bin/env python3
"""
Transport Environment Learning Demo.

This script demonstrates neural network-based policy learning for a minimal
transport environment:
- 5 nodes forming a small road network
- 1 human agent (learns to walk to goal)
- 0 vehicles
- Goal: human reaches a target node

The demo uses the nn_based.transport module to:
1. Create a small transport environment
2. Train neural network (Q-network) to learn human policy for reaching goals
3. Run rollouts showing the learned policy

This is a minimal example to verify that the learning infrastructure works.
No vehicles or complex routing - just a human learning to navigate to a goal.
"""

import os
import time
import random
import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from empo.world_specific_helpers.transport import (
    TransportEnvWrapper,
    TransportGoal,
    TransportGoalSampler,
)
from empo.learning_based.transport import (
    train_transport_neural_policy_prior,
)


# ============================================================================
# Configuration
# ============================================================================

# Quick vs full mode
N_EPISODES_QUICK = 50
N_EPISODES_FULL = 500
N_ROLLOUTS_QUICK = 3
N_ROLLOUTS_FULL = 10
MAX_STEPS = 20


# ============================================================================
# Create minimal network
# ============================================================================

def create_simple_5node_network():
    """
    Create a simple 5-node network for testing.
    
    Layout:
        0 --- 1 --- 2
        |     |     |
        3 --- 4 ----+
        
    Node 0 is at (0, 0)
    Node 1 is at (1, 0)  
    Node 2 is at (2, 0)
    Node 3 is at (0, 1)
    Node 4 is at (1, 1)
    
    All edges are bidirectional.
    """
    G = nx.DiGraph()
    
    # Add nodes with positions and names
    # Use 'x' and 'y' attributes for the vendored renderer
    positions = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (2.0, 0.0),
        3: (0.0, 1.0),
        4: (1.0, 1.0),
    }
    
    for node, pos in positions.items():
        G.add_node(node, x=pos[0], y=pos[1], name=f"node_{node}")
    
    # Add bidirectional edges with lengths
    edges = [
        (0, 1), (1, 0),  # 0-1
        (1, 2), (2, 1),  # 1-2
        (0, 3), (3, 0),  # 0-3
        (1, 4), (4, 1),  # 1-4
        (3, 4), (4, 3),  # 3-4
        (2, 4), (4, 2),  # 2-4 (diagonal)
    ]
    
    for u, v in edges:
        pos_u = positions[u]
        pos_v = positions[v]
        length = ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)**0.5
        G.add_edge(u, v, length=length, speed=50.0, capacity=10)
    
    return G


# ============================================================================
# Visualization
# ============================================================================

def render_network_state(env, agent_name='human_0', goal_node=None, value_dict=None, title=""):
    """
    Render the network state using the vendored ai_transport rendering.
    
    This method uses the improved rendering in the ai_transport environment which:
    - Shows agents moving continuously along roads
    - Shows bidirectional roads as separate lanes
    - Shows vehicles as blue rectangles with passengers
    - Shows humans as red dots
    - Highlights goal with dashed outline and star marker
    - Draws dashed line from agent to goal (like multigrid)
    
    Args:
        env: TransportEnvWrapper
        agent_name: Name of agent with the goal
        goal_node: Target node (highlighted)
        value_dict: Optional dict mapping nodes to V-values
        title: Plot title
    
    Returns:
        RGB array of the rendered image
    """
    # Use the vendored render_frame method with goal info
    goal_info = None
    if goal_node is not None:
        goal_info = {
            'agent': agent_name,
            'node': goal_node,
            'type': 'node'
        }
    
    # Render using the underlying env's render_frame method
    frame = env.env.render_frame(
        goal_info=goal_info,
        value_dict=value_dict,
        title=title
    )
    
    return frame


def compute_value_for_nodes(q_network, env, agent_idx, goal_node, device='cpu'):
    """
    Compute V-values for all nodes using the Q-network.
    
    V(s) = Σ_a π(a|s,g) * Q(s,a,g) where π = softmax(β*Q)
    """
    values = {}
    network = env.env.network
    
    # Create goal object
    goal = TransportGoal(env, agent_idx, goal_node)
    
    # Save current agent position
    agent_name = env.agents[agent_idx]
    original_pos = env.env.agent_positions.get(agent_name)
    
    with torch.no_grad():
        for node in network.nodes():
            # Temporarily move agent to this node to compute value
            env.env.agent_positions[agent_name] = node
            
            # If agent is at goal, value is 1.0
            if node == goal_node:
                values[node] = 1.0
                continue
            
            # Get Q-values
            q_values = q_network.forward(
                None, env, agent_idx, goal, device
            )
            
            # V = E_π[Q]
            probs = q_network.get_policy(q_values)
            v_value = (probs * q_values).sum().item()
            values[node] = v_value
    
    # Restore original position
    env.env.agent_positions[agent_name] = original_pos
    
    return values


# ============================================================================
# Rollout
# ============================================================================

def run_rollout(env, neural_prior, goal_node, agent_idx=0, beta=5.0, device='cpu'):
    """
    Run a single rollout with the learned policy.
    
    Returns list of frames and whether goal was reached.
    """
    env.reset()
    frames = []
    goal = TransportGoal(env, agent_idx, goal_node)
    agent_name = env.agents[agent_idx]
    
    for step in range(MAX_STEPS):
        # Compute values for visualization
        value_dict = compute_value_for_nodes(neural_prior.q_network, env, agent_idx, goal_node, device)
        
        # Check if at goal
        agent_pos = env.env.agent_positions.get(agent_name)
        at_goal = agent_pos == goal_node
        
        title = f"Step {step} | Goal: node {goal_node}"
        if at_goal:
            title += " | GOAL REACHED!"
        
        frame = render_network_state(env, agent_name=agent_name, goal_node=goal_node, value_dict=value_dict, title=title)
        frames.append(frame)
        
        if at_goal:
            break
        
        # Sample action from the neural policy prior
        action = neural_prior.sample(None, agent_idx, goal, apply_action_mask=True, beta=beta)
        
        # Execute action for all agents (only 1 human, no vehicles)
        actions = [action]  # Only one agent
        env.step(actions)
    
    # Final frame
    value_dict = compute_value_for_nodes(neural_prior.q_network, env, agent_idx, goal_node, device)
    agent_pos = env.env.agent_positions.get(agent_name)
    at_goal = agent_pos == goal_node
    title = f"Step {MAX_STEPS} | Goal: node {goal_node}"
    if at_goal:
        title += " | GOAL REACHED!"
    frame = render_network_state(env, agent_name=agent_name, goal_node=goal_node, value_dict=value_dict, title=title)
    frames.append(frame)
    
    return frames, at_goal


# ============================================================================
# Movie Creation
# ============================================================================

def create_movie(all_frames, goal_nodes, output_path, n_rollouts):
    """Create animation from rollout frames."""
    print(f"Creating movie with {len(all_frames)} rollouts...")
    
    frames = []
    frame_info = []
    
    for rollout_idx, (rollout_frames, goal_node) in enumerate(zip(all_frames, goal_nodes)):
        for frame_idx, frame in enumerate(rollout_frames):
            frames.append(frame)
            frame_info.append((rollout_idx, frame_idx, goal_node))
    
    if len(frames) == 0:
        print("No frames to create movie!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        rollout_idx, step_idx, goal_node = frame_info[frame_idx]
        im.set_array(frames[frame_idx])
        title.set_text(f'Rollout {rollout_idx + 1}/{n_rollouts} | Step {step_idx} | '
                      f'Goal: node {goal_node}\n'
                      f'Colors = learned V-values | ★ = goal | H = human')
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
    n_episodes = N_EPISODES_QUICK if quick_mode else N_EPISODES_FULL
    n_rollouts = N_ROLLOUTS_QUICK if quick_mode else N_ROLLOUTS_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print("=" * 70)
    print("Transport Environment Learning Demo")
    print(f"  [{mode_str}]")
    print("Minimal example: 5 nodes, 1 human, no vehicles")
    print("Goal: Human learns to walk to target node")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom network
    print("Creating simple 5-node network...")
    network = create_simple_5node_network()
    print(f"  Nodes: {list(network.nodes())}")
    print(f"  Edges: {list(network.edges())}")
    print()
    
    # Create environment with custom network
    print("Creating transport environment...")
    
    env = TransportEnvWrapper(
        num_humans=1,
        num_vehicles=0,
        network=network,
        num_clusters=0,
        render_mode=None,
    )
    env.reset(seed=42)
    
    print(f"  Agents: {env.agents}")
    print(f"  Number of nodes: {len(env.env.network.nodes())}")
    print(f"  Max steps: {MAX_STEPS}")
    print()
    
    # Get goal nodes (all 5 nodes)
    goal_nodes = list(env.env.network.nodes())
    print(f"Possible goal nodes: {goal_nodes}")
    print()
    
    # Create goal sampler
    goal_sampler = TransportGoalSampler(env, seed=42)
    human_agent_indices = [0]  # Only one human
    
    # Train neural network
    print(f"Training neural network for {n_episodes} episodes...")
    print("  Learning: Q-values for reaching each goal node")
    print("  Using: GNN-based state encoding")
    print()
    
    device = 'cpu'
    beta = 5.0
    
    t0 = time.time()
    neural_prior = train_transport_neural_policy_prior(
        env=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=n_episodes,
        steps_per_episode=MAX_STEPS,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.99,
        beta=beta,
        buffer_capacity=10000,
        target_update_freq=50,
        state_feature_dim=64,
        goal_feature_dim=16,
        hidden_dim=64,
        num_gnn_layers=2,
        gnn_type='gcn',
        device=device,
        verbose=True,
        reward_shaping=False,  # Use base reward only, Q-values bounded in [0, 1]
        epsilon=0.3,
        updates_per_episode=2,
        max_nodes=10,
        num_clusters=0,
    )
    
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    print()
    
    # Select goal nodes for rollouts
    print(f"Running {n_rollouts} rollouts with learned policy...")
    random.seed(42)
    selected_goals = random.choices(goal_nodes, k=n_rollouts)
    print(f"  Selected goals: {selected_goals}")
    print()
    
    all_frames = []
    successes = 0
    
    for i, goal_node in enumerate(selected_goals):
        print(f"  Rollout {i + 1}/{n_rollouts}: Goal = node {goal_node}")
        
        frames, reached = run_rollout(
            env=env,
            neural_prior=neural_prior,
            goal_node=goal_node,
            agent_idx=0,
            beta=beta,
            device=device
        )
        all_frames.append(frames)
        
        if reached:
            successes += 1
            print(f"    ✓ Goal reached in {len(frames) - 1} steps")
        else:
            print(f"    ✗ Goal not reached")
    
    print()
    print(f"Success rate: {successes}/{n_rollouts} ({100*successes/n_rollouts:.1f}%)")
    print()
    
    # Create movie
    movie_path = os.path.join(output_dir, 'transport_learning_demo.mp4')
    create_movie(all_frames, selected_goals, movie_path, n_rollouts=n_rollouts)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transport Learning Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    args = parser.parse_args()
    main(quick_mode=args.quick)
