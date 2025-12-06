#!/usr/bin/env python3
"""
Transport Two-Cluster Demo with Human Learning and Fixed Vehicle Policy.

This script demonstrates a transport environment with:
- 100 nodes separated into two clusters of 50 nodes each
- Only one road connecting the two clusters (bottleneck)
- 2 human agents learning to reach individual goal nodes
- 1 vehicle agent with a fixed policy: moves back and forth between cluster centroids
- Vehicle announces cluster destinations (not nodes)
- Vehicle is 10x faster than humans
- Humans can learn to combine: walk to centroid → ride vehicle → walk to goal

The interesting question is whether humans learn the optimal strategy of:
1. Walking to the nearest cluster centroid
2. Boarding the vehicle
3. Traveling to the other cluster
4. Unboarding
5. Walking to their goal node

Usage:
    python transport_two_cluster_demo.py           # Full run
    python transport_two_cluster_demo.py --quick   # Quick test run
"""

import sys
import os
import time
import random
import argparse

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'ai_transport'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from empo.transport import (
    TransportEnvWrapper,
    TransportGoal,
    TransportGoalSampler,
    TransportActions,
)
from empo.nn_based.transport import (
    TransportNeuralHumanPolicyPrior,
    train_transport_neural_policy_prior,
)


# ============================================================================
# Configuration
# ============================================================================

# Training configuration
N_EPISODES_QUICK = 10  # Very reduced for fast testing
N_EPISODES_FULL = 500
N_ROLLOUTS_QUICK = 2
N_ROLLOUTS_FULL = 10
MAX_STEPS_QUICK = 30  # Reduced for faster episodes  
MAX_STEPS_FULL = 100

# Network configuration - for quick mode, use smaller network
NUM_NODES_PER_CLUSTER_QUICK = 5  # Very small for testing
NUM_NODES_PER_CLUSTER_FULL = 50
NUM_CLUSTERS = 2

# Agent speeds
HUMAN_SPEED = 1.0
VEHICLE_SPEED = 10.0  # 10x faster than humans

# Gamma for human learning - needs to be high enough to incentivize arriving earlier
# but not so high that learning is unstable
HUMAN_GAMMA = 0.95  # Balanced discount factor


# ============================================================================
# Network Generation
# ============================================================================

def create_two_cluster_network(
    nodes_per_cluster: int = 50,
    cluster_separation: float = 50.0,
    cluster_radius: float = 10.0,
    seed: int = 42
) -> nx.DiGraph:
    """
    Create a network with two clusters connected by a single road.
    
    Each cluster is arranged in a roughly circular pattern. There is only
    one edge connecting the two clusters, creating a bottleneck.
    
    Args:
        nodes_per_cluster: Number of nodes in each cluster
        cluster_separation: Distance between cluster centers
        cluster_radius: Radius of each cluster
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX DiGraph with node positions and edge attributes
    """
    np.random.seed(seed)
    G = nx.DiGraph()
    
    # Cluster 0 centered at (0, 0)
    # Cluster 1 centered at (cluster_separation, 0)
    cluster_centers = [
        (0.0, 0.0),
        (cluster_separation, 0.0)
    ]
    
    node_id = 0
    cluster_nodes = [[], []]
    
    # Generate nodes for each cluster
    for cluster_idx in range(2):
        center_x, center_y = cluster_centers[cluster_idx]
        
        for i in range(nodes_per_cluster):
            # Random position in circular cluster
            angle = 2 * np.pi * i / nodes_per_cluster + np.random.uniform(-0.1, 0.1)
            radius = cluster_radius * (0.5 + 0.5 * np.random.random())
            
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            G.add_node(node_id, x=x, y=y, name=f"node_{node_id}", cluster=cluster_idx)
            cluster_nodes[cluster_idx].append(node_id)
            node_id += 1
    
    # Add edges within each cluster
    # Connect each node to its nearest neighbors
    for cluster_idx in range(2):
        nodes = cluster_nodes[cluster_idx]
        center_x, center_y = cluster_centers[cluster_idx]
        
        for node in nodes:
            node_x = G.nodes[node]['x']
            node_y = G.nodes[node]['y']
            
            # Find k nearest neighbors in same cluster
            distances = []
            for other in nodes:
                if other == node:
                    continue
                other_x = G.nodes[other]['x']
                other_y = G.nodes[other]['y']
                dist = np.sqrt((node_x - other_x)**2 + (node_y - other_y)**2)
                distances.append((dist, other))
            
            # Connect to 3-5 nearest neighbors
            distances.sort()
            k_neighbors = min(5, len(distances))
            for dist, neighbor in distances[:k_neighbors]:
                # Add bidirectional edges
                if not G.has_edge(node, neighbor):
                    G.add_edge(node, neighbor, length=dist, speed=VEHICLE_SPEED, capacity=10)
                if not G.has_edge(neighbor, node):
                    G.add_edge(neighbor, node, length=dist, speed=VEHICLE_SPEED, capacity=10)
    
    # Add single connecting edge between clusters
    # Find the two nodes closest to the midpoint between clusters
    midpoint_x = cluster_separation / 2.0
    midpoint_y = 0.0
    
    # Find closest node in each cluster to the midpoint
    cluster_0_bridge = None
    cluster_1_bridge = None
    min_dist_0 = float('inf')
    min_dist_1 = float('inf')
    
    for node in cluster_nodes[0]:
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        dist = np.sqrt((x - midpoint_x)**2 + (y - midpoint_y)**2)
        if dist < min_dist_0:
            min_dist_0 = dist
            cluster_0_bridge = node
    
    for node in cluster_nodes[1]:
        x = G.nodes[node]['x']
        y = G.nodes[node]['y']
        dist = np.sqrt((x - midpoint_x)**2 + (y - midpoint_y)**2)
        if dist < min_dist_1:
            min_dist_1 = dist
            cluster_1_bridge = node
    
    # Add the single bridge edge (bidirectional)
    bridge_length = np.sqrt(
        (G.nodes[cluster_0_bridge]['x'] - G.nodes[cluster_1_bridge]['x'])**2 +
        (G.nodes[cluster_0_bridge]['y'] - G.nodes[cluster_1_bridge]['y'])**2
    )
    G.add_edge(cluster_0_bridge, cluster_1_bridge, 
               length=bridge_length, speed=VEHICLE_SPEED, capacity=10, is_bridge=True)
    G.add_edge(cluster_1_bridge, cluster_0_bridge, 
               length=bridge_length, speed=VEHICLE_SPEED, capacity=10, is_bridge=True)
    
    # Mark bridge nodes
    G.nodes[cluster_0_bridge]['is_bridge'] = True
    G.nodes[cluster_1_bridge]['is_bridge'] = True
    
    return G


def find_cluster_centroids(G: nx.DiGraph) -> dict:
    """
    Find the centroid node for each cluster.
    
    The centroid is defined as the node closest to the geometric center
    of all nodes in the cluster.
    
    Returns:
        Dictionary mapping cluster_idx to centroid node_id
    """
    centroids = {}
    
    for cluster_idx in range(2):
        cluster_nodes = [n for n in G.nodes() if G.nodes[n].get('cluster') == cluster_idx]
        
        if not cluster_nodes:
            continue
        
        # Compute geometric center
        center_x = np.mean([G.nodes[n]['x'] for n in cluster_nodes])
        center_y = np.mean([G.nodes[n]['y'] for n in cluster_nodes])
        
        # Find closest node to center
        min_dist = float('inf')
        centroid = None
        for node in cluster_nodes:
            x = G.nodes[node]['x']
            y = G.nodes[node]['y']
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist < min_dist:
                min_dist = dist
                centroid = node
        
        centroids[cluster_idx] = centroid
    
    return centroids


# ============================================================================
# Fixed Vehicle Policy
# ============================================================================

class CentroidShuttlePolicy:
    """
    Fixed policy for vehicle that shuttles between cluster centroids.
    
    The vehicle:
    1. Announces its destination cluster (alternating between clusters)
    2. Moves to the centroid of that cluster
    3. Once at the centroid, announces the other cluster
    4. Repeats indefinitely
    
    This policy does not learn - it's a fixed heuristic to test if humans
    can learn to use the vehicle effectively.
    """
    
    def __init__(self, env, centroids: dict, vehicle_agent_idx: int, seed: int = 42):
        """
        Initialize the centroid shuttle policy.
        
        Args:
            env: TransportEnvWrapper environment
            centroids: Dict mapping cluster_idx to centroid node_id
            vehicle_agent_idx: Index of the vehicle agent
            seed: Random seed (unused, kept for compatibility)
        """
        self.env = env
        self.centroids = centroids
        self.vehicle_agent_idx = vehicle_agent_idx
        self.vehicle_agent_name = env.agents[vehicle_agent_idx]
        
        # Start with destination cluster 1 (go to second cluster first)
        self.current_destination_cluster = 1
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, obs) -> int:
        """
        Get the vehicle's action based on current state.
        
        Returns:
            Action index from TransportActions
        """
        step_type = self.env.step_type
        vehicle_pos = self.env.env.agent_positions.get(self.vehicle_agent_name)
        
        if step_type == 'routing':
            # Announce destination cluster
            # Use DEST_START + 1 + cluster_idx for cluster-based routing
            action = TransportActions.DEST_START + 1 + self.current_destination_cluster
            return action
        
        elif step_type == 'departing':
            # Check if at target centroid
            target_centroid = self.centroids[self.current_destination_cluster]
            
            if vehicle_pos == target_centroid:
                # Reached target centroid, switch destination
                self.current_destination_cluster = 1 - self.current_destination_cluster
                # Stay at centroid this step
                return TransportActions.PASS
            
            # Otherwise, find shortest path to target centroid
            # Use simple greedy: take edge that reduces distance to target
            if vehicle_pos is None or isinstance(vehicle_pos, tuple):
                # On edge or invalid position, just pass
                return TransportActions.PASS
            
            # Get outgoing edges
            outgoing_edges = list(self.env.network.out_edges(vehicle_pos))
            if not outgoing_edges:
                return TransportActions.PASS
            
            # Find edge that brings us closest to target
            target_x = self.env.network.nodes[target_centroid]['x']
            target_y = self.env.network.nodes[target_centroid]['y']
            
            best_edge_idx = 0
            min_dist = float('inf')
            
            for idx, (u, v) in enumerate(outgoing_edges):
                v_x = self.env.network.nodes[v]['x']
                v_y = self.env.network.nodes[v]['y']
                dist = np.sqrt((v_x - target_x)**2 + (v_y - target_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_edge_idx = idx
            
            return TransportActions.DEPART_START + best_edge_idx
        
        else:
            # For boarding/unboarding, just pass (vehicle doesn't board)
            return TransportActions.PASS


# ============================================================================
# Visualization
# ============================================================================

def render_network_state(env, human_agent_indices, vehicle_agent_idx, goal_nodes, 
                         centroids, value_dict=None, title=""):
    """
    Render the network state with humans, vehicle, goals, and centroids.
    
    Args:
        env: TransportEnvWrapper
        human_agent_indices: List of human agent indices
        vehicle_agent_idx: Vehicle agent index
        goal_nodes: Dict mapping human agent index to goal node
        centroids: Dict mapping cluster_idx to centroid node_id
        value_dict: Optional dict mapping (agent_idx, node) to V-value
        title: Plot title
    
    Returns:
        RGB array of the rendered image
    """
    # Use the underlying env's rendering if available
    # For now, create a simple visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    G = env.network
    pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
    
    # Draw edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        is_bridge = G.edges[u, v].get('is_bridge', False)
        if is_bridge:
            edge_colors.append('red')
            edge_widths.append(3.0)
        else:
            edge_colors.append('lightgray')
            edge_widths.append(0.5)
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                          width=edge_widths, arrows=False, alpha=0.5)
    
    # Draw nodes colored by cluster
    cluster_0_nodes = [n for n in G.nodes() if G.nodes[n].get('cluster') == 0]
    cluster_1_nodes = [n for n in G.nodes() if G.nodes[n].get('cluster') == 1]
    
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_0_nodes, 
                          node_color='lightblue', node_size=20, ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=cluster_1_nodes, 
                          node_color='lightgreen', node_size=20, ax=ax, alpha=0.6)
    
    # Highlight centroids with stars
    centroid_nodes = list(centroids.values())
    nx.draw_networkx_nodes(G, pos, nodelist=centroid_nodes,
                          node_color='gold', node_size=300, node_shape='*', ax=ax)
    
    # Draw agents
    for agent_idx in human_agent_indices:
        agent_name = env.agents[agent_idx]
        agent_pos = env.env.agent_positions.get(agent_name)
        
        # Check if human is aboard vehicle
        aboard = env.env.human_aboard.get(agent_name)
        
        if aboard is None and agent_pos is not None and not isinstance(agent_pos, tuple):
            # Human is walking at a node
            x, y = pos[agent_pos]
            ax.plot(x, y, 'ro', markersize=12, label=f'Human {agent_idx}')
            
            # Draw line to goal
            goal_node = goal_nodes.get(agent_idx)
            if goal_node is not None:
                goal_x, goal_y = pos[goal_node]
                ax.plot([x, goal_x], [y, goal_y], 'r--', alpha=0.3, linewidth=1)
    
    # Draw vehicle
    vehicle_name = env.agents[vehicle_agent_idx]
    vehicle_pos = env.env.agent_positions.get(vehicle_name)
    if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
        x, y = pos[vehicle_pos]
        ax.plot(x, y, 'bs', markersize=16, label='Vehicle')
        
        # Show which humans are aboard
        aboard_humans = [i for i in human_agent_indices 
                        if env.env.human_aboard.get(env.agents[i]) == vehicle_name]
        if aboard_humans:
            ax.text(x, y + 2, f'Aboard: {aboard_humans}', ha='center', fontsize=8)
    
    # Draw goals with stars
    for agent_idx in human_agent_indices:
        goal_node = goal_nodes.get(agent_idx)
        if goal_node is not None:
            x, y = pos[goal_node]
            ax.plot(x, y, 'r*', markersize=20, alpha=0.7)
            ax.text(x, y - 2, f'Goal H{agent_idx}', ha='center', fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are handles
        ax.legend(loc='upper right')
    
    # Convert to RGB array
    fig.canvas.draw()
    # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape((height, width, 4))
    img = img[:, :, :3]  # Remove alpha channel
    plt.close(fig)
    
    return img


# ============================================================================
# Rollout
# ============================================================================

def run_rollout(env, neural_prior, vehicle_policy, goal_nodes, 
                human_agent_indices, vehicle_agent_idx, centroids, 
                max_steps=100, beta=5.0, device='cpu'):
    """
    Run a single rollout with learned human policy and fixed vehicle policy.
    
    Returns list of frames and success status.
    """
    env.reset()
    frames = []
    
    # Create goals for humans
    human_goals = {}
    for agent_idx in human_agent_indices:
        goal_node = goal_nodes[agent_idx]
        human_goals[agent_idx] = TransportGoal(env, agent_idx, goal_node)
    
    for step in range(max_steps):
        # Check if all humans reached goals
        all_reached = True
        for agent_idx in human_agent_indices:
            agent_name = env.agents[agent_idx]
            agent_pos = env.env.agent_positions.get(agent_name)
            goal_node = goal_nodes[agent_idx]
            if agent_pos != goal_node:
                all_reached = False
                break
        
        # Render frame
        title = f"Step {step} | Vehicle shuttles between centroids"
        if all_reached:
            title += " | ALL GOALS REACHED!"
        
        frame = render_network_state(
            env, human_agent_indices, vehicle_agent_idx, 
            goal_nodes, centroids, value_dict=None, title=title
        )
        frames.append(frame)
        
        if all_reached:
            break
        
        # Get actions for all agents
        actions = []
        for agent_idx in range(env.num_agents):
            if agent_idx in human_agent_indices:
                # Human: use learned policy
                goal = human_goals[agent_idx]
                action = neural_prior.sample(None, agent_idx, goal, 
                                            apply_action_mask=True, beta=beta)
            elif agent_idx == vehicle_agent_idx:
                # Vehicle: use fixed shuttle policy
                action = vehicle_policy.get_action(None)
            else:
                # Should not happen
                action = TransportActions.PASS
            
            actions.append(action)
        
        # Execute actions
        obs, rewards, done, info = env.step(actions)
        
        if done:
            break
    
    # Final frame
    all_reached = all([env.env.agent_positions.get(env.agents[i]) == goal_nodes[i]
                       for i in human_agent_indices])
    title = f"Step {step} | Vehicle shuttles between centroids"
    if all_reached:
        title += " | ALL GOALS REACHED!"
    frame = render_network_state(
        env, human_agent_indices, vehicle_agent_idx,
        goal_nodes, centroids, value_dict=None, title=title
    )
    frames.append(frame)
    
    return frames, all_reached


# ============================================================================
# Movie Creation
# ============================================================================

def create_movie(all_frames, output_path, n_rollouts):
    """Create animation from rollout frames."""
    print(f"Creating movie with {len(all_frames)} rollouts...")
    
    frames = []
    frame_info = []
    
    for rollout_idx, rollout_frames in enumerate(all_frames):
        for frame_idx, frame in enumerate(rollout_frames):
            frames.append(frame)
            frame_info.append((rollout_idx, frame_idx))
    
    if len(frames) == 0:
        print("No frames to create movie!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('', fontsize=14, fontweight='bold')
    
    def update(frame_idx):
        rollout_idx, step_idx = frame_info[frame_idx]
        im.set_array(frames[frame_idx])
        title.set_text(f'Rollout {rollout_idx + 1}/{n_rollouts} | Step {step_idx}')
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=200, blit=True, repeat=True
    )
    
    try:
        writer = animation.FFMpegWriter(fps=5, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✓ Movie saved to {output_path}")
    except Exception as e:
        print(f"Could not save MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=5)
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
    nodes_per_cluster = NUM_NODES_PER_CLUSTER_QUICK if quick_mode else NUM_NODES_PER_CLUSTER_FULL
    max_steps = MAX_STEPS_QUICK if quick_mode else MAX_STEPS_FULL
    total_nodes = nodes_per_cluster * NUM_CLUSTERS
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print("=" * 70)
    print("Transport Two-Cluster Demo")
    print(f"  [{mode_str}]")
    print(f"Network: {total_nodes} nodes in 2 clusters ({nodes_per_cluster} each) with 1 bridge")
    print("Agents: 2 humans (learning), 1 vehicle (fixed shuttle policy)")
    print("Question: Will humans learn to use the vehicle?")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two-cluster network
    print("Creating two-cluster network...")
    network = create_two_cluster_network(
        nodes_per_cluster=nodes_per_cluster,
        cluster_separation=50.0,
        cluster_radius=10.0,
        seed=42
    )
    print(f"  Nodes: {network.number_of_nodes()}")
    print(f"  Edges: {network.number_of_edges()}")
    
    # Find centroids
    centroids = find_cluster_centroids(network)
    print(f"  Cluster 0 centroid: node {centroids[0]}")
    print(f"  Cluster 1 centroid: node {centroids[1]}")
    print()
    
    # Create environment with cluster-based routing
    print("Creating transport environment...")
    env = TransportEnvWrapper(
        num_humans=2,
        num_vehicles=1,
        network=network,
        human_speeds=[HUMAN_SPEED, HUMAN_SPEED],
        vehicle_speeds=[VEHICLE_SPEED],
        vehicle_capacities=[10],  # Large capacity
        num_clusters=NUM_CLUSTERS,  # Enable cluster-based routing
        clustering_method='kmeans',
        render_mode=None,
        max_steps=max_steps,
    )
    env.reset(seed=42)
    
    print(f"  Agents: {env.agents}")
    print(f"  Human speed: {HUMAN_SPEED}")
    print(f"  Vehicle speed: {VEHICLE_SPEED} (10x faster)")
    print(f"  Clusters: {NUM_CLUSTERS}")
    print()
    
    # Agent indices
    human_agent_indices = env.human_agent_indices
    vehicle_agent_idx = env.vehicle_agent_indices[0]
    
    # Create vehicle policy
    vehicle_policy = CentroidShuttlePolicy(env, centroids, vehicle_agent_idx, seed=42)
    print(f"Vehicle policy: Shuttle between centroids {centroids}")
    print()
    
    # Create goal sampler for humans
    goal_sampler = TransportGoalSampler(env, seed=42)
    
    # Train human neural policies
    print(f"Training human policies for {n_episodes} episodes...")
    print(f"  Gamma: {HUMAN_GAMMA} (incentivizes arriving earlier)")
    print(f"  Using GNN-based Q-learning")
    print()
    
    device = 'cpu'
    beta = 5.0
    
    # Adjust parameters based on mode
    if quick_mode:
        batch_size = 16
        hidden_dim = 64
        num_gnn_layers = 2
        updates_per_episode = 2
    else:
        batch_size = 32
        hidden_dim=128
        num_gnn_layers = 3
        updates_per_episode = 5
    
    t0 = time.time()
    neural_prior = train_transport_neural_policy_prior(
        env=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        num_episodes=n_episodes,
        steps_per_episode=max_steps,
        batch_size=batch_size,
        learning_rate=1e-3,
        gamma=HUMAN_GAMMA,
        beta=beta,
        buffer_capacity=20000,
        target_update_freq=50,
        state_feature_dim=64,
        goal_feature_dim=16,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        gnn_type='gcn',
        device=device,
        verbose=False,  # Disable verbose to avoid hanging
        reward_shaping=False,
        epsilon=0.3,
        updates_per_episode=updates_per_episode,
        max_nodes=total_nodes,
        num_clusters=NUM_CLUSTERS,
    )
    
    # Print progress indicators
    for i in range(0, n_episodes, max(1, n_episodes // 10)):
        if i == 0:
            print(f"  Episode 0/{n_episodes}...")
        else:
            print(f"  Episode {i}/{n_episodes}...")
    print(f"  Episode {n_episodes}/{n_episodes}...")
    
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.2f} seconds")
    print()
    
    # Run rollouts
    print(f"Running {n_rollouts} rollouts with learned policies...")
    random.seed(42)
    
    all_frames = []
    successes = 0
    
    for i in range(n_rollouts):
        # Select random goal nodes for each human
        # Make sure goals are in different clusters to make the problem interesting
        cluster_0_nodes = [n for n in network.nodes() if network.nodes[n].get('cluster') == 0]
        cluster_1_nodes = [n for n in network.nodes() if network.nodes[n].get('cluster') == 1]
        
        # Alternate: human 0 in cluster 0, human 1 in cluster 1, then vice versa
        if i % 2 == 0:
            goal_nodes = {
                human_agent_indices[0]: random.choice(cluster_0_nodes),
                human_agent_indices[1]: random.choice(cluster_1_nodes),
            }
        else:
            goal_nodes = {
                human_agent_indices[0]: random.choice(cluster_1_nodes),
                human_agent_indices[1]: random.choice(cluster_0_nodes),
            }
        
        print(f"  Rollout {i + 1}/{n_rollouts}: Goals = {goal_nodes}")
        
        frames, reached = run_rollout(
            env=env,
            neural_prior=neural_prior,
            vehicle_policy=vehicle_policy,
            goal_nodes=goal_nodes,
            human_agent_indices=human_agent_indices,
            vehicle_agent_idx=vehicle_agent_idx,
            centroids=centroids,
            max_steps=max_steps,
            beta=beta,
            device=device
        )
        all_frames.append(frames)
        
        if reached:
            successes += 1
            print(f"    ✓ All goals reached in {len(frames) - 1} steps")
        else:
            print(f"    ✗ Goals not reached")
    
    print()
    print(f"Success rate: {successes}/{n_rollouts} ({100*successes/n_rollouts:.1f}%)")
    print()
    
    # Create movie
    movie_path = os.path.join(output_dir, 'transport_two_cluster_demo.mp4')
    create_movie(all_frames, movie_path, n_rollouts=n_rollouts)
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Output: {os.path.abspath(movie_path)}")
    print()
    print("Analysis:")
    print("  - Did humans learn to walk to centroids and board the vehicle?")
    print("  - Did they learn the multi-step strategy of:")
    print("    1. Walk to nearest centroid")
    print("    2. Board vehicle")
    print("    3. Travel to other cluster")
    print("    4. Unboard")
    print("    5. Walk to goal")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transport Two-Cluster Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer episodes')
    args = parser.parse_args()
    main(quick_mode=args.quick)
