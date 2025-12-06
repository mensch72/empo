#!/usr/bin/env python3
"""
Transport Stress Test Demo - 100 Nodes, 100 Vehicles, 100 Passengers.

This script creates a large-scale transport scenario with:
- 100 randomly placed nodes
- 100 vehicles (capacity 3 each)
- 100 human passengers
- Random actions with high boarding/unboarding probability
- Long rollout to showcase many interactions

Rewritten from scratch using transport_two_cluster_demo.py as a template
to ensure correct video rendering.

Usage:
    python transport_stress_test_demo.py
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'ai_transport'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import networkx as nx

from empo.transport import (
    TransportEnvWrapper,
    TransportGoal,
    TransportGoalSampler,
    TransportActions,
)


# ============================================================================
# Configuration
# ============================================================================

NUM_NODES = 50  # Reasonable size for demonstration
NUM_VEHICLES = 25  # Half the nodes
NUM_HUMANS = 25  # Half the nodes
VEHICLE_CAPACITY = 3
VEHICLE_SPEED = 5.0
HUMAN_SPEED = 1.0
MAX_STEPS = 50  # Reasonable rollout length
BOARDING_PROB = 0.8  # High probability to board when vehicles available
UNBOARDING_PROB = 0.2  # Low probability to unboard (passengers stay on longer)


# ============================================================================
# Network Generation
# ============================================================================

def create_random_network(num_nodes=100, seed=42):
    """
    Create a random network with nodes placed randomly in 2D space.
    
    Args:
        num_nodes: Number of nodes to create
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX DiGraph with node positions and edges
    """
    np.random.seed(seed)
    G = nx.DiGraph()
    
    # Generate random node positions in a 100x100 square
    node_positions = {}
    for i in range(num_nodes):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        node_positions[i] = (x, y)
        G.add_node(i, x=x, y=y, name=f"N{i}")
    
    # Create edges between nearby nodes (within distance threshold)
    distance_threshold = 20.0
    
    # First pass: connect nearby nodes
    for i in range(num_nodes):
        x1, y1 = node_positions[i]
        neighbors = []
        
        for j in range(num_nodes):
            if i == j:
                continue
            x2, y2 = node_positions[j]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist <= distance_threshold:
                neighbors.append((j, dist))
        
        # Sort by distance and connect to closest neighbors
        neighbors.sort(key=lambda x: x[1])
        for j, dist in neighbors[:5]:  # Connect to up to 5 closest neighbors
            if not G.has_edge(i, j):
                G.add_edge(i, j, length=dist, speed=1.0, capacity=10.0)
            if not G.has_edge(j, i):
                G.add_edge(j, i, length=dist, speed=1.0, capacity=10.0)
    
    # Second pass: ensure connectivity
    G_undirected = G.to_undirected()
    components = list(nx.connected_components(G_undirected))
    
    if len(components) > 1:
        # Connect components by linking closest nodes from different components
        for i in range(len(components) - 1):
            comp1 = list(components[i])
            comp2 = list(components[i + 1])
            
            # Find closest pair of nodes between components
            min_dist = float('inf')
            best_pair = None
            
            for n1 in comp1:
                x1, y1 = node_positions[n1]
                for n2 in comp2:
                    x2, y2 = node_positions[n2]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (n1, n2)
            
            if best_pair:
                n1, n2 = best_pair
                G.add_edge(n1, n2, length=min_dist, speed=1.0, capacity=10.0)
                G.add_edge(n2, n1, length=min_dist, speed=1.0, capacity=10.0)
    
    return G


# ============================================================================
# Random Policy for Stress Test
# ============================================================================

class RandomStressTestPolicy:
    """
    Random policy with biased boarding/unboarding for stress testing.
    """
    
    def __init__(self, env, agent_idx, is_vehicle=False, seed=42):
        self.env = env
        self.agent_idx = agent_idx
        self.is_vehicle = is_vehicle
        self.rng = np.random.RandomState(seed + agent_idx)
    
    def get_action(self):
        """Get a random action based on current phase."""
        step_type = self.env.step_type
        agent_name = self.env.agents[self.agent_idx]
        
        if step_type == 'routing':
            if self.is_vehicle:
                # Vehicle announces random destination
                num_nodes = len(self.env.network.nodes())
                dest_node = self.rng.randint(0, num_nodes)
                return TransportActions.DEST_START + dest_node
            else:
                # Human doesn't announce
                return TransportActions.PASS
        
        elif step_type == 'boarding':
            if not self.is_vehicle:
                # Human: high probability to board if vehicles available
                # Check if there are vehicles at current location
                agent_pos = self.env.env.agent_positions.get(agent_name)
                if agent_pos and not isinstance(agent_pos, tuple):
                    # Get vehicles at same node
                    vehicles_here = []
                    for i, other_agent in enumerate(self.env.agents):
                        if i != self.agent_idx:
                            other_pos = self.env.env.agent_positions.get(other_agent)
                            # Check if it's a vehicle agent (vehicles come after humans)
                            is_vehicle_agent = 'vehicle' in other_agent
                            if other_pos == agent_pos and is_vehicle_agent:
                                vehicles_here.append(i)
                    
                    if vehicles_here and self.rng.random() < BOARDING_PROB:
                        # Try to board a random vehicle
                        vehicle_idx = self.rng.choice(vehicles_here)
                        return TransportActions.BOARD_START + vehicle_idx
            
            return TransportActions.PASS
        
        elif step_type == 'departing':
            # All agents: random movement
            agent_pos = self.env.env.agent_positions.get(agent_name)
            
            if agent_pos is None or isinstance(agent_pos, tuple):
                # On edge or invalid position
                return TransportActions.PASS
            
            # Get available neighbors
            neighbors = list(self.env.network.successors(agent_pos))
            if not neighbors:
                return TransportActions.PASS
            
            # Random movement
            next_node = self.rng.choice(neighbors)
            edge_idx = list(self.env.network.out_edges(agent_pos)).index((agent_pos, next_node))
            return TransportActions.DEPART_START + edge_idx
        
        elif step_type == 'unboarding':
            if not self.is_vehicle:
                # Human: low probability to unboard (stay on longer)
                # Check if aboard a vehicle
                agent_pos = self.env.env.agent_positions.get(agent_name)
                if agent_pos is None:
                    # May be aboard - check vehicle passengers
                    for vehicle_name in self.env.env.vehicle_agents:
                        if agent_name in self.env.env.vehicle_passengers.get(vehicle_name, set()):
                            if self.rng.random() < UNBOARDING_PROB:
                                return TransportActions.UNBOARD
                            break
            
            return TransportActions.PASS
        
        return TransportActions.PASS


# ============================================================================
# Rollout Function
# ============================================================================

def run_stress_test_rollout(env, policies, max_steps=100):
    """
    Run a single stress test rollout with random actions.
    
    This follows the same pattern as transport_two_cluster_demo.py
    to ensure proper video rendering.
    """
    env.reset()
    
    # Render initial state
    title = f"Stress Test | Step 0 (Initial) | {NUM_HUMANS} humans, {NUM_VEHICLES} vehicles"
    env.env.render(goal_info=None, value_dict=None, title=title)
    
    boarding_count = 0
    unboarding_count = 0
    
    for step in range(max_steps):
        # Get actions for all agents
        actions = []
        for agent_idx in range(env.num_agents):
            action = policies[agent_idx].get_action()
            actions.append(action)
            
            # Track boarding/unboarding
            step_type = env.step_type
            if step_type == 'boarding' and action >= TransportActions.BOARD_START:
                boarding_count += 1
            elif step_type == 'unboarding' and action == TransportActions.UNBOARD:
                unboarding_count += 1
        
        # Execute actions
        obs, rewards, done, info = env.step(actions)
        
        # Render after step (THIS IS CRITICAL FOR VIDEO CAPTURE)
        step_type = env.step_type
        title = f"Stress Test | Step {step + 1} | Phase: {step_type} | Boardings: {boarding_count} | Unboardings: {unboarding_count}"
        env.env.render(goal_info=None, value_dict=None, title=title)
        
        if done:
            break
    
    print(f"Rollout complete: {step + 1} steps")
    print(f"  Total boardings: {boarding_count}")
    print(f"  Total unboardings: {unboarding_count}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Transport Stress Test Demo")
    print(f"Network: {NUM_NODES} randomly placed nodes")
    print(f"Agents: {NUM_HUMANS} humans + {NUM_VEHICLES} vehicles")
    print(f"Actions: Random with high boarding probability ({BOARDING_PROB})")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create random network
    print(f"Creating random network with {NUM_NODES} nodes...")
    network = create_random_network(num_nodes=NUM_NODES, seed=42)
    print(f"  Nodes: {network.number_of_nodes()}")
    print(f"  Edges: {network.number_of_edges()}")
    
    # Create environment
    print(f"\nCreating transport environment with {NUM_HUMANS} humans and {NUM_VEHICLES} vehicles...")
    
    human_speeds = [HUMAN_SPEED] * NUM_HUMANS
    vehicle_speeds = [VEHICLE_SPEED] * NUM_VEHICLES
    vehicle_capacities = [VEHICLE_CAPACITY] * NUM_VEHICLES
    
    env = TransportEnvWrapper(
        num_humans=NUM_HUMANS,
        num_vehicles=NUM_VEHICLES,
        network=network,
        human_speeds=human_speeds,
        vehicle_speeds=vehicle_speeds,
        vehicle_capacities=vehicle_capacities,
        render_mode='human',  # Enable rendering for video recording
        max_steps=MAX_STEPS,
    )
    env.reset(seed=42)
    
    print(f"  Total agents: {env.num_agents}")
    print(f"    Humans: {NUM_HUMANS}")
    print(f"    Vehicles: {NUM_VEHICLES}")
    
    # Create random policies for all agents
    print("\nInitializing random policies...")
    policies = []
    for agent_idx in range(env.num_agents):
        is_vehicle = agent_idx >= NUM_HUMANS  # Vehicles come after humans
        policy = RandomStressTestPolicy(env, agent_idx, is_vehicle=is_vehicle, seed=42)
        policies.append(policy)
    
    # Start video recording
    video_path = os.path.join(output_dir, 'transport_stress_test_demo.mp4')
    print(f"\nStarting video recording to: {video_path}")
    env.env.start_video_recording()
    
    # Run rollout
    print(f"\nRunning stress test rollout ({MAX_STEPS} max steps)...")
    print("-" * 70)
    run_stress_test_rollout(env, policies, max_steps=MAX_STEPS)
    print("-" * 70)
    
    # Save video
    print(f"\nSaving video to: {video_path}")
    env.env.save_video(video_path, fps=10)
    print(f"Video saved successfully!")
    
    # Verify video has frames by checking file size
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path) / 1024 / 1024  # MB
        print(f"Video file size: {file_size:.2f} MB")
        if file_size < 0.1:
            print("WARNING: Video file is very small, may not contain frames!")
    
    print("\n" + "=" * 70)
    print("Stress test complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
