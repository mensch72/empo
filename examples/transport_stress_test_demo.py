#!/usr/bin/env python3
"""
Transport Stress Test Demo - 100 Nodes, 100 Vehicles, 100 Passengers.

This script creates a large-scale transport scenario with:
- 100 randomly placed nodes
- 100 vehicles (capacity 3 each)
- 100 human passengers
- Random actions with high boarding/unboarding probability
- Long rollout to showcase many interactions

The visualization demonstrates the rendering system's ability to handle
complex scenarios with many agents, vehicles, passengers, and destination
announcements all happening simultaneously.

Usage:
    python transport_stress_test_demo.py

Requirements:
    - ai_transport (vendored in vendor/ai_transport)
    - networkx
    - numpy
    - matplotlib
    - PIL (for GIF generation)
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'ai_transport'))

import numpy as np
import networkx as nx
from ai_transport import parallel_env


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
    # This creates a connected network without too many edges
    distance_threshold = 20.0
    min_degree = 2  # Ensure each node has at least this many connections
    
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
    
    # Second pass: ensure connectivity by connecting isolated components
    # Find connected components (treating as undirected for this check)
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
    
    print(f"Created network with {num_nodes} nodes and {len(G.edges())} directed edges")
    return G


def run_stress_test_demo():
    """
    Run a stress test demo with many agents taking random actions.
    """
    print("=" * 70)
    print("Transport Stress Test Demo - 100 Nodes, 100 Vehicles, 100 Passengers")
    print("=" * 70)
    
    # Create the random network
    num_nodes = 100
    network = create_random_network(num_nodes=num_nodes, seed=42)
    print(f"Network: {num_nodes} nodes, {len(network.edges())} directed edges")
    
    # Create environment with 100 humans and 100 vehicles
    num_humans = 100
    num_vehicles = 100
    vehicle_capacity = 3
    
    print(f"\nCreating environment:")
    print(f"  - {num_humans} human passengers")
    print(f"  - {num_vehicles} vehicles (capacity {vehicle_capacity} each)")
    
    # Create environment with random initial positions
    env = parallel_env(
        network=network,
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        vehicle_capacities=[vehicle_capacity] * num_vehicles,
        vehicle_speeds=[5.0] * num_vehicles,  # Vehicles are 5x faster than humans
        observation_scenario='full',
        render_mode='human',  # Enable rendering
    )
    
    observations, infos = env.reset(seed=123)
    agents = env.agents
    
    # Place agents at random nodes to ensure co-location for boarding opportunities
    np.random.seed(789)
    node_list = list(network.nodes())
    
    # Place vehicles at random nodes
    for i, vehicle_agent in enumerate([a for a in agents if a.startswith("vehicle_")]):
        random_node = np.random.choice(node_list)
        env.agent_positions[vehicle_agent] = random_node
    
    # Place humans at random nodes (some will be co-located with vehicles)
    for i, human_agent in enumerate([a for a in agents if a.startswith("human_")]):
        random_node = np.random.choice(node_list)
        env.agent_positions[human_agent] = random_node
        env.human_aboard[human_agent] = None  # Ensure not aboard initially
    
    print(f"Placed agents at random nodes")
    
    # Assign random goal nodes to humans
    np.random.seed(456)
    human_goals = {}
    for agent in agents:
        if agent.startswith("human_"):
            goal_node = np.random.choice(list(network.nodes()))
            human_goals[agent] = goal_node
            idx = int(agent.split("_")[1])
            human_goals[idx] = goal_node
    
    print(f"Assigned random goal nodes to {len([a for a in agents if a.startswith('human_')])} humans")
    
    # Count initial co-locations
    human_nodes = {}
    vehicle_nodes = {}
    for agent in agents:
        pos = env.agent_positions.get(agent)
        if agent.startswith("human_"):
            human_nodes.setdefault(pos, []).append(agent)
        elif agent.startswith("vehicle_"):
            vehicle_nodes.setdefault(pos, []).append(agent)
    
    colocated_nodes = set(human_nodes.keys()) & set(vehicle_nodes.keys())
    num_colocated_humans = sum(len(human_nodes[n]) for n in colocated_nodes)
    print(f"Initial co-locations: {num_colocated_humans} humans at same nodes as vehicles")
    
    # Start video recording
    print("\nStarting video recording...")
    env.start_video_recording()
    
    # Render initial state (Step 0)
    env.render(goal_info=human_goals, title="Stress Test Demo | Step 0")
    
    # Run rollout with random actions
    print("Running rollout with random actions...")
    print("(Completely random - boarding happens naturally when humans and vehicles co-locate)\n")
    
    step = 0
    max_steps = 100  # Long rollout to see lots of activity
    done = False
    boarding_events = 0
    unboarding_events = 0
    
    while not done and step < max_steps:
        # Track step type
        step_type = env.step_type if hasattr(env, 'step_type') else 'unknown'
        
        # Generate random actions for each agent
        actions = {}
        
        for agent in agents:
            action_mask = observations[agent].get('action_mask', [])
            if len(action_mask) == 0:
                actions[agent] = 0
                continue
            
            # Choose random valid action
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                actions[agent] = action
                
                # Track boarding/unboarding
                if step_type == 'boarding' and agent.startswith("human_") and action > 0:
                    boarding_events += 1
                elif step_type == 'unboarding' and agent.startswith("human_") and action > 0:
                    unboarding_events += 1
            else:
                actions[agent] = 0  # idle if no valid actions
        
        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render the new state (this captures the frame if video recording is active)
        env.render(goal_info=human_goals, title=f"Stress Test Demo | Step {step+1} ({step_type})")
        
        step += 1
        
        # Check if done
        done = all(terminations.values()) or all(truncations.values())
        
        # Print progress every 10 steps
        if step % 10 == 0:
            num_aboard = sum(1 for agent in agents 
                           if agent.startswith("human_") and 
                           env.human_aboard.get(agent) is not None)
            print(f"Step {step}: {num_aboard}/{num_humans} humans aboard | "
                  f"Boardings: {boarding_events} | Unboardings: {unboarding_events} | "
                  f"Phase: {step_type}")
    
    print(f"\nRollout completed after {step} steps")
    print(f"Total boarding events: {boarding_events}")
    print(f"Total unboarding events: {unboarding_events}")
    
    # Save video to outputs folder
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'transport_stress_test_demo.mp4')
    
    print(f"\nSaving video to {output_file}...")
    env.save_video(output_file, fps=2)
    print(f"Video saved successfully!")
    print(f"The video shows {step} steps with many simultaneous interactions:")
    print(f"  - Vehicles moving and announcing destinations (blue arcs)")
    print(f"  - Humans boarding and unboarding vehicles")
    print(f"  - Passengers visible inside vehicles (small red dots)")
    print(f"  - Continuous movement along roads")
    
    env.close()


if __name__ == "__main__":
    run_stress_test_demo()
