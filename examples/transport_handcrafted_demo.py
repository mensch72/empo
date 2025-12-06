#!/usr/bin/env python3
"""
Minimal Transport Demo with Hand-Crafted Actions.

This script demonstrates a simple triangle network where:
1. Human walks from node 0 to node 1
2. Human boards a vehicle at node 1
3. Vehicle (with human passenger) travels from node 1 to node 2
4. Human unboards at node 2
5. Human walks from node 2 to goal node 3

All actions are hand-crafted (no policies), and a video is generated showing
the continuous movement.

Usage:
    python transport_handcrafted_demo.py

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


def create_triangle_network():
    """
    Create a simple triangle network with 4 nodes:
    
         0
        / \
       1---2
            \
             3
    
    Node layout forms a triangle (0-1-2) plus an extension to node 3.
    """
    G = nx.DiGraph()  # Use directed graph for the transport environment
    
    # Add nodes with positions and names
    # Scale coordinates by 15 to avoid overlapping nodes in rendering
    scale = 15
    node_positions = {
        0: (0 * scale, 1 * scale),       # Top vertex
        1: (-1 * scale, 0 * scale),      # Bottom left
        2: (1 * scale, 0 * scale),       # Bottom right
        3: (2 * scale, -1 * scale),      # Extension (goal)
    }
    
    for node, (x, y) in node_positions.items():
        G.add_node(node, x=x, y=y, name=f"Node_{node}")
    
    # Add edges (bidirectional roads) with required attributes
    # Calculate proper edge lengths based on Euclidean distance
    edges = [
        (0, 1),  # Left edge of triangle
        (1, 0),  # Reverse
        (0, 2),  # Right edge of triangle
        (2, 0),  # Reverse
        (1, 2),  # Bottom edge of triangle
        (2, 1),  # Reverse
        (2, 3),  # Extension to goal
        (3, 2),  # Reverse
    ]
    
    for u, v in edges:
        x1, y1 = node_positions[u]
        x2, y2 = node_positions[v]
        euclidean_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        G.add_edge(u, v, length=euclidean_length, speed=1.0, capacity=10.0)
    
    return G


def run_handcrafted_demo():
    """
    Run a demo with hand-crafted actions showing the full multi-modal journey.
    """
    print("=" * 70)
    print("Minimal Transport Demo - Hand-Crafted Actions")
    print("=" * 70)
    
    # Create the triangle network
    network = create_triangle_network()
    print(f"\nNetwork: 4 nodes, {len(network.edges())} directed edges")
    print("Nodes: 0 (start), 1 (board point), 2 (unboard point), 3 (goal)")
    
    # Create environment with 1 human and 1 vehicle
    env = parallel_env(
        network=network,
        num_humans=1,
        num_vehicles=1,
        human_speeds=[1.0],  # List of speeds for each human
        vehicle_speeds=[2.0],  # Vehicle is 2x faster
    )
    
    # Set human goal
    human_goals = {0: 3}  # Human 0 wants to reach node 3
    
    # Reset environment
    observations, infos = env.reset(seed=42)
    
    # Initialize agents at specific locations
    env.unwrapped.agent_positions['human_0'] = 0  # Human starts at node 0
    env.unwrapped.agent_positions['vehicle_0'] = 1  # Vehicle starts at node 1
    
    print(f"\nInitial positions:")
    print(f"  Human 0: node {env.unwrapped.agent_positions['human_0']}")
    print(f"  Vehicle 0: node {env.unwrapped.agent_positions['vehicle_0']}")
    print(f"  Human goal: node {human_goals[0]}")
    
    # Start video recording
    env.unwrapped.start_video_recording()
    
    # Render initial state (Step 0)
    env.unwrapped.render(goal_info=human_goals, title="Hand-Crafted Demo | Step 0")
    
    # Define the sequence of hand-crafted actions
    # The environment cycles through step types: routing -> unboarding -> boarding -> departing
    # Actions are simple integers:
    #   - For routing: 0 = no destination, 1+ = node index + 1
    #   - For boarding/unboarding: 0 = pass, 1+ = board/unboard vehicle index + 1
    #   - For departing: 0 = wait, 1+ = depart to node index + 1
    
    # Step type cycle index
    step_types = ['routing', 'unboarding', 'boarding', 'departing']
    
    # Plan:
    # 1. Human walks from 0 to 1 (action 1 from node 0 = edge (0,1))
    # 2. Human boards vehicle at 1 (action 1 = board vehicle 0)
    # 3. Vehicle travels from 1 to 2 (action 2 from node 1 = edge (1,2))
    # 4. Human unboards at 2 (action 1 = unboard)
    # 5. Human walks from 2 to 3 (action 3 from node 2 = edge (2,3))
    
    action_plan = [
        # Cycle 1: routing(pass) -> unboarding(pass) -> boarding(pass) -> departing(human to node 1 via edge (0,1))
        ({'human_0': 0, 'vehicle_0': 0}, 'routing', "Vehicle: no routing"),
        ({'human_0': 0, 'vehicle_0': 0}, 'unboarding', "Human: nothing to unboard from"),
        ({'human_0': 0, 'vehicle_0': 0}, 'boarding', "Human: not at vehicle yet"),
        ({'human_0': 1, 'vehicle_0': 0}, 'departing', "Human: depart via edge (0,1)"),
        
        # Cycle 2: routing -> unboarding -> boarding(human boards) -> departing(pass, let human arrive)
        ({'human_0': 0, 'vehicle_0': 0}, 'routing', "Vehicle: no routing"),
        ({'human_0': 0, 'vehicle_0': 0}, 'unboarding', "Human: still traveling"),
        ({'human_0': 1, 'vehicle_0': 0}, 'boarding', "Human: board vehicle_0"),
        ({'human_0': 0, 'vehicle_0': 0}, 'departing', "Wait for boarding to complete"),
        
        # Cycle 3: routing(vehicle announces node 2) -> unboarding -> boarding -> departing(vehicle to node 2 via edge (1,2))
        ({'human_0': 0, 'vehicle_0': 2}, 'routing', "Vehicle: announce destination node 2"),
        ({'human_0': 0, 'vehicle_0': 0}, 'unboarding', "Human: aboard vehicle"),
        ({'human_0': 0, 'vehicle_0': 0}, 'boarding', "Human: already aboard"),
        ({'human_0': 0, 'vehicle_0': 2}, 'departing', "Vehicle: depart via edge (1,2) with passenger"),
        
        # Cycle 4: routing -> unboarding(human unboards) -> boarding -> departing(pass, let vehicle arrive)
        ({'human_0': 0, 'vehicle_0': 0}, 'routing', "Vehicle: no routing"),
        ({'human_0': 1, 'vehicle_0': 0}, 'unboarding', "Human: unboard at node 2"),
        ({'human_0': 0, 'vehicle_0': 0}, 'boarding', "Human: not boarding"),
        ({'human_0': 0, 'vehicle_0': 0}, 'departing', "Wait for unboarding"),
        
        # Cycle 5: routing -> unboarding -> boarding -> departing(human to node 3 via edge (2,3))
        ({'human_0': 0, 'vehicle_0': 0}, 'routing', "Vehicle: no routing"),
        ({'human_0': 0, 'vehicle_0': 0}, 'unboarding', "Human: not aboard"),
        ({'human_0': 0, 'vehicle_0': 0}, 'boarding', "Human: not boarding"),
        ({'human_0': 3, 'vehicle_0': 0}, 'departing', "Human: depart via edge (2,3) to goal"),
        
        # Cycle 6: Let human arrive at goal
        ({'human_0': 0, 'vehicle_0': 0}, 'routing', "Vehicle: no routing"),
        ({'human_0': 0, 'vehicle_0': 0}, 'unboarding', "Human: not aboard"),
        ({'human_0': 0, 'vehicle_0': 0}, 'boarding', "Human: not boarding"),
        ({'human_0': 0, 'vehicle_0': 0}, 'departing', "Human: wait (arriving at goal)"),
    ]
    
    print(f"\n{'Cycle':<7} {'Step Type':<12} {'Actions':<50} {'Human Position':<20}")
    print("-" * 100)
    
    # Execute the action sequence
    cycle = 0
    for step_num, (actions, expected_step_type, description) in enumerate(action_plan):
        # Verify we're in the expected step type
        actual_step_type = env.unwrapped.step_type
        if actual_step_type != expected_step_type:
            print(f"Warning: Expected {expected_step_type}, got {actual_step_type}")
        
        # Track cycle number
        if expected_step_type == 'routing':
            cycle += 1
        
        # Execute step
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Get human position
        human_pos = env.unwrapped.agent_positions.get('human_0', 'unknown')
        if isinstance(human_pos, tuple) and len(human_pos) == 2:
            # On an edge
            edge, progress = human_pos
            if isinstance(edge, tuple):
                human_pos_str = f"Edge {edge[0]}→{edge[1]}"
            else:
                human_pos_str = f"Edge {edge}"
        else:
            human_pos_str = f"Node {human_pos}"
        
        print(f"{cycle:<7} {expected_step_type:<12} {description:<50} {human_pos_str:<20}")
        
        # Render the state after departing step (when movement occurs)
        if expected_step_type == 'departing':
            env.unwrapped.render(
                goal_info=human_goals,
                title=f"Hand-Crafted Demo | Cycle {cycle}, After Departing"
            )
        
        # Check if human reached goal
        if env.unwrapped.agent_positions.get('human_0') == human_goals[0]:
            print(f"\n✓ Human reached goal after cycle {cycle}!")
            # Render final state
            env.unwrapped.render(
                goal_info=human_goals,
                title=f"Hand-Crafted Demo | GOAL REACHED!"
            )
            break
    
    # Save video
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'transport_handcrafted_demo.mp4')
    
    print(f"\nSaving video to {output_file}...")
    env.unwrapped.save_video(output_file, fps=2)
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print(f"Output: {output_file}")
    print("\nThis demo showed:")
    print("  1. Human walking from node 0 to node 1")
    print("  2. Human boarding vehicle at node 1")
    print("  3. Vehicle transporting human from node 1 to node 2")
    print("  4. Human unboarding at node 2")
    print("  5. Human walking from node 2 to goal node 3")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    run_handcrafted_demo()
