"""
Example demonstrating random 2D network generation.

This example shows:
1. Creating a random 2D network using Delaunay triangulation
2. Initializing random agent positions
3. Visualizing the network structure
"""

import numpy as np
import networkx as nx
from ai_transport import parallel_env


def print_network_stats(network):
    """Print statistics about the network"""
    print(f"  Nodes: {len(network.nodes())}")
    print(f"  Edges: {len(network.edges())}")
    
    # Count bidirectional edges
    bidir_count = 0
    for u, v in network.edges():
        if network.has_edge(v, u):
            bidir_count += 1
    bidir_count //= 2  # Each pair counted twice
    
    print(f"  Bidirectional edges: {bidir_count}")
    print(f"  Unidirectional edges: {len(network.edges()) - 2 * bidir_count}")
    
    # Edge statistics
    lengths = [network[u][v]['length'] for u, v in network.edges()]
    speeds = [network[u][v]['speed'] for u, v in network.edges()]
    capacities = [network[u][v]['capacity'] for u, v in network.edges()]
    
    print(f"  Edge lengths: min={min(lengths):.2f}, max={max(lengths):.2f}, avg={np.mean(lengths):.2f}")
    print(f"  Edge speeds: min={min(speeds):.2f}, max={max(speeds):.2f}, avg={np.mean(speeds):.2f}")
    print(f"  Edge capacities: min={min(capacities):.2f}, max={max(capacities):.2f}, avg={np.mean(capacities):.2f}")


def print_position_stats(env):
    """Print statistics about agent positions"""
    at_nodes = 0
    on_edges = 0
    
    for agent, pos in env.agent_positions.items():
        if isinstance(pos, tuple):
            on_edges += 1
        else:
            at_nodes += 1
    
    print(f"  Agents at nodes: {at_nodes}")
    print(f"  Agents on edges: {on_edges}")


def main():
    print("=" * 70)
    print("AI Transport Environment - Random 2D Network Demo")
    print("=" * 70)
    
    # Create environment
    env = parallel_env(num_humans=5, num_vehicles=3)
    
    # Example 1: Small network with low bidirectional probability
    print("\n" + "=" * 70)
    print("Example 1: Small Network (10 nodes, 10% bidirectional)")
    print("=" * 70)
    
    network1 = env.create_random_2d_network(
        num_nodes=10,
        bidirectional_prob=0.1,
        speed_mean=5.0,
        capacity_mean=10.0,
        coord_std=10.0,
        seed=42
    )
    
    print_network_stats(network1)
    
    # Show some node coordinates
    print("\nSample node coordinates:")
    for i in range(min(5, len(network1.nodes()))):
        node_data = network1.nodes[i]
        print(f"  Node {i} ({node_data['name']}): x={node_data['x']:.2f}, y={node_data['y']:.2f}")
    
    # Example 2: Larger network with high bidirectional probability
    print("\n" + "=" * 70)
    print("Example 2: Larger Network (20 nodes, 70% bidirectional)")
    print("=" * 70)
    
    network2 = env.create_random_2d_network(
        num_nodes=20,
        bidirectional_prob=0.7,
        speed_mean=8.0,
        capacity_mean=15.0,
        coord_std=15.0,
        seed=123
    )
    
    print_network_stats(network2)
    
    # Example 3: Using random network with environment
    print("\n" + "=" * 70)
    print("Example 3: Environment with Random Network and Positions")
    print("=" * 70)
    
    network3 = env.create_random_2d_network(
        num_nodes=15,
        bidirectional_prob=0.5,
        seed=456
    )
    
    # Create environment with this network
    env3 = parallel_env(
        num_humans=5,
        num_vehicles=3,
        network=network3
    )
    
    env3.reset(seed=456)
    
    # Initialize random positions
    env3.initialize_random_positions(seed=789)
    
    print("\nNetwork statistics:")
    print_network_stats(network3)
    
    print("\nAgent positions:")
    print_position_stats(env3)
    
    print("\nAgent details:")
    for agent in env3.agents[:5]:  # Show first 5 agents
        pos = env3.agent_positions[agent]
        if isinstance(pos, tuple):
            edge, coord = pos
            print(f"  {agent}: on edge {edge} at coordinate {coord:.2f}")
        else:
            print(f"  {agent}: at node {pos}")
    
    # Test environment functionality
    print("\n" + "=" * 70)
    print("Testing Environment Functionality")
    print("=" * 70)
    
    env3.step_type = 'routing'
    actions = {agent: 0 for agent in env3.agents}
    obs, rewards, terms, truncs, infos = env3.step(actions)
    
    print("\nStep completed successfully!")
    print(f"  Observations returned: {len(obs)}")
    print(f"  Rewards returned: {len(rewards)}")
    
    # Example 4: Very sparse network
    print("\n" + "=" * 70)
    print("Example 4: Sparse Network (5 nodes, 0% bidirectional)")
    print("=" * 70)
    
    network4 = env.create_random_2d_network(
        num_nodes=5,
        bidirectional_prob=0.0,
        coord_std=5.0,
        seed=999
    )
    
    print_network_stats(network4)
    
    # Verify all edges are unidirectional
    print("\nEdge directions:")
    for u, v in list(network4.edges())[:10]:  # Show first 10
        has_reverse = "↔" if network4.has_edge(v, u) else "→"
        print(f"  {u} {has_reverse} {v}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nNote: The create_random_2d_network method generates networks using:")
    print("  - 2D Gaussian distribution for node coordinates")
    print("  - Delaunay triangulation for connectivity")
    print("  - Exponential distributions for edge speeds and capacities")
    print("  - Configurable bidirectional probability")


if __name__ == "__main__":
    main()
