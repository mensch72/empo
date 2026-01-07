"""
Feature extraction for transport neural network encoders.

This module provides functions to extract features from transport environment
state and convert them into tensors suitable for GNN processing.

The main entry point is `observation_to_graph_data()` which converts a
transport environment observation into a PyTorch Geometric Data object.
"""

import torch
from typing import Dict, Tuple

from .constants import (
    NUM_STEP_TYPES,
    MAX_CLUSTERS,
    MAX_PARKING_SLOTS,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    AGENT_FEATURE_DIM,
)


def extract_node_features(
    env,
    query_agent_idx: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract features for each node in the network.
    
    This function is designed to scale for hundreds of agents:
    - Human counts are aggregated per node
    - Vehicles are tracked individually in parking slots (limited slots per node)
    
    Args:
        env: TransportEnvWrapper instance
        query_agent_idx: Index of the agent being queried
        device: Torch device
    
    Returns:
        Tensor of shape (num_nodes, NODE_FEATURE_DIM)
    """
    network = env.env.network
    nodes = list(network.nodes())
    num_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    features = torch.zeros(num_nodes, NODE_FEATURE_DIM, device=device)
    
    # Get agent positions
    agent_positions = env.env.agent_positions
    query_agent = env.agents[query_agent_idx]
    query_pos = agent_positions.get(query_agent)
    
    # Get cluster info if available
    cluster_info = env.cluster_info if env.use_clusters else None
    
    # Get destination for query agent (if vehicle)
    query_destination = None
    if query_agent in env.env.vehicle_agents:
        query_destination = env.env.vehicle_destinations.get(query_agent)
    
    # Get sorted vehicle names for consistent slot assignment
    vehicle_agents = sorted(env.env.vehicle_agents)
    vehicle_to_slot = {v: i for i, v in enumerate(vehicle_agents) if i < MAX_PARKING_SLOTS}
    
    # Pre-compute agents at each node for efficiency
    humans_at_node = {node: 0 for node in nodes}
    vehicles_at_node = {node: [] for node in nodes}
    
    for agent_name, pos in agent_positions.items():
        if pos is not None and not isinstance(pos, tuple):
            if pos in humans_at_node:
                if agent_name in env.env.human_agents:
                    humans_at_node[pos] += 1
                else:
                    vehicles_at_node[pos].append(agent_name)
    
    for i, node in enumerate(nodes):
        network.nodes[node]
        
        # Cluster one-hot encoding
        if cluster_info is not None:
            cluster_id = cluster_info['node_to_cluster'].get(node, 0)
            if cluster_id < MAX_CLUSTERS:
                features[i, cluster_id] = 1.0
        
        feature_offset = MAX_CLUSTERS
        
        # Is query agent at this node?
        if query_pos is not None and not isinstance(query_pos, tuple) and query_pos == node:
            features[i, feature_offset] = 1.0
        feature_offset += 1
        
        # Aggregate human count at this node
        num_humans = humans_at_node[node]
        features[i, feature_offset] = float(num_humans)
        feature_offset += 1
        
        # Individual vehicle slots (up to MAX_PARKING_SLOTS per node)
        # Each vehicle has a consistent slot index across all nodes
        for vehicle_name in vehicles_at_node[node]:
            if vehicle_name in vehicle_to_slot:
                slot_idx = vehicle_to_slot[vehicle_name]
                if slot_idx < MAX_PARKING_SLOTS:
                    features[i, feature_offset + slot_idx] = 1.0
        feature_offset += MAX_PARKING_SLOTS
        
        # Is destination?
        if query_destination is not None and query_destination == node:
            features[i, feature_offset] = 1.0
        feature_offset += 1
        
        # Number of outgoing and incoming edges
        out_edges = network.out_degree(node)
        in_edges = network.in_degree(node)
        features[i, feature_offset] = float(out_edges)
        feature_offset += 1
        features[i, feature_offset] = float(in_edges)
    
    return features


def extract_edge_features(
    env,
    query_agent_idx: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features for each edge in the network.
    
    This function is designed to scale for hundreds of agents using
    raw agent counts on edges.
    
    Args:
        env: TransportEnvWrapper instance
        query_agent_idx: Index of the agent being queried
        device: Torch device
    
    Returns:
        Tuple of:
        - edge_index: Tensor of shape (2, num_edges) with source/target node indices
        - edge_features: Tensor of shape (num_edges, EDGE_FEATURE_DIM)
    """
    network = env.env.network
    nodes = list(network.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    edges = list(network.edges())
    num_edges = len(edges)
    
    edge_index = torch.zeros(2, num_edges, dtype=torch.long, device=device)
    edge_features = torch.zeros(num_edges, EDGE_FEATURE_DIM, device=device)
    
    # Get agent positions
    agent_positions = env.env.agent_positions
    query_agent = env.agents[query_agent_idx]
    agent_positions.get(query_agent)
    
    # Pre-compute agents on each edge for efficiency with many agents
    edge_to_key = {(u, v): i for i, (u, v) in enumerate(edges)}
    agents_on_edge = {i: 0 for i in range(num_edges)}
    query_on_edge_idx = None
    
    for agent_name, pos in agent_positions.items():
        if isinstance(pos, tuple):
            edge_info, progress = pos
            if len(edge_info) >= 2:
                edge_u = edge_info[0]
                edge_v = edge_info[1] if isinstance(edge_info[1], int) else edge_info[1].item()
                key = (edge_u, edge_v)
                if key in edge_to_key:
                    edge_idx = edge_to_key[key]
                    agents_on_edge[edge_idx] += 1
                    if agent_name == query_agent:
                        query_on_edge_idx = edge_idx
    
    for i, (u, v) in enumerate(edges):
        edge_data = network.edges[u, v]
        
        # Edge indices
        edge_index[0, i] = node_to_idx[u]
        edge_index[1, i] = node_to_idx[v]
        
        # Edge length
        length = edge_data.get('length', 1.0)
        edge_features[i, 0] = float(length)
        
        # Speed limit
        speed = edge_data.get('speed_limit', edge_data.get('speed', 50.0))
        edge_features[i, 1] = float(speed)
        
        # Capacity
        capacity = edge_data.get('capacity', 1.0)
        edge_features[i, 2] = float(capacity)
        
        # Agent count on this edge
        num_on_edge = agents_on_edge[i]
        edge_features[i, 3] = float(num_on_edge)
        
        # Query agent on edge
        edge_features[i, 4] = 1.0 if query_on_edge_idx == i else 0.0
    
    return edge_index, edge_features


def extract_global_features(
    env,
    query_agent_idx: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract global environment features.
    
    Args:
        env: TransportEnvWrapper instance
        query_agent_idx: Index of the agent being queried
        device: Torch device
    
    Returns:
        Tensor of shape (GLOBAL_FEATURE_DIM,)
    """
    features = torch.zeros(GLOBAL_FEATURE_DIM, device=device)
    
    # Step type one-hot
    step_type_idx = env.step_type_idx
    if step_type_idx < NUM_STEP_TYPES:
        features[step_type_idx] = 1.0
    
    feature_offset = NUM_STEP_TYPES
    
    # Real time
    real_time = env.env.real_time
    features[feature_offset] = float(real_time)
    feature_offset += 1
    
    # Number of humans and vehicles
    features[feature_offset] = float(len(env.env.human_agents))
    feature_offset += 1
    features[feature_offset] = float(len(env.env.vehicle_agents))
    feature_offset += 1
    
    # Cluster mode
    features[feature_offset] = 1.0 if env.use_clusters else 0.0
    feature_offset += 1
    features[feature_offset] = float(env.num_clusters) if env.use_clusters else 0.0
    
    return features


def extract_agent_features(
    env,
    query_agent_idx: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract features specific to the query agent.
    
    Args:
        env: TransportEnvWrapper instance
        query_agent_idx: Index of the agent being queried
        device: Torch device
    
    Returns:
        Tensor of shape (AGENT_FEATURE_DIM,)
    """
    features = torch.zeros(AGENT_FEATURE_DIM, device=device)
    
    query_agent = env.agents[query_agent_idx]
    is_human = query_agent in env.env.human_agents
    pos = env.env.agent_positions.get(query_agent)
    
    # Agent type
    features[0] = 1.0 if is_human else 0.0
    features[1] = 0.0 if is_human else 1.0
    
    # Position type
    at_node = pos is not None and not isinstance(pos, tuple)
    on_edge = pos is not None and isinstance(pos, tuple)
    features[2] = 1.0 if at_node else 0.0
    features[3] = 1.0 if on_edge else 0.0
    
    feature_offset = 4
    
    # Current node cluster (if at node)
    cluster_info = env.cluster_info if env.use_clusters else None
    if at_node and cluster_info is not None:
        cluster_id = cluster_info['node_to_cluster'].get(pos, 0)
        if cluster_id < MAX_CLUSTERS:
            features[feature_offset + cluster_id] = 1.0
    feature_offset += MAX_CLUSTERS
    
    # Destination cluster (if vehicle with destination)
    if not is_human:
        dest = env.env.vehicle_destinations.get(query_agent)
        if dest is not None and cluster_info is not None:
            dest_cluster = cluster_info['node_to_cluster'].get(dest, 0)
            if dest_cluster < MAX_CLUSTERS:
                features[feature_offset + dest_cluster] = 1.0
    feature_offset += MAX_CLUSTERS
    
    # Aboard vehicle (if human)
    if is_human:
        aboard = env.env.human_aboard.get(query_agent)
        features[feature_offset] = 1.0 if aboard is not None else 0.0
    feature_offset += 1
    
    # Has destination (if vehicle)
    if not is_human:
        dest = env.env.vehicle_destinations.get(query_agent)
        features[feature_offset] = 1.0 if dest is not None else 0.0
    
    return features


def observation_to_graph_data(
    env,
    query_agent_idx: int,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Convert transport environment observation to graph data for GNN processing.
    
    This is the main entry point for preparing input to the GNN encoder.
    
    Args:
        env: TransportEnvWrapper instance
        query_agent_idx: Index of the agent being queried
        device: Torch device
    
    Returns:
        Dictionary containing:
        - 'node_features': Tensor (num_nodes, NODE_FEATURE_DIM)
        - 'edge_index': Tensor (2, num_edges)
        - 'edge_features': Tensor (num_edges, EDGE_FEATURE_DIM)
        - 'global_features': Tensor (GLOBAL_FEATURE_DIM,)
        - 'agent_features': Tensor (AGENT_FEATURE_DIM,)
        - 'query_node_idx': int or None (node index where query agent is, if at node)
        - 'num_nodes': int
        - 'num_edges': int
    
    Example:
        >>> env = create_transport_env(num_humans=4, num_vehicles=2, num_nodes=12)
        >>> env.reset(seed=42)
        >>> graph_data = observation_to_graph_data(env, query_agent_idx=0)
        >>> print(graph_data['node_features'].shape)
        torch.Size([12, 26])
    """
    # Extract all features
    node_features = extract_node_features(env, query_agent_idx, device)
    edge_index, edge_features = extract_edge_features(env, query_agent_idx, device)
    global_features = extract_global_features(env, query_agent_idx, device)
    agent_features = extract_agent_features(env, query_agent_idx, device)
    
    # Determine query agent's node (if at a node)
    query_agent = env.agents[query_agent_idx]
    pos = env.env.agent_positions.get(query_agent)
    query_node_idx = None
    
    if pos is not None and not isinstance(pos, tuple):
        nodes = list(env.env.network.nodes())
        if pos in nodes:
            query_node_idx = nodes.index(pos)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'global_features': global_features,
        'agent_features': agent_features,
        'query_node_idx': query_node_idx,
        'num_nodes': node_features.shape[0],
        'num_edges': edge_index.shape[1],
    }
