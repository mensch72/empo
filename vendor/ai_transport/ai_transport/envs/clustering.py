"""
Network Clustering for Transport Environment.

This module provides clustering utilities for grouping network nodes into
regions (clusters). Clusters are used to allow vehicles to announce
destination regions rather than fixed routes, increasing flexibility
and potentially passenger empowerment.

The main use case is k-means clustering on 2D node coordinates, but
the module is designed to support alternative clustering methods.

Example usage:
    >>> import networkx as nx
    >>> from ai_transport.envs.clustering import cluster_network
    >>> 
    >>> # Create a sample network with 2D positions
    >>> G = nx.DiGraph()
    >>> for i in range(20):
    ...     G.add_node(i, x=i % 5, y=i // 5)
    >>> 
    >>> # Cluster into 4 regions
    >>> cluster_info = cluster_network(G, k=4)
    >>> print(cluster_info['node_to_cluster'])
    >>> print(cluster_info['cluster_to_nodes'])
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def cluster_network(
    G: nx.DiGraph,
    k: int = 20,
    method: str = 'kmeans',
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Cluster network nodes into k regions.
    
    The goal is to find k disjoint sets S_i of network nodes such that,
    roughly, for every pair of nodes v1, v2, there is at least one S_i
    where it is plausible that when going from v1 to S_i on a plausible
    route, one will come close to node v2.
    
    Args:
        G: NetworkX directed graph with node attributes 'x' and 'y' 
           (or 'pos' as (x, y) tuple) for 2D coordinates.
        k: Number of clusters to create. Should be chosen large enough
           (e.g., k=20) to ensure good coverage of route possibilities.
        method: Clustering method to use:
            - 'kmeans': K-means clustering on 2D coordinates (default)
            - 'spectral': Spectral clustering on graph structure (TODO)
        random_state: Random seed for reproducibility.
    
    Returns:
        Dictionary with:
            - 'node_to_cluster': {node_id: cluster_id} mapping
            - 'cluster_to_nodes': {cluster_id: [node_ids]} mapping
            - 'centroids': {cluster_id: centroid_node_id} mapping
                           (nearest node to each cluster center)
            - 'cluster_centers': {cluster_id: (x, y)} mapping
                                 (actual cluster center coordinates)
            - 'num_clusters': Actual number of clusters (may be < k if 
                              there are fewer nodes than k)
    
    Raises:
        ValueError: If no nodes have valid coordinates.
        NotImplementedError: If an unsupported clustering method is requested.
    
    Example:
        >>> G = nx.DiGraph()
        >>> for i in range(10):
        ...     G.add_node(i, x=float(i % 5), y=float(i // 5))
        >>> info = cluster_network(G, k=3)
        >>> print(info['num_clusters'])
        3
    """
    if len(G.nodes()) == 0:
        return {
            'node_to_cluster': {},
            'cluster_to_nodes': {},
            'centroids': {},
            'cluster_centers': {},
            'num_clusters': 0
        }
    
    # Adjust k if there are fewer nodes than clusters requested
    actual_k = min(k, len(G.nodes()))
    
    if method == 'kmeans':
        return _cluster_kmeans(G, actual_k, random_state)
    elif method == 'spectral':
        raise NotImplementedError(
            "Spectral clustering not yet implemented. "
            "Use method='kmeans' for now."
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")


def _extract_coordinates(G: nx.DiGraph) -> Tuple[List, np.ndarray]:
    """
    Extract 2D coordinates from graph nodes.
    
    Supports two formats:
    - 'x' and 'y' attributes separately
    - 'pos' attribute as (x, y) tuple
    
    Args:
        G: NetworkX graph with coordinate attributes.
    
    Returns:
        Tuple of (node_list, coordinates_array) where coordinates_array
        has shape (num_nodes, 2).
    
    Raises:
        ValueError: If nodes don't have valid coordinate attributes.
    """
    nodes = list(G.nodes())
    coords = []
    
    for node in nodes:
        node_data = G.nodes[node]
        
        if 'x' in node_data and 'y' in node_data:
            coords.append([float(node_data['x']), float(node_data['y'])])
        elif 'pos' in node_data:
            pos = node_data['pos']
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                coords.append([float(pos[0]), float(pos[1])])
            else:
                raise ValueError(
                    f"Node {node} has invalid 'pos' attribute: {pos}"
                )
        else:
            raise ValueError(
                f"Node {node} missing coordinate attributes. "
                f"Expected 'x'/'y' or 'pos'. Got: {list(node_data.keys())}"
            )
    
    return nodes, np.array(coords)


def _cluster_kmeans(
    G: nx.DiGraph,
    k: int,
    random_state: int
) -> Dict[str, Any]:
    """
    Perform k-means clustering on node coordinates.
    
    Args:
        G: NetworkX graph with coordinate attributes.
        k: Number of clusters.
        random_state: Random seed.
    
    Returns:
        Clustering result dictionary.
    """
    from sklearn.cluster import KMeans
    
    nodes, coords = _extract_coordinates(G)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)
    
    # Build node_to_cluster mapping
    node_to_cluster = {node: int(labels[i]) for i, node in enumerate(nodes)}
    
    # Build cluster_to_nodes mapping
    cluster_to_nodes: Dict[int, List] = {}
    for node, cluster_id in node_to_cluster.items():
        cluster_to_nodes.setdefault(cluster_id, []).append(node)
    
    # Find centroid nodes (nearest node to each cluster center)
    centroids: Dict[int, Any] = {}
    cluster_centers: Dict[int, Tuple[float, float]] = {}
    
    for cluster_id in range(k):
        cluster_nodes = cluster_to_nodes.get(cluster_id, [])
        if not cluster_nodes:
            continue
        
        center = kmeans.cluster_centers_[cluster_id]
        cluster_centers[cluster_id] = (float(center[0]), float(center[1]))
        
        # Find node nearest to cluster center
        min_dist = float('inf')
        nearest_node = cluster_nodes[0]
        
        for i, node in enumerate(nodes):
            if node_to_cluster[node] == cluster_id:
                dist = np.linalg.norm(coords[i] - center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
        
        centroids[cluster_id] = nearest_node
    
    return {
        'node_to_cluster': node_to_cluster,
        'cluster_to_nodes': cluster_to_nodes,
        'centroids': centroids,
        'cluster_centers': cluster_centers,
        'num_clusters': len(cluster_to_nodes)
    }


def get_cluster_for_node(
    node: Any,
    cluster_info: Dict[str, Any]
) -> Optional[int]:
    """
    Get the cluster ID for a given node.
    
    Args:
        node: Node identifier.
        cluster_info: Result from cluster_network().
    
    Returns:
        Cluster ID, or None if node not in clustering.
    """
    return cluster_info['node_to_cluster'].get(node)


def get_nodes_in_cluster(
    cluster_id: int,
    cluster_info: Dict[str, Any]
) -> List:
    """
    Get all nodes belonging to a cluster.
    
    Args:
        cluster_id: Cluster identifier.
        cluster_info: Result from cluster_network().
    
    Returns:
        List of node IDs in the cluster, or empty list if cluster doesn't exist.
    """
    return cluster_info['cluster_to_nodes'].get(cluster_id, [])


def get_cluster_centroid(
    cluster_id: int,
    cluster_info: Dict[str, Any]
) -> Optional[Any]:
    """
    Get the centroid node of a cluster.
    
    The centroid is the node nearest to the cluster's center point.
    
    Args:
        cluster_id: Cluster identifier.
        cluster_info: Result from cluster_network().
    
    Returns:
        Node ID of the centroid, or None if cluster doesn't exist.
    """
    return cluster_info['centroids'].get(cluster_id)


def visualize_clusters(
    G: nx.DiGraph,
    cluster_info: Dict[str, Any],
    ax=None,
    node_size: int = 100,
    show_labels: bool = True,
    show_centroids: bool = True,
    cmap: str = 'tab20'
):
    """
    Visualize network with cluster coloring.
    
    Args:
        G: NetworkX graph.
        cluster_info: Result from cluster_network().
        ax: Matplotlib axes (created if None).
        node_size: Size of node markers.
        show_labels: Whether to show node labels.
        show_centroids: Whether to highlight centroid nodes.
        cmap: Colormap name for cluster colors.
    
    Returns:
        Matplotlib axes object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    nodes, coords = _extract_coordinates(G)
    node_to_cluster = cluster_info['node_to_cluster']
    num_clusters = cluster_info['num_clusters']
    centroids = cluster_info['centroids']
    
    # Create color map
    colormap = cm.get_cmap(cmap)
    colors = [colormap(node_to_cluster[node] / max(1, num_clusters - 1)) 
              for node in nodes]
    
    # Draw edges
    for u, v in G.edges():
        u_idx = nodes.index(u)
        v_idx = nodes.index(v)
        ax.plot(
            [coords[u_idx, 0], coords[v_idx, 0]],
            [coords[u_idx, 1], coords[v_idx, 1]],
            'k-', alpha=0.2, linewidth=0.5
        )
    
    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=node_size, zorder=2)
    
    # Highlight centroids
    if show_centroids:
        for cluster_id, centroid_node in centroids.items():
            idx = nodes.index(centroid_node)
            ax.scatter(
                coords[idx, 0], coords[idx, 1],
                c='black', s=node_size * 2, marker='*', zorder=3,
                edgecolors='white', linewidths=1
            )
    
    # Add labels
    if show_labels:
        for i, node in enumerate(nodes):
            ax.annotate(
                str(node),
                (coords[i, 0], coords[i, 1]),
                fontsize=6, ha='center', va='bottom'
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Network Clusters (k={num_clusters})')
    ax.set_aspect('equal')
    
    return ax
