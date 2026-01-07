"""
Goal encoder for transport environment.

This module provides encoders for transport goals (reaching nodes or clusters).
Goals are encoded into fixed-size feature vectors for use in Q-networks and
policy networks.
"""

import torch
import torch.nn as nn
from typing import Any

from empo.learning_based.phase1.goal_encoder import BaseGoalEncoder
from .constants import MAX_CLUSTERS


class TransportGoalEncoder(BaseGoalEncoder):
    """
    Encoder for transport environment goals.
    
    Supports two types of goals:
    1. Node goals: Agent reaches a specific node
    2. Cluster goals: Agent reaches any node in a cluster
    
    Goals are encoded as:
    - Node ID one-hot (up to max_nodes)
    - Cluster ID one-hot (up to num_clusters)
    - Goal type (is_cluster_goal)
    - Node coordinates (x, y) normalized
    
    Args:
        max_nodes: Maximum number of nodes in network
        num_clusters: Number of clusters (0 if not using clusters)
        feature_dim: Output feature dimension
    
    Example:
        >>> encoder = TransportGoalEncoder(max_nodes=100, num_clusters=10)
        >>> # Encode a node goal
        >>> goal_tensor = encoder.tensorize_goal(goal, device='cpu')
        >>> encoded = encoder(goal_tensor)
        >>> print(encoded.shape)
        torch.Size([1, 32])
    """
    
    def __init__(
        self,
        max_nodes: int = 100,
        num_clusters: int = 0,
        feature_dim: int = 32
    ):
        super().__init__(feature_dim)
        self.max_nodes = max_nodes
        self.num_clusters = num_clusters if num_clusters > 0 else MAX_CLUSTERS
        
        # Input dimension:
        # - node one-hot: max_nodes
        # - cluster one-hot: num_clusters
        # - is_cluster_goal: 1
        # - coordinates (x, y): 2
        input_dim = max_nodes + self.num_clusters + 3
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, goal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode goal tensor (neural network forward pass).
        
        Args:
            goal_tensor: (batch, input_dim) goal input tensor
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        return self.fc(goal_tensor)
    
    def tensorize_goal(
        self,
        goal: Any,
        device: str = 'cpu',
        env: Any = None
    ) -> torch.Tensor:
        """
        Convert goal to input tensor (preprocessing, NOT neural network encoding).
        
        Handles goal formats:
        1. TransportGoal: tensorize target_node
        2. TransportClusterGoal: tensorize target_cluster
        3. int (node ID): tensorize as node goal
        4. tuple (cluster_id, 'cluster'): tensorize as cluster goal
        
        Args:
            goal: Goal object or node/cluster ID
            device: Torch device
            env: Optional TransportEnvWrapper for coordinate lookup
        
        Returns:
            Tensor (1, input_dim) ready for forward()
        """
        input_dim = self.max_nodes + self.num_clusters + 3
        tensor = torch.zeros(1, input_dim, device=device)
        
        # Determine goal type and extract target
        is_cluster_goal = False
        target_node = None
        target_cluster = None
        coords = (0.0, 0.0)
        
        if hasattr(goal, 'target_cluster'):
            # TransportClusterGoal
            is_cluster_goal = True
            target_cluster = goal.target_cluster
        elif hasattr(goal, 'target_node'):
            # TransportGoal
            target_node = goal.target_node
        elif isinstance(goal, tuple) and len(goal) == 2 and goal[1] == 'cluster':
            # (cluster_id, 'cluster') tuple
            is_cluster_goal = True
            target_cluster = goal[0]
        elif isinstance(goal, int):
            # Direct node ID
            target_node = goal
        
        # Encode node one-hot
        if target_node is not None and 0 <= target_node < self.max_nodes:
            tensor[0, target_node] = 1.0
            
            # Get coordinates if env available
            if env is not None:
                network = env.env.network
                if target_node in network.nodes():
                    node_data = network.nodes[target_node]
                    if 'x' in node_data and 'y' in node_data:
                        coords = (node_data['x'], node_data['y'])
                    elif 'pos' in node_data:
                        coords = node_data['pos'][:2]
        
        # Encode cluster one-hot
        offset = self.max_nodes
        if target_cluster is not None and 0 <= target_cluster < self.num_clusters:
            tensor[0, offset + target_cluster] = 1.0
            
            # Get cluster centroid coordinates if env available
            if env is not None and env.cluster_info is not None:
                cluster_center = env.cluster_info['cluster_centers'].get(target_cluster)
                if cluster_center is not None:
                    coords = cluster_center
        
        # Is cluster goal
        offset += self.num_clusters
        tensor[0, offset] = 1.0 if is_cluster_goal else 0.0
        
        # Coordinates (normalized - assuming coordinates are in reasonable range)
        tensor[0, offset + 1] = coords[0] / 100.0  # Normalize x
        tensor[0, offset + 2] = coords[1] / 100.0  # Normalize y
        
        return tensor
    
    def get_config(self) -> dict:
        """Return configuration for save/load."""
        return {
            'max_nodes': self.max_nodes,
            'num_clusters': self.num_clusters,
            'feature_dim': self.feature_dim,
        }
