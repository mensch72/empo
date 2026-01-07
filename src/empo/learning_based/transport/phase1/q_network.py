"""
Q-Network for transport environment.

This module provides a Q-network that combines the GNN-based state encoder
and goal encoder for learning action-value functions in the transport
environment.

The Q-network outputs Q-values for all actions, which can be masked based
on the action mask from the environment.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from empo.learning_based.phase1.q_network import BaseQNetwork
from .state_encoder import TransportStateEncoder
from .goal_encoder import TransportGoalEncoder
from .constants import NUM_TRANSPORT_ACTIONS, DEFAULT_OUTPUT_FEATURE_DIM


class TransportQNetwork(BaseQNetwork):
    """
    Q-Network for transport environment using GNN-based state encoding.
    
    Extends BaseQNetwork with transport-specific GNN encoding.
    
    Architecture:
    1. State encoder (GNN) -> state features
    2. Goal encoder (MLP) -> goal features
    3. Combined MLP -> Q-values for all actions
    
    Supports action masking for invalid actions based on step type and
    agent state.
    
    Args:
        state_encoder: TransportStateEncoder instance (or None to create default)
        goal_encoder: TransportGoalEncoder instance (or None to create default)
        num_actions: Number of actions in action space
        hidden_dim: Hidden dimension for Q-value MLP
        beta: Temperature for Boltzmann policy
        feasible_range: Optional tuple (a, b) for Q-value bounds
        max_nodes: Max nodes (used if state_encoder is None)
        num_clusters: Number of clusters (used if encoders are None)
        state_feature_dim: State encoder output dim (used if state_encoder is None)
        goal_feature_dim: Goal encoder output dim (used if goal_encoder is None)
    
    Example:
        >>> q_network = TransportQNetwork(
        ...     max_nodes=100, num_clusters=10, num_actions=42
        ... )
        >>> graph_data = observation_to_graph_data(env, query_agent_idx=0)
        >>> goal_tensor = q_network.goal_encoder.tensorize_goal(goal)
        >>> q_values = q_network.forward_graph(graph_data, goal_tensor)
    """
    
    def __init__(
        self,
        state_encoder: Optional[TransportStateEncoder] = None,
        goal_encoder: Optional[TransportGoalEncoder] = None,
        num_actions: int = NUM_TRANSPORT_ACTIONS,
        hidden_dim: int = DEFAULT_OUTPUT_FEATURE_DIM,
        beta: float = 1.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        # Parameters for creating default encoders
        max_nodes: int = 100,
        num_clusters: int = 0,
        state_feature_dim: int = 128,
        goal_feature_dim: int = 32,
        num_gnn_layers: int = 3,
        gnn_type: str = 'gcn',
    ):
        super().__init__(num_actions, beta, feasible_range)
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self._num_clusters = num_clusters
        
        # Create encoders if not provided
        if state_encoder is None:
            state_encoder = TransportStateEncoder(
                num_clusters=num_clusters,
                max_nodes=max_nodes,
                feature_dim=state_feature_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                gnn_type=gnn_type,
            )
        
        if goal_encoder is None:
            goal_encoder = TransportGoalEncoder(
                max_nodes=max_nodes,
                num_clusters=num_clusters,
                feature_dim=goal_feature_dim,
            )
        
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        
        # Combine state and goal features
        combined_dim = state_encoder.feature_dim + goal_encoder.feature_dim
        
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def _network_forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Internal: Compute Q-values from pre-encoded tensors.
        
        Args:
            graph_data: Dictionary from observation_to_graph_data()
            goal_tensor: Encoded goal tensor from goal_encoder.tensorize_goal()
            action_mask: Optional (batch, num_actions) boolean mask, True = valid
        
        Returns:
            Q-values tensor (batch, num_actions), soft-clamped if feasible_range is set
        """
        # Encode state
        state_features = self.state_encoder(
            graph_data['node_features'],
            graph_data['edge_index'],
            graph_data['edge_features'],
            graph_data['global_features'],
            graph_data['agent_features'],
            graph_data.get('query_node_idx'),
        )
        
        # Encode goal
        goal_features = self.goal_encoder(goal_tensor)
        
        # Combine features
        combined = torch.cat([state_features, goal_features], dim=-1)
        
        # Compute Q-values
        q_values = self.q_head(combined)
        
        # Apply soft clamping if feasible_range is set
        q_values = self.apply_soft_clamp(q_values)
        
        # Apply action mask if provided (set invalid actions to very negative)
        if action_mask is not None:
            mask_float = action_mask.float()
            q_values = q_values * mask_float + (1 - mask_float) * (-1e9)
        
        return q_values
    
    def forward(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute Q-values.
        
        Args:
            state: Ignored (transport env doesn't have separate state)
            world_model: TransportEnvWrapper instance
            query_agent_idx: Index of the agent making decisions
            goal: Goal object
            device: Torch device
        
        Returns:
            Q-values (1, num_actions)
        """
        from .feature_extraction import observation_to_graph_data
        
        graph_data = observation_to_graph_data(world_model, query_agent_idx, device)
        goal_tensor = self.goal_encoder.tensorize_goal(goal, device, env=world_model)
        
        return self._network_forward(graph_data, goal_tensor)
    
    def get_q_value(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensor: torch.Tensor,
        action: int
    ) -> torch.Tensor:
        """Get Q-value for a specific action."""
        q_values = self._network_forward(graph_data, goal_tensor)
        return q_values[:, action]
    
    def select_action(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            graph_data: Dictionary from observation_to_graph_data()
            goal_tensor: Encoded goal tensor
            action_mask: Optional (num_actions,) boolean mask
            epsilon: Exploration probability
        
        Returns:
            Selected action index
        """
        import random
        
        if random.random() < epsilon:
            # Random action (from valid actions if mask provided)
            if action_mask is not None:
                valid_actions = torch.where(action_mask)[0].tolist()
                if valid_actions:
                    return random.choice(valid_actions)
            return random.randint(0, self.num_actions - 1)
        
        # Greedy action
        with torch.no_grad():
            q_values = self._network_forward(
                graph_data, goal_tensor,
                action_mask.unsqueeze(0) if action_mask is not None else None
            )
            return q_values.argmax(dim=-1).item()
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'max_nodes': self.max_nodes,
            'num_clusters': self._num_clusters,
            'num_actions': self.num_actions,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta,
            'feasible_range': self.feasible_range,
            'state_encoder_config': self.state_encoder.get_config(),
            'goal_encoder_config': self.goal_encoder.get_config(),
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TransportQNetwork':
        """Create Q-network from configuration."""
        state_encoder = TransportStateEncoder(**config['state_encoder_config'])
        goal_encoder = TransportGoalEncoder(**config['goal_encoder_config'])
        return cls(
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            num_actions=config['num_actions'],
            hidden_dim=config.get('hidden_dim', DEFAULT_OUTPUT_FEATURE_DIM),
            beta=config.get('beta', 1.0),
            feasible_range=config.get('feasible_range'),
        )
