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
from typing import Any, Dict, Optional

from .state_encoder import TransportStateEncoder
from .goal_encoder import TransportGoalEncoder
from .constants import NUM_TRANSPORT_ACTIONS, DEFAULT_OUTPUT_FEATURE_DIM


class TransportQNetwork(nn.Module):
    """
    Q-Network for transport environment using GNN-based state encoding.
    
    Architecture:
    1. State encoder (GNN) -> state features
    2. Goal encoder (MLP) -> goal features
    3. Combined MLP -> Q-values for all actions
    
    Supports action masking for invalid actions based on step type and
    agent state.
    
    Args:
        state_encoder: TransportStateEncoder instance
        goal_encoder: TransportGoalEncoder instance
        num_actions: Number of actions in action space
        hidden_dim: Hidden dimension for Q-value MLP
    
    Example:
        >>> state_encoder = TransportStateEncoder(num_clusters=10)
        >>> goal_encoder = TransportGoalEncoder(max_nodes=100, num_clusters=10)
        >>> q_network = TransportQNetwork(state_encoder, goal_encoder)
        >>> 
        >>> # Get Q-values
        >>> graph_data = observation_to_graph_data(env, query_agent_idx=0)
        >>> goal_tensor = goal_encoder.encode_goal(goal)
        >>> q_values = q_network(graph_data, goal_tensor)
        >>> print(q_values.shape)
        torch.Size([1, 42])
    """
    
    def __init__(
        self,
        state_encoder: TransportStateEncoder,
        goal_encoder: TransportGoalEncoder,
        num_actions: int = NUM_TRANSPORT_ACTIONS,
        hidden_dim: int = DEFAULT_OUTPUT_FEATURE_DIM
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.num_actions = num_actions
        
        # Combine state and goal features
        combined_dim = state_encoder.feature_dim + goal_encoder.feature_dim
        
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Args:
            graph_data: Dictionary from observation_to_graph_data()
            goal_tensor: Encoded goal tensor from goal_encoder.encode_goal()
            action_mask: Optional (batch, num_actions) boolean mask, True = valid
            return_features: If True, also return state and goal features
        
        Returns:
            Q-values tensor (batch, num_actions)
            If return_features: tuple of (q_values, state_features, goal_features)
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
        
        # Apply action mask if provided (set invalid actions to very negative)
        if action_mask is not None:
            # Convert mask to float and apply
            mask_float = action_mask.float()
            q_values = q_values * mask_float + (1 - mask_float) * (-1e9)
        
        if return_features:
            return q_values, state_features, goal_features
        return q_values
    
    def get_q_value(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensor: torch.Tensor,
        action: int
    ) -> torch.Tensor:
        """
        Get Q-value for a specific action.
        
        Args:
            graph_data: Dictionary from observation_to_graph_data()
            goal_tensor: Encoded goal tensor
            action: Action index
        
        Returns:
            Q-value for the specified action
        """
        q_values = self.forward(graph_data, goal_tensor)
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
            q_values = self.forward(
                graph_data, goal_tensor,
                action_mask.unsqueeze(0) if action_mask is not None else None
            )
            return q_values.argmax(dim=-1).item()
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'state_encoder_config': self.state_encoder.get_config(),
            'goal_encoder_config': self.goal_encoder.get_config(),
            'num_actions': self.num_actions,
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
        )
