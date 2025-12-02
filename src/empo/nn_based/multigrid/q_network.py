"""
Q-Network for multigrid environments.

Combines state encoder and goal encoder to predict action Q-values.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from ..q_network import BaseQNetwork
from .state_encoder_unified import MultiGridStateEncoder
from .goal_encoder import MultiGridGoalEncoder


class MultiGridQNetwork(BaseQNetwork):
    """
    Q-Network for multigrid environments.
    
    Uses unified MultiGridStateEncoder for complete state encoding
    and MultiGridGoalEncoder for goal encoding.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_actions: Number of possible actions.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dim.
        goal_feature_dim: Goal encoder output dim.
        hidden_dim: Hidden layer dimension.
        beta: Temperature for Boltzmann policy.
        max_kill_buttons: Max KillButtons.
        max_pause_switches: Max PauseSwitches.
        max_disabling_switches: Max DisablingSwitches.
        max_control_buttons: Max ControlButtons.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_actions: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        goal_feature_dim: int = 32,
        hidden_dim: int = 256,
        beta: float = 1.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__(num_actions, beta)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.hidden_dim = hidden_dim
        
        # Unified state encoder
        self.state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=num_agent_colors,
            feature_dim=state_feature_dim,
            max_kill_buttons=max_kill_buttons,
            max_pause_switches=max_pause_switches,
            max_disabling_switches=max_disabling_switches,
            max_control_buttons=max_control_buttons
        )
        
        # Goal encoder (separate from state - not part of world state)
        self.goal_encoder = MultiGridGoalEncoder(
            grid_height, grid_width, goal_feature_dim
        )
        
        # Combined feature dimension
        combined_dim = state_feature_dim + goal_feature_dim
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor,
        goal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values.
        
        Args:
            grid_tensor: (batch, channels, H, W)
            global_features: (batch, 4)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            goal_coords: (batch, 2)
        
        Returns:
            Q-values (batch, num_actions)
        """
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        goal_emb = self.goal_encoder(goal_coords)
        
        combined = torch.cat([state_features, goal_emb], dim=1)
        return self.q_head(combined)
    
    def encode_and_forward(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute Q-values.
        
        Args:
            state: Environment state tuple.
            world_model: Environment.
            query_agent_idx: Index of query agent.
            goal: Goal object.
            device: Torch device.
        
        Returns:
            Q-values (1, num_actions)
        """
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.encode_state(state, world_model, query_agent_idx, device)
        goal_coords = self.goal_encoder.encode_goal(goal, device)
        
        return self.forward(
            grid_tensor, global_features, agent_features,
            interactive_features, goal_coords
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        config = self.state_encoder.get_config()
        config.update({
            'num_actions': self.num_actions,
            'state_feature_dim': self.state_encoder.feature_dim,
            'goal_feature_dim': self.goal_encoder.feature_dim,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta,
        })
        return config
