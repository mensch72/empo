"""
Multigrid-specific Robot Q-Network for Phase 2.

Implements Q_r(s, a_r) from equation (4) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ...phase2.robot_q_network import BaseRobotQNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridRobotQNetwork(BaseRobotQNetwork):
    """
    Robot Q-Network for multigrid environments.
    
    Uses MultiGridStateEncoder for state encoding, then predicts Q-values
    for the joint robot action space.
    
    For a robot fleet of K robots, each with A actions:
    - Total joint actions = A^K
    - Joint action a_r = (a_{r_1}, ..., a_{r_K}) is a tuple
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_robot_actions: Number of actions per individual robot.
        num_robots: Number of robots in the fleet.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dim.
        hidden_dim: Hidden layer dimension.
        beta_r: Power-law policy exponent.
        feasible_range: Optional Q-value bounds (typically negative).
        dropout: Dropout rate for hidden layers (0 = no dropout).
        max_kill_buttons: Max KillButtons.
        max_pause_switches: Max PauseSwitches.
        max_disabling_switches: Max DisablingSwitches.
        max_control_buttons: Max ControlButtons.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_robot_actions: int,
        num_robots: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        beta_r: float = 10.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__(
            num_actions=num_robot_actions,
            num_robots=num_robots,
            beta_r=beta_r,
            feasible_range=feasible_range
        )
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # State encoder (reuse existing multigrid encoder)
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
        
        # Q-value head for joint actions with optional dropout
        if dropout > 0.0:
            self.q_head = nn.Sequential(
                nn.Linear(state_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_action_combinations),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(state_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_action_combinations),
            )
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q_r(s, a_r) for all joint robot actions.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Q-values tensor (batch, num_action_combinations) with Q_r < 0.
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Compute raw Q-values
        raw_q = self.q_head(state_features)
        
        # Ensure Q_r < 0
        q_values = self.ensure_negative(raw_q)
        
        return q_values
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute Q-values.
        
        For Phase 2, we don't need a specific query agent - we encode
        the full state from a neutral perspective (using robot 0 as query).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Q-values tensor (1, num_action_combinations).
        """
        # Use first robot as query agent for encoding
        # (state encoding doesn't depend much on query agent for Q_r)
        query_agent_idx = 0
        if hasattr(world_model, 'robot_indices') and world_model.robot_indices:
            query_agent_idx = world_model.robot_indices[0]
        
        # Encode state
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.encode_state(state, world_model, query_agent_idx, device)
        
        return self.forward(grid_tensor, global_features, agent_features, interactive_features)
    
    def forward_from_encoded(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with pre-encoded state features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Q-values tensor (batch, num_action_combinations) with Q_r < 0.
        """
        return self.forward(grid_tensor, global_features, agent_features, interactive_features)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'num_robot_actions': self.num_actions,
            'num_robots': self.num_robots,
            'hidden_dim': self.hidden_dim,
            'beta_r': self.beta_r,
            'feasible_range': self.feasible_range,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
        }
