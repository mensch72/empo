"""
Multigrid-specific Robot Value Network for Phase 2.

Implements V_r(s) from equation (9) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ...phase2.robot_value_network import BaseRobotValueNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridRobotValueNetwork(BaseRobotValueNetwork):
    """
    Robot Value Network for multigrid environments.
    
    Estimates V_r(s) = U_r(s) + E_{a_r ~ π_r}[Q_r(s, a_r)] - the robot's
    value function representing expected long-term aggregate human power.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dimension.
        hidden_dim: Hidden layer dimension.
        gamma_r: Robot discount factor.
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
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        gamma_r: float = 0.99,
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        state_encoder: Optional[MultiGridStateEncoder] = None
    ):
        super().__init__(gamma_r=gamma_r)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Use shared state encoder or create own
        if state_encoder is not None:
            self.state_encoder = state_encoder
        else:
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
        
        # V_r value head with optional dropout
        # Use actual encoder feature_dim (may differ from state_feature_dim when use_encoders=False)
        actual_state_dim = self.state_encoder.feature_dim
        if dropout > 0.0:
            self.value_head = nn.Sequential(
                nn.Linear(actual_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(actual_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_r(s).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            V_r values tensor (batch,) with V_r < 0.
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Compute raw value
        raw_value = self.value_head(state_features).squeeze(-1)
        
        # Ensure V_r < 0
        return self.ensure_negative(raw_value)
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute V_r.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            V_r tensor of shape (1,).
        """
        # Encode state (agent-agnostic)
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.tensorize_state(state, world_model, device)
        
        return self.forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
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
            V_r values tensor (batch,) with V_r < 0.
        """
        return self.forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict V_r (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            V_r tensor of shape (1,).
        """
        with torch.no_grad():
            return self.encode_and_forward(state, world_model, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'hidden_dim': self.hidden_dim,
            'gamma_r': self.gamma_r,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
        }
