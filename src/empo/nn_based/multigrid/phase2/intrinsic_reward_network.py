"""
Multigrid-specific Intrinsic Reward Network for Phase 2.

Implements U_r(s) from equation (8) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ...phase2.intrinsic_reward_network import BaseIntrinsicRewardNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridIntrinsicRewardNetwork(BaseIntrinsicRewardNetwork):
    """
    Intrinsic Reward Network for multigrid environments.
    
    Estimates U_r(s) = -(E_h[X_h(s)^{-ξ}])^η - the robot's intrinsic reward
    based on aggregate human power.
    
    Network predicts log(y-1) where y = E_h[X_h^{-ξ}], then computes U_r = -y^η.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dimension.
        hidden_dim: Hidden layer dimension.
        xi: Inter-human inequality aversion parameter.
        eta: Intertemporal inequality aversion parameter.
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
        xi: float = 1.0,
        eta: float = 1.1,
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        state_encoder: Optional[MultiGridStateEncoder] = None
    ):
        super().__init__(xi=xi, eta=eta)
        
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
        
        # Network predicts log(y-1) for numerical stability with optional dropout
        # y = 1 + exp(log(y-1)) ensures y > 1
        if dropout > 0.0:
            self.y_head = nn.Sequential(
                nn.Linear(state_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.y_head = nn.Sequential(
                nn.Linear(state_feature_dim, hidden_dim),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute y and U_r(s).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Tuple (y, U_r) where:
            - y: intermediate value (batch,), y > 1
            - U_r: intrinsic reward (batch,), U_r < 0
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Predict log(y-1)
        log_y_minus_1 = self.y_head(state_features).squeeze(-1)
        
        # Convert to y
        y = self.log_y_minus_1_to_y(log_y_minus_1)
        
        # Compute U_r = -y^η
        u_r = self.y_to_u_r(y)
        
        return y, u_r
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state and compute y and U_r.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Tuple (y, U_r), each of shape (1,).
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with pre-encoded state features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Tuple (y, U_r) where:
            - y: intermediate value (batch,), y > 1
            - U_r: intrinsic reward (batch,), U_r < 0
        """
        return self.forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict y and U_r (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Tuple (y, U_r), each of shape (1,).
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
            'xi': self.xi,
            'eta': self.eta,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
        }
