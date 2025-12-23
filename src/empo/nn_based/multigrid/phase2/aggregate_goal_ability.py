"""
Multigrid-specific Aggregate Goal Ability Network for Phase 2.

Implements X_h(s) from equation (7) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ...phase2.aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """
    Aggregate Goal Ability Network for multigrid environments.
    
    Estimates X_h(s) = E_{g_h}[V_h^e(s, g_h)^ζ] - the aggregate ability
    of human h to achieve various goals.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dimension.
        hidden_dim: Hidden layer dimension.
        zeta: Risk/reliability preference parameter.
        feasible_range: Output bounds for X_h.
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
        zeta: float = 2.0,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        
        # State encoder
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
        
        # X_h value head
        # Note: X_h depends on both state and human identity. For Phase 2,
        # we encode the state from the human's perspective (using human_agent_idx as query).
        self.value_head = nn.Sequential(
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
    ) -> torch.Tensor:
        """
        Compute X_h(s).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            X_h values tensor (batch,) in (0, 1].
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Compute raw value
        raw_value = self.value_head(state_features).squeeze(-1)
        
        # Apply soft clamp to keep in (0, 1]
        return self.apply_soft_clamp(raw_value)
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute X_h for a specific human.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h tensor of shape (1,).
        """
        # Encode state from human's perspective
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.encode_state(state, world_model, human_agent_idx, device)
        
        return self.forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict X_h with hard clamping (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h tensor with strict (0, 1] bounds.
        """
        with torch.no_grad():
            x_h = self.encode_and_forward(state, world_model, human_agent_idx, device)
            return self.apply_hard_clamp(x_h)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'hidden_dim': self.hidden_dim,
            'zeta': self.zeta,
            'feasible_range': self.feasible_range,
            'state_encoder_config': self.state_encoder.get_config(),
        }
