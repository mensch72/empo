"""
Multigrid-specific Human Goal Achievement Network for Phase 2.

Implements V_h^e(s, g_h) from equation (6) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from ...phase2.human_goal_ability import BaseHumanGoalAchievementNetwork
from ..state_encoder import MultiGridStateEncoder
from ..goal_encoder import MultiGridGoalEncoder


class MultiGridHumanGoalAchievementNetwork(BaseHumanGoalAchievementNetwork):
    """
    Human Goal Achievement Network for multigrid environments.
    
    Estimates V_h^e(s, g_h) - the probability that human h achieves goal g_h
    under the current robot policy π_r.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dimension.
        goal_feature_dim: Goal encoder output dimension.
        hidden_dim: Hidden layer dimension.
        gamma_h: Human discount factor.
        feasible_range: Output bounds for V_h^e.
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
        goal_feature_dim: int = 64,
        hidden_dim: int = 256,
        gamma_h: float = 0.99,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__(gamma_h=gamma_h, feasible_range=feasible_range)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.goal_feature_dim = goal_feature_dim
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
        
        # Goal encoder
        self.goal_encoder = MultiGridGoalEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=goal_feature_dim
        )
        
        # Value head: combines state + goal features
        combined_dim = state_feature_dim + goal_feature_dim
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
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
        interactive_features: torch.Tensor,
        goal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_h^e(s, g_h).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            goal_features: (batch, goal_feature_dim)
        
        Returns:
            V_h^e values tensor (batch,) in [0, 1].
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Combine state and goal features
        combined = torch.cat([state_features, goal_features], dim=-1)
        
        # Compute raw value
        raw_value = self.value_head(combined).squeeze(-1)
        
        # Apply soft clamp to keep in [0, 1]
        return self.apply_soft_clamp(raw_value)
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and goal, then compute V_h^e.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            goal: The goal g_h for this human.
            device: Torch device.
        
        Returns:
            V_h^e tensor of shape (1,).
        """
        # Encode state using human as query agent
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.encode_state(state, world_model, human_agent_idx, device)
        
        # Encode goal
        goal_features = self.goal_encoder.encode_goal(goal, device)
        
        return self.forward(
            grid_tensor, global_features, agent_features,
            interactive_features, goal_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict V_h^e with hard clamping (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            goal: The goal g_h for this human.
            device: Torch device.
        
        Returns:
            V_h^e tensor with strict [0, 1] bounds.
        """
        with torch.no_grad():
            v_h_e = self.encode_and_forward(
                state, world_model, human_agent_idx, goal, device
            )
            return self.apply_hard_clamp(v_h_e)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'goal_feature_dim': self.goal_feature_dim,
            'hidden_dim': self.hidden_dim,
            'gamma_h': self.gamma_h,
            'feasible_range': self.feasible_range,
            'state_encoder_config': self.state_encoder.get_config(),
            'goal_encoder_config': self.goal_encoder.get_config(),
        }
