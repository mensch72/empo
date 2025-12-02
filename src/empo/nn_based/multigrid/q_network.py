"""
Q-network for multigrid environments.

This network combines all multigrid encoders to predict Q-values for actions.
It takes the encoded state, agent, goal, and interactive object features and
outputs Q-values for each possible action.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .state_encoder import MultiGridStateEncoder
from .agent_encoder import MultiGridAgentEncoder
from .goal_encoder import MultiGridGoalEncoder
from .interactive_encoder import MultiGridInteractiveObjectEncoder


class SoftClamp(nn.Module):
    """Soft clamp layer to bound outputs to a range."""
    
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low = low
        self.high = high
        self.range = high - low
        self.mid = (low + high) / 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid squashes to [0, 1], then scale to [low, high]
        return self.low + self.range * torch.sigmoid(x - self.mid + 0.5)


class MultiGridQNetwork(nn.Module):
    """
    Q-network for multigrid environments.
    
    Combines StateEncoder, AgentEncoder, GoalEncoder, and InteractiveObjectEncoder
    to predict Q-values for each action.
    
    Args:
        state_encoder: Grid-based state encoder.
        agent_encoder: List-based agent encoder.
        goal_encoder: Goal position encoder.
        interactive_encoder: Interactive object encoder.
        num_actions: Number of possible actions.
        hidden_dim: Hidden layer dimension.
        feasible_range: Optional (low, high) tuple to clamp Q-values.
    """
    
    def __init__(
        self,
        state_encoder: MultiGridStateEncoder,
        agent_encoder: MultiGridAgentEncoder,
        goal_encoder: MultiGridGoalEncoder,
        interactive_encoder: MultiGridInteractiveObjectEncoder,
        num_actions: int,
        hidden_dim: int = 256,
        feasible_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.agent_encoder = agent_encoder
        self.goal_encoder = goal_encoder
        self.interactive_encoder = interactive_encoder
        self.num_actions = num_actions
        self.feasible_range = feasible_range
        
        # Combined feature dimension
        combined_dim = (
            state_encoder.feature_dim +
            agent_encoder.feature_dim +
            goal_encoder.feature_dim +
            interactive_encoder.feature_dim
        )
        
        # Q-value head
        layers = [
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        ]
        
        if feasible_range is not None:
            layers.append(SoftClamp(*feasible_range))
        
        self.q_head = nn.Sequential(*layers)
    
    def forward(
        self,
        # State encoder inputs
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        # Agent encoder inputs
        query_position: torch.Tensor,
        query_direction: torch.Tensor,
        query_abilities: torch.Tensor,
        query_carried: torch.Tensor,
        query_status: torch.Tensor,
        all_agent_positions: torch.Tensor,
        all_agent_directions: torch.Tensor,
        all_agent_abilities: torch.Tensor,
        all_agent_carried: torch.Tensor,
        all_agent_status: torch.Tensor,
        agent_color_indices: torch.Tensor,
        # Goal encoder inputs
        goal_coords: torch.Tensor,
        # Interactive object encoder inputs
        kill_buttons: torch.Tensor,
        pause_switches: torch.Tensor,
        disabling_switches: torch.Tensor,
        control_buttons: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Returns:
            Q-values tensor of shape (batch, num_actions)
        """
        # Encode state
        state_feat = self.state_encoder(grid_tensor, global_features)
        
        # Encode agents
        agent_feat = self.agent_encoder(
            query_position, query_direction, query_abilities, query_carried, query_status,
            all_agent_positions, all_agent_directions, all_agent_abilities,
            all_agent_carried, all_agent_status, agent_color_indices
        )
        
        # Encode goal
        goal_feat = self.goal_encoder(goal_coords)
        
        # Encode interactive objects
        interactive_feat = self.interactive_encoder(
            kill_buttons, pause_switches, disabling_switches, control_buttons
        )
        
        # Combine features
        combined = torch.cat([state_feat, agent_feat, goal_feat, interactive_feat], dim=1)
        
        # Compute Q-values
        q_values = self.q_head(combined)
        
        return q_values
    
    def get_policy(self, q_values: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute Boltzmann policy from Q-values.
        
        Args:
            q_values: Q-values tensor of shape (batch, num_actions)
            beta: Inverse temperature parameter
        
        Returns:
            Policy tensor of shape (batch, num_actions) - probabilities
        """
        return torch.softmax(beta * q_values, dim=-1)
