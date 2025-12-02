"""
Policy prior network for multigrid environments.

This network computes the marginal policy prior by marginalizing over possible
goals. It wraps a Q-network and computes the expected policy under a goal
distribution.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .q_network import MultiGridQNetwork


class MultiGridPolicyPriorNetwork(nn.Module):
    """
    Policy prior network for multigrid environments.
    
    Computes the marginal policy prior h_phi(s, h) by marginalizing over goals:
        h_phi(s, h) = E_g[softmax(beta * Q(s, h, g))]
    
    Args:
        q_network: The Q-network to use for computing goal-specific policies.
        num_actions: Number of possible actions.
    """
    
    def __init__(
        self,
        q_network: MultiGridQNetwork,
        num_actions: int
    ):
        super().__init__()
        self.q_network = q_network
        self.num_actions = num_actions
    
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
        # Goal encoder inputs (can be batched goals for marginalization)
        goal_coords_list: List[torch.Tensor],
        goal_weights: Optional[torch.Tensor],
        # Interactive object encoder inputs
        kill_buttons: torch.Tensor,
        pause_switches: torch.Tensor,
        disabling_switches: torch.Tensor,
        control_buttons: torch.Tensor,
        # Temperature
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Compute marginal policy prior by marginalizing over goals.
        
        Args:
            goal_coords_list: List of goal coordinate tensors, each (batch, 4)
            goal_weights: Optional weights for each goal, shape (num_goals,)
                         If None, uniform weighting is used.
            beta: Inverse temperature for Boltzmann policy
        
        Returns:
            Policy tensor of shape (batch, num_actions) - marginal probabilities
        """
        batch_size = grid_tensor.shape[0]
        device = grid_tensor.device
        num_goals = len(goal_coords_list)
        
        # Initialize weighted policy sum
        marginal_policy = torch.zeros(batch_size, self.num_actions, device=device)
        
        # Use uniform weights if not provided
        if goal_weights is None:
            goal_weights = torch.ones(num_goals, device=device) / num_goals
        
        # Compute policy for each goal and accumulate weighted average
        for i, goal_coords in enumerate(goal_coords_list):
            q_values = self.q_network(
                grid_tensor, global_features,
                query_position, query_direction, query_abilities, query_carried, query_status,
                all_agent_positions, all_agent_directions, all_agent_abilities,
                all_agent_carried, all_agent_status, agent_color_indices,
                goal_coords,
                kill_buttons, pause_switches, disabling_switches, control_buttons
            )
            
            policy = self.q_network.get_policy(q_values, beta)
            marginal_policy += goal_weights[i] * policy
        
        return marginal_policy
