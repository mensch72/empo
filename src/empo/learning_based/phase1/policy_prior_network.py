"""
Base policy prior network class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

from .q_network import BaseQNetwork


class BasePolicyPriorNetwork(nn.Module, ABC):
    """
    Base class for policy prior networks.
    
    Computes marginal action probabilities by averaging over goals:
    π(a|s) = E_g[π(a|s,g)] = Σ_g P(g) * π(a|s,g)
    
    Contains generic marginalization logic.
    """
    
    def __init__(self, q_network: BaseQNetwork):
        super().__init__()
        self.q_network = q_network
        self.num_actions = q_network.num_actions
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute marginal action probabilities."""
    
    def compute_marginal_from_policies(
        self,
        goal_policies: torch.Tensor,
        goal_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute marginal policy by weighted average over goal-specific policies.
        
        Args:
            goal_policies: (batch, num_goals, num_actions) policies for each goal
            goal_weights: Optional (batch, num_goals) goal probabilities.
                         If None, uniform weights are used.
        
        Returns:
            Marginal policy (batch, num_actions)
        """
        batch_size, num_goals, num_actions = goal_policies.shape
        device = goal_policies.device
        
        if goal_weights is None:
            goal_weights = torch.ones(batch_size, num_goals, device=device) / num_goals
        
        # Weighted average: (batch, num_actions)
        weights = goal_weights.unsqueeze(-1)  # (batch, num_goals, 1)
        marginal = (goal_policies * weights).sum(dim=1)
        
        return marginal
    
    def compute_marginal_from_q_values(
        self,
        goal_q_values: torch.Tensor,
        goal_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute marginal policy from Q-values for multiple goals.
        
        Args:
            goal_q_values: (batch, num_goals, num_actions) Q-values for each goal
            goal_weights: Optional (batch, num_goals) goal probabilities.
        
        Returns:
            Marginal policy (batch, num_actions)
        """
        # Convert Q-values to policies
        goal_policies = self.q_network.get_policy(goal_q_values)
        return self.compute_marginal_from_policies(goal_policies, goal_weights)
