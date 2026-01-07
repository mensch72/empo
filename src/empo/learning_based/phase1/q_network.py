"""
Base Q-network class.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from empo.learning_based.util.soft_clamp import SoftClamp


class BaseQNetwork(nn.Module, ABC):
    """
    Base class for Q-networks.
    
    Q-networks estimate action values Q(s, a, g) for state s, action a, and goal g.
    Contains the generic Boltzmann policy computation.
    
    Args:
        num_actions: Number of possible actions.
        beta: Temperature for Boltzmann policy.
        feasible_range: Optional tuple (a, b) for theoretical Q-value bounds.
            When provided, Q-values are soft-clamped to prevent gradient explosion
            for values outside the expected range.
    """
    
    def __init__(
        self,
        num_actions: int,
        beta: float = 1.0,
        feasible_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__()
        self.num_actions = num_actions
        self.beta = beta
        self.feasible_range = feasible_range
        
        # Create soft clamp module if feasible_range is specified
        self.soft_clamp: Optional[SoftClamp] = None
        if feasible_range is not None:
            self.soft_clamp = SoftClamp(a=feasible_range[0], b=feasible_range[1])
    
    @abstractmethod
    def forward(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Encode state and compute Q-values. Returns (1, num_actions)."""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
    
    def get_policy(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Convert Q-values to action probabilities using Boltzmann policy.
        
        π(a|s,g) = softmax(β * Q(s,a,g))
        
        If feasible_range is set, Q-values are hard-clamped before policy computation
        to ensure numerical stability.
        
        Args:
            q_values: Tensor of shape (..., num_actions)
        
        Returns:
            Tensor of shape (..., num_actions) with action probabilities
        """
        if self.feasible_range is not None:
            q_values = torch.clamp(
                q_values, 
                self.feasible_range[0], 
                self.feasible_range[1]
            )
        
        if self.beta == np.inf:
            # Greedy policy
            policy = torch.zeros_like(q_values)
            max_indices = q_values.argmax(dim=-1, keepdim=True)
            policy.scatter_(-1, max_indices, 1.0)
            return policy
        else:
            q_values = q_values - q_values.max(dim=-1, keepdim=True).values  # For numerical stability
            return torch.softmax(self.beta * q_values, dim=-1)
    
    def apply_soft_clamp(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Apply soft clamping to Q-values during training.
        
        This preserves gradients while bounding Q-values to prevent explosion
        for values far outside the expected range.
        
        Args:
            q_values: Raw Q-values from the network.
        
        Returns:
            Soft-clamped Q-values.
        """
        if self.soft_clamp is not None:
            return self.soft_clamp(q_values)
        return q_values
    
    def get_value(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute state value V(s,g) = Σ_a π(a|s,g) * Q(s,a,g)
        
        Args:
            q_values: Tensor of shape (..., num_actions)
        
        Returns:
            Value tensor of shape (...)
        """
        policy = self.get_policy(q_values)
        return (policy * q_values).sum(dim=-1)
