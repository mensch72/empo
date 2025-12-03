"""
Base Q-network class.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseQNetwork(nn.Module, ABC):
    """
    Base class for Q-networks.
    
    Q-networks estimate action values Q(s, a, g) for state s, action a, and goal g.
    Contains the generic Boltzmann policy computation.
    """
    
    def __init__(self, num_actions: int, beta: float = 1.0):
        super().__init__()
        self.num_actions = num_actions
        self.beta = beta
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute Q-values for all actions. Returns (batch, num_actions)."""
        pass
    
    @abstractmethod
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Encode state and compute Q-values. Returns (1, num_actions)."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
    
    def get_policy(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Convert Q-values to action probabilities using Boltzmann policy.
        
        π(a|s,g) = softmax(β * Q(s,a,g))
        
        Args:
            q_values: Tensor of shape (..., num_actions)
        
        Returns:
            Tensor of shape (..., num_actions) with action probabilities
        """
        if self.beta == np.inf:
            # Greedy policy
            policy = torch.zeros_like(q_values)
            max_indices = q_values.argmax(dim=-1, keepdim=True)
            policy.scatter_(-1, max_indices, 1.0)
            return policy
        else:
            q_values -= q_values.max(dim=-1, keepdim=True).values  # For numerical stability
            return torch.softmax(self.beta * q_values, dim=-1)
    
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
