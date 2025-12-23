"""
Base Aggregate Goal Achievement Ability Network for Phase 2.

Implements X_h(s) from equation (7) of the EMPO paper:
    X_h(s) ← E_{g_h ~ possible_goal_sampler(h)}[V_h^e(s, g_h)^ζ]

This network estimates the aggregate ability of human h to achieve various goals,
which is then used to compute the robot's intrinsic reward U_r.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseAggregateGoalAbilityNetwork(nn.Module, ABC):
    """
    Base class for X_h networks in Phase 2.
    
    X_h(s) = E_{g_h}[V_h^e(s, g_h)^ζ]
    
    where:
    - V_h^e(s, g_h) is the probability human h achieves goal g_h
    - ζ (zeta) is the risk/reliability preference parameter
    
    Key properties:
    - X_h ∈ (0, 1] since V_h^e ∈ [0, 1] and we take expected value of powers
    - When V_h^e = 1 for some goal, X_h can be close to 1
    - When V_h^e is low for all goals, X_h is close to 0
    - ζ > 1 introduces risk aversion (prefer certain outcomes)
    
    The network directly predicts X_h rather than computing it from V_h^e,
    which would require sampling many goals. This is trained via Monte Carlo
    targets: target_x_h = V_h^e(s, g_h)^ζ for sampled goal g_h.
    
    Args:
        zeta: Risk/reliability preference parameter (ζ ≥ 1, 1 = neutral).
    """
    
    def __init__(self, zeta: float = 2.0):
        super().__init__()
        self.zeta = zeta
        
        if zeta < 1.0:
            raise ValueError(f"zeta must be >= 1.0, got {zeta}")
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute X_h values.
        
        Returns:
            Tensor of shape (batch,) with X_h ∈ (0, 1].
        """
        pass
    
    @abstractmethod
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
            state: Environment state.
            world_model: Environment/world model.
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            Tensor of shape (1,) with X_h(s) ∈ (0, 1].
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
    
    def ensure_positive_bounded(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Ensure X_h values are in (0, 1].
        
        Uses sigmoid to bound to (0, 1), then scales slightly to allow reaching 1.
        
        Args:
            raw_values: Unbounded network output.
        
        Returns:
            Values in (0, 1].
        """
        # Sigmoid gives (0, 1), we use it directly since X_h is an expected
        # value of V_h^e^ζ where V_h^e ∈ [0, 1]
        return torch.sigmoid(raw_values)
    
    def compute_target(self, v_h_e: torch.Tensor) -> torch.Tensor:
        """
        Compute target X_h from V_h^e values (Monte Carlo).
        
        For a sampled goal g_h:
            target_x_h = V_h^e(s, g_h)^ζ
        
        This is an unbiased estimate of E[V_h^e^ζ] when goals are sampled uniformly.
        
        Args:
            v_h_e: V_h^e(s, g_h) value for sampled goal, shape (batch,).
        
        Returns:
            Target X_h values, shape (batch,).
        """
        # V_h^e^ζ
        return v_h_e ** self.zeta
    
    def compute_from_v_h_e_samples(
        self,
        v_h_e_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute X_h from multiple V_h^e samples (for evaluation).
        
        X_h = E[V_h^e^ζ] ≈ mean(V_h^e^ζ)
        
        Args:
            v_h_e_samples: V_h^e values for multiple goals, shape (batch, num_goals).
        
        Returns:
            X_h values, shape (batch,).
        """
        # Compute V_h^e^ζ for each goal
        v_h_e_powered = v_h_e_samples ** self.zeta
        
        # Mean across goals
        return v_h_e_powered.mean(dim=-1)
