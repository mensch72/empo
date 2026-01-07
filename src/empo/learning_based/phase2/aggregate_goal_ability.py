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

from empo.learning_based.util.soft_clamp import SoftClamp


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
        feasible_range: Tuple (a, b) for soft clamping bounds, default (0, 1).
    """
    
    def __init__(
        self,
        zeta: float = 2.0,
        feasible_range: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        self.zeta = zeta
        self.feasible_range = feasible_range
        
        if zeta < 1.0:
            raise ValueError(f"zeta must be >= 1.0, got {zeta}")
        
        # Use SoftClamp for bounding to (0, 1] - maintains gradients unlike sigmoid
        self.soft_clamp = SoftClamp(a=feasible_range[0], b=feasible_range[1])
    
    @abstractmethod
    def forward(
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
    
    def apply_clamp(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Apply clamping based on training mode.
        
        During training (self.training=True): Uses soft clamp which is linear
        in [a, b] and exponential outside, preserving gradients.
        
        During eval (self.training=False): Uses hard clamp to ensure values
        are strictly within bounds. This is important for target networks
        which are always in eval mode.
        
        Args:
            raw_values: Unbounded network output.
        
        Returns:
            Clamped values.
        """
        if self.training:
            return self.soft_clamp(raw_values)
        else:
            return self.apply_hard_clamp(raw_values)
    
    def apply_hard_clamp(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply hard clamping during prediction/inference.
        
        Hard clamp ensures values are strictly within bounds. Use this
        during prediction when gradients are not needed.
        
        Args:
            values: Values to clamp (typically soft-clamped during forward).
        
        Returns:
            Hard-clamped values in [a, b].
        """
        return torch.clamp(
            values,
            self.feasible_range[0],
            self.feasible_range[1]
        )
    
    def compute_target(
        self,
        v_h_e: torch.Tensor,
        goal_weight: "float | torch.Tensor" = 1.0
    ) -> torch.Tensor:
        """
        Compute target X_h from V_h^e value and goal weight.
        
        For a sampled goal g_h with weight w_h:
            target_x_h = w_h * V_h^e(s, g_h)^ζ
        
        Args:
            v_h_e: V_h^e(s, g_h) value for sampled goal, shape (batch,) or scalar.
            goal_weight: Weight for this goal sample, scalar or tensor matching v_h_e shape.
        
        Returns:
            Target X_h values, shape (batch,) or scalar.
        """
        return goal_weight * (v_h_e ** self.zeta)
    
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
