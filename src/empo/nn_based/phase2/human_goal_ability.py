"""
Base Human Goal Achievement Ability Network for Phase 2.

Implements V_h^e(s, g_h) from equation (6) of the EMPO paper:
    V_h^e(s, g_h) ← E_{g_{-h}} E_{a_H ~ π_H(s,g)} E_{a_r ~ π_r(s)} E_{s'|s,a} [U_h(s', g_h) + γ_h V_h^e(s', g_h)]

This network estimates the probability that human h achieves goal g_h
under the actual robot policy π_r (not the worst-case assumption used in Phase 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ..soft_clamp import SoftClamp


class BaseHumanGoalAchievementNetwork(nn.Module, ABC):
    """
    Base class for V_h^e networks in Phase 2.
    
    V_h^e(s, g_h) estimates the (discounted) probability that human h achieves
    goal g_h, given:
    - The current robot policy π_r
    - Human policies π_H derived from Phase 1
    - The current state s
    
    Key properties:
    - V_h^e ∈ [0, 1] since it's a discounted probability
    - Different from Phase 1's V_h^m which used worst-case robot assumption
    - Depends on the robot policy, creating a mutual dependency that requires
      joint training of V_h^e and π_r
    
    Args:
        gamma_h: Human discount factor.
        feasible_range: Output bounds, defaults to (0, 1) for probabilities.
    """
    
    def __init__(
        self,
        gamma_h: float = 0.99,
        feasible_range: Tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__()
        self.gamma_h = gamma_h
        self.feasible_range = feasible_range
        
        # Soft clamp to keep values in [0, 1]
        self.soft_clamp = SoftClamp(a=feasible_range[0], b=feasible_range[1])
    
    @abstractmethod
    def forward(
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
            state: Environment state.
            world_model: Environment/world model.
            human_agent_idx: Index of the human agent.
            goal: The goal g_h for this human.
            device: Torch device.
        
        Returns:
            Tensor of shape (1,) with V_h^e(s, g_h) in [0, 1].
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
    
    def apply_clamp(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply clamping based on training mode.
        
        During training (self.training=True): Uses soft clamp which preserves
        gradients while bounding values.
        
        During eval (self.training=False): Uses hard clamp to ensure values
        are strictly within [0, 1]. This is important for target networks
        which are always in eval mode.
        
        Args:
            values: Raw network output.
        
        Returns:
            Clamped values in feasible_range.
        """
        if self.training:
            return self.soft_clamp(values)
        else:
            return self.apply_hard_clamp(values)
    
    def apply_hard_clamp(self, values: torch.Tensor) -> torch.Tensor:
        """
        Apply hard clamping during prediction/inference.
        
        Hard clamp ensures values are strictly within bounds. Use during
        prediction when gradients are not needed.
        
        Args:
            values: Values to clamp.
        
        Returns:
            Hard-clamped values in [a, b].
        """
        return torch.clamp(
            values,
            self.feasible_range[0],
            self.feasible_range[1]
        )
    
    def compute_td_target(
        self,
        goal_achieved: torch.Tensor,
        next_v_h_e: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD target for V_h^e using truncated Bellman equation.
        
        From equation (6), using the truncated form (like V_h^m in Phase 1):
            - If goal achieved: target = 1 (no continuation)
            - If goal not achieved: target = γ_h * V_h^e(s', g_h)
        
        This is truncated because when U_h(s', g_h) = 1 (goal achieved),
        the episode ends for that human and we don't add future value.
        Only when reward is 0 do we add the discounted continuation.
        
        Args:
            goal_achieved: Boolean/float tensor (1.0 if achieved, 0.0 otherwise).
            next_v_h_e: V_h^e(s', g_h) from target network.
        
        Returns:
            TD target values.
        """
        # Truncated Bellman: only add future value when reward is 0
        # If goal achieved (reward=1): target = 1 (episode ends, no continuation)
        # If goal not achieved (reward=0): target = γ_h * V_h^e(s')
        reward = goal_achieved.float()
        continuation = (1.0 - reward) * self.gamma_h * next_v_h_e
        
        return reward + continuation
