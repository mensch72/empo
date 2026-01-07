"""
Base Robot State Value Network for Phase 2.

Implements V_r(s) from equation (9) of the EMPO paper:
    V_r(s) ← U_r(s) + E_{a_r ~ π_r(s)} Q_r(s, a_r)

This is the robot's value function, representing the expected long-term
aggregate human power starting from state s.

When use_z_space=True, the network internally represents values in z-space:
    z = (-V)^{-1/(ηξ)} ∈ (0, 1]
This makes it easier to represent values across orders of magnitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict


def compute_v_r_from_components(
    u_r: torch.Tensor,
    q_r: torch.Tensor,
    pi_r: torch.Tensor
) -> torch.Tensor:
    """
    Compute V_r directly from U_r, Q_r, and π_r (equation 9).
    
    V_r(s) = U_r(s) + Σ_{a_r} π_r(s, a_r) * Q_r(s, a_r)
    
    This is a standalone function that can be used without the V_r network,
    which is useful when v_r_use_network=False.
    
    Args:
        u_r: Intrinsic reward U_r(s), shape (batch,) or scalar.
        q_r: Q-values Q_r(s, a_r), shape (batch, num_actions) or (num_actions,).
        pi_r: Policy probabilities π_r(s), shape (batch, num_actions) or (num_actions,).
    
    Returns:
        V_r values, shape (batch,) or scalar.
    """
    # Expected Q under policy
    expected_q = (pi_r * q_r).sum(dim=-1)
    
    # V_r = U_r + E[Q_r]
    return u_r + expected_q


class BaseRobotValueNetwork(nn.Module, ABC):
    """
    Base class for V_r networks in Phase 2.
    
    V_r(s) = U_r(s) + E_{a_r ~ π_r}[Q_r(s, a_r)]
    
    where:
    - U_r(s) is the intrinsic robot reward (aggregate human power)
    - Q_r(s, a_r) is the discounted future value from taking action a_r
    
    Key properties:
    - V_r < 0 always (since U_r < 0 and Q_r < 0)
    - This can be computed from U_r and Q_r directly, or learned separately
    
    When use_z_space=True:
    - Network internally stores z = (-V)^{-1/(ηξ)} ∈ (0, 1]
    - forward() returns V-values (converted from z)
    
    Args:
        gamma_r: Robot discount factor.
        use_z_space: If True, use z-space representation internally.
        eta: η parameter for z-space transformation (default 1.1).
        xi: ξ parameter for z-space transformation (default 1.0).
    """
    
    def __init__(
        self,
        gamma_r: float = 0.99,
        use_z_space: bool = False,
        eta: float = 1.1,
        xi: float = 1.0
    ):
        super().__init__()
        self.gamma_r = gamma_r
        self.use_z_space = use_z_space
        self.eta = eta
        self.xi = xi
    
    @abstractmethod
    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute V_r.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            device: Torch device.
        
        Returns:
            Tensor of shape (1,) with V_r(s) < 0.
        """
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
    
    def raw_to_z(self, raw_values: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """Convert raw network output to z ∈ (0, 1)."""
        return torch.sigmoid(raw_values).clamp(min=eps, max=1.0 - eps)
    
    def z_to_v(self, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert z-values to V-values: V = -z^{-ηξ}."""
        z_clamped = z.clamp(min=eps)
        exponent = -self.eta * self.xi
        return -torch.pow(z_clamped, exponent)
    
    def v_to_z(self, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert V-values to z-values: z = (-V)^{-1/(ηξ)}."""
        v_clamped = v.clamp(max=-eps)
        exponent = -1.0 / (self.eta * self.xi)
        return torch.pow(-v_clamped, exponent)
    
    def ensure_negative(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Ensure V_r values are negative.
        
        When use_z_space=True, converts raw → z → V.
        Otherwise, uses -softplus(-x) which maps R -> (-∞, 0).
        
        Args:
            raw_values: Unbounded network output.
        
        Returns:
            Negative values.
        """
        if self.use_z_space:
            z = self.raw_to_z(raw_values)
            return self.z_to_v(z)
        else:
            # -softplus(-x) maps R -> (-∞, 0)
            # When x is large positive, -softplus(-x) ≈ x (large negative)
            # When x is large negative, -softplus(-x) ≈ 0 (approaches 0 from below)
            return -F.softplus(-raw_values)
    
    def compute_from_components(
        self,
        u_r: torch.Tensor,
        q_r: torch.Tensor,
        pi_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_r directly from U_r, Q_r, and π_r (equation 9).
        
        V_r(s) = U_r(s) + Σ_{a_r} π_r(s, a_r) * Q_r(s, a_r)
        
        Args:
            u_r: Intrinsic reward U_r(s), shape (batch,).
            q_r: Q-values Q_r(s, a_r), shape (batch, num_actions).
            pi_r: Policy probabilities π_r(s), shape (batch, num_actions).
        
        Returns:
            V_r values, shape (batch,).
        """
        # Expected Q under policy
        expected_q = (pi_r * q_r).sum(dim=-1)
        
        # V_r = U_r + E[Q_r]
        return u_r + expected_q
    
    def compute_td_target(
        self,
        u_r: torch.Tensor,
        next_v_r: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TD target for V_r.
        
        V_r(s) = U_r(s) + γ_r * E[V_r(s')]
        
        Args:
            u_r: Current intrinsic reward U_r(s).
            next_v_r: V_r(s') from target network.
        
        Returns:
            TD target values.
        """
        return u_r + self.gamma_r * next_v_r
