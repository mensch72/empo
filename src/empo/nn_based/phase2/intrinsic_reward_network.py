"""
Base Intrinsic Robot Reward Network for Phase 2.

Implements U_r(s) from equation (8) of the EMPO paper:
    U_r(s) ← -(E_h[X_h(s)^{-ξ}])^η

This is the robot's intrinsic reward based on aggregate human power.
It implements inequality aversion across humans and time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseIntrinsicRewardNetwork(nn.Module, ABC):
    """
    Base class for U_r networks in Phase 2.
    
    U_r(s) = -(E_h[X_h(s)^{-ξ}])^η
    
    We decompose this into predicting an intermediate value y:
        y = E_h[X_h(s)^{-ξ}]
        U_r = -y^η
    
    Key properties:
    - U_r < 0 always (negative reward representing "cost" of disempowerment)
    - y ∈ [1, ∞) since X_h ∈ (0, 1] and X_h^{-ξ} ≥ 1 for ξ > 0
    - When humans have low power (small X_h), y is large → U_r is very negative
    - ξ controls inter-human inequality aversion
    - η controls intertemporal inequality aversion
    
    Network architecture:
    - Predicts log(y-1) for numerical stability (since y > 1)
    - Then y = 1 + exp(log(y-1))
    - Finally U_r = -y^η
    
    Args:
        xi: Inter-human inequality aversion parameter (ξ ≥ 1).
        eta: Intertemporal inequality aversion parameter (η ≥ 1).
    """
    
    def __init__(
        self,
        xi: float = 1.0,
        eta: float = 1.1
    ):
        super().__init__()
        self.xi = xi
        self.eta = eta
        
        if xi < 1.0:
            raise ValueError(f"xi must be >= 1.0, got {xi}")
        if eta < 1.0:
            raise ValueError(f"eta must be >= 1.0, got {eta}")
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute y and U_r values.
        
        Returns:
            Tuple of (y, U_r) where:
            - y: intermediate value, shape (batch,), y > 1
            - U_r: intrinsic reward, shape (batch,), U_r < 0
        """
        pass
    
    @abstractmethod
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state and compute y and U_r.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            device: Torch device.
        
        Returns:
            Tuple of (y, U_r) tensors, each shape (1,).
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
    
    def log_y_minus_1_to_y(self, log_y_minus_1: torch.Tensor) -> torch.Tensor:
        """
        Convert network output (log(y-1)) to y.
        
        y = 1 + exp(log(y-1))
        
        This parameterization ensures y > 1.
        
        Args:
            log_y_minus_1: Raw network output.
        
        Returns:
            y values > 1.
        """
        return 1.0 + torch.exp(log_y_minus_1)
    
    def y_to_log_y_minus_1(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert y to log(y-1) for loss computation in log-space.
        
        This is the inverse of log_y_minus_1_to_y.
        
        Args:
            y: Target y values > 1.
        
        Returns:
            log(y-1) values.
        """
        # Clamp y-1 to avoid log(0) when y is very close to 1
        return torch.log(torch.clamp(y - 1.0, min=1e-6))
    
    def y_to_u_r(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert y to U_r.
        
        U_r = -y^η
        
        Args:
            y: Intermediate values > 1.
        
        Returns:
            U_r values < 0.
        """
        return -(y ** self.eta)
    
    def compute_target_y(self, x_h: torch.Tensor) -> torch.Tensor:
        """
        Compute target y from X_h values.
        
        For a single sampled human h:
            target_y = X_h^{-ξ}
        
        This is used for Monte Carlo training where we sample one human
        per transition and use their X_h to compute the target.
        
        Args:
            x_h: X_h(s) value for sampled human, shape (batch,).
        
        Returns:
            Target y values, shape (batch,).
        """
        # X_h^{-ξ} = 1 / X_h^ξ
        # Since X_h ∈ (0, 1], X_h^{-ξ} ≥ 1
        return x_h ** (-self.xi)
    
    def compute_from_x_h(self, x_h_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute y and U_r directly from X_h values for all humans.
        
        y = E_h[X_h^{-ξ}] = mean(X_h^{-ξ})
        U_r = -y^η
        
        Args:
            x_h_values: X_h values for all humans, shape (batch, num_humans).
        
        Returns:
            Tuple of (y, U_r), each shape (batch,).
        """
        # Compute X_h^{-ξ} for each human
        x_h_powered = x_h_values ** (-self.xi)
        
        # Expected value (mean) across humans
        y = x_h_powered.mean(dim=-1)
        
        # U_r = -y^η
        u_r = self.y_to_u_r(y)
        
        return y, u_r
