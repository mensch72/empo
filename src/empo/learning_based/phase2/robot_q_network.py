"""
Base Robot Q-Network for Phase 2.

Implements Q_r(s, a_r) from equation (4) of the EMPO paper:
    Q_r(s, a_r) ← E_g E_{a_H ~ π_H(s,g)} E_{s'|s,a} [γ_r V_r(s')]

This network outputs Q-values for all robot action combinations.
Since Q_r < 0 (because V_r < 0), we use appropriate output bounding.

When use_z_space=True, the network internally represents values in z-space:
    z = (-Q)^{-1/(ηξ)} ∈ (0, 1]
This makes it easier to represent values across orders of magnitude.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseRobotQNetwork(nn.Module, ABC):
    """
    Base class for robot Q-networks in Phase 2.
    
    Q_r(s, a_r) estimates the expected discounted future value when
    the robot fleet takes joint action a_r in state s.
    
    Key properties:
    - Q_r < 0 always (since V_r < 0)
    - Output shape is (batch, num_action_combinations) where
      num_action_combinations = num_actions^num_robots
    
    The robot policy π_r is derived using a power-law softmax (eq. 5):
        π_r(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
    
    When use_z_space=True:
    - Network internally stores z = (-Q)^{-1/(ηξ)} ∈ (0, 1]
    - forward() returns Q-values (converted from z)
    - forward_z() returns raw z-values (for loss computation)
    - Policy can be computed directly from z for numerical stability
    
    Args:
        num_actions: Number of actions per robot.
        num_robots: Number of robots in the fleet.
        beta_r: Power-law exponent for policy derivation.
        feasible_range: Optional (min, max) bounds for Q-values.
        use_z_space: If True, use z-space representation internally.
        eta: η parameter for z-space transformation (default 1.1).
        xi: ξ parameter for z-space transformation (default 1.0).
    """
    
    def __init__(
        self,
        num_actions: int,
        num_robots: int,
        beta_r: float = 10.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        use_z_space: bool = False,
        eta: float = 1.1,
        xi: float = 1.0
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_robots = num_robots
        self.beta_r = beta_r
        self.feasible_range = feasible_range
        self.use_z_space = use_z_space
        self.eta = eta
        self.xi = xi
        
        # Number of joint action combinations
        self.num_action_combinations = num_actions ** num_robots
    
    @abstractmethod
    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute Q_r values.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            device: Torch device.
        
        Returns:
            Tensor of shape (1, num_action_combinations) with Q_r < 0.
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
    
    def raw_to_z(self, raw_values: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Convert raw network output to z ∈ (0, 1).
        
        Uses sigmoid to map R → (0, 1).
        
        Args:
            raw_values: Raw network output (unbounded).
            eps: Small value to keep z away from boundaries.
        
        Returns:
            z-values in (eps, 1-eps).
        """
        return torch.sigmoid(raw_values).clamp(min=eps, max=1.0 - eps)
    
    def z_to_q(self, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert z-values to Q-values: Q = -z^{-ηξ}.
        
        Args:
            z: z-values in (0, 1].
            eps: Clamping epsilon.
        
        Returns:
            Q-values < 0.
        """
        z_clamped = z.clamp(min=eps)
        exponent = -self.eta * self.xi
        return -torch.pow(z_clamped, exponent)
    
    def q_to_z(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert Q-values to z-values: z = (-Q)^{-1/(ηξ)}.
        
        Args:
            q: Q-values, must be negative (< 0).
            eps: Clamping epsilon.
        
        Returns:
            z-values in (0, ∞).
        """
        q_clamped = q.clamp(max=-eps)
        exponent = -1.0 / (self.eta * self.xi)
        return torch.pow(-q_clamped, exponent)
    
    def ensure_negative(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Ensure Q-values are negative (since V_r < 0 implies Q_r < 0).
        
        When use_z_space=True, converts raw → z → Q.
        Otherwise, uses -softplus to map any real value to (-∞, 0).
        
        Args:
            raw_values: Unbounded network output.
        
        Returns:
            Negative Q-values.
        """
        if self.use_z_space:
            z = self.raw_to_z(raw_values)
            return self.z_to_q(z)
        else:
            # -softplus(x) maps R -> (-∞, 0)
            # softplus(x) = log(1 + exp(x)) is always positive
            return -F.softplus(raw_values)
    
    def get_z_values(self, raw_values: torch.Tensor) -> torch.Tensor:
        """
        Get z-values from raw network output.
        
        Only valid when use_z_space=True.
        
        Args:
            raw_values: Raw network output.
        
        Returns:
            z-values in (0, 1).
        """
        if not self.use_z_space:
            raise ValueError("get_z_values() is only valid when use_z_space=True")
        return self.raw_to_z(raw_values)
    
    def get_policy(
        self,
        q_values: Optional[torch.Tensor],
        beta_r: Optional[float] = None,
        z_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute robot policy using power-law softmax (equation 5).
        
        π_r(a_r) ∝ (-Q_r(s, a_r))^{-β_r}
        
        Since Q_r < 0, we have -Q_r > 0, so the expression is well-defined.
        Higher β_r makes the policy more deterministic.
        When β_r = 0 or q_values is None, the policy is uniform.
        
        When use_z_space=True and z_values is provided, computes policy directly
        from z-values for numerical stability:
            π ∝ z^{-ηξβ_r}  (since |Q| = z^{-ηξ})
        
        Args:
            q_values: Tensor of shape (..., num_action_combinations), all negative.
                     If None, returns uniform distribution (requires beta_r=0).
                     Ignored if z_values is provided.
            beta_r: Optional override for policy concentration. If None, use self.beta_r.
            z_values: Optional z-values (only when use_z_space=True). More numerically
                     stable than using Q-values directly.
        
        Returns:
            Action probabilities of shape (..., num_action_combinations).
        """
        if beta_r is None:
            beta_r = self.beta_r
        
        # Special case: beta_r = 0 means uniform random policy (no Q_r needed)
        if beta_r == 0.0:
            if q_values is None and z_values is None:
                # Return uniform distribution without knowing batch size
                # Caller must handle this case or provide a template tensor
                uniform = torch.ones(self.num_action_combinations)
                return uniform / uniform.sum()
            else:
                template = z_values if z_values is not None else q_values
                uniform = torch.ones_like(template)
                return uniform / uniform.sum(dim=-1, keepdim=True)
        
        # Use z-values if provided (more numerically stable for large |Q|)
        if z_values is not None:
            # π ∝ |Q|^{β_r} = (z^{-ηξ})^{β_r} = z^{-ηξβ_r}
            # log π ∝ -ηξβ_r * log(z)
            z_clamped = z_values.clamp(min=1e-10)
            exponent = -self.eta * self.xi * beta_r
            log_unnormalized = exponent * torch.log(z_clamped)
            
            # Subtract max for numerical stability
            log_unnormalized = log_unnormalized - log_unnormalized.max(dim=-1, keepdim=True)[0]
            unnormalized = torch.exp(log_unnormalized)
            policy = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
            return torch.clamp(policy, min=1e-10, max=1.0)
        
        # For non-zero beta_r without z_values, q_values is required
        if q_values is None:
            raise ValueError("q_values cannot be None when beta_r != 0 and z_values not provided")
        
        # q_values are negative, so -q_values are positive
        neg_q = -q_values
        
        # Numerical stability: clamp to avoid division by zero
        neg_q = torch.clamp(neg_q, min=1e-10)
        
        # Power-law: (-Q)^{-β} = 1 / (-Q)^β
        # Use log-space computation for numerical stability to avoid inf/nan
        # log((-Q)^{-β}) = -β * log(-Q)
        log_unnormalized = -beta_r * torch.log(neg_q)
        
        # Subtract max for numerical stability before exp (log-sum-exp trick)
        log_unnormalized = log_unnormalized - log_unnormalized.max(dim=-1, keepdim=True)[0]
        
        # Exponentiate and normalize
        unnormalized = torch.exp(log_unnormalized)
        
        # Normalize to get probabilities
        policy = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
        
        # Final safety clamp to ensure valid probabilities
        policy = torch.clamp(policy, min=1e-10, max=1.0)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        return policy
    
    def get_value(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Q-value under the policy.
        
        E_{a_r ~ π_r}[Q_r(s, a_r)]
        
        Args:
            q_values: Tensor of shape (..., num_action_combinations).
        
        Returns:
            Expected value tensor of shape (...).
        """
        policy = self.get_policy(q_values)
        return (policy * q_values).sum(dim=-1)
    
    def action_index_to_tuple(self, index: int) -> Tuple[int, ...]:
        """
        Convert flat action index to tuple of per-robot actions.
        
        Args:
            index: Flat index in [0, num_action_combinations).
        
        Returns:
            Tuple of actions, one per robot.
        """
        actions = []
        remaining = index
        for _ in range(self.num_robots):
            actions.append(remaining % self.num_actions)
            remaining //= self.num_actions
        return tuple(actions)
    
    def action_tuple_to_index(self, actions: Tuple[int, ...]) -> int:
        """
        Convert tuple of per-robot actions to flat index.
        
        Args:
            actions: Tuple of actions, one per robot.
        
        Returns:
            Flat index in [0, num_action_combinations).
        """
        index = 0
        multiplier = 1
        for action in actions:
            index += action * multiplier
            multiplier *= self.num_actions
        return index
    
    def sample_action(
        self,
        q_values: torch.Tensor,
        beta_r: Optional[float] = None
    ) -> Tuple[int, ...]:
        """
        Sample a joint action from the policy.
        
        Note: Epsilon-greedy exploration is handled by the trainer, not here.
        
        Args:
            q_values: Q-values of shape (1, num_action_combinations).
            beta_r: Optional override for policy concentration. If None, use self.beta_r.
                   Use beta_r=0 for uniform random policy (during warm-up).
        
        Returns:
            Tuple of actions, one per robot.
        """
        # Sample from policy (with optional beta_r override)
        policy = self.get_policy(q_values, beta_r=beta_r)
        flat_idx = torch.multinomial(policy.squeeze(0), 1).item()
        
        return self.action_index_to_tuple(flat_idx)
