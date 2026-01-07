"""
Value transformations for Phase 2 networks.

Provides theory-grounded transformations for Q_r, V_r, U_r values which are
guaranteed negative by EMPO theory. The transformation maps Q < 0 to z ∈ (0, 1],
making it easier for neural networks to represent values across orders of magnitude.

The transformation uses the power parameters η (eta) and ξ (xi) from the EMPO equations:
    z = f(Q) = (-Q)^{-1/(ηξ)}
    Q = f^{-1}(z) = -z^{-ηξ}

This maps:
    Q = -1    → z = 1
    Q = -10   → z ≈ 0.095 (for η=1.1, ξ=1.0)
    Q = -100  → z ≈ 0.012
    Q = -1000 → z ≈ 0.0016
    Q → -∞   → z → 0

For the U_r intermediate value y ∈ [1, ∞), we use:
    z = y^{-1/ξ}
    y = z^{-ξ}

This maps:
    y = 1    → z = 1
    y = 10   → z ≈ 0.1 (for ξ=1.0)
    y = 100  → z = 0.01
    y → ∞   → z → 0
"""

import torch
import torch.nn.functional as F
from typing import Optional


def to_z_space(
    q: torch.Tensor,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Transform Q-values to z-space: z = (-Q)^{-1/(ηξ)}.
    
    Maps Q ∈ (-∞, 0) to z ∈ (0, ∞), with Q = -1 mapping to z = 1.
    For typical U_r ∈ (-∞, -1], we get z ∈ (0, 1].
    
    Args:
        q: Q-values, must be negative (< 0).
        eta: η parameter from EMPO theory (intertemporal inequality aversion).
        xi: ξ parameter from EMPO theory (inter-human inequality aversion).
        eps: Small value to clamp Q away from 0.
    
    Returns:
        z-values in (0, ∞).
    """
    # Ensure Q < 0 by clamping away from 0
    q_clamped = q.clamp(max=-eps)
    
    # z = (-Q)^{-1/(ηξ)}
    exponent = -1.0 / (eta * xi)
    return torch.pow(-q_clamped, exponent)


def from_z_space(
    z: torch.Tensor,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Transform z-values back to Q-space: Q = -z^{-ηξ}.
    
    Maps z ∈ (0, ∞) to Q ∈ (-∞, 0), with z = 1 mapping to Q = -1.
    
    Args:
        z: z-values, must be positive (> 0).
        eta: η parameter from EMPO theory.
        xi: ξ parameter from EMPO theory.
        eps: Small value to clamp z away from 0.
    
    Returns:
        Q-values, all negative.
    """
    # Ensure z > 0 by clamping away from 0
    z_clamped = z.clamp(min=eps)
    
    # Q = -z^{-ηξ}
    exponent = -eta * xi
    return -torch.pow(z_clamped, exponent)


def y_to_z_space(
    y: torch.Tensor,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Transform y-values (U_r intermediate) to z-space: z = y^{-1/ξ}.
    
    Maps y ∈ [1, ∞) to z ∈ (0, 1], with y = 1 mapping to z = 1.
    
    The y value is y = E_h[X_h^{-ξ}] ≥ 1 (since X_h ∈ (0,1] implies X_h^{-ξ} ≥ 1).
    
    Args:
        y: y-values, must be ≥ 1.
        xi: ξ parameter from EMPO theory (inter-human inequality aversion).
        eps: Small value to clamp y away from 0.
    
    Returns:
        z-values in (0, 1].
    """
    # Ensure y >= 1
    y_clamped = y.clamp(min=1.0)
    
    # z = y^{-1/ξ}
    exponent = -1.0 / xi
    return torch.pow(y_clamped, exponent)


def z_to_y_space(
    z: torch.Tensor,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Transform z-values back to y-space: y = z^{-ξ}.
    
    Maps z ∈ (0, 1] to y ∈ [1, ∞), with z = 1 mapping to y = 1.
    
    Args:
        z: z-values, must be positive (> 0).
        xi: ξ parameter from EMPO theory.
        eps: Small value to clamp z away from 0.
    
    Returns:
        y-values ≥ 1.
    """
    # Ensure z > 0
    z_clamped = z.clamp(min=eps)
    
    # y = z^{-ξ}
    exponent = -xi
    return torch.pow(z_clamped, exponent)


def raw_to_z(raw: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert raw network output to z ∈ (0, 1).
    
    Uses sigmoid to map R → (0, 1). This is preferred over softplus
    because z should be bounded in (0, 1] for typical Q-values ≤ -1.
    
    Args:
        raw: Raw network output (unbounded).
        eps: Small value to keep z away from exactly 0 or 1.
    
    Returns:
        z-values in (eps, 1-eps).
    """
    # sigmoid maps R → (0, 1)
    # Clamp to avoid numerical issues at boundaries
    return torch.sigmoid(raw).clamp(min=eps, max=1.0 - eps)


def z_to_q(
    z: torch.Tensor,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Convert z-values to Q-values. Alias for from_z_space.
    
    Args:
        z: z-values in (0, 1].
        eta: η parameter.
        xi: ξ parameter.
        eps: Clamping epsilon.
    
    Returns:
        Q-values < 0.
    """
    return from_z_space(z, eta, xi, eps)


def compute_z_space_loss(
    z_pred: torch.Tensor,
    q_target: torch.Tensor,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute MSE loss in z-space.
    
    Converts Q targets to z-space and computes MSE against z predictions.
    This gives balanced gradients across different Q-value scales.
    
    Args:
        z_pred: Predicted z-values from network.
        q_target: Target Q-values (negative).
        eta: η parameter.
        xi: ξ parameter.
        eps: Clamping epsilon.
    
    Returns:
        Scalar MSE loss.
    """
    z_target = to_z_space(q_target, eta, xi, eps)
    return F.mse_loss(z_pred, z_target)


def compute_q_space_loss(
    z_pred: torch.Tensor,
    q_target: torch.Tensor,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute MSE loss in Q-space.
    
    Converts z predictions to Q-space and computes MSE against Q targets.
    Use during 1/t decay phase for proper Robbins-Monro convergence.
    
    Args:
        z_pred: Predicted z-values from network.
        q_target: Target Q-values (negative).
        eta: η parameter.
        xi: ξ parameter.
        eps: Clamping epsilon.
    
    Returns:
        Scalar MSE loss.
    """
    q_pred = from_z_space(z_pred, eta, xi, eps)
    return F.mse_loss(q_pred, q_target)


def compute_policy_from_z(
    z: torch.Tensor,
    beta_r: float,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute robot policy from z-values using power-law softmax.
    
    The policy is π ∝ |Q_r|^{β_r} = (-Q_r)^{β_r} = z^{-ηξβ_r}.
    
    This is numerically stable because we work with log(z) rather than
    computing the huge Q-values directly.
    
    Args:
        z: z-values for each action, shape (..., num_actions).
        beta_r: Power-law policy exponent.
        eta: η parameter.
        xi: ξ parameter.
        eps: Clamping epsilon.
    
    Returns:
        Policy probabilities, shape (..., num_actions).
    """
    if beta_r == 0.0:
        # Uniform policy
        return torch.ones_like(z) / z.shape[-1]
    
    # Clamp z away from 0 to avoid -inf in log
    z_clamped = z.clamp(min=eps)
    
    # log(|Q|^{β_r}) = log(z^{-ηξ·β_r}) = -ηξβ_r * log(z)
    log_unnorm_policy = -eta * xi * beta_r * torch.log(z_clamped)
    
    # Softmax for proper probabilities
    return F.softmax(log_unnorm_policy, dim=-1)


def get_policy_log_probs_from_z(
    z: torch.Tensor,
    beta_r: float,
    eta: float,
    xi: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Get log-probabilities of robot policy from z-values.
    
    Returns log π_r(a) for each action, useful for entropy computation.
    
    Args:
        z: z-values for each action, shape (..., num_actions).
        beta_r: Power-law policy exponent.
        eta: η parameter.
        xi: ξ parameter.
        eps: Clamping epsilon.
    
    Returns:
        Log-probabilities, shape (..., num_actions).
    """
    if beta_r == 0.0:
        # Uniform policy: log(1/n) for each action
        n_actions = z.shape[-1]
        return torch.full_like(z, -torch.log(torch.tensor(n_actions, dtype=z.dtype)))
    
    z_clamped = z.clamp(min=eps)
    log_unnorm_policy = -eta * xi * beta_r * torch.log(z_clamped)
    
    # log_softmax for numerical stability
    return F.log_softmax(log_unnorm_policy, dim=-1)
