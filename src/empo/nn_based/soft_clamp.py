"""
Soft clamping module for neural networks.
"""

import torch
import torch.nn as nn


class SoftClamp(nn.Module):
    """
    Implements a soft clamp that is linear in [a, b] and exponential outside.
    
    The function smoothly transitions from the linear region to exponential
    tails, ensuring differentiability while preventing gradient explosion
    for values far outside the expected range.
    
    For Z in [a, b]: Q(Z) = Z (linear)
    For Z < a: Q(Z) approaches a - R asymptotically
    For Z > b: Q(Z) approaches b + R asymptotically
    
    where R = b - a is the range size.
    
    This is useful for bounding Q-values based on the theoretical feasible
    range determined by the reward structure (e.g., shaping rewards).
    
    Args:
        a: Lower bound of linear region (default: 0.5)
        b: Upper bound of linear region (default: 1.5)
    """
    
    def __init__(self, a: float = 0.5, b: float = 1.5):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.R = self.b - self.a
        
        if self.R <= 0:
            raise ValueError("b must be greater than a for the linear region [a, b].")
        
        self.relu = nn.ReLU()
    
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Apply soft clamping to the input tensor.
        
        Args:
            Z: Input tensor of any shape.
        
        Returns:
            Soft-clamped tensor of same shape.
        """
        # Term 1: Linear Core (The Clip Function)
        # T1 = ReLU(Z - a) - ReLU(Z - b) + a
        # This is Z in [a, b], 'a' for Z < a, and 'b' for Z > b.
        term1_linear_core = self.relu(Z - self.a) - self.relu(Z - self.b) + self.a
        
        # Term 2: Upper Exponential Tail (Z > b)
        # T2 = R * (1 - exp(-(1/R) * ReLU(Z - b)))
        # This is 0 for Z <= b, approaches R for Z >> b.
        term2_tail_gt_b = self.R * (1.0 - torch.exp(-(1.0 / self.R) * self.relu(Z - self.b)))
        
        # Term 3: Lower Exponential Tail (Z < a)
        # T3 = R * (exp(-(1/R) * ReLU(a - Z)) - 1)
        # This is 0 for Z >= a, approaches -R for Z << a.
        term3_tail_lt_a = self.R * (torch.exp(-(1.0 / self.R) * self.relu(self.a - Z)) - 1.0)
        
        return term1_linear_core + term2_tail_gt_b + term3_tail_lt_a
