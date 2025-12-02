"""
Goal encoder for multigrid environments.

Simple encoder for goal positions. Goals are typically represented as
target positions (x, y coordinates).
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from empo.possible_goal import PossibleGoal


class MultiGridGoalEncoder(nn.Module):
    """
    Encoder for goal positions in multigrid environments.
    
    Goals are encoded as raw (x, y) coordinates, optionally with a bounding
    box for rectangular goals.
    
    Args:
        feature_dim: Output feature dimension.
    """
    
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Input: 4 coordinates (x1, y1, x2, y2) for rectangular goals
        # or (x, y, x, y) for point goals
        self.fc = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates into feature vector.
        
        Args:
            goal_coords: (batch, 4) goal coordinates [x1, y1, x2, y2]
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        return self.fc(goal_coords)
    
    def encode_goal(
        self,
        goal: PossibleGoal,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode a single goal into tensor.
        
        Args:
            goal: PossibleGoal instance
            device: Torch device
        
        Returns:
            Goal coordinates tensor of shape (1, 4)
        """
        if hasattr(goal, 'target_pos'):
            target = goal.target_pos
            x, y = float(target[0]), float(target[1])
            return torch.tensor([[x, y, x, y]], device=device, dtype=torch.float32)
        elif hasattr(goal, 'target_region'):
            # Rectangular goal region
            x1, y1, x2, y2 = goal.target_region
            return torch.tensor([[float(x1), float(y1), float(x2), float(y2)]], 
                              device=device, dtype=torch.float32)
        else:
            # Unknown goal type - return zeros
            return torch.zeros(1, 4, device=device, dtype=torch.float32)
