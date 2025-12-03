"""
Goal encoder for multigrid environments.

Encodes goal positions into feature vectors.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from ..goal_encoder import BaseGoalEncoder


class MultiGridGoalEncoder(BaseGoalEncoder):
    """
    Encoder for position-based goals in multigrid.
    
    Goals are typically target positions. Encodes raw (x, y) coordinates.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        feature_dim: int = 32
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        
        # Input: x, y coordinates (raw integers)
        self.fc = nn.Sequential(
            nn.Linear(2, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates.
        
        Args:
            goal_coords: (batch, 2) with x, y coordinates
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        return self.fc(goal_coords)
    
    def encode_goal(
        self,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode a goal object.
        
        Args:
            goal: Goal object with target position.
            device: Torch device.
        
        Returns:
            Tensor (1, 2) with goal coordinates.
        """
        # Extract position from goal
        if hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
        elif hasattr(goal, 'position'):
            x, y = goal.position
        elif isinstance(goal, (tuple, list)) and len(goal) >= 2:
            x, y = goal[0], goal[1]
        else:
            x, y = 0, 0
        
        coords = torch.tensor([[float(x), float(y)]], device=device)
        return coords
