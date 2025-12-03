"""
Goal encoder for multigrid environments.

Encodes goal regions into feature vectors.
All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive coordinates.
Point goals are represented as (x, y, x, y).
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union

from ..goal_encoder import BaseGoalEncoder


class MultiGridGoalEncoder(BaseGoalEncoder):
    """
    Encoder for region-based goals in multigrid.
    
    All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive
    coordinates. Point goals are represented as (x, y, x, y).
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        feature_dim: Output feature dimension.
        support_rectangles: If True, encode as bounding box (x1, y1, x2, y2).
            If False, only encode point position (x, y) for backward compatibility.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        feature_dim: int = 32,
        support_rectangles: bool = True
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.support_rectangles = support_rectangles
        
        if support_rectangles:
            # Input: bounding box (x1, y1, x2, y2) with inclusive coordinates
            # Point goals are (x, y, x, y)
            self.fc = nn.Sequential(
                nn.Linear(4, feature_dim),
                nn.ReLU(),
            )
        else:
            # Input: x, y coordinates (raw integers)
            self.fc = nn.Sequential(
                nn.Linear(2, feature_dim),
                nn.ReLU(),
            )
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates.
        
        Args:
            goal_coords: (batch, 2) or (batch, 4) with goal coordinates.
                For backward compat point-only mode: (x, y)
                For rectangle mode: bounding box (x1, y1, x2, y2)
        
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
        Encode a goal object as a bounding box.
        
        Handles goal formats:
        1. Rectangle goal with target_rect: (x1, y1, x2, y2)
        2. Point goal with target_pos: (x, y) -> encoded as (x, y, x, y)
        3. Tuple/list goal: (x1, y1, x2, y2) or (x, y) -> (x, y, x, y)
        
        Args:
            goal: Goal object with target position or rectangle.
            device: Torch device.
        
        Returns:
            Tensor (1, 4) with bounding box (x1, y1, x2, y2) if support_rectangles,
            or (1, 2) with (x, y) otherwise.
        """
        # Extract goal coordinates as bounding box (x1, y1, x2, y2)
        if hasattr(goal, 'target_rect'):
            # Rectangle goal: (x1, y1, x2, y2)
            x1, y1, x2, y2 = goal.target_rect
            # Normalize coordinates
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
        elif hasattr(goal, 'target_pos'):
            # Point goal -> bounding box (x, y, x, y)
            x, y = goal.target_pos
            x1, y1, x2, y2 = float(x), float(y), float(x), float(y)
        elif hasattr(goal, 'position'):
            x, y = goal.position
            x1, y1, x2, y2 = float(x), float(y), float(x), float(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                # Rectangle goal
                x1, y1, x2, y2 = goal
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
            elif len(goal) >= 2:
                # Point goal -> bounding box (x, y, x, y)
                x1, y1 = float(goal[0]), float(goal[1])
                x2, y2 = x1, y1
            else:
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        
        if self.support_rectangles:
            coords = torch.tensor(
                [[float(x1), float(y1), float(x2), float(y2)]], 
                device=device
            )
        else:
            # For backward compatibility, use first corner as point
            coords = torch.tensor(
                [[float(x1), float(y1)]], 
                device=device
            )
        
        return coords
