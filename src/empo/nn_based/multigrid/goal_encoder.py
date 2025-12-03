"""
Goal encoder for multigrid environments.

Encodes goal positions into feature vectors.
Supports both point goals (x, y) and rectangle goals (x1, y1, x2, y2).
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union

from ..goal_encoder import BaseGoalEncoder


class MultiGridGoalEncoder(BaseGoalEncoder):
    """
    Encoder for position-based goals in multigrid.
    
    Supports two types of goals:
    - Point goals: target position (x, y)
    - Rectangle goals: target region (x1, y1, x2, y2)
    
    For rectangle goals, encodes the center and size of the rectangle.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        feature_dim: Output feature dimension.
        support_rectangles: If True, encode rectangles with center + size.
            If False, only encode point position (for backward compatibility).
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
            # Input: center_x, center_y, width, height (4 values)
            # For point goals, width=height=0
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
                For point goals: (x, y)
                For rectangle goals: (center_x, center_y, width, height)
        
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
        
        Handles three goal formats:
        1. Point goal with target_pos: (x, y)
        2. Rectangle goal with target_rect: (x1, y1, x2, y2)
        3. Tuple/list goal: (x, y) or (x1, y1, x2, y2)
        
        Args:
            goal: Goal object with target position or rectangle.
            device: Torch device.
        
        Returns:
            Tensor (1, 4) with (center_x, center_y, width, height) if support_rectangles,
            or (1, 2) with (x, y) otherwise.
        """
        # Extract goal coordinates
        if hasattr(goal, 'target_rect'):
            # Rectangle goal: (x1, y1, x2, y2)
            x1, y1, x2, y2 = goal.target_rect
            # Normalize coordinates
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            is_rectangle = True
        elif hasattr(goal, 'target_pos'):
            # Point goal
            x, y = goal.target_pos
            center_x, center_y = float(x), float(y)
            width, height = 0.0, 0.0
            is_rectangle = False
        elif hasattr(goal, 'position'):
            x, y = goal.position
            center_x, center_y = float(x), float(y)
            width, height = 0.0, 0.0
            is_rectangle = False
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                # Rectangle goal
                x1, y1, x2, y2 = goal
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                is_rectangle = True
            elif len(goal) >= 2:
                center_x, center_y = float(goal[0]), float(goal[1])
                width, height = 0.0, 0.0
                is_rectangle = False
            else:
                center_x, center_y = 0.0, 0.0
                width, height = 0.0, 0.0
                is_rectangle = False
        else:
            center_x, center_y = 0.0, 0.0
            width, height = 0.0, 0.0
            is_rectangle = False
        
        if self.support_rectangles:
            coords = torch.tensor(
                [[center_x, center_y, width, height]], 
                device=device
            )
        else:
            coords = torch.tensor(
                [[center_x, center_y]], 
                device=device
            )
        
        return coords
