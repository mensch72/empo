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
        
        # Input: bounding box (x1, y1, x2, y2) with inclusive coordinates
        # Point goals are (x, y, x, y)
        self.fc = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates.
        
        Args:
            goal_coords: (batch, 4) with bounding box (x1, y1, x2, y2)
        
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
            Tensor (1, 4) with bounding box (x1, y1, x2, y2)
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
        
        coords = torch.tensor(
            [[float(x1), float(y1), float(x2), float(y2)]], 
            device=device
        )
        
        return coords
    
    @staticmethod
    def compute_goal_weight(goal: Any) -> float:
        """
        Compute the aggregation weight for a goal based on its area.
        
        Weight = (1 + x2 - x1) * (1 + y2 - y1)
        
        For point goals (x, y, x, y), this returns 1.0.
        For rectangles, it returns the number of cells in the bounding box.
        
        Args:
            goal: Goal object with target_rect or target_pos.
        
        Returns:
            Aggregation weight (area of bounding box).
        """
        # Extract bounding box coordinates
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
            # Normalize
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = x, y, x, y
        elif hasattr(goal, 'position'):
            x, y = goal.position
            x1, y1, x2, y2 = x, y, x, y
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = goal
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
            elif len(goal) >= 2:
                x1, y1 = goal[0], goal[1]
                x2, y2 = x1, y1
            else:
                return 1.0
        else:
            return 1.0
        
        # Weight = area = (1 + x2 - x1) * (1 + y2 - y1)
        return float((1 + x2 - x1) * (1 + y2 - y1))
