"""
Goal encoder for multigrid environments.

Encodes goal regions into feature vectors.
All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive coordinates.
Point goals are represented as (x, y, x, y).

Also provides weight-proportional goal sampling where goals are sampled with
probability proportional to their area weight (1+x2-x1)*(1+y2-y1).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple, Union, List

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
    
    @staticmethod
    def _compute_cumulative_weights(n: int) -> np.ndarray:
        """
        Compute cumulative weights for sampling coordinate pairs (c1, c2) where
        c1 <= c2 and weight(c1, c2) = (1 + c2 - c1).
        
        For dimension of size n (coordinates 0 to n-1):
        - Marginal P(c1) ∝ (n - c1)(n - c1 + 1) / 2  (sum of weights 1,2,...,n-c1)
        - Conditional P(c2 | c1) ∝ (1 + c2 - c1) for c2 in [c1, n-1]
        
        Returns cumulative marginal weights for c1.
        """
        # Marginal weight for c1 = k is (n-k)(n-k+1)/2 for k=0,...,n-1
        # This is the sum of weights (1 + c2 - c1) for c2 from c1 to n-1
        marginal_weights = np.zeros(n)
        for c1 in range(n):
            k = n - c1  # number of valid c2 values: c1, c1+1, ..., n-1
            marginal_weights[c1] = k * (k + 1) / 2  # sum of 1, 2, ..., k
        
        # Compute cumulative sum
        cumsum = np.cumsum(marginal_weights)
        return cumsum
    
    @staticmethod
    def sample_coordinate_pair_weighted(n: int, rng: np.random.Generator = None) -> Tuple[int, int]:
        """
        Sample (c1, c2) with c1 <= c2 from [0, n-1] with probability
        proportional to weight = (1 + c2 - c1).
        
        Uses inverse transform sampling without rejection:
        1. Sample c1 from marginal P(c1) ∝ (n - c1)(n - c1 + 1) / 2
        2. Sample c2 from conditional P(c2 | c1) ∝ (1 + c2 - c1)
        
        Args:
            n: Size of coordinate range [0, n-1]
            rng: Optional numpy random generator
        
        Returns:
            Tuple (c1, c2) with 0 <= c1 <= c2 <= n-1
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if n <= 0:
            return (0, 0)
        
        if n == 1:
            return (0, 0)
        
        # Step 1: Sample c1 from marginal distribution
        # P(c1 = k) ∝ (n - k)(n - k + 1) / 2  for k = 0, ..., n-1
        # Compute cumulative sum for inverse transform sampling
        marginal_weights = np.zeros(n)
        for c1 in range(n):
            k = n - c1  # k = n - c1
            marginal_weights[c1] = k * (k + 1) / 2
        
        cumsum = np.cumsum(marginal_weights)
        total = cumsum[-1]
        
        # Sample c1 using inverse transform
        u1 = rng.uniform(0, total)
        c1 = int(np.searchsorted(cumsum, u1, side='left'))
        c1 = min(c1, n - 1)  # Ensure valid index
        
        # Step 2: Sample c2 from conditional P(c2 | c1) ∝ (1 + c2 - c1)
        # For c2 in [c1, n-1], weight is (1 + c2 - c1) = 1, 2, ..., (n - c1)
        num_options = n - c1
        
        if num_options <= 1:
            return (c1, c1)
        
        # Weights are 1, 2, ..., num_options
        # Cumulative: 1, 3, 6, 10, ... = k(k+1)/2
        cond_cumsum = np.zeros(num_options)
        for i in range(num_options):
            cond_cumsum[i] = (i + 1) * (i + 2) / 2
        
        total_cond = cond_cumsum[-1]
        u2 = rng.uniform(0, total_cond)
        delta = int(np.searchsorted(cond_cumsum, u2, side='left'))
        delta = min(delta, num_options - 1)  # Ensure valid
        
        c2 = c1 + delta
        
        return (c1, c2)
    
    @staticmethod
    def sample_rectangle_weighted(
        valid_x_range: Tuple[int, int],
        valid_y_range: Tuple[int, int],
        rng: np.random.Generator = None
    ) -> Tuple[int, int, int, int]:
        """
        Sample a rectangle (x1, y1, x2, y2) with probability proportional
        to its area weight (1+x2-x1)*(1+y2-y1).
        
        Uses the product structure: (x1, x2) is sampled independently from (y1, y2),
        each with weight proportional to (1 + c2 - c1).
        
        Args:
            valid_x_range: (x_min, x_max) inclusive
            valid_y_range: (y_min, y_max) inclusive
            rng: Optional numpy random generator
        
        Returns:
            Tuple (x1, y1, x2, y2) with valid_x_range[0] <= x1 <= x2 <= valid_x_range[1]
            and similarly for y.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        x_min, x_max = valid_x_range
        y_min, y_max = valid_y_range
        
        # Sample (x1, x2) offset within [0, x_max - x_min]
        nx = x_max - x_min + 1
        x1_offset, x2_offset = MultiGridGoalEncoder.sample_coordinate_pair_weighted(nx, rng)
        x1 = x_min + x1_offset
        x2 = x_min + x2_offset
        
        # Sample (y1, y2) offset within [0, y_max - y_min]
        ny = y_max - y_min + 1
        y1_offset, y2_offset = MultiGridGoalEncoder.sample_coordinate_pair_weighted(ny, rng)
        y1 = y_min + y1_offset
        y2 = y_min + y2_offset
        
        return (x1, y1, x2, y2)
