"""
Goal encoder for multigrid environments.

Encodes goal regions into feature vectors.
All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive coordinates.
Point goals are represented as (x, y, x, y).

The encoder supports internal caching of goal coordinate extraction (before NN forward)
to avoid redundant computation when the same goal is encoded multiple times.

For goal sampling and rendering, see empo.multigrid module.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

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
        
        # Internal cache for goal coordinate extraction (before NN forward)
        # Keys are goal object ids, values are coordinate tensors
        self._raw_cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_cache(self):
        """Clear the internal coordinate cache."""
        self._raw_cache.clear()
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) cache statistics."""
        return self._cache_hits, self._cache_misses
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        self._cache_hits = 0
        self._cache_misses = 0
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates through the neural network.
        
        Args:
            goal_coords: (batch, 4) with bounding box (x1, y1, x2, y2)
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        return self.fc(goal_coords)
    
    def tensorize_goal(
        self,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Convert goal to input tensor (preprocessing, NOT neural network encoding).
        
        This method extracts raw coordinates from the goal object. Results are
        cached by goal object id to avoid redundant extraction. Call forward()
        on these coordinates to get the actual neural network encoding.
        
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
        # Check cache first
        cache_key = id(goal)
        if cache_key in self._raw_cache:
            self._cache_hits += 1
            # Clone to avoid in-place operation conflicts during gradient computation
            return self._raw_cache[cache_key].clone()
        
        self._cache_misses += 1
        
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
        
        self._raw_cache[cache_key] = coords
        return coords
    
    @staticmethod
    def compute_goal_weight(goal: Any) -> float:
        """
        Compute the weight of a goal based on its area.
        
        For rectangle goals: weight = (x2 - x1 + 1) * (y2 - y1 + 1)
        For point goals: weight = 1
        
        Args:
            goal: Goal object with target_rect or target_pos.
        
        Returns:
            Weight as float (area of the goal region).
        """
        bbox = MultiGridGoalEncoder.get_goal_bounding_box(goal)
        x1, y1, x2, y2 = bbox
        return float((x2 - x1 + 1) * (y2 - y1 + 1))
    
    @staticmethod
    def get_goal_bounding_box(goal: Any) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from a goal object.
        
        Handles goal formats:
        1. Rectangle goal with target_rect: (x1, y1, x2, y2)
        2. Point goal with target_pos: (x, y) -> (x, y, x, y)
        3. Tuple/list goal: (x1, y1, x2, y2) or (x, y) -> (x, y, x, y)
        
        Args:
            goal: Goal object.
        
        Returns:
            Bounding box as (x1, y1, x2, y2) with normalized coordinates (x1<=x2, y1<=y2).
        """
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = goal
            elif len(goal) >= 2:
                x1, y1 = int(goal[0]), int(goal[1])
                x2, y2 = x1, y1
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        # Normalize coordinates
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def sample_coordinate_pair_weighted(n: int, rng) -> Tuple[int, int]:
        """
        Sample a coordinate pair (c1, c2) with c1 <= c2 from [0, n-1].
        
        Uses weight-proportional sampling where larger intervals are more likely.
        The weight for interval (c1, c2) is (c2 - c1 + 1).
        
        Args:
            n: Size of the coordinate space (0 to n-1).
            rng: Random number generator with integers() method.
        
        Returns:
            Tuple (c1, c2) where c1 <= c2.
        """
        if n <= 0:
            return (0, 0)
        
        # Compute total weight
        total_weight = 0
        for c1 in range(n):
            k = n - c1  # number of possible c2 values
            total_weight += k * (k + 1) // 2
        
        r = rng.integers(0, total_weight)
        
        cumulative = 0
        c1 = 0
        for c1_candidate in range(n):
            k = n - c1_candidate
            weight_c1 = k * (k + 1) // 2
            if cumulative + weight_c1 > r:
                c1 = c1_candidate
                r -= cumulative
                break
            cumulative += weight_c1
        
        # Now sample c2 given c1
        k = n - c1
        cumulative = 0
        c2 = c1
        for offset in range(k):
            weight_offset = offset + 1
            if cumulative + weight_offset > r:
                c2 = c1 + offset
                break
            cumulative += weight_offset
        else:
            c2 = n - 1
        
        return (c1, c2)
    
    @staticmethod
    def sample_rectangle_weighted(
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        rng
    ) -> Tuple[int, int, int, int]:
        """
        Sample a rectangle (x1, y1, x2, y2) with weight-proportional sampling.
        
        Weight = (x2 - x1 + 1) * (y2 - y1 + 1), i.e., the area of the rectangle.
        
        Args:
            x_range: (x_min, x_max) range for x coordinates.
            y_range: (y_min, y_max) range for y coordinates.
            rng: Random number generator with integers() method.
        
        Returns:
            Rectangle (x1, y1, x2, y2).
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        x_size = x_max - x_min + 1
        x1_offset, x2_offset = MultiGridGoalEncoder.sample_coordinate_pair_weighted(x_size, rng)
        x1 = x_min + x1_offset
        x2 = x_min + x2_offset
        
        y_size = y_max - y_min + 1
        y1_offset, y2_offset = MultiGridGoalEncoder.sample_coordinate_pair_weighted(y_size, rng)
        y1 = y_min + y1_offset
        y2 = y_min + y2_offset
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def closest_point_on_rectangle(
        rect: Tuple[int, int, int, int],
        px: float,
        py: float,
        tile_size: int,
        inset: float = 0.08
    ) -> Tuple[float, float]:
        """
        Find the closest point on a rectangle boundary to a given point.
        
        Args:
            rect: Rectangle (x1, y1, x2, y2) in grid coordinates.
            px: x pixel coordinate of the point.
            py: y pixel coordinate of the point.
            tile_size: Size of each grid tile in pixels.
            inset: Inset fraction from tile edges.
        
        Returns:
            (cx, cy) pixel coordinates of the closest point on the rectangle boundary.
        """
        x1, y1, x2, y2 = rect
        
        # Convert rectangle to pixel coordinates with inset
        left = x1 * tile_size + tile_size * inset
        right = (x2 + 1) * tile_size - tile_size * inset
        top = y1 * tile_size + tile_size * inset
        bottom = (y2 + 1) * tile_size - tile_size * inset
        
        # Clamp point to rectangle bounds
        cx = max(left, min(right, px))
        cy = max(top, min(bottom, py))
        
        # If point is inside, find closest edge
        if left < px < right and top < py < bottom:
            dist_left = px - left
            dist_right = right - px
            dist_top = py - top
            dist_bottom = bottom - py
            
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
            if min_dist == dist_left:
                cx = left
            elif min_dist == dist_right:
                cx = right
            elif min_dist == dist_top:
                cy = top
            else:
                cy = bottom
        
        return (cx, cy)
    
    @staticmethod
    def render_goal_overlay(
        ax,
        goal: Any,
        agent_pos: Tuple[float, float],
        agent_idx: int,
        tile_size: int = 32,
        goal_color: Tuple[float, float, float, float] = (0.0, 0.4, 1.0, 0.7),
        line_width: float = 2.5,
        inset: float = 0.08
    ) -> None:
        """
        Render goal overlay on a matplotlib axes.
        
        Draws a dashed rectangle around the goal region and a line connecting
        the agent to the closest point on the rectangle.
        
        Args:
            ax: Matplotlib axes to draw on.
            goal: Goal object or tuple (x1, y1, x2, y2).
            agent_pos: Agent position (x, y) in grid coordinates.
            agent_idx: Agent index (for labeling).
            tile_size: Size of each grid tile in pixels.
            goal_color: RGBA color tuple for the goal overlay.
            line_width: Width of the lines.
            inset: Inset fraction from tile edges.
        """
        import matplotlib.patches as patches
        
        bbox = MultiGridGoalEncoder.get_goal_bounding_box(goal)
        x1, y1, x2, y2 = bbox
        
        # Calculate pixel coordinates with inset
        left = x1 * tile_size + tile_size * inset
        top = y1 * tile_size + tile_size * inset
        width = (x2 - x1 + 1) * tile_size - 2 * tile_size * inset
        height = (y2 - y1 + 1) * tile_size - 2 * tile_size * inset
        
        # Draw dashed rectangle
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=line_width,
            edgecolor=goal_color,
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Draw line from agent to closest point on rectangle
        agent_px = agent_pos[0] * tile_size + tile_size / 2
        agent_py = agent_pos[1] * tile_size + tile_size / 2
        
        closest = MultiGridGoalEncoder.closest_point_on_rectangle(
            bbox, agent_px, agent_py, tile_size, inset
        )
        
        ax.plot(
            [agent_px, closest[0]],
            [agent_py, closest[1]],
            color=goal_color,
            linewidth=line_width,
            linestyle='--'
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'feature_dim': self.feature_dim,
        }
    
