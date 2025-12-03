"""
Goal encoder for multigrid environments.

Encodes goal regions into feature vectors.
All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive coordinates.
Point goals are represented as (x, y, x, y).

Also provides:
- Weight-proportional goal sampling where goals are sampled with
  probability proportional to their area weight (1+x2-x1)*(1+y2-y1).
- Goal rendering for visualization with dashed rectangle boundaries and
  agent-to-goal connection lines.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Tuple, Union, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.axes

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
    
    @staticmethod
    def get_goal_bounding_box(goal: Any) -> Tuple[int, int, int, int]:
        """
        Extract bounding box coordinates from a goal object.
        
        Args:
            goal: Goal object with target_rect, target_pos, position, or tuple.
        
        Returns:
            Tuple (x1, y1, x2, y2) with inclusive coordinates.
        """
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif hasattr(goal, 'position'):
            x, y = goal.position
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = int(goal[0]), int(goal[1]), int(goal[2]), int(goal[3])
            elif len(goal) >= 2:
                x1, y1 = int(goal[0]), int(goal[1])
                x2, y2 = x1, y1
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        # Normalize order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        return (x1, y1, x2, y2)
    
    @staticmethod
    def closest_point_on_rectangle(
        rect: Tuple[int, int, int, int],
        px: float,
        py: float,
        tile_size: int = 32,
        inset: float = 0.08
    ) -> Tuple[float, float]:
        """
        Find the closest point on the rectangle boundary to a given point.
        
        The rectangle is defined by cell coordinates (x1, y1, x2, y2) with inclusive
        coordinates. The rectangle boundary is slightly inside the cells.
        
        Args:
            rect: Bounding box (x1, y1, x2, y2) in cell coordinates.
            px, py: Point position in pixel coordinates.
            tile_size: Size of each grid cell in pixels.
            inset: Fraction of cell to inset the rectangle boundary (0.08 = 8%).
        
        Returns:
            (closest_x, closest_y) in pixel coordinates.
        """
        x1, y1, x2, y2 = rect
        
        # Convert to pixel coordinates with inset
        left = x1 * tile_size + tile_size * inset
        right = (x2 + 1) * tile_size - tile_size * inset
        top = y1 * tile_size + tile_size * inset
        bottom = (y2 + 1) * tile_size - tile_size * inset
        
        # If point is inside the rectangle, find closest edge
        if left <= px <= right and top <= py <= bottom:
            # Distances to each edge
            d_left = px - left
            d_right = right - px
            d_top = py - top
            d_bottom = bottom - py
            
            min_d = min(d_left, d_right, d_top, d_bottom)
            
            if min_d == d_left:
                return (left, py)
            elif min_d == d_right:
                return (right, py)
            elif min_d == d_top:
                return (px, top)
            else:
                return (px, bottom)
        
        # Clamp point to rectangle boundary
        cx = max(left, min(px, right))
        cy = max(top, min(py, bottom))
        
        return (cx, cy)
    
    @staticmethod
    def render_goal_overlay(
        ax: 'matplotlib.axes.Axes',
        goal: Any,
        agent_pos: Tuple[float, float],
        agent_idx: int,
        tile_size: int = 32,
        goal_color: Tuple[float, float, float, float] = (0.0, 0.4, 1.0, 0.7),
        line_width: float = 2.5,
        inset: float = 0.08
    ) -> None:
        """
        Render a goal region on the given matplotlib axes.
        
        Draws:
        1. A non-filled rectangle with dashed, semi-transparent blue boundary
           lines slightly inside the cell boundaries.
        2. A dashed, semi-transparent line connecting the agent to the closest
           point on the rectangle boundary.
        
        Args:
            ax: Matplotlib axes to draw on.
            goal: Goal object with target_rect or target_pos.
            agent_pos: (x, y) position of the agent in cell coordinates.
            agent_idx: Index of the agent (used for color variation if needed).
            tile_size: Size of each grid cell in pixels.
            goal_color: RGBA color tuple for the goal visualization.
            line_width: Width of the dashed lines.
            inset: Fraction of cell to inset rectangle from cell edges (0.08 = 8%).
        """
        import matplotlib.patches as patches
        import matplotlib.lines as mlines
        
        # Get bounding box
        x1, y1, x2, y2 = MultiGridGoalEncoder.get_goal_bounding_box(goal)
        
        # Calculate pixel coordinates for rectangle with inset
        # Inset slightly inside the cell boundaries so rectangle lines don't
        # overlay the cell boundary lines
        left = x1 * tile_size + tile_size * inset
        top = y1 * tile_size + tile_size * inset
        width = (x2 - x1 + 1) * tile_size - 2 * tile_size * inset
        height = (y2 - y1 + 1) * tile_size - 2 * tile_size * inset
        
        # Draw dashed rectangle boundary
        rect = patches.Rectangle(
            (left, top), width, height,
            linewidth=line_width,
            edgecolor=goal_color,
            facecolor='none',
            linestyle='--',
            alpha=goal_color[3] if len(goal_color) > 3 else 0.7
        )
        ax.add_patch(rect)
        
        # Calculate agent pixel position (center of cell)
        agent_px = agent_pos[0] * tile_size + tile_size / 2
        agent_py = agent_pos[1] * tile_size + tile_size / 2
        
        # Find closest point on rectangle boundary to agent
        rect_coords = (x1, y1, x2, y2)
        closest_x, closest_y = MultiGridGoalEncoder.closest_point_on_rectangle(
            rect_coords, agent_px, agent_py, tile_size, inset
        )
        
        # Draw dashed line from agent to closest point on rectangle
        line = mlines.Line2D(
            [agent_px, closest_x],
            [agent_py, closest_y],
            linewidth=line_width * 0.8,
            color=goal_color[:3] if len(goal_color) >= 3 else goal_color,
            linestyle='--',
            alpha=goal_color[3] * 0.8 if len(goal_color) > 3 else 0.5
        )
        ax.add_line(line)
    
    @staticmethod
    def render_goals_on_frame(
        env,
        agent_goals: Dict[int, Any],
        tile_size: int = 32,
        goal_color: Tuple[float, float, float, float] = (0.0, 0.4, 1.0, 0.7)
    ) -> np.ndarray:
        """
        Render the environment with goal overlays for all agents.
        
        This is a convenience method that:
        1. Renders the base environment to an image
        2. Overlays goal rectangles and agent-to-goal lines
        3. Returns the combined image
        
        Args:
            env: The multigrid environment.
            agent_goals: Dict mapping agent indices to their goal objects.
            tile_size: Size of each grid cell in pixels.
            goal_color: RGBA color for goal visualization.
        
        Returns:
            RGB image array of the rendered frame.
        """
        import matplotlib.pyplot as plt
        
        # Render base environment
        img = env.render(mode='rgb_array', highlight=False)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        
        # Get agent positions from environment state
        state = env.get_state()
        _, agent_states, _, _ = state
        
        # Render each agent's goal
        for agent_idx, goal in agent_goals.items():
            if agent_idx < len(agent_states):
                agent_state = agent_states[agent_idx]
                agent_pos = (float(agent_state[0]), float(agent_state[1]))
                
                MultiGridGoalEncoder.render_goal_overlay(
                    ax=ax,
                    goal=goal,
                    agent_pos=agent_pos,
                    agent_idx=agent_idx,
                    tile_size=tile_size,
                    goal_color=goal_color
                )
        
        ax.axis('off')
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        # Convert to array
        buf = np.asarray(fig.canvas.buffer_rgba())
        buf = buf[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        return buf
