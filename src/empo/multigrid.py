"""
Multigrid-specific goal types, samplers, policies, and rendering utilities.

This module provides goal classes and samplers for use with multigrid environments:
- ReachCellGoal: Goal to reach a specific cell
- ReachRectangleGoal: Goal to reach any cell in a rectangle region
- MultiGridGoalSampler: Weight-proportional goal sampler for multigrid
- RandomPolicy: Random action policy with configurable probabilities

Also provides goal rendering utilities:
- get_goal_bounding_box(): Extract bounding box from goal objects
- closest_point_on_rectangle(): Find closest point on goal boundary
- render_goal_overlay(): Draw goal visualization on matplotlib axes
- render_goals_on_frame(): Render full frame with goal overlays

All goals are represented as bounding boxes (x1, y1, x2, y2) with inclusive
coordinates. Point goals are represented as (x, y, x, y).

The MultiGridGoalSampler samples goals with probability proportional to their
area weight (1+x2-x1)*(1+y2-y1), which is required for correct phi network
training where the marginal policy must account for goal weights.
"""

import numpy as np
from typing import Tuple, Optional, Any, Dict, TYPE_CHECKING

from empo.possible_goal import PossibleGoal, PossibleGoalSampler

if TYPE_CHECKING:
    import matplotlib.axes
    from empo.world_model import WorldModel


class ReachCellGoal(PossibleGoal):
    """
    A goal where an agent tries to reach a specific cell.
    
    This is a convenience class that wraps a point goal as a bounding box
    (x, y, x, y) for consistency with the rectangle goal representation.
    
    Args:
        world_model: The environment.
        human_agent_index: Index of the human agent.
        target_pos: Tuple (x, y) defining the target cell.
    """
    
    def __init__(
        self,
        world_model: 'WorldModel',
        human_agent_index: int,
        target_pos: Tuple[int, int]
    ):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        # For consistency with rectangle goals
        self.target_rect = (self.target_pos[0], self.target_pos[1],
                           self.target_pos[0], self.target_pos[1])
    
    def is_achieved(self, state) -> int:
        """Check if agent is at the target cell."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            x, y = int(agent_state[0]), int(agent_state[1])
            if x == self.target_pos[0] and y == self.target_pos[1]:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachCell({self.target_pos})"
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_pos))
    
    def __eq__(self, other):
        if not isinstance(other, ReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and
                self.target_pos == other.target_pos)


class ReachRectangleGoal(PossibleGoal):
    """
    A goal where an agent tries to reach any cell in a rectangle region.
    
    Args:
        world_model: The environment.
        human_agent_index: Index of the human agent.
        target_rect: Tuple (x1, y1, x2, y2) defining the rectangle with
                     inclusive coordinates.
    """
    
    def __init__(
        self,
        world_model: 'WorldModel',
        human_agent_index: int,
        target_rect: Tuple[int, int, int, int]
    ):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        # Normalize rectangle coordinates
        x1, y1, x2, y2 = target_rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        self.target_rect = (x1, y1, x2, y2)
        # For compatibility with point goal interface
        self.target_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def is_achieved(self, state) -> int:
        """Check if agent is inside the rectangle."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            x, y = int(agent_state[0]), int(agent_state[1])
            x1, y1, x2, y2 = self.target_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return 1
        return 0
    
    def __str__(self):
        return f"ReachRect({self.target_rect})"
    
    def __hash__(self):
        return hash((self.human_agent_index, self.target_rect))
    
    def __eq__(self, other):
        if not isinstance(other, ReachRectangleGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and
                self.target_rect == other.target_rect)


class MultiGridGoalSampler(PossibleGoalSampler):
    """
    Weight-proportional goal sampler for multigrid environments.
    
    Samples rectangle goals with probability proportional to their area weight:
    P(goal) ∝ (1+x2-x1)*(1+y2-y1)
    
    This ensures that larger rectangles are sampled proportionally more often,
    which is required for correct phi network training where the marginal policy
    must account for goal weights.
    
    The sampling uses the product structure: (x1, x2) is sampled independently
    from (y1, y2), each using efficient inverse transform sampling without rejection.
    
    Args:
        world_model: The multigrid environment.
        valid_x_range: Optional (x_min, x_max) for valid goal coordinates.
                       Defaults to (1, width-2) to exclude outer walls.
        valid_y_range: Optional (y_min, y_max) for valid goal coordinates.
                       Defaults to (1, height-2) to exclude outer walls.
        seed: Optional random seed for reproducibility.
    
    Example:
        >>> sampler = MultiGridGoalSampler(env)
        >>> goal, weight = sampler.sample(state, human_agent_index=0)
        >>> # weight is 1.0 since sampling already accounts for goal weights
    """
    
    def __init__(
        self,
        world_model: 'WorldModel',
        valid_x_range: Optional[Tuple[int, int]] = None,
        valid_y_range: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None
    ):
        super().__init__(world_model)
        self._rng = np.random.default_rng(seed)
        self._custom_x_range = valid_x_range
        self._custom_y_range = valid_y_range
        self._update_valid_range()
    
    def _update_valid_range(self):
        """Update valid coordinate ranges for goal placement."""
        env = self.world_model
        if self._custom_x_range is not None:
            self._x_range = self._custom_x_range
        else:
            # Default: exclude outer walls
            self._x_range = (1, env.width - 2)
        
        if self._custom_y_range is not None:
            self._y_range = self._custom_y_range
        else:
            self._y_range = (1, env.height - 2)
    
    def set_world_model(self, world_model: 'WorldModel'):
        """Update world model and refresh valid ranges."""
        self.world_model = world_model
        self._update_valid_range()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
    
    @staticmethod
    def _sample_coordinate_pair_weighted(n: int, rng: np.random.Generator) -> Tuple[int, int]:
        """
        Sample (c1, c2) with c1 <= c2 from [0, n-1] with probability
        proportional to weight = (1 + c2 - c1).
        
        Uses inverse transform sampling without rejection:
        1. Sample c1 from marginal P(c1) ∝ (n - c1)(n - c1 + 1) / 2
        2. Sample c2 from conditional P(c2 | c1) ∝ (1 + c2 - c1)
        
        Args:
            n: Size of coordinate range [0, n-1]
            rng: numpy random generator
        
        Returns:
            Tuple (c1, c2) with 0 <= c1 <= c2 <= n-1
        """
        if n <= 0:
            return (0, 0)
        
        if n == 1:
            return (0, 0)
        
        # Step 1: Sample c1 from marginal distribution
        # P(c1 = k) ∝ (n - k)(n - k + 1) / 2  for k = 0, ..., n-1
        marginal_weights = np.zeros(n)
        for c1 in range(n):
            k = n - c1
            marginal_weights[c1] = k * (k + 1) / 2
        
        cumsum = np.cumsum(marginal_weights)
        total = cumsum[-1]
        
        # Sample c1 using inverse transform
        u1 = rng.uniform(0, total)
        c1 = int(np.searchsorted(cumsum, u1, side='left'))
        c1 = min(c1, n - 1)  # Ensure valid index
        
        # Step 2: Sample c2 from conditional P(c2 | c1) ∝ (1 + c2 - c1)
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
        delta = min(delta, num_options - 1)
        
        c2 = c1 + delta
        
        return (c1, c2)
    
    def sample_rectangle(self) -> Tuple[int, int, int, int]:
        """
        Sample a rectangle (x1, y1, x2, y2) with probability proportional
        to its area weight (1+x2-x1)*(1+y2-y1).
        
        Returns:
            Tuple (x1, y1, x2, y2) with valid coordinates.
        """
        x_min, x_max = self._x_range
        y_min, y_max = self._y_range
        
        # Sample (x1, x2) offset within [0, x_max - x_min]
        nx = x_max - x_min + 1
        x1_offset, x2_offset = self._sample_coordinate_pair_weighted(nx, self._rng)
        x1 = x_min + x1_offset
        x2 = x_min + x2_offset
        
        # Sample (y1, y2) offset within [0, y_max - y_min]
        ny = y_max - y_min + 1
        y1_offset, y2_offset = self._sample_coordinate_pair_weighted(ny, self._rng)
        y1 = y_min + y1_offset
        y2 = y_min + y2_offset
        
        return (x1, y1, x2, y2)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """
        Sample a rectangle goal with probability proportional to its area.
        
        The returned weight is 1.0 since the sampling already accounts for
        goal weights. This means when averaging over sampled goals, a simple
        average correctly represents the weighted expectation.
        
        Args:
            state: Current world state (not used for sampling, but required by interface).
            human_agent_index: Index of the human agent for the goal.
        
        Returns:
            Tuple of (goal, weight) where:
                - goal: ReachRectangleGoal instance
                - weight: 1.0 (sampling already accounts for weights)
        """
        x1, y1, x2, y2 = self.sample_rectangle()
        goal = ReachRectangleGoal(self.world_model, human_agent_index, (x1, y1, x2, y2))
        return goal, 1.0
    
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


# ============================================================================
# Goal Rendering Utilities
# ============================================================================

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
    x1, y1, x2, y2 = get_goal_bounding_box(goal)
    
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
    closest_x, closest_y = closest_point_on_rectangle(
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


def render_goals_on_frame(
    env,
    agent_goals: Dict[int, Any],
    tile_size: int = 32,
    goal_color: Tuple[float, float, float, float] = (0.0, 0.4, 1.0, 0.7),
    dpi: int = 100
) -> np.ndarray:
    """
    Render the environment with goal overlays for all agents.
    
    This is a convenience function that:
    1. Renders the base environment to an image
    2. Overlays goal rectangles and agent-to-goal lines
    3. Returns the combined image
    
    Args:
        env: The multigrid environment.
        agent_goals: Dict mapping agent indices to their goal objects.
        tile_size: Size of each grid cell in pixels.
        goal_color: RGBA color for goal visualization.
        dpi: DPI for the output image (determines final size).
    
    Returns:
        RGB image array of the rendered frame with consistent size.
    """
    import matplotlib.pyplot as plt
    
    # Render base environment
    img = env.render(mode='rgb_array', highlight=False, tile_size=tile_size)
    
    # Calculate figure size to produce consistent output dimensions
    # Output will be img.shape[1] x img.shape[0] pixels
    fig_width = img.shape[1] / dpi
    fig_height = img.shape[0] / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(img)
    
    # Get agent positions from environment state
    state = env.get_state()
    _, agent_states, _, _ = state
    
    # Render each agent's goal
    for agent_idx, goal in agent_goals.items():
        if agent_idx < len(agent_states):
            agent_state = agent_states[agent_idx]
            agent_pos = (float(agent_state[0]), float(agent_state[1]))
            
            render_goal_overlay(
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
    
    # Use constrained layout for consistent sizing
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.canvas.draw()
    
    # Convert to array - use exact pixel dimensions from canvas
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf = buf[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    return buf


# ============================================================================
# Random Policy for Agents
# ============================================================================

class RandomPolicy:
    """
    A simple random policy for agents in multigrid environments.
    
    Samples actions according to a configurable probability distribution.
    Default distribution is biased toward forward movement:
    - 6% still (action 0)
    - 18% left (action 1)
    - 18% right (action 2)
    - 58% forward (action 3)
    
    This distribution encourages exploration while preferring forward movement,
    which is typical behavior for goal-seeking agents.
    
    Args:
        action_probs: Optional array of action probabilities.
                     If None, uses default [0.06, 0.18, 0.18, 0.58].
                     Must sum to 1.0 and have length matching action space.
    
    Example:
        >>> policy = RandomPolicy()  # Use default distribution
        >>> action = policy.sample()  # Sample a random action
        
        >>> # Custom uniform distribution
        >>> policy = RandomPolicy(action_probs=[0.25, 0.25, 0.25, 0.25])
        >>> action = policy.sample()
    """
    
    # Default probabilities: 6% still, 18% left, 18% right, 58% forward
    DEFAULT_PROBS = np.array([0.06, 0.18, 0.18, 0.58])
    
    def __init__(self, action_probs: Optional[np.ndarray] = None):
        """
        Initialize the random policy.
        
        Args:
            action_probs: Optional array of action probabilities.
                         If None, uses default biased distribution.
        """
        if action_probs is not None:
            self._probs = np.array(action_probs, dtype=np.float64)
            # Normalize to ensure it sums to 1.0
            self._probs = self._probs / self._probs.sum()
        else:
            self._probs = self.DEFAULT_PROBS.copy()
        
        self._num_actions = len(self._probs)
    
    @property
    def action_probs(self) -> np.ndarray:
        """Get the action probability distribution."""
        return self._probs.copy()
    
    @property
    def num_actions(self) -> int:
        """Get the number of actions in this policy."""
        return self._num_actions
    
    def sample(self) -> int:
        """
        Sample a random action from the policy distribution.
        
        Returns:
            int: Sampled action index.
        """
        return int(np.random.choice(self._num_actions, p=self._probs))
    
    def __call__(self) -> int:
        """Alias for sample() to allow policy() syntax."""
        return self.sample()
    
    def __repr__(self) -> str:
        return f"RandomPolicy(action_probs={self._probs.tolist()})"


# ============================================================================
# Test Map Value Visualization
# ============================================================================

def render_test_map_values(
    test_maps: list,
    goal_specs: list,
    trainer,
    human_indices: list,
    robot_indices: list,
    tile_size: int = 64,
    annotation_panel_width: int = 280,
    annotation_font_size: int = 10,
) -> list:
    """
    Render frames showing predicted values for each test map state.
    
    For each test map:
    1. Creates a new MultiGridEnv with that map
    2. Renders the state with V_h^e values overlaid on goal cells
    3. Shows X_h, U_r, V_r, and Q_r values in annotation panel
    
    This is useful for visualizing what the trained networks predict
    for specific states without running any episodes.
    
    Args:
        test_maps: List of map strings OR list of (map_string, description) tuples.
                   If tuples, the description is shown at the top of the frame.
        goal_specs: List of goal specifications. Each spec is a tuple:
                   - Point goal: (human_idx, (x, y)) → ReachCellGoal
                   - Rectangle goal: (human_idx, ((x1, y1), (x2, y2))) → ReachRectangleGoal
                   Goals are created fresh for each test map's environment.
        trainer: Phase 2 trainer with convenience methods (get_v_h_e, etc.).
        human_indices: List of human agent indices.
        robot_indices: List of robot agent indices.
        tile_size: Pixel size of each grid cell.
        annotation_panel_width: Width of the annotation panel in pixels.
        annotation_font_size: Font size for annotation text.
    
    Returns:
        List of RGB image arrays (numpy), one per test map.
    
    Example:
        >>> frames = render_test_map_values(
        ...     test_maps=TEST_MAPS,
        ...     goal_specs=[
        ...         (1, (2, 1)),                    # Point goal at (2,1)
        ...         (1, ((1, 1), (3, 2))),          # Rectangle goal from (1,1) to (3,2)
        ...     ],
        ...     trainer=trainer,
        ...     human_indices=[1],
        ...     robot_indices=[0],
        ... )
        >>> # Save as video
        >>> env.start_video_recording()
        >>> env._video_frames = frames
        >>> env.save_video('test_map_values.mp4', fps=1)
    """
    import torch
    from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
    
    frames = []
    
    # Action names for annotation
    single_action_names = ['still', 'left', 'right', 'forward']
    num_robots = len(robot_indices)
    
    # Generate joint action names
    import itertools
    combinations = list(itertools.product(single_action_names, repeat=num_robots))
    joint_action_names = [', '.join(combo) for combo in combinations]
    
    for map_idx, test_map_entry in enumerate(test_maps):
        # Handle both plain map strings and (map, description) tuples
        if isinstance(test_map_entry, tuple):
            test_map, description = test_map_entry
        else:
            test_map = test_map_entry
            description = None
        
        # Create new environment with this test map
        env = MultiGridEnv(
            map=test_map,
            max_steps=10,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        env.reset()
        state = env.get_state()
        
        # Create fresh goals for this environment
        goals = []
        for human_idx, target_spec in goal_specs:
            # Detect if this is a rectangle spec ((x1,y1), (x2,y2)) or point spec (x, y)
            if (isinstance(target_spec, (tuple, list)) and len(target_spec) == 2 and
                isinstance(target_spec[0], (tuple, list)) and isinstance(target_spec[1], (tuple, list))):
                # Rectangle goal: ((x1, y1), (x2, y2))
                (x1, y1), (x2, y2) = target_spec
                goal = ReachRectangleGoal(env, human_idx, (x1, y1, x2, y2))
            else:
                # Point goal: (x, y)
                goal = ReachCellGoal(env, human_idx, target_spec)
            goals.append(goal)
        
        # Use trainer convenience methods to get values
        q_np = trainer.get_q_r(state, env)
        pi_np = trainer.get_pi_r(state, env)
        
        x_h_vals = []
        for h in human_indices:
            x_h = trainer.get_x_h(state, env, h)
            x_h_vals.append(x_h)
        
        u_r_val = trainer.get_u_r(state, env)
        v_r_val = trainer.get_v_r(state, env)
        
        # Compute V_h^e for each goal
        v_h_e_values = {}
        for goal in goals:
            h = goal.human_agent_index
            v_h_e_val = trainer.get_v_h_e(state, env, h, goal)
            # Store by goal's target position for overlay
            if hasattr(goal, 'target_rect'):
                key = goal.target_rect
            elif hasattr(goal, 'target_pos'):
                key = goal.target_pos
            else:
                key = str(goal)
            v_h_e_values[key] = v_h_e_val
        
        # Build annotation text
        lines = []
        lines.append(f"Test Map {map_idx + 1}/{len(test_maps)}")
        if description:
            lines.append(f"  {description}")
        lines.append("")
        lines.append(f"U_r: {u_r_val:.4f}")
        lines.append(f"V_r: {v_r_val:.4f}")
        lines.append("")
        for i, h in enumerate(human_indices):
            lines.append(f"X_h[{h}]: {x_h_vals[i]:.4f}")
        lines.append("")
        lines.append("Q_r values:")
        
        max_name_len = max(len(name) for name in joint_action_names) if joint_action_names else 7
        for action_idx in range(len(q_np)):
            action_name = joint_action_names[action_idx] if action_idx < len(joint_action_names) else f"a{action_idx}"
            lines.append(f" {action_name:>{max_name_len}}: {q_np[action_idx]:.3f}")
        
        lines.append("")
        lines.append("π_r probs:")
        for action_idx in range(len(pi_np)):
            action_name = joint_action_names[action_idx] if action_idx < len(joint_action_names) else f"a{action_idx}"
            lines.append(f" {action_name:>{max_name_len}}: {pi_np[action_idx]:.3f}")
        
        # Render base frame with annotation
        img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=tile_size,
            annotation_text=lines,
            annotation_panel_width=annotation_panel_width,
            annotation_font_size=annotation_font_size
        )
        
        # Overlay V_h^e values on goal cells
        img = _overlay_v_h_e_on_goals(img, v_h_e_values, goals, tile_size)
        
        frames.append(img)
    
    return frames


def _overlay_v_h_e_on_goals(
    img: np.ndarray,
    v_h_e_values: dict,
    goals: list,
    tile_size: int,
) -> np.ndarray:
    """
    Overlay V_h^e values as text on goal cells in the image.
    
    Args:
        img: RGB image array (H, W, 3).
        v_h_e_values: Dict mapping goal key (target_pos or target_rect) to V_h^e value.
        goals: List of goal objects.
        tile_size: Pixel size of each grid cell.
    
    Returns:
        Modified image with V_h^e values overlaid.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img  # PIL not available, return unchanged
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=max(10, tile_size // 5))
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", size=max(10, tile_size // 5))
        except (OSError, IOError):
            font = ImageFont.load_default()
    
    for goal in goals:
        # Get goal position
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
            key = goal.target_rect
            # Center of rectangle
            cx = (x1 + x2 + 1) / 2
            cy = (y1 + y2 + 1) / 2
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            key = goal.target_pos
            cx = x + 0.5
            cy = y + 0.5
        else:
            continue
        
        if key not in v_h_e_values:
            continue
        
        v_h_e_val = v_h_e_values[key]
        
        # Convert to pixel coordinates
        px = int(cx * tile_size)
        py = int(cy * tile_size)
        
        # Draw text with background for visibility
        text = f"{v_h_e_val:.2f}"
        
        # Get text bounding box
        bbox = draw.textbbox((px, py), text, font=font, anchor="mm")
        
        # Draw semi-transparent background
        padding = 2
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=(255, 255, 200, 200)
        )
        
        # Draw text
        draw.text((px, py), text, fill=(0, 0, 128), font=font, anchor="mm")
    
    return np.array(pil_img)
