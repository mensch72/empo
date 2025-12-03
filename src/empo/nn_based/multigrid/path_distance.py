"""
Path distance calculator for multigrid environments.

Computes path-based distances using passing difficulty scores by obstacle type.
Supports both point targets and rectangular region targets.

The distance calculation walks along precomputed shortest paths and sums up
"passing difficulty" scores based on what's currently on each cell. This enables
more accurate reward shaping that accounts for obstacles of varying difficulty.
"""

import numpy as np
from typing import Any, Dict, Optional, Set, Tuple, Union
from collections import deque


# Default passing costs for different object types
# These scores represent the difficulty of passing through/by each object type
DEFAULT_PASSING_COSTS = {
    'empty': 1,
    'door_open': 1,
    'door_closed': 2,      # Can be opened easily
    'door_locked': 25,     # Need to find a key first
    'agent': 2,
    'block': 2,
    'pickable': 3,         # key, ball, box
    'rock': 50,
    'wall': float('inf'),
    'magicwall': float('inf'),
    'lava': float('inf'),  # Deadly - impassable
    'unsteadyground': 2,
    'unknown': 2,
}


class PathDistanceCalculator:
    """
    Computes path-based distances using passing difficulty scores.
    
    At initialization, creates a "stripped down" grid version that only contains
    impassable obstacles (walls, magic walls, lava) and precomputes shortest paths
    between all passable cells using BFS.
    
    At runtime, calculates a "distance indicator" by walking along the precomputed
    shortest path and summing parameterized "passing difficulty" scores based on
    what's currently on each cell.
    
    Supports both point targets and rectangular region targets.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        world_model: Optional environment for precomputing wall grid.
        passing_costs: Optional dict mapping object types to passing costs.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        world_model: Any = None,
        passing_costs: Optional[Dict[str, float]] = None
    ):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.passing_costs = passing_costs or DEFAULT_PASSING_COSTS.copy()
        
        # Create wall-only grid and precompute shortest paths if world_model provided
        self._wall_grid: Optional[np.ndarray] = None
        self._shortest_paths: Optional[Dict] = None
        
        if world_model is not None:
            self._wall_grid = self._create_wall_grid(world_model)
            self._shortest_paths = self._precompute_shortest_paths()
        
        # Compute feasible range based on passing costs
        self.feasible_range = self._compute_feasible_range()
        
        # Caches for BFS-based distance computation (used when no world_model)
        self._cache: Dict[Tuple, np.ndarray] = {}
        self._rect_cache: Dict[Tuple, np.ndarray] = {}
    
    def _create_wall_grid(self, world_model: Any) -> np.ndarray:
        """
        Create a boolean grid where True = impassable, False = passable.
        
        Considers Wall, MagicWall, and Lava as impassable obstacles.
        """
        wall_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        if not hasattr(world_model, 'grid') or world_model.grid is None:
            return wall_grid
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type in ('wall', 'magicwall', 'lava'):
                        wall_grid[y, x] = True
        
        return wall_grid
    
    def _precompute_shortest_paths(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], list]]:
        """
        Precompute shortest paths between all pairs of empty cells using BFS.
        
        Returns a dict: source_pos -> {target_pos -> path} where path is a list
        of (x, y) coordinates from source to target (inclusive).
        """
        if self._wall_grid is None:
            return {}
        
        paths = {}
        
        # Find all empty cells (not walls)
        empty_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not self._wall_grid[y, x]:
                    empty_cells.append((x, y))
        
        # For each empty cell, compute shortest paths to all other empty cells
        for source in empty_cells:
            paths[source] = {}
            paths[source][source] = [source]  # Path to self
            
            # BFS from source
            visited = {source: [source]}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                cx, cy = current
                
                # Check 4 neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        neighbor = (nx, ny)
                        
                        if not self._wall_grid[ny, nx] and neighbor not in visited:
                            visited[neighbor] = visited[current] + [neighbor]
                            paths[source][neighbor] = visited[neighbor]
                            queue.append(neighbor)
        
        return paths
    
    def _compute_feasible_range(self) -> Tuple[float, float]:
        """
        Compute the feasible range of path costs based on grid size and passing costs.
        
        The max cost is the longest possible path times the max finite passing cost.
        
        Returns:
            Tuple of (-max_cost, max_cost) for Q-value clamping.
        """
        max_finite_cost = max(
            cost for cost in self.passing_costs.values() if cost < float('inf')
        )
        max_path_length = self.grid_width + self.grid_height
        max_cost = max_path_length * max_finite_cost
        return (-max_cost, max_cost)
    
    def get_shortest_path(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int]
    ) -> Optional[list]:
        """
        Get the precomputed shortest path from source to target.
        
        Returns None if no path exists (one or both positions are walls).
        """
        if self._shortest_paths is None:
            return None
        if source in self._shortest_paths:
            return self._shortest_paths[source].get(target)
        return None
    
    def compute_path_cost(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int],
        world_model: Any
    ) -> float:
        """
        Compute the path cost from source to target based on current grid state.
        
        Walks along the precomputed shortest path and sums up passing difficulty
        scores based on what's currently on each cell.
        
        Args:
            source: (x, y) starting position
            target: (x, y) target position
            world_model: Current environment for checking cell contents
        
        Returns:
            Total path cost (sum of passing difficulties), or inf if no path exists.
        """
        path = self.get_shortest_path(source, target)
        
        if path is None:
            # Fall back to simple BFS distance if no precomputed paths
            obstacles = self.get_obstacles_from_state(
                (0, [], [], []), world_model) if world_model else set()
            dist = self.get_distance(source, target, obstacles)
            return dist  # Simple step count
        
        if source == target:
            return 0.0
        
        # Build agent position lookup for efficiency
        agent_positions = set()
        if world_model is not None and hasattr(world_model, 'agents'):
            for agent in world_model.agents:
                if hasattr(agent, 'pos') and agent.pos is not None:
                    agent_positions.add((int(agent.pos[0]), int(agent.pos[1])))
        
        total_cost = 0.0
        
        # Sum passing costs for each cell along the path (excluding source)
        for pos in path[1:]:
            x, y = pos
            cell = world_model.grid.get(x, y) if world_model and hasattr(world_model, 'grid') else None
            cost = self._get_cell_passing_cost(cell, pos, agent_positions)
            total_cost += cost
        
        return total_cost
    
    def _get_cell_passing_cost(
        self,
        cell: Any,
        pos: Tuple[int, int],
        agent_positions: Set[Tuple[int, int]]
    ) -> float:
        """
        Get the passing cost for a cell based on its current contents.
        
        Args:
            cell: The object at this cell (or None for empty)
            pos: (x, y) position of the cell
            agent_positions: Set of (x, y) positions where agents are located
        
        Returns:
            Passing cost for this cell.
        """
        # Check for agent at this position first
        if pos in agent_positions:
            return self.passing_costs.get('agent', 2)
        
        if cell is None:
            return self.passing_costs.get('empty', 1)
        
        cell_type = getattr(cell, 'type', None)
        
        # Handle different object types
        if cell_type == 'door':
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
            if is_open:
                return self.passing_costs.get('door_open', 1)
            elif is_locked:
                return self.passing_costs.get('door_locked', 25)
            else:
                return self.passing_costs.get('door_closed', 2)
        
        elif cell_type == 'block':
            return self.passing_costs.get('block', 2)
        
        elif cell_type == 'rock':
            return self.passing_costs.get('rock', 50)
        
        elif cell_type in ('key', 'ball', 'box'):
            return self.passing_costs.get('pickable', 3)
        
        elif cell_type in ('wall', 'magicwall'):
            return self.passing_costs.get('wall', float('inf'))
        
        elif cell_type == 'lava':
            return self.passing_costs.get('lava', float('inf'))
        
        elif cell_type in ('goal', 'floor', 'switch', 'objectgoal'):
            return self.passing_costs.get('empty', 1)
        
        elif cell_type == 'unsteadyground':
            return self.passing_costs.get('unsteadyground', 2)
        
        else:
            return self.passing_costs.get('unknown', 2)
    
    def compute_potential(
        self,
        agent_pos: Tuple[int, int],
        goal: Union[Tuple[int, int], Tuple[int, int, int, int]],
        world_model: Any = None,
        max_cost: Optional[float] = None
    ) -> float:
        """
        Compute potential function value for reward shaping.
        
        Uses path cost instead of simple distance for more accurate shaping.
        
        Φ(s) = -path_cost(agent_pos, target) / max_cost
        
        The potential is maximal (0) when agent is at the goal.
        
        Args:
            agent_pos: (x, y) current agent position.
            goal: Either a point goal (x, y) or rectangle goal (x1, y1, x2, y2).
            world_model: Environment for checking cell contents.
            max_cost: Maximum cost for normalization.
        
        Returns:
            Potential value in range [-1, 0] (approximately).
        """
        if max_cost is None:
            max_cost = self.feasible_range[1]  # Use computed max cost
        
        # For rectangle goals, compute distance to closest point
        if len(goal) == 4:
            x1, y1, x2, y2 = goal
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Check if already in rectangle
            if x1 <= agent_pos[0] <= x2 and y1 <= agent_pos[1] <= y2:
                return 0.0
            
            # Find closest point in rectangle
            min_cost = float('inf')
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    target = (x, y)
                    if world_model is not None and self._shortest_paths is not None:
                        cost = self.compute_path_cost(agent_pos, target, world_model)
                    else:
                        obstacles = self.get_obstacles_from_state(
                            (0, [], [], []), world_model) if world_model else set()
                        cost = self.get_distance(agent_pos, target, obstacles)
                    if cost < min_cost:
                        min_cost = cost
            
            if min_cost == float('inf'):
                return -1.0
            return -min_cost / max_cost
        
        else:
            # Point goal
            target = goal
            if agent_pos == target:
                return 0.0
            
            if world_model is not None and self._shortest_paths is not None:
                cost = self.compute_path_cost(agent_pos, target, world_model)
            else:
                obstacles = self.get_obstacles_from_state(
                    (0, [], [], []), world_model) if world_model else set()
                cost = self.get_distance(agent_pos, target, obstacles)
            
            if cost == float('inf'):
                return -1.0
            return -cost / max_cost
    
    def compute_distances(
        self,
        target: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute shortest path distances from all cells to target.
        
        Args:
            target: Target cell (x, y).
            obstacles: Set of impassable cells.
        
        Returns:
            2D array where [y, x] is distance from (x, y) to target.
            Unreachable cells have distance inf.
        """
        cache_key = (target, frozenset(obstacles))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        distances = np.full((self.grid_height, self.grid_width), float('inf'))
        
        tx, ty = target
        if not (0 <= tx < self.grid_width and 0 <= ty < self.grid_height):
            return distances
        if target in obstacles:
            return distances
        
        distances[ty, tx] = 0
        queue = deque([(tx, ty, 0)])
        visited = {(tx, ty)}
        
        while queue:
            x, y, dist = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue
                if (nx, ny) in visited or (nx, ny) in obstacles:
                    continue
                visited.add((nx, ny))
                distances[ny, nx] = dist + 1
                queue.append((nx, ny, dist + 1))
        
        self._cache[cache_key] = distances
        return distances
    
    def get_distance(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> float:
        """Get distance from start to target."""
        distances = self.compute_distances(target, obstacles)
        sx, sy = start
        if 0 <= sx < self.grid_width and 0 <= sy < self.grid_height:
            return distances[sy, sx]
        return float('inf')
    
    def compute_distances_to_rectangle(
        self,
        rectangle: Tuple[int, int, int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Compute shortest path distances from all cells to any cell in a rectangle.
        
        The distance from a cell to the rectangle is the minimum distance to any
        cell within the rectangle bounds (inclusive).
        
        Args:
            rectangle: Target rectangle (x1, y1, x2, y2) where (x1, y1) is the
                       top-left corner and (x2, y2) is the bottom-right corner.
                       All coordinates are inclusive.
            obstacles: Set of impassable cells.
        
        Returns:
            2D array where [y, x] is distance from (x, y) to the closest cell
            in the rectangle. Unreachable cells have distance inf.
        """
        x1, y1, x2, y2 = rectangle
        
        # Normalize rectangle coordinates (ensure x1 <= x2 and y1 <= y2)
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Use normalized coordinates for cache key for consistency
        normalized_rect = (x1, y1, x2, y2)
        cache_key = (normalized_rect, frozenset(obstacles))
        if cache_key in self._rect_cache:
            return self._rect_cache[cache_key]
        
        distances = np.full((self.grid_height, self.grid_width), float('inf'))
        
        # If rectangle is completely outside the grid, return inf
        if x2 < 0 or x1 >= self.grid_width or y2 < 0 or y1 >= self.grid_height:
            return distances
        
        # Clamp rectangle to grid bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.grid_width - 1, x2)
        y2 = min(self.grid_height - 1, y2)
        
        # Initialize BFS from all cells in the rectangle
        queue = deque()
        visited = set()
        
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if (x, y) not in obstacles:
                    distances[y, x] = 0
                    queue.append((x, y, 0))
                    visited.add((x, y))
        
        # If no valid starting cells in rectangle, return inf
        if not queue:
            return distances
        
        # BFS outward from all rectangle cells
        while queue:
            x, y, dist = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                    continue
                if (nx, ny) in visited or (nx, ny) in obstacles:
                    continue
                visited.add((nx, ny))
                distances[ny, nx] = dist + 1
                queue.append((nx, ny, dist + 1))
        
        self._rect_cache[cache_key] = distances
        return distances
    
    def get_distance_to_rectangle(
        self,
        start: Tuple[int, int],
        rectangle: Tuple[int, int, int, int],
        obstacles: Set[Tuple[int, int]]
    ) -> float:
        """
        Get distance from start to the closest cell in a rectangle.
        
        Args:
            start: Starting cell (x, y).
            rectangle: Target rectangle (x1, y1, x2, y2).
            obstacles: Set of impassable cells.
        
        Returns:
            Distance from start to closest cell in rectangle, or inf if unreachable.
        """
        distances = self.compute_distances_to_rectangle(rectangle, obstacles)
        sx, sy = start
        if 0 <= sx < self.grid_width and 0 <= sy < self.grid_height:
            return distances[sy, sx]
        return float('inf')
    
    def get_distance_to_goal(
        self,
        start: Tuple[int, int],
        goal: Union[Tuple[int, int], Tuple[int, int, int, int]],
        obstacles: Set[Tuple[int, int]]
    ) -> float:
        """
        Get distance from start to a goal (point or rectangle).
        
        This is a unified method that handles both point goals and rectangle goals.
        
        Args:
            start: Starting cell (x, y).
            goal: Either a point goal (x, y) or rectangle goal (x1, y1, x2, y2).
            obstacles: Set of impassable cells.
        
        Returns:
            Distance from start to goal, or inf if unreachable.
        """
        if len(goal) == 2:
            # Point goal
            return self.get_distance(start, goal, obstacles)
        elif len(goal) == 4:
            # Rectangle goal
            return self.get_distance_to_rectangle(start, goal, obstacles)
        else:
            raise ValueError(f"Goal must be a point (x, y) or rectangle (x1, y1, x2, y2), got {goal}")
    
    def clear_cache(self):
        """Clear the distance cache."""
        self._cache.clear()
        self._rect_cache.clear()
    
    def get_obstacles_from_state(self, state: Tuple, world_model: Any) -> Set[Tuple[int, int]]:
        """
        Extract obstacle positions from state.
        
        Args:
            state: Environment state tuple.
            world_model: Environment with grid.
        
        Returns:
            Set of (x, y) positions that are obstacles.
        """
        obstacles = set()
        
        if world_model is None or not hasattr(world_model, 'grid'):
            return obstacles
        
        H = self.grid_height
        W = self.grid_width
        
        for y in range(H):
            for x in range(W):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    # Check if cell blocks movement
                    if cell_type in ('wall', 'lava'):
                        obstacles.add((x, y))
                    elif cell_type == 'door':
                        # Locked/closed doors are obstacles
                        is_open = getattr(cell, 'is_open', False)
                        if not is_open:
                            obstacles.add((x, y))
        
        # Add mobile objects as obstacles
        _, _, mobile_objects, _ = state
        if mobile_objects:
            for obj_data in mobile_objects:
                obj_x, obj_y = obj_data[1], obj_data[2]
                obstacles.add((obj_x, obj_y))
        
        return obstacles
    
    def is_in_goal(
        self,
        pos: Tuple[int, int],
        goal: Union[Tuple[int, int], Tuple[int, int, int, int]]
    ) -> bool:
        """
        Check if a position is at/in the goal.
        
        Args:
            pos: (x, y) position to check.
            goal: Either a point goal (x, y) or rectangle goal (x1, y1, x2, y2).
        
        Returns:
            True if position is at the goal (point) or inside the goal (rectangle).
        """
        if len(goal) == 2:
            return pos == goal
        elif len(goal) == 4:
            x1, y1, x2, y2 = goal
            # Normalize rectangle
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2
        else:
            raise ValueError(f"Goal must be a point (x, y) or rectangle (x1, y1, x2, y2), got {goal}")
