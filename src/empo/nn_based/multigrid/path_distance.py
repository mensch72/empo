"""
Path distance calculator for multigrid environments.

Computes shortest path distances on grids for reward shaping.
Supports both point targets and rectangular region targets.
"""

import numpy as np
from typing import Any, Dict, Set, Tuple, Union
from collections import deque


class PathDistanceCalculator:
    """
    Computes shortest path distances on a multigrid.
    
    Uses BFS accounting for obstacles. Caches results for efficiency.
    Supports both point targets and rectangular region targets.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
    """
    
    def __init__(self, grid_height: int, grid_width: int):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self._cache: Dict[Tuple, np.ndarray] = {}
        self._rect_cache: Dict[Tuple, np.ndarray] = {}
    
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
    
    def compute_potential(
        self,
        agent_pos: Tuple[int, int],
        goal: Union[Tuple[int, int], Tuple[int, int, int, int]],
        obstacles: Set[Tuple[int, int]],
        max_distance: float = None
    ) -> float:
        """
        Compute potential function value for reward shaping.
        
        Uses the potential function: Φ(s) = -distance(agent, goal) / max_distance
        
        The potential is maximal (0) when agent is at the goal.
        For rectangle goals, the distance is to the closest cell in the rectangle.
        
        Args:
            agent_pos: (x, y) current agent position.
            goal: Either a point goal (x, y) or rectangle goal (x1, y1, x2, y2).
            obstacles: Set of impassable cells.
            max_distance: Maximum distance for normalization. If None, uses grid diagonal.
        
        Returns:
            Potential value in range [-1, 0] (approximately).
        """
        if max_distance is None:
            max_distance = self.grid_width + self.grid_height
        
        distance = self.get_distance_to_goal(agent_pos, goal, obstacles)
        
        if distance == float('inf'):
            return -1.0  # No path - minimum potential
        
        return -distance / max_distance
    
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
