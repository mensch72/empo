"""
Path distance calculator for multigrid environments.

Computes shortest path distances on grids for reward shaping.
"""

import numpy as np
from typing import Any, Dict, Set, Tuple
from collections import deque


class PathDistanceCalculator:
    """
    Computes shortest path distances on a multigrid.
    
    Uses BFS accounting for obstacles. Caches results for efficiency.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
    """
    
    def __init__(self, grid_height: int, grid_width: int):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self._cache: Dict[Tuple, np.ndarray] = {}
    
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
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
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
    
    def clear_cache(self):
        """Clear the distance cache."""
        self._cache.clear()
    
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
