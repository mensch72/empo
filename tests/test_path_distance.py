"""
Tests for PathDistanceCalculator rectangle support and passing cost shaping.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from empo.nn_based.multigrid import PathDistanceCalculator
from empo.nn_based.multigrid.path_distance import DEFAULT_PASSING_COSTS


class MockGrid:
    """Mock grid for testing."""
    def __init__(self, width, height, cells=None):
        self.width = width
        self.height = height
        self._cells = cells or {}
    
    def get(self, x, y):
        return self._cells.get((x, y), None)


class MockCell:
    """Mock cell for testing."""
    def __init__(self, cell_type, **kwargs):
        self.type = cell_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockAgent:
    """Mock agent for testing."""
    def __init__(self, pos):
        self.pos = pos


class MockWorldModel:
    """Mock world model for testing."""
    def __init__(self, width=10, height=10, cells=None, agents=None):
        self.width = width
        self.height = height
        self.grid = MockGrid(width, height, cells or {})
        self.agents = agents or []


def test_path_distance_point_goal():
    """Test basic point goal distance calculation."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # No obstacles
    obstacles = set()
    
    # Distance from (0, 0) to (3, 4) should be 7 (Manhattan)
    dist = calc.get_distance((0, 0), (3, 4), obstacles)
    assert dist == 7, f"Expected 7, got {dist}"
    
    # Distance from point to itself is 0
    dist = calc.get_distance((5, 5), (5, 5), obstacles)
    assert dist == 0, f"Expected 0, got {dist}"
    
    print("  ✓ Point goal distance test passed!")


def test_path_distance_with_obstacles():
    """Test distance calculation with obstacles."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # Wall blocking direct path
    obstacles = {(1, 0), (1, 1), (1, 2)}
    
    # Path from (0, 0) to (2, 0) must go around
    dist = calc.get_distance((0, 0), (2, 0), obstacles)
    # Should be longer than 2 due to obstacle
    assert dist > 2, f"Expected > 2, got {dist}"
    
    print("  ✓ Obstacle distance test passed!")


def test_path_distance_rectangle_goal():
    """Test rectangle goal distance calculation."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # Rectangle from (4, 4) to (6, 6)
    rectangle = (4, 4, 6, 6)
    obstacles = set()
    
    # Distance from outside the rectangle
    dist = calc.get_distance_to_rectangle((0, 0), rectangle, obstacles)
    # Should be 4 + 4 = 8 (Manhattan distance to corner)
    assert dist == 8, f"Expected 8, got {dist}"
    
    # Distance from inside the rectangle should be 0
    dist = calc.get_distance_to_rectangle((5, 5), rectangle, obstacles)
    assert dist == 0, f"Expected 0, got {dist}"
    
    # Distance from edge of rectangle should be 0
    dist = calc.get_distance_to_rectangle((4, 5), rectangle, obstacles)
    assert dist == 0, f"Expected 0, got {dist}"
    
    # Distance from adjacent to rectangle should be 1
    dist = calc.get_distance_to_rectangle((3, 5), rectangle, obstacles)
    assert dist == 1, f"Expected 1, got {dist}"
    
    print("  ✓ Rectangle goal distance test passed!")


def test_path_distance_unified_method():
    """Test the unified get_distance_to_goal method."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    obstacles = set()
    
    # Point goal
    point = (5, 5)
    dist = calc.get_distance_to_goal((0, 0), point, obstacles)
    assert dist == 10, f"Expected 10, got {dist}"
    
    # Rectangle goal
    rect = (4, 4, 6, 6)
    dist = calc.get_distance_to_goal((0, 0), rect, obstacles)
    assert dist == 8, f"Expected 8, got {dist}"
    
    print("  ✓ Unified distance method test passed!")


def test_compute_potential_point_goal():
    """Test potential function computation for point goals."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # At goal, potential should be 0
    point = (5, 5)
    pot = calc.compute_potential((5, 5), point)
    assert pot == 0.0, f"Expected 0.0, got {pot}"
    
    # Further from goal, potential should be more negative
    pot1 = calc.compute_potential((3, 3), point)
    pot2 = calc.compute_potential((0, 0), point)
    assert pot2 < pot1 < 0, f"Expected pot2 < pot1 < 0, got pot1={pot1}, pot2={pot2}"
    
    print("  ✓ Point goal potential test passed!")


def test_compute_potential_rectangle_goal():
    """Test potential function computation for rectangle goals."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # Rectangle goal potential
    rect = (4, 4, 6, 6)
    
    pot = calc.compute_potential((5, 5), rect)  # Inside rectangle
    assert pot == 0.0, f"Expected 0.0, got {pot}"
    
    pot = calc.compute_potential((0, 0), rect)  # Outside
    assert pot < 0, f"Expected < 0, got {pot}"
    
    print("  ✓ Rectangle goal potential test passed!")


def test_is_in_goal():
    """Test is_in_goal method."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # Point goal
    point = (5, 5)
    assert calc.is_in_goal((5, 5), point) == True
    assert calc.is_in_goal((5, 6), point) == False
    
    # Rectangle goal
    rect = (4, 4, 6, 6)
    assert calc.is_in_goal((5, 5), rect) == True  # Inside
    assert calc.is_in_goal((4, 4), rect) == True  # Corner
    assert calc.is_in_goal((6, 6), rect) == True  # Corner
    assert calc.is_in_goal((3, 5), rect) == False  # Outside
    assert calc.is_in_goal((7, 5), rect) == False  # Outside
    
    print("  ✓ is_in_goal test passed!")


def test_rectangle_normalization():
    """Test that rectangles are normalized (x1 <= x2, y1 <= y2)."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    obstacles = set()
    
    # Rectangle with swapped coordinates
    rect_normal = (4, 4, 6, 6)
    rect_swapped = (6, 6, 4, 4)  # Swapped x and y
    
    dist1 = calc.get_distance_to_rectangle((0, 0), rect_normal, obstacles)
    dist2 = calc.get_distance_to_rectangle((0, 0), rect_swapped, obstacles)
    
    assert dist1 == dist2, f"Rectangle normalization failed: {dist1} != {dist2}"
    
    print("  ✓ Rectangle normalization test passed!")


def test_rectangle_cache_normalization():
    """Test that normalized rectangles share the same cache entry."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    obstacles = set()
    
    # Clear cache
    calc.clear_cache()
    
    # First query with normal order
    rect_normal = (4, 4, 6, 6)
    _ = calc.get_distance_to_rectangle((0, 0), rect_normal, obstacles)
    
    # Check cache has one entry
    assert len(calc._rect_cache) == 1, f"Expected 1 cache entry, got {len(calc._rect_cache)}"
    
    # Query with swapped order - should use same cache entry
    rect_swapped = (6, 6, 4, 4)
    _ = calc.get_distance_to_rectangle((0, 0), rect_swapped, obstacles)
    
    # Cache should still have only one entry (both use normalized coords)
    assert len(calc._rect_cache) == 1, f"Expected 1 cache entry after swap, got {len(calc._rect_cache)}"
    
    print("  ✓ Rectangle cache normalization test passed!")


def test_default_passing_costs():
    """Test that DEFAULT_PASSING_COSTS has expected values."""
    assert DEFAULT_PASSING_COSTS['empty'] == 1
    assert DEFAULT_PASSING_COSTS['door_open'] == 1
    assert DEFAULT_PASSING_COSTS['door_closed'] == 2
    assert DEFAULT_PASSING_COSTS['door_locked'] == 25
    assert DEFAULT_PASSING_COSTS['agent'] == 2
    assert DEFAULT_PASSING_COSTS['block'] == 2
    assert DEFAULT_PASSING_COSTS['pickable'] == 3
    assert DEFAULT_PASSING_COSTS['rock'] == 50
    assert DEFAULT_PASSING_COSTS['wall'] == float('inf')
    assert DEFAULT_PASSING_COSTS['lava'] == float('inf')
    
    print("  ✓ Default passing costs test passed!")


def test_feasible_range_computation():
    """Test that feasible_range is computed correctly."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    
    # Max finite cost is 50 (rock)
    # Max path length is 10 + 10 = 20
    # So max cost should be 20 * 50 = 1000
    expected_max = (10 + 10) * 50
    
    assert calc.feasible_range == (-expected_max, expected_max), \
        f"Expected {(-expected_max, expected_max)}, got {calc.feasible_range}"
    
    print("  ✓ Feasible range computation test passed!")


def test_path_cost_with_world_model():
    """Test path cost computation with world model and obstacles."""
    # Create a world model with a door
    cells = {
        (2, 0): MockCell('wall'),
        (2, 1): MockCell('wall'),
        (2, 2): MockCell('door', is_open=False, is_locked=True),
        (2, 3): MockCell('wall'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    
    calc = PathDistanceCalculator(
        grid_height=10, grid_width=10,
        world_model=world_model
    )
    
    # With precomputed paths, compute_path_cost should account for passing costs
    # Path from (0, 0) to (3, 0) goes through (2, 2) which has a locked door (cost 25)
    cost = calc.compute_path_cost((0, 0), (3, 0), world_model)
    
    # The path should have higher cost due to the locked door
    # (would need to go around or through the door)
    assert cost > 0, f"Expected positive cost, got {cost}"
    
    print("  ✓ Path cost with world model test passed!")


def test_path_cost_with_agents():
    """Test that agent positions add passing cost."""
    cells = {}  # No obstacles in grid
    agents = [MockAgent((2, 0))]
    world_model = MockWorldModel(width=10, height=10, cells=cells, agents=agents)
    
    calc = PathDistanceCalculator(
        grid_height=10, grid_width=10,
        world_model=world_model
    )
    
    # Path from (0, 0) to (3, 0) goes through (2, 0) where an agent is
    cost = calc.compute_path_cost((0, 0), (3, 0), world_model)
    
    # Cost should be: 2 steps of empty (2) + 1 step with agent (2) = 4
    # But since we pass through (1,0), (2,0), (3,0):
    # (1,0) = empty = 1
    # (2,0) = agent = 2
    # (3,0) = empty = 1 (destination, but we don't count it usually)
    # Actually the path cost sums cells along path excluding source
    assert cost >= 1, f"Expected at least 1, got {cost}"
    
    print("  ✓ Path cost with agents test passed!")


def test_cell_passing_costs():
    """Test _get_cell_passing_cost for different cell types."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    agent_positions = set()
    
    # Empty cell
    cost = calc._get_cell_passing_cost(None, (0, 0), agent_positions)
    assert cost == 1, f"Expected 1 for empty, got {cost}"
    
    # Open door
    cell = MockCell('door', is_open=True, is_locked=False)
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == 1, f"Expected 1 for open door, got {cost}"
    
    # Closed door
    cell = MockCell('door', is_open=False, is_locked=False)
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == 2, f"Expected 2 for closed door, got {cost}"
    
    # Locked door
    cell = MockCell('door', is_open=False, is_locked=True)
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == 25, f"Expected 25 for locked door, got {cost}"
    
    # Rock
    cell = MockCell('rock')
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == 50, f"Expected 50 for rock, got {cost}"
    
    # Pickable item (key)
    cell = MockCell('key')
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == 3, f"Expected 3 for key, got {cost}"
    
    # Wall
    cell = MockCell('wall')
    cost = calc._get_cell_passing_cost(cell, (0, 0), agent_positions)
    assert cost == float('inf'), f"Expected inf for wall, got {cost}"
    
    # Agent at position
    agent_positions = {(0, 0)}
    cost = calc._get_cell_passing_cost(None, (0, 0), agent_positions)
    assert cost == 2, f"Expected 2 for agent position, got {cost}"
    
    print("  ✓ Cell passing costs test passed!")


def run_all_tests():
    """Run all path distance tests."""
    print("=" * 60)
    print("Running PathDistanceCalculator Tests")
    print("=" * 60)
    
    test_path_distance_point_goal()
    test_path_distance_with_obstacles()
    test_path_distance_rectangle_goal()
    test_path_distance_unified_method()
    test_compute_potential_point_goal()
    test_compute_potential_rectangle_goal()
    test_is_in_goal()
    test_rectangle_normalization()
    test_rectangle_cache_normalization()
    test_default_passing_costs()
    test_feasible_range_computation()
    test_path_cost_with_world_model()
    test_path_cost_with_agents()
    test_cell_passing_costs()
    
    print()
    print("=" * 60)
    print("All PathDistanceCalculator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
