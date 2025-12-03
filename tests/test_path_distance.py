"""
Tests for PathDistanceCalculator rectangle support.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from empo.nn_based.multigrid import PathDistanceCalculator


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


def test_compute_potential():
    """Test potential function computation."""
    calc = PathDistanceCalculator(grid_height=10, grid_width=10)
    obstacles = set()
    
    # At goal, potential should be 0
    point = (5, 5)
    pot = calc.compute_potential((5, 5), point, obstacles)
    assert pot == 0.0, f"Expected 0.0, got {pot}"
    
    # Further from goal, potential should be more negative
    pot1 = calc.compute_potential((3, 3), point, obstacles)
    pot2 = calc.compute_potential((0, 0), point, obstacles)
    assert pot2 < pot1 < 0, f"Expected pot2 < pot1 < 0, got pot1={pot1}, pot2={pot2}"
    
    # Rectangle goal potential
    rect = (4, 4, 6, 6)
    pot = calc.compute_potential((5, 5), rect, obstacles)  # Inside rectangle
    assert pot == 0.0, f"Expected 0.0, got {pot}"
    
    pot = calc.compute_potential((0, 0), rect, obstacles)  # Outside
    assert pot < 0, f"Expected < 0, got {pot}"
    
    print("  ✓ Potential function test passed!")


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


def run_all_tests():
    """Run all path distance tests."""
    print("=" * 60)
    print("Running PathDistanceCalculator Tests")
    print("=" * 60)
    
    test_path_distance_point_goal()
    test_path_distance_with_obstacles()
    test_path_distance_rectangle_goal()
    test_path_distance_unified_method()
    test_compute_potential()
    test_is_in_goal()
    test_rectangle_normalization()
    test_rectangle_cache_normalization()
    
    print()
    print("=" * 60)
    print("All PathDistanceCalculator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
