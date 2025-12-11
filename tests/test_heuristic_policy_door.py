#!/usr/bin/env python3
"""
Tests for HeuristicPotentialPolicy door handling functionality.

Tests the new door handling feature that overrides potential-based action
when the agent is adjacent to an actionable door (locked with key, or closed).
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from empo.human_policy_prior import HeuristicPotentialPolicy


class MockCell:
    """Mock cell for testing."""
    def __init__(self, cell_type, **kwargs):
        self.type = cell_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockGrid:
    """Mock grid for testing."""
    def __init__(self, width, height, cells=None):
        self.width = width
        self.height = height
        self._cells = cells or {}
    
    def get(self, x, y):
        return self._cells.get((x, y), None)


class MockWorldModel:
    """Mock world model for testing."""
    def __init__(self, width=10, height=10, cells=None, agents=None):
        self.width = width
        self.height = height
        self.grid = MockGrid(width, height, cells or {})
        self.agents = agents or []


class MockPathCalculator:
    """Mock path calculator for testing."""
    def __init__(self):
        pass
    
    def compute_potential(self, pos, goal, world_model=None):
        # Simple potential: negative Manhattan distance to goal
        if isinstance(goal, tuple):
            if len(goal) == 2:
                # Point goal
                return -(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))
            elif len(goal) == 4:
                # Rectangle goal: distance to nearest edge
                x1, y1, x2, y2 = goal
                dx = max(0, x1 - pos[0], pos[0] - x2)
                dy = max(0, y1 - pos[1], pos[1] - y2)
                return -(dx + dy)
        return 0.0
    
    def is_in_goal(self, pos, goal):
        if isinstance(goal, tuple):
            if len(goal) == 2:
                return pos == goal
            elif len(goal) == 4:
                x1, y1, x2, y2 = goal
                return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2
        return False


class MockGoal:
    """Mock goal for testing."""
    def __init__(self, target_pos):
        self.target_pos = target_pos


def test_find_actionable_door_locked_with_key():
    """Test that locked door with matching key is detected."""
    print("Testing: locked door with matching key...")
    
    # Create a world with a locked red door to the east of agent
    cells = {
        (3, 2): MockCell('door', is_open=False, is_locked=True, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4
    )
    
    # Agent at (2, 2) carrying red key
    agent_x, agent_y = 2, 2
    carrying_type = 'key'
    carrying_color = 'red'
    
    door_dir = policy._find_actionable_door(agent_x, agent_y, carrying_type, carrying_color)
    
    # Should find door to the east (direction 0)
    assert door_dir == 0, f"Expected direction 0 (east), got {door_dir}"
    print("  ✓ Locked door with matching key detected correctly")


def test_find_actionable_door_locked_wrong_key():
    """Test that locked door with wrong key is NOT detected."""
    print("Testing: locked door with wrong key (should not detect)...")
    
    # Create a world with a locked red door to the east
    cells = {
        (3, 2): MockCell('door', is_open=False, is_locked=True, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4
    )
    
    # Agent at (2, 2) carrying BLUE key (wrong color)
    agent_x, agent_y = 2, 2
    carrying_type = 'key'
    carrying_color = 'blue'
    
    door_dir = policy._find_actionable_door(agent_x, agent_y, carrying_type, carrying_color)
    
    # Should NOT find the door (wrong key color)
    assert door_dir is None, f"Expected None (wrong key), got {door_dir}"
    print("  ✓ Locked door with wrong key correctly ignored")


def test_find_actionable_door_unlocked_closed():
    """Test that unlocked closed door is detected."""
    print("Testing: unlocked closed door...")
    
    # Create a world with an unlocked closed door to the south
    cells = {
        (2, 3): MockCell('door', is_open=False, is_locked=False, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4
    )
    
    # Agent at (2, 2) not carrying anything
    agent_x, agent_y = 2, 2
    carrying_type = None
    carrying_color = None
    
    door_dir = policy._find_actionable_door(agent_x, agent_y, carrying_type, carrying_color)
    
    # Should find door to the south (direction 1)
    assert door_dir == 1, f"Expected direction 1 (south), got {door_dir}"
    print("  ✓ Unlocked closed door detected correctly")


def test_find_actionable_door_open():
    """Test that open door is NOT detected (already open)."""
    print("Testing: open door (should not detect)...")
    
    # Create a world with an open door to the east
    cells = {
        (3, 2): MockCell('door', is_open=True, is_locked=False, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4
    )
    
    # Agent at (2, 2)
    agent_x, agent_y = 2, 2
    
    door_dir = policy._find_actionable_door(agent_x, agent_y, None, None)
    
    # Should NOT find the door (already open)
    assert door_dir is None, f"Expected None (door already open), got {door_dir}"
    print("  ✓ Open door correctly ignored")


def test_turn_action_calculation():
    """Test turn action calculation to face a target direction."""
    print("Testing: turn action calculation...")
    
    world_model = MockWorldModel()
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4
    )
    
    # Test cases: (current_dir, target_dir) -> expected action
    test_cases = [
        # Already facing (return forward)
        (0, 0, policy.ACTION_FORWARD),
        (1, 1, policy.ACTION_FORWARD),
        (2, 2, policy.ACTION_FORWARD),
        (3, 3, policy.ACTION_FORWARD),
        
        # Turn right once (diff == 1)
        (0, 1, policy.ACTION_RIGHT),  # East to South
        (1, 2, policy.ACTION_RIGHT),  # South to West
        (2, 3, policy.ACTION_RIGHT),  # West to North
        (3, 0, policy.ACTION_RIGHT),  # North to East
        
        # Turn left once (diff == 3)
        (1, 0, policy.ACTION_LEFT),   # South to East
        (2, 1, policy.ACTION_LEFT),   # West to South
        (3, 2, policy.ACTION_LEFT),   # North to West
        (0, 3, policy.ACTION_LEFT),   # East to North
        
        # 180 degrees (diff == 2, prefer right)
        (0, 2, policy.ACTION_RIGHT),  # East to West
        (1, 3, policy.ACTION_RIGHT),  # South to North
    ]
    
    for current_dir, target_dir, expected_action in test_cases:
        result = policy._get_turn_action_to_face(current_dir, target_dir)
        assert result == expected_action, \
            f"Turn from {current_dir} to {target_dir}: expected {expected_action}, got {result}"
    
    print("  ✓ Turn action calculation correct for all cases")


def test_policy_override_turn_toward_door():
    """Test that policy returns turn action when not facing actionable door."""
    print("Testing: policy override - turn toward door...")
    
    # Create a world with unlocked closed door to the east
    cells = {
        (3, 2): MockCell('door', is_open=False, is_locked=False, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4,
        softness=10.0
    )
    
    # Agent at (2, 2), facing south (direction 1), door is to the east (direction 0)
    # Agent state: (x, y, dir, terminated, started, paused, carrying_type, carrying_color, forced_action)
    agent_state = (2, 2, 1, False, True, False, None, None, None)
    state = (0, [agent_state], [], [])
    
    goal = MockGoal((5, 5))  # Goal far away
    
    probs = policy(state, human_agent_index=0, possible_goal=goal)
    
    # Door is to the east (dir 0), agent facing south (dir 1)
    # Need to turn left (from south to east = diff 3)
    expected_action = policy.ACTION_LEFT
    
    assert probs[expected_action] == 1.0, \
        f"Expected ACTION_LEFT (prob=1.0), got probs={probs}"
    print("  ✓ Policy correctly returns turn action toward door")


def test_policy_override_forward_when_facing_door():
    """Test that policy returns forward when facing actionable door (SmallActions)."""
    print("Testing: policy override - forward when facing door (SmallActions)...")
    
    # Create a world with unlocked closed door to the east
    cells = {
        (3, 2): MockCell('door', is_open=False, is_locked=False, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4,  # SmallActions
        softness=10.0
    )
    
    # Agent at (2, 2), facing east (direction 0) toward the door
    agent_state = (2, 2, 0, False, True, False, None, None, None)
    state = (0, [agent_state], [], [])
    
    goal = MockGoal((5, 5))
    
    probs = policy(state, human_agent_index=0, possible_goal=goal)
    
    # When facing door with SmallActions (num_actions=4), should use forward
    assert probs[policy.ACTION_FORWARD] == 1.0, \
        f"Expected ACTION_FORWARD (prob=1.0), got probs={probs}"
    print("  ✓ Policy correctly returns forward when facing door (SmallActions)")


def test_policy_override_toggle_when_facing_door():
    """Test that policy returns toggle when facing actionable door (full Actions)."""
    print("Testing: policy override - toggle when facing door (full Actions)...")
    
    # Create a world with unlocked closed door to the east
    cells = {
        (3, 2): MockCell('door', is_open=False, is_locked=False, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=8,  # Full Actions set (includes toggle at index 6)
        softness=10.0
    )
    
    # Agent at (2, 2), facing east (direction 0) toward the door
    agent_state = (2, 2, 0, False, True, False, None, None, None)
    state = (0, [agent_state], [], [])
    
    goal = MockGoal((5, 5))
    
    probs = policy(state, human_agent_index=0, possible_goal=goal)
    
    # When facing door with full Actions (num_actions>6), should use toggle (index 6)
    assert probs[policy.ACTION_TOGGLE] == 1.0, \
        f"Expected ACTION_TOGGLE (prob=1.0), got probs={probs}"
    print("  ✓ Policy correctly returns toggle when facing door (full Actions)")


def test_policy_no_door_uses_potential():
    """Test that policy uses potential-based action when no actionable door."""
    print("Testing: no actionable door - uses potential-based action...")
    
    # No doors in the world
    world_model = MockWorldModel(width=10, height=10, cells={})
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4,
        softness=1000.0  # High softness = deterministic
    )
    
    # Agent at (2, 2), facing east (direction 0), goal to the east
    agent_state = (2, 2, 0, False, True, False, None, None, None)
    state = (0, [agent_state], [], [])
    
    goal = MockGoal((5, 2))  # Goal directly to the east
    
    probs = policy(state, human_agent_index=0, possible_goal=goal)
    
    # Should prefer forward (toward goal) since no door override
    # With high softness, should be nearly 1.0 for forward
    assert probs[policy.ACTION_FORWARD] > 0.9, \
        f"Expected high prob for ACTION_FORWARD, got probs={probs}"
    print("  ✓ Policy correctly uses potential-based action when no door")


def test_policy_locked_door_with_key():
    """Test complete scenario: locked door with matching key."""
    print("Testing: complete scenario - locked door with key...")
    
    # Create a world with a locked red door to the north
    cells = {
        (2, 1): MockCell('door', is_open=False, is_locked=True, color='red'),
    }
    world_model = MockWorldModel(width=10, height=10, cells=cells)
    path_calc = MockPathCalculator()
    
    policy = HeuristicPotentialPolicy(
        world_model=world_model,
        human_agent_indices=[0],
        path_calculator=path_calc,
        num_actions=4,
        softness=10.0
    )
    
    # Agent at (2, 2), facing south (dir 1), carrying red key
    # Door is to the north (dir 3)
    agent_state = (2, 2, 1, False, True, False, 'key', 'red', None)
    state = (0, [agent_state], [], [])
    
    goal = MockGoal((2, 0))  # Goal beyond the door
    
    probs = policy(state, human_agent_index=0, possible_goal=goal)
    
    # Door to north (dir 3), agent facing south (dir 1), need to turn left (diff=2 -> right)
    # From south (1) to north (3): diff = (3-1)%4 = 2 -> ACTION_RIGHT
    assert probs[policy.ACTION_RIGHT] == 1.0, \
        f"Expected ACTION_RIGHT (prob=1.0), got probs={probs}"
    print("  ✓ Policy correctly handles locked door with matching key")


def run_all_tests():
    """Run all door handling tests."""
    print("=" * 60)
    print("Running HeuristicPotentialPolicy Door Handling Tests")
    print("=" * 60)
    print()
    
    test_find_actionable_door_locked_with_key()
    print()
    
    test_find_actionable_door_locked_wrong_key()
    print()
    
    test_find_actionable_door_unlocked_closed()
    print()
    
    test_find_actionable_door_open()
    print()
    
    test_turn_action_calculation()
    print()
    
    test_policy_override_turn_toward_door()
    print()
    
    test_policy_override_forward_when_facing_door()
    print()
    
    test_policy_override_toggle_when_facing_door()
    print()
    
    test_policy_no_door_uses_potential()
    print()
    
    test_policy_locked_door_with_key()
    print()
    
    print("=" * 60)
    print("All HeuristicPotentialPolicy door handling tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
