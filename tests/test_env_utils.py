#!/usr/bin/env python3
"""
Tests for env_utils module.

Tests the get_dag function for computing DAG structure of gym environments.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import empo
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from empo.env_utils import get_dag


class SimpleMockEnv:
    """
    A minimal mock environment for testing get_dag.
    
    This creates a simple DAG structure:
        State 0 (root)
         /    \
      State 1  State 2
         \    /
        State 3 (terminal)
    
    This demonstrates the critical case where State 3 is reachable via
    two paths of different lengths.
    """
    
    def __init__(self):
        self.current_state = 0
        self.agents = [None]  # Single agent
        self.action_space = MockActionSpace()
    
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def get_state(self):
        return self.current_state
    
    def set_state(self, state):
        self.current_state = state
    
    def transition_probabilities(self, state, actions):
        """
        Define the DAG structure:
        - State 0 -> State 1 (action 0) or State 2 (action 1)
        - State 1 -> State 3 (any action)
        - State 2 -> State 3 (any action)
        - State 3 -> None (terminal)
        """
        if state == 3:
            # Terminal state
            return None
        
        if state == 0:
            # Root can go to state 1 or state 2 depending on action
            if actions[0] == 0:
                return [(1.0, 1)]
            else:
                return [(1.0, 2)]
        elif state == 1:
            # State 1 always goes to state 3
            return [(1.0, 3)]
        elif state == 2:
            # State 2 always goes to state 3
            return [(1.0, 3)]
        else:
            return None


class MockActionSpace:
    """Mock action space with 2 actions."""
    def __init__(self):
        self.n = 2


def create_tiny_env():
    """Create a very simple mock environment for testing."""
    return SimpleMockEnv()


def test_get_dag_returns_correct_structure():
    """Test that get_dag returns the correct data structure."""
    print("Test: get_dag returns correct structure...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # Check that all return values are of correct type
    if not isinstance(states, list):
        print(f"  ✗ states should be a list, got {type(states)}")
        return False
    
    if not isinstance(state_to_idx, dict):
        print(f"  ✗ state_to_idx should be a dict, got {type(state_to_idx)}")
        return False
    
    if not isinstance(successors, list):
        print(f"  ✗ successors should be a list, got {type(successors)}")
        return False
    
    print(f"  ✓ Returns tuple of (list, dict, list)")
    print(f"  ✓ Found {len(states)} reachable states")
    return True


def test_get_dag_root_state_is_first():
    """Test that the root state is the first element."""
    print("Test: root state is first...")
    
    env = create_tiny_env()
    env.reset()
    root_state = env.get_state()
    
    states, state_to_idx, successors = get_dag(env)
    
    if states[0] == root_state:
        print("  ✓ Root state is first in list")
        return True
    else:
        print("  ✗ Root state is not first in list")
        return False


def test_get_dag_state_to_idx_consistency():
    """Test that state_to_idx mapping is consistent with states list."""
    print("Test: state_to_idx consistency...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # Check that state_to_idx has same length as states
    if len(state_to_idx) != len(states):
        print(f"  ✗ state_to_idx has {len(state_to_idx)} entries but states has {len(states)}")
        return False
    
    # Check that each state maps to its correct index
    for i, state in enumerate(states):
        if state_to_idx.get(state) != i:
            print(f"  ✗ states[{i}] maps to index {state_to_idx.get(state)} instead of {i}")
            return False
    
    print(f"  ✓ All {len(states)} states map to correct indices")
    return True


def test_get_dag_successors_consistency():
    """Test that successors list has correct length and valid indices."""
    print("Test: successors list consistency...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # Check that successors has same length as states
    if len(successors) != len(states):
        print(f"  ✗ successors has {len(successors)} entries but states has {len(states)}")
        return False
    
    # Check that all successor indices are valid
    for i, successor_list in enumerate(successors):
        if not isinstance(successor_list, list):
            print(f"  ✗ successors[{i}] is not a list")
            return False
        
        for successor_idx in successor_list:
            if not isinstance(successor_idx, int):
                print(f"  ✗ successors[{i}] contains non-integer: {successor_idx}")
                return False
            
            if successor_idx < 0 or successor_idx >= len(states):
                print(f"  ✗ successors[{i}] contains invalid index: {successor_idx}")
                return False
    
    print(f"  ✓ Successors list is valid")
    return True


def test_get_dag_topological_ordering():
    """Test that successors always come after predecessors (topological ordering)."""
    print("Test: topological ordering...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # For each state, check that all successors have higher indices
    violations = []
    for i, successor_list in enumerate(successors):
        for successor_idx in successor_list:
            if successor_idx <= i:
                violations.append((i, successor_idx))
    
    if violations:
        print(f"  ✗ Found {len(violations)} topological ordering violations")
        for pred, succ in violations[:5]:  # Show first 5
            print(f"    State {pred} -> State {succ} (successor has lower/equal index)")
        return False
    
    print(f"  ✓ All successors have higher indices than predecessors")
    return True


def test_get_dag_multiple_paths_different_lengths():
    """
    Test the CRITICAL case: when a state is reachable via paths of different lengths.
    
    This is the key insight that BFS alone would fail on. Consider this DAG:
    
        Root (A)
        /     \\
       B       C
        \\     /
          D
    
    If we discover states via BFS:
    - A is discovered first (index 0)
    - B is discovered next via A→B (index 1)
    - D is discovered next via B→D (index 2)  <-- WRONG!
    - C is discovered next via A→C (index 3)
    - But C→D is an edge, meaning C (index 3) → D (index 2)
    - This violates topological order!
    
    With proper topological sort:
    - A gets index 0
    - B and C get indices 1,2 (order doesn't matter, both depend only on A)
    - D gets index 3 (depends on both B and C)
    
    This test verifies that our implementation handles this correctly.
    """
    print("Test: multiple paths with different lengths (CRITICAL)...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # Find all states that have multiple predecessors
    predecessors = [[] for _ in range(len(states))]
    for i, succ_list in enumerate(successors):
        for succ_idx in succ_list:
            predecessors[succ_idx].append(i)
    
    states_with_multiple_paths = [i for i in range(len(states)) if len(predecessors[i]) > 1]
    
    if len(states_with_multiple_paths) == 0:
        print("  ⚠ No states with multiple predecessors found in this environment")
        print("    (Test is vacuous but passes)")
        return True
    
    print(f"  • Found {len(states_with_multiple_paths)} state(s) with multiple predecessors")
    
    # For each such state, verify ALL predecessors have lower index
    violations = []
    for state_idx in states_with_multiple_paths:
        for pred_idx in predecessors[state_idx]:
            if pred_idx >= state_idx:
                violations.append((pred_idx, state_idx))
    
    if violations:
        print(f"  ✗ Found {len(violations)} violations of topological ordering:")
        MAX_VIOLATIONS_TO_SHOW = 3
        for pred, succ in violations[:MAX_VIOLATIONS_TO_SHOW]:
            print(f"    State {pred} → State {succ} (predecessor has higher/equal index!)")
        return False
    
    # Additionally, verify that the state appears AFTER all its predecessors
    # This is a stronger check: it should appear after the MAXIMUM predecessor index
    MAX_VIOLATIONS_TO_SHOW = 3
    for state_idx in states_with_multiple_paths:
        max_pred_idx = max(predecessors[state_idx])
        if state_idx <= max_pred_idx:
            print(f"  ✗ State {state_idx} has index ≤ its maximum predecessor {max_pred_idx}")
            return False
    
    print(f"  ✓ All {len(states_with_multiple_paths)} states with multiple paths are correctly ordered")
    print(f"    (All predecessors have strictly lower indices)")
    return True


def test_get_dag_no_duplicate_successors():
    """Test that each state's successor list has no duplicates."""
    print("Test: no duplicate successors...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    duplicates_found = False
    for i, successor_list in enumerate(successors):
        if len(successor_list) != len(set(successor_list)):
            print(f"  ✗ State {i} has duplicate successors: {successor_list}")
            duplicates_found = True
    
    if not duplicates_found:
        print("  ✓ No duplicate successors found")
        return True
    
    return False


def test_get_dag_all_states_unique():
    """Test that all states in the list are unique."""
    print("Test: all states unique...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    if len(states) == len(set(states)):
        print(f"  ✓ All {len(states)} states are unique")
        return True
    else:
        print(f"  ✗ Found duplicate states: {len(states)} total, {len(set(states))} unique")
        return False


def test_get_dag_reachability():
    """Test that all states are reachable from root by following successor edges."""
    print("Test: all states reachable from root...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # BFS from root to find all reachable states
    reachable = {0}  # Root is at index 0
    queue = [0]
    
    while queue:
        current = queue.pop(0)
        for successor in successors[current]:
            if successor not in reachable:
                reachable.add(successor)
                queue.append(successor)
    
    if len(reachable) == len(states):
        print(f"  ✓ All {len(states)} states are reachable from root")
        return True
    else:
        unreachable = set(range(len(states))) - reachable
        print(f"  ✗ {len(unreachable)} states not reachable from root: {list(unreachable)[:5]}...")
        return False


def test_get_dag_with_simple_env():
    """Test get_dag with the specific diamond-shaped DAG."""
    print("Test: diamond DAG structure...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # We expect exactly 4 states: 0, 1, 2, 3
    if len(states) != 4:
        print(f"  ✗ Expected 4 states, got {len(states)}")
        return False
    
    # State 0 should be root
    if states[0] != 0:
        print(f"  ✗ Expected root state 0, got {states[0]}")
        return False
    
    # State 3 should be last (terminal state with highest index)
    if states[3] != 3:
        print(f"  ✗ Expected terminal state 3 at index 3, got {states[3]}")
        return False
    
    # States 1 and 2 should be in the middle (indices 1 and 2)
    middle_states = {states[1], states[2]}
    if middle_states != {1, 2}:
        print(f"  ✗ Expected states 1 and 2 in middle, got {middle_states}")
        return False
    
    print(f"  ✓ Diamond DAG structure is correct")
    print(f"    Topological order: {states}")
    return True


def test_get_dag_terminal_states():
    """Test that terminal states have no successors."""
    print("Test: terminal states have no successors...")
    
    env = create_tiny_env()
    states, state_to_idx, successors = get_dag(env)
    
    # Find terminal states (those where transition_probabilities returns None)
    terminal_states = []
    for i, state in enumerate(states):
        # Check if this is a terminal state by trying to get transitions
        num_agents = len(env.agents)
        actions = [0] * num_agents  # Try with action 0
        transitions = env.transition_probabilities(state, actions)
        
        if transitions is None:
            terminal_states.append(i)
    
    # Check that terminal states have no successors
    all_correct = True
    for i in terminal_states:
        if len(successors[i]) > 0:
            print(f"  ✗ Terminal state {i} has successors: {successors[i]}")
            all_correct = False
    
    if all_correct:
        print(f"  ✓ All {len(terminal_states)} terminal states have no successors")
        return True
    
    return False


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_get_dag_returns_correct_structure,
        test_get_dag_root_state_is_first,
        test_get_dag_state_to_idx_consistency,
        test_get_dag_successors_consistency,
        test_get_dag_topological_ordering,
        test_get_dag_multiple_paths_different_lengths,  # CRITICAL TEST
        test_get_dag_no_duplicate_successors,
        test_get_dag_all_states_unique,
        test_get_dag_reachability,
        test_get_dag_with_simple_env,
        test_get_dag_terminal_states,
    ]
    
    print("=" * 60)
    print("Running get_dag tests")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ Test raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
