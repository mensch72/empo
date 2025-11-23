#!/usr/bin/env python3
"""
Example script showcasing the get_dag function and DAG visualization.

This demonstrates how to:
1. Create a simple environment with a diamond-shaped DAG structure
2. Compute the DAG using get_dag()
3. Visualize the DAG using plot_dag()
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from empo.env_utils import get_dag, plot_dag


class DiamondDAGEnv:
    """
    A simple environment with a diamond-shaped DAG structure.
    
    This demonstrates the critical case where a state (State 3) is reachable
    via two paths of different lengths:
    
        State 0 (root)
         /    \\
      State 1  State 2
         \\    /
        State 3 (terminal)
    
    - From State 0, action 0 leads to State 1, action 1 leads to State 2
    - From State 1, any action leads to State 3
    - From State 2, any action leads to State 3
    - State 3 is terminal
    """
    
    def __init__(self):
        self.current_state = 0
        self.agents = [None]  # Single agent
        self.action_space = MockActionSpace()
    
    def reset(self):
        """Reset to initial state."""
        self.current_state = 0
        return self.current_state
    
    def get_state(self):
        """Get current state."""
        return self.current_state
    
    def set_state(self, state):
        """Set current state."""
        self.current_state = state
    
    def transition_probabilities(self, state, actions):
        """
        Return possible transitions from the given state.
        
        Returns:
            List of (probability, next_state) tuples, or None if terminal
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


def main():
    """Main example function."""
    print("=" * 60)
    print("DAG Computation and Visualization Example")
    print("=" * 60)
    print()
    
    # Create environment
    print("Creating diamond-shaped DAG environment...")
    env = DiamondDAGEnv()
    print("  ✓ Environment created")
    print()
    
    # Compute DAG
    print("Computing DAG structure...")
    states, state_to_idx, successors = get_dag(env)
    print(f"  ✓ Found {len(states)} states")
    print()
    
    # Display results
    print("DAG Structure:")
    print("-" * 60)
    print(f"States (topological order): {states}")
    print(f"State-to-index mapping: {state_to_idx}")
    print()
    
    print("Successor relationships:")
    for idx, state in enumerate(states):
        succ_states = [states[s] for s in successors[idx]]
        print(f"  State {state} (index {idx}) -> {succ_states if succ_states else 'TERMINAL'}")
    print()
    
    # Verify topological ordering
    print("Verifying topological ordering...")
    violations = []
    for i, succ_list in enumerate(successors):
        for succ_idx in succ_list:
            if succ_idx <= i:
                violations.append((i, succ_idx))
    
    if violations:
        print(f"  ✗ Found {len(violations)} violations!")
        for pred, succ in violations:
            print(f"    State {pred} -> State {succ} (INVALID)")
    else:
        print("  ✓ All successors have higher indices than predecessors")
        print("  ✓ Topological ordering is correct!")
    print()
    
    # Highlight the critical case
    print("Critical Test: Multiple Paths to Same State")
    print("-" * 60)
    print("State 3 is reachable via two paths:")
    print("  Path 1: 0 -> 1 -> 3 (length 2)")
    print("  Path 2: 0 -> 2 -> 3 (length 2)")
    print()
    print("With naive BFS, if we discovered state 3 via path 0->1->3,")
    print("we might assign it index 2, but state 2 would get index 3,")
    print("creating an invalid edge 2->3 where predecessor > successor!")
    print()
    print("Our topological sort ensures state 3 gets the highest index,")
    print(f"so both states 1 and 2 have lower indices: {state_to_idx}")
    print()
    
    # Plot DAG
    print("Generating visualization...")
    try:
        # Create custom labels for better readability
        state_labels = {
            0: "Root\\n(State 0)",
            1: "State 1",
            2: "State 2", 
            3: "Terminal\\n(State 3)"
        }
        
        output_file = plot_dag(
            states,
            state_to_idx,
            successors,
            output_file="diamond_dag",
            format="png",
            state_labels=state_labels,
            rankdir="TB"
        )
        print(f"  ✓ DAG visualization saved to: {output_file}")
        print()
        print("Legend:")
        print("  • Green node  = Root state")
        print("  • Red node    = Terminal state")
        print("  • Blue node   = Intermediate state")
        print("  • Arrow       = Possible transition")
        
    except ImportError:
        print("  ⚠ graphviz Python package not installed, skipping visualization")
        print("    Install with: pip install graphviz")
    except Exception as e:
        if "Graphviz" in str(e) or "dot" in str(e):
            print("  ⚠ graphviz system package not installed, skipping visualization")
            print("    On Ubuntu/Debian: sudo apt-get install graphviz")
            print("    On macOS: brew install graphviz")
        else:
            print(f"  ⚠ Could not generate visualization: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
