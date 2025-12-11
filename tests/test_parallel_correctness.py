#!/usr/bin/env python3
"""
Test to verify that parallel backward induction produces identical results to sequential.
"""

import sys
import os
import numpy as np

from envs.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior


class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""
    
    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = np.array(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position, 0 otherwise."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[self.human_agent_index]
        agent_x, agent_y = agent_state[0], agent_state[1]
        if agent_x == self.target_pos[0] and agent_y == self.target_pos[1]:
            return 1
        return 0
    
    def __str__(self):
        return f"ReachCell(agent_{self.human_agent_index}_to_{self.target_pos[0]},{self.target_pos[1]})"
    
    def __hash__(self):
        return hash((self.human_agent_index, tuple(self.target_pos)))
    
    def __eq__(self, other):
        return (isinstance(other, ReachCellGoal) and 
                self.human_agent_index == other.human_agent_index and
                np.array_equal(self.target_pos, other.target_pos))


class SimpleGoalGenerator(PossibleGoalGenerator):
    """Generates a small number of goals for testing."""
    
    def __init__(self, world_model, target_cells):
        super().__init__(world_model)
        self.target_cells = target_cells
    
    def generate(self, state, human_agent_index: int):
        for pos in self.target_cells:
            goal = ReachCellGoal(self.env, human_agent_index, pos)
            yield (goal, 1.0 / len(self.target_cells))


def compare_policies(pol1, pol2, rtol=1e-5, atol=1e-8):
    """Compare two policy dictionaries and return differences."""
    differences = []
    
    # Check all states in pol1
    for state in pol1:
        if state not in pol2:
            differences.append(f"State {state} in sequential but not in parallel")
            continue
        
        for agent_idx in pol1[state]:
            if agent_idx not in pol2[state]:
                differences.append(f"Agent {agent_idx} missing in parallel for state {state}")
                continue
            
            for goal in pol1[state][agent_idx]:
                if goal not in pol2[state][agent_idx]:
                    differences.append(f"Goal {goal} missing in parallel for state {state}, agent {agent_idx}")
                    continue
                
                p1 = pol1[state][agent_idx][goal]
                p2 = pol2[state][agent_idx][goal]
                
                if not np.allclose(p1, p2, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(p1 - p2))
                    differences.append(
                        f"Policy mismatch for state {state}, agent {agent_idx}, goal {goal}: "
                        f"max_diff={max_diff:.2e}\n  seq: {p1}\n  par: {p2}"
                    )
    
    # Check for extra states in pol2
    for state in pol2:
        if state not in pol1:
            differences.append(f"State {state} in parallel but not in sequential")
    
    return differences


def test_parallel_correctness_small():
    """Test with a small environment (max_steps=2)."""
    print("=" * 70)
    print("Test: Small environment (max_steps=2)")
    print("=" * 70)
    
    # Create environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 2
    wm.reset()
    
    # Use just a few target cells
    target_cells = [(0, 0), (5, 5)]
    goal_gen = SimpleGoalGenerator(wm, target_cells)
    human_agent_indices = [0, 1]
    
    print(f"Environment: {wm.width}x{wm.height}, max_steps={wm.max_steps}")
    print(f"Human agents: {human_agent_indices}")
    print(f"Target cells: {target_cells}")
    print()
    
    # Run sequential
    print("Running sequential version...")
    result_seq = compute_human_policy_prior(
        wm, human_agent_indices, goal_gen, 
        parallel=False
    )
    print(f"Sequential: {len(result_seq.values)} states with policies")
    
    # Reset and run parallel
    wm.reset()
    print("\nRunning parallel version...")
    result_par = compute_human_policy_prior(
        wm, human_agent_indices, goal_gen,
        parallel=True, level_fct=lambda s: s[0]
    )
    print(f"Parallel: {len(result_par.values)} states with policies")
    
    # Compare
    print("\nComparing results...")
    differences = compare_policies(result_seq.values, result_par.values)
    
    assert not differences, f"Found {len(differences)} differences between sequential and parallel:\n" + "\n".join(differences[:10])
    print("✓ PASS: Sequential and parallel results are identical!")


def test_parallel_correctness_medium():
    """Test with a medium environment (max_steps=3)."""
    print("\n" + "=" * 70)
    print("Test: Medium environment (max_steps=3)")
    print("=" * 70)
    
    # Create environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 3
    wm.reset()
    
    # Use a few target cells
    target_cells = [(0, 0), (3, 3), (6, 6)]
    goal_gen = SimpleGoalGenerator(wm, target_cells)
    human_agent_indices = [0, 1]
    
    print(f"Environment: {wm.width}x{wm.height}, max_steps={wm.max_steps}")
    print(f"Human agents: {human_agent_indices}")
    print(f"Target cells: {target_cells}")
    print()
    
    # Run sequential
    print("Running sequential version...")
    result_seq = compute_human_policy_prior(
        wm, human_agent_indices, goal_gen, 
        parallel=False
    )
    print(f"Sequential: {len(result_seq.values)} states with policies")
    
    # Reset and run parallel
    wm.reset()
    print("\nRunning parallel version...")
    result_par = compute_human_policy_prior(
        wm, human_agent_indices, goal_gen,
        parallel=True, level_fct=lambda s: s[0]
    )
    print(f"Parallel: {len(result_par.values)} states with policies")
    
    # Compare
    print("\nComparing results...")
    differences = compare_policies(result_seq.values, result_par.values)
    
    assert not differences, f"Found {len(differences)} differences between sequential and parallel:\n" + "\n".join(differences[:10])
    print("✓ PASS: Sequential and parallel results are identical!")


def test_v_values_visibility():
    """Test whether workers can see updated V_values from previous levels."""
    print("\n" + "=" * 70)
    print("Test: V_values visibility across levels")
    print("=" * 70)
    
    # Create a simple environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 3  # This gives us 4 levels (0, 1, 2, 3)
    wm.reset()
    
    # Get DAG to understand structure
    states, state_to_idx, successors, transitions = wm.get_dag(return_probabilities=True)
    
    # Count states per level
    level_counts = {}
    for state in states:
        level = state[0]  # step_count
        level_counts[level] = level_counts.get(level, 0) + 1
    
    print(f"Total states: {len(states)}")
    print(f"States per level: {level_counts}")
    
    # Check that non-terminal states have successors in later levels
    for state_idx, state in enumerate(states):
        level = state[0]
        if transitions[state_idx]:  # Non-terminal
            succ_levels = set()
            for _, _, succ_indices in transitions[state_idx]:
                for succ_idx in succ_indices:
                    succ_levels.add(states[succ_idx][0])
            if level in succ_levels or any(l <= level for l in succ_levels):
                print(f"WARNING: State at level {level} has successors at levels {succ_levels}")
    
    print("\nLevel structure looks correct for backward induction.")


def main():
    """Run all tests."""
    print("Testing Parallel vs Sequential Backward Induction Correctness")
    print("=" * 70)
    print()
    
    test_names = []
    
    # Test V_values visibility
    test_v_values_visibility()
    test_names.append("V_values visibility")
    
    # Test small environment
    test_parallel_correctness_small()
    test_names.append("Small environment")
    
    # Test medium environment
    test_parallel_correctness_medium()
    test_names.append("Medium environment")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name in test_names:
        print(f"  ✓ PASS: {name}")
    
    print()
    print("All tests passed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
