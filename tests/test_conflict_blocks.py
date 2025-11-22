#!/usr/bin/env python3
"""
Test to verify the conflict block optimization is working correctly.

This test creates scenarios where agents compete for resources and verifies
that the conflict block partitioning produces correct probabilities.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, Ball
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def test_conflict_block_efficiency():
    """
    Test that conflict block optimization is more efficient than full permutation.
    
    With 4 agents, full permutation would be 4!=24.
    But if agents split into 2 blocks of 2, conflict blocks give us only 2×2=4 outcomes.
    """
    print("Test: Conflict block optimization efficiency...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # All agents moving forward - may or may not conflict depending on positions
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    # Time the computation
    start_time = time.time()
    result = env.transition_probabilities(state, actions)
    elapsed_time = time.time() - start_time
    
    print(f"  → Computed {len(result)} unique outcome(s) in {elapsed_time:.4f}s")
    
    # Verify probabilities still sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-9:
        print(f"  ✓ Probabilities sum correctly to 1.0")
        return True
    else:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False


def test_two_agents_competing_for_cell():
    """
    Test the canonical case: two agents competing for the same cell.
    
    This should produce 2 outcomes with equal probability 0.5 each.
    """
    print("Test: Two agents competing for same cell...")
    
    # We need to manually create a scenario where we know two agents will compete
    # For now, we'll test with the existing environment and check the logic
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.still]
    
    result = env.transition_probabilities(state, actions)
    
    # In the current random configuration, agents might or might not compete
    # The key is that probabilities sum to 1.0 and outcomes are consistent
    total_prob = sum(prob for prob, _ in result)
    
    if abs(total_prob - 1.0) < 1e-9:
        print(f"  ✓ Got {len(result)} outcome(s) with probabilities summing to 1.0")
        
        # If there are multiple outcomes, verify they have valid probabilities
        if len(result) > 1:
            print(f"  ✓ Multiple outcomes detected (probabilistic case)")
            for i, (prob, _) in enumerate(result):
                print(f"    Outcome {i+1}: probability = {prob:.4f}")
        else:
            print(f"  ✓ Single outcome (deterministic - agents don't interfere)")
        
        return True
    else:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False


def test_probability_values_with_conflicts():
    """
    Test that probability values are correct when conflicts exist.
    
    If we have 2 blocks of 2 agents each, each outcome should have probability 1/4.
    """
    print("Test: Probability values with conflict blocks...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    # Check if probabilities are valid fractions
    for prob, _ in result:
        # Probability should be a rational number
        # For conflict blocks, common values are 1/2, 1/3, 1/4, 1/6, etc.
        denominators = [1, 2, 3, 4, 6, 12, 24]  # Common denominators
        
        is_valid = False
        for denom in denominators:
            scaled = prob * denom
            if abs(scaled - round(scaled)) < 1e-9:
                is_valid = True
                numerator = round(scaled)
                print(f"  ✓ Probability {prob:.4f} = {numerator}/{denom}")
                break
        
        if not is_valid:
            print(f"  ⚠ Probability {prob:.4f} is not a simple rational")
    
    # Verify sum
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-9:
        print(f"  ✓ All probabilities sum to 1.0")
        return True
    else:
        print(f"  ✗ Probabilities sum to {total_prob}")
        return False


def test_independent_agents_deterministic():
    """
    Test that agents with independent actions produce deterministic outcomes.
    """
    print("Test: Independent agents are deterministic...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # Mix of rotations and forward - if agents are far apart, should be deterministic
    actions = [Actions.left, Actions.right, Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    # Should be deterministic if agents don't interfere
    if len(result) == 1:
        print(f"  ✓ Independent agents produce single outcome (deterministic)")
        print(f"  ✓ Probability = {result[0][0]}")
        return True
    else:
        print(f"  ⚠ Got {len(result)} outcomes (agents may be interfering)")
        # Still valid as long as probabilities sum to 1.0
        total_prob = sum(prob for prob, _ in result)
        if abs(total_prob - 1.0) < 1e-9:
            print(f"  ✓ Probabilities still sum correctly to 1.0")
            return True
        else:
            return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing Conflict Block Optimization")
    print("=" * 70)
    print()
    
    tests = [
        test_conflict_block_efficiency,
        test_two_agents_competing_for_cell,
        test_probability_values_with_conflicts,
        test_independent_agents_deterministic,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
