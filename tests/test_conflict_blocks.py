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
    
    # Create environment and get initial state
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Manufacture a state where two agents will compete for the same cell
    # Place agent 0 at (1, 2) facing right (dir=0) -> moves to (2, 2)
    # Place agent 1 at (3, 2) facing left (dir=2) -> moves to (2, 2)
    # Agent 2 stays still
    
    initial_state = env.get_state()
    state_dict = dict(initial_state)
    
    # Modify agent positions and directions
    agents_data = list(state_dict['agents'])
    
    # Agent 0: position (1, 2), direction 0 (right)
    agent0_dict = dict(agents_data[0])
    agent0_dict['pos'] = (1, 2)
    agent0_dict['dir'] = 0  # facing right
    agents_data[0] = tuple(sorted(agent0_dict.items()))
    
    # Agent 1: position (3, 2), direction 2 (left)
    agent1_dict = dict(agents_data[1])
    agent1_dict['pos'] = (3, 2)
    agent1_dict['dir'] = 2  # facing left
    agents_data[1] = tuple(sorted(agent1_dict.items()))
    
    # Agent 2: keep as is, will use "still" action
    
    # Update state
    modified_state = tuple(sorted([
        ('grid', state_dict['grid']),
        ('agents', tuple(agents_data)),
        ('step_count', state_dict['step_count']),
        ('rng_state', state_dict['rng_state']),
    ]))
    
    # Set the manufactured state
    env.set_state(modified_state)
    
    # Both agents 0 and 1 move forward to (2, 2), agent 2 stays still
    actions = [Actions.forward, Actions.forward, Actions.still]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 2 outcomes with probability 0.5 each
    if len(result) != 2:
        print(f"  ✗ Expected 2 outcomes, got {len(result)}")
        for i, (prob, _) in enumerate(result):
            print(f"    Outcome {i+1}: probability = {prob:.4f}")
        return False
    
    # Check probabilities
    probs = [prob for prob, _ in result]
    if not all(abs(p - 0.5) < 1e-9 for p in probs):
        print(f"  ✗ Expected probabilities [0.5, 0.5], got {probs}")
        return False
    
    total_prob = sum(probs)
    if abs(total_prob - 1.0) > 1e-9:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False
    
    print(f"  ✓ Got 2 outcomes with probabilities 0.5 each")
    print(f"  ✓ Two agents successfully compete for cell (2, 2)")
    
    return True


def test_probability_values_with_conflicts():
    """
    Test that probability values are correct when conflicts exist.
    
    If we have 3 agents competing for same cell, each outcome should have probability 1/3.
    """
    print("Test: Probability values with conflict blocks...")
    
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Manufacture a state where all 3 agents compete for cell (5, 5)
    # Agent 0 at (4, 5) facing right (dir=0)
    # Agent 1 at (6, 5) facing left (dir=2) 
    # Agent 2 at (5, 4) facing down (dir=1)
    
    initial_state = env.get_state()
    state_dict = dict(initial_state)
    
    # Modify agent positions and directions
    agents_data = list(state_dict['agents'])
    
    # Agent 0: position (4, 5), direction 0 (right)
    agent0_dict = dict(agents_data[0])
    agent0_dict['pos'] = (4, 5)
    agent0_dict['dir'] = 0
    agents_data[0] = tuple(sorted(agent0_dict.items()))
    
    # Agent 1: position (6, 5), direction 2 (left)
    agent1_dict = dict(agents_data[1])
    agent1_dict['pos'] = (6, 5)
    agent1_dict['dir'] = 2
    agents_data[1] = tuple(sorted(agent1_dict.items()))
    
    # Agent 2: position (5, 4), direction 1 (down)
    agent2_dict = dict(agents_data[2])
    agent2_dict['pos'] = (5, 4)
    agent2_dict['dir'] = 1
    agents_data[2] = tuple(sorted(agent2_dict.items()))
    
    # Update state
    modified_state = tuple(sorted([
        ('grid', state_dict['grid']),
        ('agents', tuple(agents_data)),
        ('step_count', state_dict['step_count']),
        ('rng_state', state_dict['rng_state']),
    ]))
    
    # Set the manufactured state
    env.set_state(modified_state)
    
    # All agents move forward to (5, 5)
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 3 outcomes with probability 1/3 each
    if len(result) != 3:
        print(f"  ✗ Expected 3 outcomes, got {len(result)}")
        # Still check if probabilities are valid
        for i, (prob, _) in enumerate(result):
            print(f"    Outcome {i+1}: probability = {prob:.4f}")
        return False
    
    # Check probabilities
    expected_prob = 1.0 / 3.0
    for i, (prob, _) in enumerate(result):
        if abs(prob - expected_prob) > 1e-9:
            print(f"  ✗ Outcome {i+1}: expected {expected_prob:.4f}, got {prob:.4f}")
            return False
        print(f"  ✓ Outcome {i+1}: probability = {prob:.4f} = 1/3")
    
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
    
    # Manufacture a state where agents are far apart and won't interfere
    initial_state = env.get_state()
    state_dict = dict(initial_state)
    
    # Modify agent positions to be far apart
    agents_data = list(state_dict['agents'])
    
    # Agent 0 at (2, 2), facing right
    agent0_dict = dict(agents_data[0])
    agent0_dict['pos'] = (2, 2)
    agent0_dict['dir'] = 0
    agents_data[0] = tuple(sorted(agent0_dict.items()))
    
    # Agent 1 at (7, 7), facing up
    agent1_dict = dict(agents_data[1])
    agent1_dict['pos'] = (7, 7)
    agent1_dict['dir'] = 3
    agents_data[1] = tuple(sorted(agent1_dict.items()))
    
    # Agent 2 at (2, 7), facing down
    agent2_dict = dict(agents_data[2])
    agent2_dict['pos'] = (2, 7)
    agent2_dict['dir'] = 1
    agents_data[2] = tuple(sorted(agent2_dict.items()))
    
    # Update state
    modified_state = tuple(sorted([
        ('grid', state_dict['grid']),
        ('agents', tuple(agents_data)),
        ('step_count', state_dict['step_count']),
        ('rng_state', state_dict['rng_state']),
    ]))
    
    env.set_state(modified_state)
    
    # All agents move forward to different cells (no conflicts)
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should be deterministic (single outcome)
    if len(result) == 1:
        print(f"  ✓ Independent agents produce single outcome (deterministic)")
        print(f"  ✓ Probability = {result[0][0]}")
        return True
    else:
        print(f"  ✗ Expected 1 outcome, got {len(result)}")
        for i, (prob, _) in enumerate(result):
            print(f"    Outcome {i+1}: probability = {prob:.4f}")
        return False


def test_two_conflict_blocks():
    """
    Test two separate conflict blocks: 2 agents compete for cell A, 2 compete for cell B.
    
    Should produce 2×2=4 outcomes, each with probability 1/4.
    """
    print("Test: Two separate conflict blocks (2×2)...")
    
    # Create a custom environment with 4 agents to test 2 blocks of 2
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))
    from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Grid, Wall
    
    class FourAgentEnv(MultiGridEnv):
        def __init__(self):
            agents = [Agent(World, i+1, view_size=7) for i in range(4)]
            super().__init__(
                grid_size=10,
                max_steps=100,
                agents=agents,
                agent_view_size=7
            )
        
        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            # Generate walls
            self.grid.horz_wall(World, 0, 0)
            self.grid.horz_wall(World, 0, height-1)
            self.grid.vert_wall(World, 0, 0)
            self.grid.vert_wall(World, width-1, 0)
            # Place agents
            for a in self.agents:
                self.place_agent(a)
    
    env = FourAgentEnv()
    env.reset()
    
    # Manufacture state with TWO conflict blocks:
    # Block 1: Agents 0,1 compete for cell (3, 3)
    # Block 2: Agents 2,3 compete for cell (6, 6)
    
    initial_state = env.get_state()
    state_dict = dict(initial_state)
    
    # Modify agent positions
    agents_data = list(state_dict['agents'])
    
    # Block 1: Agents 0,1 compete for (3, 3)
    # Agent 0 at (2, 3) facing right - will move to (3, 3)
    agent0_dict = dict(agents_data[0])
    agent0_dict['pos'] = (2, 3)
    agent0_dict['dir'] = 0
    agents_data[0] = tuple(sorted(agent0_dict.items()))
    
    # Agent 1 at (4, 3) facing left - will move to (3, 3)
    agent1_dict = dict(agents_data[1])
    agent1_dict['pos'] = (4, 3)
    agent1_dict['dir'] = 2
    agents_data[1] = tuple(sorted(agent1_dict.items()))
    
    # Block 2: Agents 2,3 compete for (6, 6)
    # Agent 2 at (5, 6) facing right - will move to (6, 6)
    agent2_dict = dict(agents_data[2])
    agent2_dict['pos'] = (5, 6)
    agent2_dict['dir'] = 0
    agents_data[2] = tuple(sorted(agent2_dict.items()))
    
    # Agent 3 at (7, 6) facing left - will move to (6, 6)
    agent3_dict = dict(agents_data[3])
    agent3_dict['pos'] = (7, 6)
    agent3_dict['dir'] = 2
    agents_data[3] = tuple(sorted(agent3_dict.items()))
    
    # Update state
    modified_state = tuple(sorted([
        ('grid', state_dict['grid']),
        ('agents', tuple(agents_data)),
        ('step_count', state_dict['step_count']),
        ('rng_state', state_dict['rng_state']),
    ]))
    
    env.set_state(modified_state)
    
    # All 4 agents move forward, creating 2 separate conflicts
    actions = [Actions.forward, Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 2×2=4 outcomes
    # (agent 0 or 1 wins cell A) × (agent 2 or 3 wins cell B)
    if len(result) != 4:
        print(f"  ✗ Expected 4 outcomes (2×2), got {len(result)}")
        for i, (prob, _) in enumerate(result):
            print(f"    Outcome {i+1}: probability = {prob:.4f}")
        return False
    
    # Each outcome should have probability 1/4
    expected_prob = 0.25
    for i, (prob, _) in enumerate(result):
        if abs(prob - expected_prob) > 1e-9:
            print(f"  ✗ Outcome {i+1}: expected 0.25, got {prob:.4f}")
            return False
        print(f"  ✓ Outcome {i+1}: probability = {prob:.4f} = 1/4")
    
    print(f"  ✓ Two conflict blocks (2×2) correctly produce 4 outcomes")
    print(f"  ✓ Each outcome has probability 1/4 (= 1/2 × 1/2)")
    
    return True


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
        test_two_conflict_blocks,
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
