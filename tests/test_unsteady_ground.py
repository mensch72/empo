#!/usr/bin/env python3
"""
Test for unsteady ground transition probabilities.

Tests that the transition_probabilities method correctly handles unsteady ground
agents with stumbling mechanics, particularly when multiple stumbling agents
compete for the same target cell.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, UnsteadyGround


class UnsteadyGroundTestEnv(MultiGridEnv):
    """
    Minimal test environment with two agents on unsteady ground.
    
    Layout (5x5):
    W W W W W
    W A . . W
    W U U T W
    W . . . W
    W W W W W
    
    Where:
    - W = Wall
    - A = Agent 0 (facing down)
    - U = Unsteady ground (50% stumble probability)
    - T = Target cell (empty)
    - . = Empty
    
    Agent 1 is placed on the left unsteady cell, facing right.
    Both agents attempt forward action, which can result in them competing
    for the target cell.
    """
    
    def __init__(self):
        self.agents = [Agent(World, i) for i in range(2)]
        super().__init__(
            width=5,
            height=5,
            max_steps=100,
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height-1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width-1, j, Wall(World))
        
        # Place unsteady ground cells
        unsteady1 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
        unsteady2 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
        self.grid.set(1, 2, unsteady1)
        self.grid.set(2, 2, unsteady2)
        
        # Place agent 0 at (1, 1) facing down (dir=1)
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 1  # facing down
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        self.grid.set(1, 1, self.agents[0])
        
        # Place agent 1 at (1, 2) on unsteady ground, facing right (dir=0)
        self.agents[1].pos = np.array([1, 2])
        self.agents[1].dir = 0  # facing right
        self.agents[1].init_pos = self.agents[1].pos.copy()
        self.agents[1].init_dir = self.agents[1].dir
        self.agents[1].on_unsteady_ground = True  # Agent is on unsteady ground
        self.grid.set(1, 2, self.agents[1])


def test_unsteady_ground_probabilities():
    """
    Test that transition probabilities are correctly computed for unsteady ground.
    
    In this test:
    - Agent 0 moves forward from (1,1) to the unsteady ground at (1,2)
    - Agent 1 is on unsteady ground at (1,2) and attempts forward to (2,2)
    - Agent 1 may stumble and turn, affecting its target cell
    
    Expected outcomes depend on:
    1. Agent 0's execution (moves to (1,2))
    2. Agent 1's stumbling (3 outcomes: forward, left+forward, right+forward)
    """
    print("Test: Unsteady ground transition probabilities...")
    
    env = UnsteadyGroundTestEnv()
    env.reset()
    
    # Get initial state
    state = env.get_state()
    
    # Both agents attempt forward
    actions = [Actions.forward, Actions.forward]
    
    # Compute transition probabilities
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ✗ Got None result")
        return False
    
    print(f"  Number of possible outcomes: {len(result)}")
    
    # Check that probabilities sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-10:
        print(f"  ✓ Probabilities sum to 1.0")
    else:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False
    
    # Print all outcomes
    print("\n  Outcomes:")
    for i, (prob, succ_state) in enumerate(result):
        state_dict = dict(succ_state)
        agents_data = state_dict['agents']
        agent0_data = dict(agents_data[0])
        agent1_data = dict(agents_data[1])
        
        print(f"    {i+1}. Prob={prob:.4f}: Agent0@{agent0_data['pos']}, Agent1@{agent1_data['pos']} dir={agent1_data['dir']}")
    
    # Since agent 1 is on unsteady ground, we expect outcomes based on:
    # - Agent 1 stumbling (3 outcomes: forward, left-forward, right-forward)
    # The total should account for these 3 outcomes
    
    # Check that we have reasonable number of outcomes
    if len(result) >= 3:
        print(f"\n  ✓ Found {len(result)} outcomes (expected at least 3 for stumbling)")
    else:
        print(f"\n  ✗ Found only {len(result)} outcomes (expected at least 3)")
        return False
    
    return True


def test_two_stumbling_agents_same_target():
    """
    Test transition probabilities when two agents on unsteady ground compete for same cell.
    
    In this scenario:
    - Both agents are on unsteady ground
    - Both attempt forward action
    - They may compete for the same target cell depending on stumbling outcomes
    
    This tests the conflict resolution logic for unsteady-forward agents.
    """
    print("\nTest: Two stumbling agents competing for same target...")
    
    # Create a custom environment for this test
    class TwoStumblingAgentsEnv(MultiGridEnv):
        """
        Layout (5x5):
        W W W W W
        W U T U W
        W . . . W
        W . . . W
        W W W W W
        
        Agent 0 at (1,1) on unsteady ground, facing right
        Agent 1 at (3,1) on unsteady ground, facing left
        Both attempting forward will target the middle cell (2,1)
        But stumbling can redirect them to other cells
        """
        
        def __init__(self):
            self.agents = [Agent(World, i) for i in range(2)]
            super().__init__(
                width=5,
                height=5,
                max_steps=100,
                agents=self.agents,
                partial_obs=False,
                objects_set=World
            )
        
        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            
            # Add walls
            for i in range(width):
                self.grid.set(i, 0, Wall(World))
                self.grid.set(i, height-1, Wall(World))
            for j in range(height):
                self.grid.set(0, j, Wall(World))
                self.grid.set(width-1, j, Wall(World))
            
            # Place unsteady ground cells
            unsteady1 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
            unsteady2 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
            self.grid.set(1, 1, unsteady1)
            self.grid.set(3, 1, unsteady2)
            
            # Place agent 0 at (1, 1) facing right (dir=0)
            self.agents[0].pos = np.array([1, 1])
            self.agents[0].dir = 0  # facing right
            self.agents[0].init_pos = self.agents[0].pos.copy()
            self.agents[0].init_dir = self.agents[0].dir
            self.agents[0].on_unsteady_ground = True  # Agent is on unsteady ground
            self.grid.set(1, 1, self.agents[0])
            
            # Place agent 1 at (3, 1) facing left (dir=2)
            self.agents[1].pos = np.array([3, 1])
            self.agents[1].dir = 2  # facing left
            self.agents[1].init_pos = self.agents[1].pos.copy()
            self.agents[1].init_dir = self.agents[1].dir
            self.agents[1].on_unsteady_ground = True  # Agent is on unsteady ground
            self.grid.set(3, 1, self.agents[1])
    
    env = TwoStumblingAgentsEnv()
    env.reset()
    
    # Get initial state
    state = env.get_state()
    
    # Both agents attempt forward
    actions = [Actions.forward, Actions.forward]
    
    # Compute transition probabilities
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ✗ Got None result")
        return False
    
    print(f"  Number of possible outcomes: {len(result)}")
    
    # Check that probabilities sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-10:
        print(f"  ✓ Probabilities sum to 1.0")
    else:
        print(f"  ✗ Probabilities sum to {total_prob}, not 1.0")
        return False
    
    # With two agents on unsteady ground, each has 3 outcomes
    # Total combinations: 3 * 3 = 9
    # But some may lead to the same state due to conflicts
    
    print(f"\n  Outcomes breakdown:")
    for i, (prob, succ_state) in enumerate(result):
        state_dict = dict(succ_state)
        agents_data = state_dict['agents']
        agent0_data = dict(agents_data[0])
        agent1_data = dict(agents_data[1])
        
        print(f"    {i+1}. Prob={prob:.4f}: "
              f"Agent0@{agent0_data['pos']} dir={agent0_data['dir']}, "
              f"Agent1@{agent1_data['pos']} dir={agent1_data['dir']}")
    
    # We expect up to 9 outcomes (3 x 3), but some may collapse to the same state
    if len(result) >= 3 and len(result) <= 9:
        print(f"\n  ✓ Found {len(result)} unique outcomes (expected between 3 and 9)")
    else:
        print(f"\n  ✗ Found {len(result)} outcomes (expected between 3 and 9)")
        return False
    
    # Check that each outcome has reasonable probability
    # With 3x3 = 9 total combinations, each base outcome should have prob ~1/9
    # But conflicts may aggregate probabilities
    expected_base_prob = 1.0 / 9.0
    for prob, _ in result:
        # Each outcome should be a multiple of the base probability
        # (due to aggregation of identical outcomes)
        ratio = prob / expected_base_prob
        if abs(ratio - round(ratio)) < 0.01:  # Check if close to integer multiple
            print(f"  ✓ Outcome probability {prob:.4f} is {round(ratio)}x base probability")
        else:
            print(f"  Note: Outcome probability {prob:.4f} is {ratio:.2f}x base probability")
    
    return True


def run_all_tests():
    """Run all unsteady ground tests."""
    print("=" * 70)
    print("Unsteady Ground Transition Probability Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_unsteady_ground_probabilities,
        test_two_stumbling_agents_same_target,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    print("=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
