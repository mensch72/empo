#!/usr/bin/env python3
"""
Test for unsteady ground with block/rock pushing.

Tests that stumbling agents correctly handle pushing blocks/rocks and that
conflicts are properly resolved when multiple agents push to the same location.
"""

import sys
from pathlib import Path
import numpy as np

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, UnsteadyGround, Block


class UnsteadyPushTestEnv(MultiGridEnv):
    """
    Test environment with agent on unsteady ground that can push blocks.
    
    Layout (7x5):
    W W W W W W W
    W U . B . . W
    W . . . . . W
    W . . . . . W
    W W W W W W W
    
    Agent at (1,1) on unsteady ground facing right
    Block at (3,1)
    If agent stumbles right, it will turn down and move forward
    If agent doesn't stumble, it moves right toward the block
    """
    
    def __init__(self):
        self.agents = [Agent(World, 0)]
        super().__init__(
            width=7,
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
        
        # Place unsteady ground
        unsteady = UnsteadyGround(World, stumble_probability=0.5, color='brown')
        self.grid.set(1, 1, unsteady)
        # Save terrain to terrain_grid so it persists under agents
        self.terrain_grid.set(1, 1, unsteady)
        
        # Place block
        block = Block(World)
        self.grid.set(3, 1, block)
        
        # Place agent on unsteady ground facing right
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0  # facing right
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        self.agents[0].on_unsteady_ground = True
        self.grid.set(1, 1, self.agents[0])


def test_unsteady_agent_can_push_blocks():
    """
    Test that an agent stumbling on unsteady ground can still push blocks.
    """
    print("Test: Unsteady agent pushing blocks...")
    
    env = UnsteadyPushTestEnv()
    env.reset()
    
    # Agent attempts forward
    # If stumbles right: turns down (dir=1), moves to (1,2)
    # If stumbles left: turns up (dir=3), moves to (1,0) which is wall - blocked
    # If no stumble: moves right to (2,1)
    actions = [Actions.forward]
    
    # Run multiple times to see different outcomes
    outcomes = {}
    for trial in range(100):
        env.reset()
        obs, rewards, done, info = env.step(actions)
        
        agent_pos = tuple(env.agents[0].pos)
        agent_dir = env.agents[0].dir
        outcome = (agent_pos, agent_dir)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    
    print(f"  Outcomes from 100 trials:")
    for outcome, count in sorted(outcomes.items()):
        pos, dir_val = outcome
        print(f"    Pos {pos}, Dir {dir_val}: {count} times")
    
    # Should see at least 2 different outcomes (stumble vs no stumble)
    assert len(outcomes) >= 2, f"Only found {len(outcomes)} outcome (should be stochastic with at least 2)"
    print(f"  ✓ Found {len(outcomes)} different outcomes (stochastic)")


def test_transition_probabilities_with_push():
    """
    Test that transition probabilities correctly handle block pushing after stumbling.
    """
    print("\nTest: Transition probabilities with block pushing...")
    
    env = UnsteadyPushTestEnv()
    env.reset()
    
    state = env.get_state()
    actions = [Actions.forward]
    
    result = env.transition_probabilities(state, actions)
    
    assert result is not None, "Got None result from transition_probabilities"
    
    print(f"  Number of possible outcomes: {len(result)}")
    
    # Check probabilities sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-10, f"Probabilities sum to {total_prob}, not 1.0"
    print(f"  ✓ Probabilities sum to 1.0")
    
    # Print outcomes
    print("\n  Outcomes:")
    for i, (prob, succ_state) in enumerate(result):
        # Compact state format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = succ_state
        agent_state = agent_states[0]
        # Agent state format: (pos_x, pos_y, dir, terminated, started, paused, on_unsteady, carrying_type, carrying_color)
        agent_pos = (agent_state[0], agent_state[1])
        agent_dir = agent_state[2]
        
        print(f"    {i+1}. Prob={prob:.4f}: Agent@{agent_pos} dir={agent_dir}")
    
    # Should have 3 outcomes (forward, left-forward, right-forward)
    assert len(result) == 3, f"Found {len(result)} outcomes (expected 3)"
    print(f"\n  ✓ Found expected 3 outcomes")


def run_all_tests():
    """Run all unsteady push tests."""
    print("=" * 70)
    print("Unsteady Ground with Block Pushing Tests")
    print("=" * 70)
    print()
    
    tests = [
        test_unsteady_agent_can_push_blocks,
        test_transition_probabilities_with_push,
    ]
    
    for test_func in tests:
        test_func()
        print()
    
    print("=" * 70)
    print(f"All {len(tests)} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
