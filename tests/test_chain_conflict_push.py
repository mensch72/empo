#!/usr/bin/env python3
"""
Test for chain conflict prevention when pushing blocks.

Tests that an agent cannot push blocks onto a cell that was occupied by another 
agent at the beginning of the step, even if that agent moves away during the step.
This ensures deterministic conflict resolution.
"""

import sys
import numpy as np

import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, Block


class ChainConflictPushTestEnv(MultiGridEnv):
    """
    Test environment where agent tries to push two blocks onto a cell 
    that another agent is vacating.
    
    Layout (8x5):
    W W W W W W W W
    W . . . . . . W
    W A B B . . T W  <- Agent 0 at (1,2) facing right, Blocks at (2,2) and (3,2), Agent 1 (T) at (6,2)
    W . . . . . . W
    W W W W W W W W
    
    Agent 0 at (1,2) facing right - will try to push blocks
    Blocks at (2,2) and (3,2) - two consecutive blocks
    Agent 1 at (6,2) facing right - will try to move away to (7,2) but that's a wall
    
    Actually let me redesign to make Agent 1 able to move:
    
    Layout (9x5):
    W W W W W W W W W
    W . . . . . . . W
    W A B B . T . . W  <- Agent 0 at (1,2), Blocks at (2,2) and (3,2), Agent 1 at (4,2)
    W . . . . . . . W
    W W W W W W W W W
    
    Agent 0 at (1,2) facing right - pushes blocks from (2,2) to (3,2), blocks would go to (3,2) and (4,2)
    Agent 1 at (4,2) facing right - moves to (5,2)
    
    The push would land the second block at (4,2), which was Agent 1's initial position.
    Even though Agent 1 moves away, the push should be blocked.
    """
    
    def __init__(self):
        self.agents = [Agent(World, i % 6) for i in range(2)]
        super().__init__(
            width=9,
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
        
        # Place two consecutive blocks at (2,2) and (3,2)
        block1 = Block(World)
        block2 = Block(World)
        self.grid.set(2, 2, block1)
        self.grid.set(3, 2, block2)
        
        # Agent 0 at (1,2) facing right - will push blocks
        self.agents[0].pos = np.array([1, 2])
        self.agents[0].dir = 0  # facing right
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        self.grid.set(1, 2, self.agents[0])
        
        # Agent 1 at (4,2) facing right - will move away
        # The second block would land at (4,2) if push succeeds
        self.agents[1].pos = np.array([4, 2])
        self.agents[1].dir = 0  # facing right
        self.agents[1].init_pos = self.agents[1].pos.copy()
        self.agents[1].init_dir = self.agents[1].dir
        self.grid.set(4, 2, self.agents[1])


def test_cannot_push_blocks_onto_vacated_cell():
    """
    Test that pushing blocks onto a cell occupied by another agent at step start
    is blocked, even if that agent moves away during the same step.
    
    This tests the chain conflict prevention: the push should fail because
    the target cell (4,2) was occupied by Agent 1 at the start of the step,
    regardless of whether Agent 1 moves away or not.
    """
    print("Test: Cannot push blocks onto cell vacated by another agent in same step...")
    
    env = ChainConflictPushTestEnv()
    env.reset()
    
    print("\nInitial Setup:")
    print(f"  Agent 0: pos={env.agents[0].pos}, dir={env.agents[0].dir} (facing right)")
    print(f"  Agent 1: pos={env.agents[1].pos}, dir={env.agents[1].dir} (facing right)")
    print(f"  Block 1 at: (2, 2)")
    print(f"  Block 2 at: (3, 2)")
    print()
    print("Actions:")
    print("  Agent 0: forward (attempts to push blocks)")
    print("  Agent 1: forward (moves from (4,2) to (5,2))")
    print()
    print("Expected behavior:")
    print("  - Agent 1 moves to (5,2)")
    print("  - Agent 0's push is BLOCKED because (4,2) was occupied at step start")
    print("  - Blocks stay at (2,2) and (3,2)")
    print("  - Agent 0 stays at (1,2)")
    print()
    
    # Get initial state
    state = env.get_state()
    
    # Both agents attempt forward
    actions = [Actions.forward, Actions.forward]
    
    # Test with transition_probabilities
    result = env.transition_probabilities(state, actions)
    
    assert result is not None, "Got None result from transition_probabilities"
    
    print(f"Number of possible outcomes: {len(result)}")
    
    # Should be deterministic (1 outcome) since no stochastic elements
    assert len(result) == 1, f"Expected 1 deterministic outcome, got {len(result)}"
    
    prob, succ_state = result[0]
    assert abs(prob - 1.0) < 1e-10, f"Expected probability 1.0, got {prob}"
    
    # Parse successor state
    step_count, agent_states, mobile_objects, mutable_objects = succ_state
    
    agent0_state = agent_states[0]
    agent1_state = agent_states[1]
    
    agent0_pos = (agent0_state[0], agent0_state[1])
    agent1_pos = (agent1_state[0], agent1_state[1])
    
    # Get block positions
    block_positions = []
    for obj in mobile_objects:
        if obj[0] == 'block':
            block_positions.append((obj[1], obj[2]))
    
    print("Outcome:")
    print(f"  Agent 0: {agent0_pos}")
    print(f"  Agent 1: {agent1_pos}")
    print(f"  Blocks: {block_positions}")
    
    # Verify expected outcome
    # Agent 1 should have moved to (5,2)
    assert agent1_pos == (5, 2), f"Agent 1 should have moved to (5,2), but is at {agent1_pos}"
    print("  ✓ Agent 1 correctly moved to (5,2)")
    
    # Agent 0 should have stayed at (1,2) because push was blocked
    assert agent0_pos == (1, 2), f"Agent 0 should have stayed at (1,2), but is at {agent0_pos}"
    print("  ✓ Agent 0 correctly stayed at (1,2) (push blocked)")
    
    # Blocks should not have moved
    assert (2, 2) in block_positions, f"Block 1 should be at (2,2), blocks are at {block_positions}"
    assert (3, 2) in block_positions, f"Block 2 should be at (3,2), blocks are at {block_positions}"
    print("  ✓ Blocks correctly stayed at (2,2) and (3,2)")
    
    # Also verify with actual step() execution
    print("\nVerifying with step() execution...")
    env.reset()
    
    # Run step multiple times to verify consistency
    for trial in range(10):
        env.reset()
        obs, rewards, done, info = env.step(actions)
        
        trial_agent0_pos = tuple(env.agents[0].pos)
        trial_agent1_pos = tuple(env.agents[1].pos)
        
        assert trial_agent0_pos == (1, 2), f"Trial {trial}: Agent 0 should be at (1,2), got {trial_agent0_pos}"
        assert trial_agent1_pos == (5, 2), f"Trial {trial}: Agent 1 should be at (5,2), got {trial_agent1_pos}"
    
    print("  ✓ 10 step() executions all produced consistent results")
    print("\n✓ Test passed: Chain conflict prevention works correctly for block pushing")


def test_can_push_blocks_onto_empty_cell():
    """
    Control test: verify that pushing blocks onto a truly empty cell still works.
    """
    print("\nTest: Can push blocks onto empty cell (control test)...")
    
    # Create a simpler environment with no agent at the target position
    class SimplePushTestEnv(MultiGridEnv):
        def __init__(self):
            self.agents = [Agent(World, 0)]
            super().__init__(
                width=8,
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
            
            # Place two consecutive blocks
            block1 = Block(World)
            block2 = Block(World)
            self.grid.set(2, 2, block1)
            self.grid.set(3, 2, block2)
            
            # Agent at (1,2) facing right
            self.agents[0].pos = np.array([1, 2])
            self.agents[0].dir = 0  # facing right
            self.agents[0].init_pos = self.agents[0].pos.copy()
            self.agents[0].init_dir = self.agents[0].dir
            self.grid.set(1, 2, self.agents[0])
    
    env = SimplePushTestEnv()
    env.reset()
    
    actions = [Actions.forward]
    
    obs, rewards, done, info = env.step(actions)
    
    agent_pos = tuple(env.agents[0].pos)
    
    # Agent should have pushed blocks and moved to (2,2)
    assert agent_pos == (2, 2), f"Agent should have moved to (2,2), got {agent_pos}"
    
    # Check block positions
    block1_cell = env.grid.get(3, 2)
    block2_cell = env.grid.get(4, 2)
    
    assert block1_cell is not None and block1_cell.type == 'block', "Block should be at (3,2)"
    assert block2_cell is not None and block2_cell.type == 'block', "Block should be at (4,2)"
    
    print("  ✓ Agent successfully pushed blocks onto empty cells")
    print("  ✓ Control test passed")


def run_all_tests():
    """Run all chain conflict push tests."""
    print("=" * 70)
    print("Chain Conflict Prevention Tests for Block Pushing")
    print("=" * 70)
    print()
    
    tests = [
        test_cannot_push_blocks_onto_vacated_cell,
        test_can_push_blocks_onto_empty_cell,
    ]
    
    for test_func in tests:
        test_func()
        print()
    
    print("=" * 70)
    print(f"All {len(tests)} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
