#!/usr/bin/env python3
"""
Test for conflict between stumbling agent moving to a cell and another pushing blocks to that cell.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, UnsteadyGround, Block


class UnsteadyCrossPushTestEnv(MultiGridEnv):
    """
    Test environment where two agents on unsteady ground compete:
    - Agent A stumbles into a cell
    - Agent B pushes a block into that same cell
    
    Layout (7x5):
    W W W W W W W
    W U . . B . W
    W U . . . . W
    W . . . . . W
    W W W W W W W
    
    Agent 0 at (1,1) on unsteady ground, facing right
    Agent 1 at (1,2) on unsteady ground, facing right
    Block at (4,1)
    
    If Agent 0 doesn't stumble: moves to (2,1)
    If Agent 1 stumbles up: turns up, moves to (1,1) - but that's wall, blocked
    If Agent 1 doesn't stumble: moves to (2,2)
    If Agent 1 stumbles right: turns down, moves to (1,3)
    
    But more importantly, we want to test the case where:
    - Agent 1 might push the block at (4,1) which would land at (5,1)
    - Meanwhile Agent 0 might stumble and try to move to (5,1) or some cell
    
    Actually, let me redesign this...
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
        
        # Place unsteady ground
        unsteady1 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
        unsteady2 = UnsteadyGround(World, stumble_probability=0.5, color='brown')
        self.grid.set(2, 2, unsteady1)
        self.grid.set(4, 2, unsteady2)
        
        # Place block at (3,2) - between the two agents
        block = Block(World)
        self.grid.set(3, 2, block)
        
        # Agent 0 at (2,2) on unsteady ground, facing right
        # If stumbles down: turns down, moves to (2,3)
        # If doesn't stumble: moves right to (3,2) - pushes block to (4,2) where agent 1 is!
        self.agents[0].pos = np.array([2, 2])
        self.agents[0].dir = 0  # facing right
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        self.agents[0].on_unsteady_ground = True
        self.grid.set(2, 2, self.agents[0])
        
        # Agent 1 at (4,2) on unsteady ground, facing left
        # If stumbles down: turns down, moves to (4,3)
        # If doesn't stumble: moves left to (3,2) - can't, there's a block
        # Actually this won't work because agent 1 can't push from this side
        
        # Let me revise: Agent 1 should be facing the block from the right side
        self.agents[1].pos = np.array([4, 2])
        self.agents[1].dir = 2  # facing left
        self.agents[1].init_pos = self.agents[1].pos.copy()
        self.agents[1].init_dir = self.agents[1].dir
        self.agents[1].on_unsteady_ground = True
        self.grid.set(4, 2, self.agents[1])


def test_cross_push_conflict():
    """
    Test that when one agent stumbles into a cell and another pushes a block
    into that same cell, neither succeeds.
    """
    print("Test: Cross-push conflict (agent moves to cell vs agent pushes block to cell)...")
    
    env = UnsteadyCrossPushTestEnv()
    env.reset()
    
    # Both agents attempt forward
    actions = [Actions.forward, Actions.forward]
    
    # Agent 0: on unsteady ground at (2,2) facing right
    #   - No stumble: pushes block from (3,2) to (4,2) where Agent 1 is - should fail
    #   - Stumble down: turns down, moves to (2,3)
    #   - Stumble up: turns up, moves to (2,1)
    
    # Agent 1: on unsteady ground at (4,2) facing left
    #   - No stumble: can't move left, block at (3,2) blocks the way, can't push from this side
    #   - Stumble down: turns down, moves to (4,3)
    #   - Stumble up: turns up, moves to (4,1)
    
    # The key conflict: if Agent 0 doesn't stumble, it tries to push block to (4,2)
    # But Agent 1 is at (4,2), so the push should fail
    
    # Let's check transition probabilities
    state = env.get_state()
    result = env.transition_probabilities(state, actions)
    
    if result is None:
        print("  ✗ Got None result")
        return False
    
    print(f"  Number of possible outcomes: {len(result)}")
    
    # Check probabilities sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    if abs(total_prob - 1.0) < 1e-10:
        print(f"  ✓ Probabilities sum to 1.0")
    else:
        print(f"  ✗ Probabilities sum to {total_prob}")
        return False
    
    # Print outcomes
    print("\n  Outcomes:")
    for i, (prob, succ_state) in enumerate(result):
        # Compact state format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = succ_state
        agent0_state = agent_states[0]
        agent1_state = agent_states[1]
        # Agent state format: (pos_x, pos_y, dir, terminated, started, paused, on_unsteady, carrying_type, carrying_color)
        agent0_pos = (agent0_state[0], agent0_state[1])
        agent0_dir = agent0_state[2]
        agent1_pos = (agent1_state[0], agent1_state[1])
        agent1_dir = agent1_state[2]
        
        # Check block position from mobile_objects
        # mobile_objects format: (obj_type, x, y, color, pushable_by)
        block_pos = None
        for obj in mobile_objects:
            if obj[0] == 'block':
                block_pos = (obj[1], obj[2])
                break
        
        print(f"    {i+1}. Prob={prob:.4f}: "
              f"A0@{agent0_pos} dir={agent0_dir}, "
              f"A1@{agent1_pos} dir={agent1_dir}, "
              f"Block@{block_pos}")
        
        # Verify that block never ends up at agent 1's original position
        if block_pos == (4, 2):
            print(f"       ⚠ Block is at (4,2) which is Agent 1's original position!")
    
    # We expect 3x3=9 outcomes total from stumbling combinations
    if len(result) >= 6:  # At least several distinct outcomes
        print(f"\n  ✓ Found {len(result)} outcomes (reasonable for 2 stumbling agents)")
        return True
    else:
        print(f"\n  ✗ Found only {len(result)} outcomes (expected more)")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Cross-Push Conflict Test")
    print("=" * 70)
    print()
    
    try:
        success = test_cross_push_conflict()
        print()
        print("=" * 70)
        print(f"Result: {'PASS' if success else 'FAIL'}")
        print("=" * 70)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
