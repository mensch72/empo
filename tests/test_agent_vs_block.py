#!/usr/bin/env python3
"""
Test for the specific case: one stumbling agent moves to cell X,
another stumbling agent pushes a block that would land in cell X.
Both should fail.
"""

import sys
import numpy as np
from pathlib import Path
import gymnasium as gym

sys.modules["gym"] = gym

current_file = Path(__file__).resolve()
repo_root = current_file.parent.parent
multigrid_path = repo_root / "vendor" / "multigrid"

# Avoiding unfound modules
for p in [str(repo_root), str(multigrid_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from gym_multigrid.multigrid import (
    MultiGridEnv,
    Agent,
    World,
    Actions,
    Grid,
    Wall,
    UnsteadyGround,
    Block,
)


class AgentVsBlockPushTestEnv(MultiGridEnv):
    """
    Test environment where:
    - Agent 0 is on unsteady ground at (2,2) facing right
    - Agent 1 is on unsteady ground at (2,4) facing up
    - Block is at (3,3)

    Layout (7x6):
    W W W W W W W
    W . . . . . W
    W . U . . . W  <- Agent 0 at (2,2) on unsteady, facing right
    W . . B . . W  <- Block at (3,3)
    W . U . . . W  <- Agent 1 at (2,4) on unsteady, facing up
    W W W W W W W

    Scenario:
    - Agent 0 (at 2,2 facing right):
        * No stumble: moves to (3,2)
        * Stumbles right (down): turns down, moves to (2,3)
        * Stumbles left (up): turns up, moves to (2,1)

    - Agent 1 (at 2,4 facing up):
        * No stumble: moves to (2,3)
        * Stumbles right: turns right (facing right), moves to (3,4)
        * Stumbles left: turns left (facing left), moves to (1,4)

    KEY CONFLICT: Both agents can target (2,3):
    - Agent 0 stumbles down -> (2,3)
    - Agent 1 doesn't stumble -> (2,3)
    Both should fail!

    But we want to test block pushing too. Let me adjust...
    """

    def __init__(self):
        self.agents = [Agent(World, i % 6) for i in range(2)]
        super().__init__(
            width=8,
            height=6,
            max_steps=100,
            agents=self.agents,
            partial_obs=False,
            objects_set=World,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Add walls
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height - 1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width - 1, j, Wall(World))

        # Place unsteady ground
        unsteady1 = UnsteadyGround(World, stumble_probability=0.5, color="brown")
        unsteady2 = UnsteadyGround(World, stumble_probability=0.5, color="brown")
        self.grid.set(2, 2, unsteady1)  # For Agent 0
        self.grid.set(5, 2, unsteady2)  # For Agent 1
        # Save terrain to terrain_grid so it persists under agents
        self.terrain_grid.set(2, 2, unsteady1)
        self.terrain_grid.set(5, 2, unsteady2)

        # Place block at (4,2)
        block = Block(World)
        self.grid.set(4, 2, block)

        # Agent 0 at (2,2) on unsteady ground, facing right
        # - No stumble: moves to (3,2)
        # - Stumbles down: turns down, moves to (2,3)
        # - Stumbles up: turns up, moves to (2,1)
        self.agents[0].pos = np.array([2, 2])
        self.agents[0].dir = 0  # facing right
        self.agents[0].init_pos = self.agents[0].pos.copy()
        self.agents[0].init_dir = self.agents[0].dir
        self.agents[0].on_unsteady_ground = True
        self.grid.set(2, 2, self.agents[0])

        # Agent 1 at (5,2) on unsteady ground, facing right
        # - No stumble: pushes block from (4,2) forward, block goes to (3,2), agent moves to (4,2)
        #   NO WAIT - agent is at (5,2) facing right, so forward is (6,2) which is wall
        # Let me make agent face LEFT
        # - No stumble (facing left): pushes block from (4,2) to (3,2), agent moves to (4,2)
        # - Stumbles down (left->down): turns down, moves to (5,3)
        # - Stumbles up (left->up): turns up, moves to (5,1)

        # THIS IS THE KEY: Agent 1 no-stumble pushes block to (3,2), which is where Agent 0 might move!
        self.agents[1].pos = np.array([5, 2])
        self.agents[1].dir = 2  # facing left toward the block
        self.agents[1].init_pos = self.agents[1].pos.copy()
        self.agents[1].init_dir = self.agents[1].dir
        self.agents[1].on_unsteady_ground = True
        self.grid.set(5, 2, self.agents[1])


def test_agent_vs_pushed_block():
    """
    Test that when one agent moves to cell X and another pushes a block to cell X,
    neither succeeds.
    """
    print("Test: Agent moving to cell X vs Agent pushing block to cell X...")

    env = AgentVsBlockPushTestEnv()
    env.reset()

    print("\nSetup:")
    print(f"  Agent 0: pos={env.agents[0].pos}, dir={env.agents[0].dir} (facing right)")
    print(f"  Agent 1: pos={env.agents[1].pos}, dir={env.agents[1].dir} (facing left)")
    print("  Block at: (4, 2)")
    print()
    print("Expected behaviors:")
    print("  Agent 0 no-stumble: moves to (3,2)")
    print("  Agent 1 no-stumble: pushes block from (4,2) to (3,2), moves to (4,2)")
    print(
        "  CONFLICT: Both target (3,2) - Agent 0 moves there, Agent 1 pushes block there"
    )
    print("  Result: Neither should succeed!")
    print()

    # Both agents attempt forward
    actions = [Actions.forward, Actions.forward]

    state = env.get_state()
    result = env.transition_probabilities(state, actions)

    assert result is not None, "Got None result from transition_probabilities"

    print(f"Number of possible outcomes: {len(result)}")

    # Check probabilities sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-10, f"Probabilities sum to {total_prob}, not 1.0"
    print("✓ Probabilities sum to 1.0\n")

    # Print outcomes
    print("Outcomes:")
    conflict_handled_correctly = True
    for i, (prob, succ_state) in enumerate(result):
        # Compact state format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, mobile_objects, mutable_objects = succ_state

        # Agent states format: (pos_x, pos_y, dir, terminated, started, paused, on_unsteady, carrying_type, carrying_color)
        agent0_state = agent_states[0]
        agent1_state = agent_states[1]

        agent0_pos = (agent0_state[0], agent0_state[1])
        agent0_dir = agent0_state[2]
        agent1_pos = (agent1_state[0], agent1_state[1])
        agent1_dir = agent1_state[2]

        # Check block position from mobile_objects
        # mobile_objects format: (obj_type, x, y, color)
        block_pos = None
        for obj in mobile_objects:
            if obj[0] == "block":
                block_pos = (obj[1], obj[2])
                break

        a0_moved = agent0_pos != (2, 2)
        a1_moved = agent1_pos != (5, 2)
        block_moved = block_pos != (4, 2)

        print(
            f"  {i + 1}. Prob={prob:.4f}: "
            f"A0@{agent0_pos} dir={agent0_dir} {'MOVED' if a0_moved else 'STAYED'}, "
            f"A1@{agent1_pos} dir={agent1_dir} {'MOVED' if a1_moved else 'STAYED'}, "
            f"Block@{block_pos} {'MOVED' if block_moved else 'STAYED'}"
        )

        # Check for the conflict case: both don't stumble
        # Agent 0 dir should be 0 (no turn), Agent 1 dir should be 2 (no turn)
        if agent0_dir == 0 and agent1_dir == 2:
            print("     -> This is the NO-STUMBLE case for both agents")
            # Check if both were blocked
            if agent0_pos == (2, 2) and agent1_pos == (5, 2) and block_pos == (4, 2):
                print("     -> ✓ CORRECT: Both agents blocked, block stayed")
            elif agent0_pos == (3, 2):
                print("     -> ✗ ERROR: Agent 0 moved to (3,2)")
                conflict_handled_correctly = False
            elif block_pos == (3, 2):
                print("     -> ✗ ERROR: Block was pushed to (3,2)")
                conflict_handled_correctly = False

    print()
    assert conflict_handled_correctly, (
        "Conflict NOT handled correctly: agent or block occupies contested cell"
    )
    print(
        "✓ Conflict handled correctly: neither agent nor block occupies contested cell"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Agent vs Pushed Block Conflict Test")
    print("=" * 80)
    print()

    test_agent_vs_pushed_block()
    print()
    print("=" * 80)
    print("Result: PASS")
    print("=" * 80)
