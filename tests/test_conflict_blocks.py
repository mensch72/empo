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

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Actions, Grid, Wall, Ball
from gym_multigrid.envs import CollectGame4HEnv10x10N2


def get_agent_positions_from_compact_state(compact_state):
    """Extract agent positions from compact state format."""
    step_count, agent_states, mobile_objects, mutable_objects = compact_state
    positions = []
    for agent_state in agent_states:
        pos_x, pos_y, dir_, terminated, started, paused, on_unsteady, carrying_type, carrying_color = agent_state
        positions.append((pos_x, pos_y, dir_))
    return positions


def test_conflict_block_efficiency():
    """
    Test that conflict block optimization is more efficient than full permutation.
    
    With 4 agents, full permutation would be 4!=24.
    But if agents split into 2 blocks of 2, conflict blocks give us only 2×2=4 outcomes.
    """
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    state = env.get_state()
    
    # All agents moving forward - may or may not conflict depending on positions
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    # Time the computation
    start_time = time.time()
    result = env.transition_probabilities(state, actions)
    elapsed_time = time.time() - start_time
    
    # Verify probabilities still sum to 1.0
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-9, f"Probabilities sum to {total_prob}, not 1.0"


def test_two_agents_competing_for_cell():
    """
    Test the canonical case: two agents competing for the same cell.
    
    This should produce 2 outcomes with equal probability 0.5 each.
    """
    # Create environment and get initial state
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Manufacture a state where two agents will compete for the same cell
    # Place agent 0 at (1, 2) facing right (dir=0) -> moves to (2, 2)
    # Place agent 1 at (3, 2) facing left (dir=2) -> moves to (2, 2)
    # Agent 2 stays still
    
    # Clear old agent positions from grid
    for agent in env.agents:
        if agent.pos is not None:
            cell = env.grid.get(*agent.pos)
            if cell is agent:
                env.grid.set(*agent.pos, None)
    
    # Agent 0: position (1, 2), direction 0 (right)
    env.agents[0].pos = np.array([1, 2])
    env.agents[0].dir = 0  # facing right
    env.grid.set(1, 2, env.agents[0])
    
    # Agent 1: position (3, 2), direction 2 (left)
    env.agents[1].pos = np.array([3, 2])
    env.agents[1].dir = 2  # facing left
    env.grid.set(3, 2, env.agents[1])
    
    # Agent 2: keep somewhere safe
    env.agents[2].pos = np.array([5, 5])
    env.agents[2].dir = 0
    env.grid.set(5, 5, env.agents[2])
    
    # Get the modified state
    modified_state = env.get_state()
    
    # Both agents 0 and 1 move forward to (2, 2), agent 2 stays still
    actions = [Actions.forward, Actions.forward, Actions.still]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 2 outcomes with probability 0.5 each
    assert len(result) == 2, f"Expected 2 outcomes, got {len(result)}"
    
    # Check probabilities
    probs = [prob for prob, _ in result]
    assert all(abs(p - 0.5) < 1e-9 for p in probs), f"Expected probabilities [0.5, 0.5], got {probs}"
    
    total_prob = sum(probs)
    assert abs(total_prob - 1.0) < 1e-9, f"Probabilities sum to {total_prob}, not 1.0"


def test_probability_values_with_conflicts():
    """
    Test that probability values are correct when conflicts exist.
    
    If we have 3 agents competing for same cell, each outcome should have probability 1/3.
    """
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Manufacture a state where all 3 agents compete for cell (5, 5)
    # Agent 0 at (4, 5) facing right (dir=0)
    # Agent 1 at (6, 5) facing left (dir=2) 
    # Agent 2 at (5, 4) facing down (dir=1)
    
    # Clear old agent positions from grid
    for agent in env.agents:
        if agent.pos is not None:
            cell = env.grid.get(*agent.pos)
            if cell is agent:
                env.grid.set(*agent.pos, None)
    
    # Agent 0: position (4, 5), direction 0 (right)
    env.agents[0].pos = np.array([4, 5])
    env.agents[0].dir = 0
    env.grid.set(4, 5, env.agents[0])
    
    # Agent 1: position (6, 5), direction 2 (left)
    env.agents[1].pos = np.array([6, 5])
    env.agents[1].dir = 2
    env.grid.set(6, 5, env.agents[1])
    
    # Agent 2: position (5, 4), direction 1 (down)
    env.agents[2].pos = np.array([5, 4])
    env.agents[2].dir = 1
    env.grid.set(5, 4, env.agents[2])
    
    # Get the modified state
    modified_state = env.get_state()
    
    # All agents move forward to (5, 5)
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 3 outcomes with probability 1/3 each
    assert len(result) == 3, f"Expected 3 outcomes, got {len(result)}"
    
    # Check probabilities
    expected_prob = 1.0 / 3.0
    for i, (prob, _) in enumerate(result):
        assert abs(prob - expected_prob) < 1e-9, f"Outcome {i+1}: expected {expected_prob:.4f}, got {prob:.4f}"
    
    # Verify sum
    total_prob = sum(prob for prob, _ in result)
    assert abs(total_prob - 1.0) < 1e-9, f"Probabilities sum to {total_prob}, not 1.0"


def test_independent_agents_deterministic():
    """
    Test that agents with independent actions produce deterministic outcomes.
    """
    env = CollectGame4HEnv10x10N2()
    env.reset()
    
    # Manufacture a state where agents are far apart and won't interfere
    # Clear old agent positions from grid
    for agent in env.agents:
        if agent.pos is not None:
            cell = env.grid.get(*agent.pos)
            if cell is agent:
                env.grid.set(*agent.pos, None)
    
    # Agent 0 at (2, 2), facing right
    env.agents[0].pos = np.array([2, 2])
    env.agents[0].dir = 0
    env.grid.set(2, 2, env.agents[0])
    
    # Agent 1 at (7, 7), facing up
    env.agents[1].pos = np.array([7, 7])
    env.agents[1].dir = 3
    env.grid.set(7, 7, env.agents[1])
    
    # Agent 2 at (2, 7), facing down
    env.agents[2].pos = np.array([2, 7])
    env.agents[2].dir = 1
    env.grid.set(2, 7, env.agents[2])
    
    # Get the modified state
    modified_state = env.get_state()
    
    # All agents move forward to different cells (no conflicts)
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should be deterministic (single outcome)
    assert len(result) == 1, f"Expected 1 outcome (deterministic), got {len(result)}"
    assert result[0][0] == 1.0, f"Expected probability 1.0, got {result[0][0]}"


def test_two_conflict_blocks():
    """
    Test two separate conflict blocks: 2 agents compete for cell A, 2 compete for cell B.
    
    Should produce 2×2=4 outcomes, each with probability 1/4.
    """
    # Create a custom environment with 4 agents to test 2 blocks of 2
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
    
    # Clear old agent positions from grid
    for agent in env.agents:
        if agent.pos is not None:
            cell = env.grid.get(*agent.pos)
            if cell is agent:
                env.grid.set(*agent.pos, None)
    
    # Block 1: Agents 0,1 compete for (3, 3)
    # Agent 0 at (2, 3) facing right - will move to (3, 3)
    env.agents[0].pos = np.array([2, 3])
    env.agents[0].dir = 0
    env.grid.set(2, 3, env.agents[0])
    
    # Agent 1 at (4, 3) facing left - will move to (3, 3)
    env.agents[1].pos = np.array([4, 3])
    env.agents[1].dir = 2
    env.grid.set(4, 3, env.agents[1])
    
    # Block 2: Agents 2,3 compete for (6, 6)
    # Agent 2 at (5, 6) facing right - will move to (6, 6)
    env.agents[2].pos = np.array([5, 6])
    env.agents[2].dir = 0
    env.grid.set(5, 6, env.agents[2])
    
    # Agent 3 at (7, 6) facing left - will move to (6, 6)
    env.agents[3].pos = np.array([7, 6])
    env.agents[3].dir = 2
    env.grid.set(7, 6, env.agents[3])
    
    # Get the modified state
    modified_state = env.get_state()
    
    # All 4 agents move forward, creating 2 separate conflicts
    actions = [Actions.forward, Actions.forward, Actions.forward, Actions.forward]
    
    result = env.transition_probabilities(modified_state, actions)
    
    # Should have 2×2=4 outcomes
    # (agent 0 or 1 wins cell A) × (agent 2 or 3 wins cell B)
    assert len(result) == 4, f"Expected 4 outcomes (2×2), got {len(result)}"
    
    # Each outcome should have probability 1/4
    expected_prob = 0.25
    for i, (prob, _) in enumerate(result):
        assert abs(prob - expected_prob) < 1e-9, f"Outcome {i+1}: expected 0.25, got {prob:.4f}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
