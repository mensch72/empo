#!/usr/bin/env python3
"""Replay the MCTS actions to see what actually happened."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

# From the MCTS output - first few actions
# "Forward -> Forward -> Forward -> Still -> Forward -> Still -> Forward -> Right -> Right -> Still -> Still -> Right -> Forward -> Forward -> Forward -> Forward -> Right -> Right -> Forward -> Left"

action_map = {
    'Left': 0,
    'Right': 1,
    'Forward': 2,
    'Still': 3,
    'Pickup': 4,
    'Drop': 5,
    'Toggle': 6
}

# Simulate a few key steps to see what happens
print("=== INITIAL STATE ===")
print(f"Robot at {tuple(env.agents[0].pos)}, dir={env.agents[0].dir}")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")

# Let's trace the first 20 actions to see the pattern
actions_str = "Forward Forward Forward Still Forward Still Forward Right Right Still Still Right Forward Forward Forward Forward Right Right Forward Left".split()

print("\n=== SIMULATING FIRST 20 ROBOT ACTIONS ===")
for i, action_name in enumerate(actions_str[:20]):
    action = action_map.get(action_name, 3)

    # Only robot acts (agent 0)
    actions = [action] + [3] * (len(env.agents) - 1)  # Robot action + humans still

    old_pos = tuple(env.agents[0].pos)
    old_dir = env.agents[0].dir

    env.step(actions)

    new_pos = tuple(env.agents[0].pos)
    new_dir = env.agents[0].dir

    if i < 20:
        print(f"Step {i:2d}: {action_name:8s} | Robot: {old_pos} dir={old_dir} -> {new_pos} dir={new_dir}")

print("\n=== FINAL ROCK POSITIONS (after 20 steps) ===")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")

print("\n=== CHECKING: Did it push any rocks? ===")
# Initial rocks were at (3,4) and (7,4)
# If robot pushed right rock, it should be at (8,4) or (9,4)
# If robot pushed left rock, it should be at (2,4) or (1,4)
