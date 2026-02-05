#!/usr/bin/env python3
"""Visualize the current map."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

print("=== MAP VISUALIZATION ===\n")
for y in range(env.height):
    line = f"{y}: "
    for x in range(env.width):
        cell = env.grid.get(x, y)
        if cell is None:
            char = ".."
        elif hasattr(cell, 'type'):
            if cell.type == 'wall':
                char = "WW"
            elif cell.type == 'rock':
                char = "RO"
            else:
                char = cell.type[:2].upper()
        else:
            char = "??"

        # Check for agents
        for idx, agent in enumerate(env.agents):
            if hasattr(agent, 'pos') and agent.pos is not None:
                ax, ay = int(agent.pos[0]), int(agent.pos[1])
                if ax == x and ay == y:
                    if agent.color == 'grey':
                        char = "Rb"
                    elif agent.color == 'yellow':
                        char = f"H{idx}"

        line += char + " "
    print(line)

print("\n   ", end="")
for x in range(env.width):
    print(f"{x:2d} ", end="")
print()

print("\n=== AGENTS ===")
for idx, agent in enumerate(env.agents):
    pos = tuple(agent.pos)
    color = agent.color
    print(f"Agent {idx} ({color}): {pos}")

print("\n=== ROCKS ===")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")
