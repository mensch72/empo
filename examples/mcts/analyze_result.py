#!/usr/bin/env python3
"""Analyze the MCTS result to see what strategy was followed."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

print("=== INITIAL STATE ===")
print(f"Robot position: {env.agents[0].pos}")
print(f"Robot direction: {env.agents[0].dir}")

print("\n=== INITIAL ROCKS ===")
rocks = []
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            rocks.append((x, y))
            print(f"Rock at ({x}, {y})")

print("\n=== INITIAL HUMANS ===")
for idx in [1, 2]:
    agent = env.agents[idx]
    print(f"Human {idx} at {tuple(agent.pos)}")

# Simulate a simple test: push left rock twice
print("\n=== TESTING: What if robot pushes LEFT rock twice? ===")
print("Expected: Opens path for Human at (3,6) to access more goals")

# Check what cells Human at (3,6) can reach before vs after pushing left rock
from collections import deque

def get_reachable_cells(human_pos, rock_positions):
    """BFS to find all reachable cells."""
    reachable = set()
    queue = deque([human_pos])
    visited = set([human_pos])

    while queue:
        x, y = queue.popleft()
        reachable.add((x, y))

        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if (nx, ny) in rock_positions:
                continue

            cell = env.grid.get(nx, ny)
            if cell is not None and hasattr(cell, 'type') and cell.type == 'wall':
                continue

            visited.add((nx, ny))
            queue.append((nx, ny))

    return reachable

# Before pushing
human_pos = (3, 6)
rock_positions = {(3, 4), (7, 4)}
before = get_reachable_cells(human_pos, rock_positions)
print(f"Human at {human_pos} can reach {len(before)} cells BEFORE pushing left rock")

# After pushing left rock twice (moves from 3,4 to 1,4)
rock_positions_after = {(1, 4), (7, 4)}
after = get_reachable_cells(human_pos, rock_positions_after)
print(f"Human at {human_pos} can reach {len(after)} cells AFTER pushing left rock twice")
print(f"Gain: +{len(after) - len(before)} cells")

# Check right rock
print("\n=== TESTING: What if robot pushes RIGHT rock twice? ===")
human_pos_right = (7, 5)
rock_positions_right_before = {(3, 4), (7, 4)}
before_right = get_reachable_cells(human_pos_right, rock_positions_right_before)
print(f"Human at {human_pos_right} can reach {len(before_right)} cells BEFORE pushing right rock")

rock_positions_right_after = {(3, 4), (9, 4)}
after_right = get_reachable_cells(human_pos_right, rock_positions_right_after)
print(f"Human at {human_pos_right} can reach {len(after_right)} cells AFTER pushing right rock twice")
print(f"Gain: +{len(after_right) - len(before_right)} cells")

# Check both rocks
print("\n=== TESTING: What if robot pushes BOTH rocks? ===")
rock_positions_both = {(1, 4), (9, 4)}
left_both = get_reachable_cells(human_pos, rock_positions_both)
right_both = get_reachable_cells(human_pos_right, rock_positions_both)
total_both = len(left_both) + len(right_both)
total_before = len(before) + len(before_right)
print(f"Total reachable cells (both humans) BEFORE: {total_before}")
print(f"Total reachable cells (both humans) AFTER pushing both: {total_both}")
print(f"Gain: +{total_both - total_before} cells")

print("\n=== EXPECTED OPTIMAL STRATEGY ===")
print("1. Human at (7,5) moves down to escape danger")
print("2. Robot pushes right rock twice: (7,4) -> (8,4) -> (9,4)")
print("3. Robot goes to left side and pushes left rock twice: (3,4) -> (2,4) -> (1,4)")
print("4. Both humans gain maximum mobility")
