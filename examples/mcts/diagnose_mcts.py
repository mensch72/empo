#!/usr/bin/env python3
"""Diagnose MCTS to see why it's not pushing rocks."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig

# Load environment
env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

print("=== INITIAL STATE ===")
robot_pos = tuple(env.agents[0].pos)
robot_dir = env.agents[0].dir
print(f"Robot at {robot_pos}, facing direction {robot_dir}")

# Direction mapping: 0=Right, 1=Down, 2=Left, 3=Up
dir_names = ['Right', 'Down', 'Left', 'Up']
print(f"Robot facing: {dir_names[robot_dir]}")

# Find rocks
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")

# Create MCTS planner
from empo.mcts import MinRobotRiskHumanPolicy

config = MCTSConfig(
    num_simulations=500,
    max_depth=20,
    verbose=False,
    use_transition_probabilities=False,
    zeta=2.0,
    xi=1.0,
    eta=1.1,
    beta_r=5.0
)

human_prior = MinRobotRiskHumanPolicy(env, human_agent_indices=[1, 2], robot_agent_indices=[0])

planner = MCTSPlanner(
    world_model=env,
    human_policy_prior=human_prior,
    human_agent_indices=[1, 2],
    robot_agent_indices=[0],
    config=config
)

# Get current state
state = env.get_state()

# Manually check what happens if robot pushes right rock
print("\n=== MANUAL TEST: What if robot pushes RIGHT rock? ===")
print("To push right rock at (7,4), robot needs to:")
print("1. Move from (5,4) to (6,4)")
print("2. Face Right (direction 0)")
print("3. Move Forward to push rock from (7,4) to (8,4)")

# Compute U_r for different scenarios
from collections import deque

def compute_reachable(human_pos, rock_positions, grid_obj):
    """Simplified reachability check."""
    visited = set()
    queue = deque([human_pos])
    visited.add(human_pos)

    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited or (nx, ny) in rock_positions:
                continue
            cell = grid_obj.get(nx, ny)
            if cell is not None and hasattr(cell, 'type') and cell.type == 'wall':
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))

    return len(visited)

# Current state
rocks_current = {(3, 4), (7, 4)}
h1_reach = compute_reachable((7, 5), rocks_current, env.grid)
h2_reach = compute_reachable((3, 6), rocks_current, env.grid)
print(f"\nCurrent: H1 can reach {h1_reach}, H2 can reach {h2_reach}, Total: {h1_reach + h2_reach}")

# After pushing right rock once
rocks_right1 = {(3, 4), (8, 4)}
h1_reach_r1 = compute_reachable((7, 5), rocks_right1, env.grid)
h2_reach_r1 = compute_reachable((3, 6), rocks_right1, env.grid)
print(f"After right rock x1: H1 can reach {h1_reach_r1}, H2 can reach {h2_reach_r1}, Total: {h1_reach_r1 + h2_reach_r1}")

# After pushing right rock twice
rocks_right2 = {(3, 4), (9, 4)}
h1_reach_r2 = compute_reachable((7, 5), rocks_right2, env.grid)
h2_reach_r2 = compute_reachable((3, 6), rocks_right2, env.grid)
print(f"After right rock x2: H1 can reach {h1_reach_r2}, H2 can reach {h2_reach_r2}, Total: {h1_reach_r2 + h2_reach_r2}")

# After pushing left rock twice
rocks_left2 = {(1, 4), (7, 4)}
h1_reach_l2 = compute_reachable((7, 5), rocks_left2, env.grid)
h2_reach_l2 = compute_reachable((3, 6), rocks_left2, env.grid)
print(f"After left rock x2: H1 can reach {h1_reach_l2}, H2 can reach {h2_reach_l2}, Total: {h1_reach_l2 + h2_reach_l2}")

# After pushing both rocks
rocks_both = {(1, 4), (9, 4)}
h1_reach_both = compute_reachable((7, 5), rocks_both, env.grid)
h2_reach_both = compute_reachable((3, 6), rocks_both, env.grid)
print(f"After both rocks: H1 can reach {h1_reach_both}, H2 can reach {h2_reach_both}, Total: {h1_reach_both + h2_reach_both}")

# Run a single MCTS planning step to see what it chooses
print("\n=== RUNNING MCTS (100 sims, depth 15) ===")
result = planner.search_with_result(state)
action = result.best_action[0]  # Get robot action from tuple
print(f"MCTS chose action: {action}")
print(f"Action meaning: {['Left', 'Right', 'Forward', 'Still', 'Pickup', 'Drop', 'Toggle'][action]}")
print(f"Total simulations: {result.total_simulations}")
print(f"Search time: {result.search_time_secs:.2f}s")
print(f"Aggregate power (U_r): {result.aggregate_power:.4f}")

# Check action values at root
print("\nRoot action values:")
action_names = ['Left', 'Right', 'Forward', 'Still', 'Pickup', 'Drop', 'Toggle']
for action_tuple, q_val in result.q_values.items():
    a = action_tuple[0]  # Robot action
    n_val = result.visit_counts.get(action_tuple, 0)
    print(f"  Action {a} ({action_names[a]:7s}): Q={q_val:8.4f}, N={n_val:4d}")
