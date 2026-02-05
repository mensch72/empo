#!/usr/bin/env python3
"""Test if pushing rocks actually improves U_r as measured by MCTS."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig, MinRobotRiskHumanPolicy

# Load environment
env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

config = MCTSConfig(
    num_simulations=50,
    max_depth=10,
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

print("=== INITIAL STATE ===")
state0 = env.get_state()
u_r_0 = planner._compute_intrinsic_reward(state0)
print(f"U_r at start: {u_r_0:.4f}")

# Manually move robot and push right rock
print("\n=== SIMULATING: Robot moves forward (6,4) ===")
env.step([2, 3, 3])  # Robot forward, humans still
state1 = env.get_state()
u_r_1 = planner._compute_intrinsic_reward(state1)
print(f"Robot position: {tuple(env.agents[0].pos)}")
print(f"U_r after moving: {u_r_1:.4f} (change: {u_r_1 - u_r_0:+.4f})")

# Check rocks
print("\nRock positions:")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"  Rock at ({x}, {y})")

print("\n=== SIMULATING: Robot pushes right rock (7,4) -> (8,4) ===")
env.step([2, 3, 3])  # Robot forward, humans still
state2 = env.get_state()
u_r_2 = planner._compute_intrinsic_reward(state2)
print(f"Robot position: {tuple(env.agents[0].pos)}")
print(f"U_r after pushing: {u_r_2:.4f} (change: {u_r_2 - u_r_1:+.4f})")

print("\nRock positions after push:")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"  Rock at ({x}, {y})")

print("\n=== SUMMARY ===")
print(f"Initial U_r:      {u_r_0:.4f}")
print(f"After move:       {u_r_1:.4f} ({u_r_1 - u_r_0:+.4f})")
print(f"After push:       {u_r_2:.4f} ({u_r_2 - u_r_0:+.4f})")
print(f"\nExpected: U_r should INCREASE significantly after pushing rock")
print(f"Actual: U_r {'INCREASED' if u_r_2 > u_r_0 else 'DECREASED or STAYED SAME'}")

if u_r_2 <= u_r_0:
    print("\n⚠️  PROBLEM: Pushing rock didn't improve U_r!")
    print("This explains why MCTS isn't choosing to push rocks.")
    print("The reward signal is broken or humans aren't benefiting from more space.")
else:
    print("\n✓ Good: Pushing rock improves U_r")
    print("MCTS should be finding this strategy with enough simulations.")
