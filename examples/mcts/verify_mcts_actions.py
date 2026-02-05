#!/usr/bin/env python3
"""Verify MCTS is correctly evaluating rock-pushing actions."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig, MinRobotRiskHumanPolicy

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

# Correct action mapping for MultiGridEnv
ACTIONS = {
    0: "Do Nothing",
    1: "Turn Right",
    2: "Turn Left",
    3: "Move Forward"
}

print("=== INITIAL STATE ===")
print(f"Robot at {tuple(env.agents[0].pos)}, direction {env.agents[0].dir} (0=Right)")

config = MCTSConfig(
    num_simulations=200,
    max_depth=15,
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

print("\n=== RUNNING MCTS (200 sims, depth 15) ===")
state = env.get_state()
result = planner.search_with_result(state)

print(f"\nBest action: {result.best_action[0]} ({ACTIONS[result.best_action[0]]})")
print(f"Total simulations: {result.total_simulations}")
print(f"Aggregate power (U_r): {result.aggregate_power:.4f}")

print("\nAction values:")
for action_tuple, q_val in sorted(result.q_values.items(), key=lambda x: x[1], reverse=True):
    a = action_tuple[0]
    n_val = result.visit_counts.get(action_tuple, 0)
    print(f"  Action {a} ({ACTIONS[a]:15s}): Q={q_val:8.4f}, N={n_val:4d}")

print("\n=== ANALYSIS ===")
print("Robot is at (5,4) facing Right (direction 0)")
print("Right rock is at (7,4)")
print("\nTo push right rock:")
print("  1. Action 3 (Move Forward) -> Robot moves to (6,4)")
print("  2. Action 3 (Move Forward) -> Robot pushes rock from (7,4) to (8,4)")
print("\nExpected: Action 3 should have the best Q-value")
best_action = result.best_action[0]
print(f"Actual: Action {best_action} ({ACTIONS[best_action]}) was chosen")

if best_action == 3:
    print("✓ CORRECT: MCTS chose to move forward!")
else:
    print(f"✗ WRONG: MCTS chose {ACTIONS[best_action]} instead of Move Forward")
    print(f"   This suggests MCTS isn't exploring deep enough to see the benefit of pushing rocks")
