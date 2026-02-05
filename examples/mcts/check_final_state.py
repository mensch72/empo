#!/usr/bin/env python3
"""Check final state after running MCTS for 20 steps."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig, MinRobotRiskHumanPolicy

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

config = MCTSConfig(
    num_simulations=300,
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

print("=== INITIAL STATE ===")
print(f"Robot: {tuple(env.agents[0].pos)}")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")

# Run MCTS for 20 steps
print("\n=== RUNNING MCTS FOR 20 STEPS ===")
for step in range(20):
    state = env.get_state()
    result = planner.search_with_result(state)
    robot_action = result.best_action

    # Get human actions
    human_actions = []
    for hidx in [1, 2]:
        h_action = human_prior.best_action(state, hidx)
        human_actions.append(h_action)

    # Combine actions
    full_actions = [robot_action[0], human_actions[0], human_actions[1]]
    env.step(full_actions)

    if step < 5 or step % 5 == 0:
        action_names = {0: "DoNothing", 1: "TurnRight", 2: "TurnLeft", 3: "MoveForward"}
        print(f"Step {step:2d}: Robot action {robot_action[0]} ({action_names[robot_action[0]]}), pos={tuple(env.agents[0].pos)}")

print("\n=== FINAL STATE (after 20 steps) ===")
print(f"Robot: {tuple(env.agents[0].pos)}")
for x in range(env.width):
    for y in range(env.height):
        cell = env.grid.get(x, y)
        if cell and hasattr(cell, 'type') and cell.type == 'rock':
            print(f"Rock at ({x}, {y})")

print("\n=== ANALYSIS ===")
print("Initial rocks: (3,4) and (7,4)")
print("If robot pushed right rock: should be at (8,4) or (9,4)")
print("If robot pushed left rock: should be at (2,4) or (1,4)")
print("If robot pushed both: rocks at (1,4) or (2,4) AND (8,4) or (9,4)")
