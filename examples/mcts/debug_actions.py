#!/usr/bin/env python3
"""Debug action space to understand why Forward isn't working."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_simple.yaml', partial_obs=False)
env.reset()

print("=== ACTION SPACE INFO ===")
print(f"Number of agents: {len(env.agents)}")
print(f"Action space: {env.action_space}")

print("\n=== INITIAL ROBOT STATE ===")
robot = env.agents[0]
print(f"Robot position: {tuple(robot.pos)}")
print(f"Robot direction: {robot.dir} (0=Right, 1=Down, 2=Left, 3=Up)")

print("\n=== TESTING ACTION 2 (Forward?) ===")
# Try action 2 for robot only
actions = [2, 3, 3]
obs, reward, done, info = env.step(actions)

print(f"Robot position after step: {tuple(env.agents[0].pos)}")
print(f"Robot direction after step: {env.agents[0].dir}")

print("\n=== TESTING ALL ACTIONS (0-3) ===")
for action_idx in range(4):
    env.reset()
    robot_before = tuple(env.agents[0].pos), env.agents[0].dir

    actions = [action_idx, 3, 3]
    env.step(actions)

    robot_after = tuple(env.agents[0].pos), env.agents[0].dir

    action_names = ['Left', 'Right', 'Forward', '???']
    print(f"Action {action_idx} ({action_names[action_idx]:7s}): {robot_before} -> {robot_after}")
