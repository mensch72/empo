#!/usr/bin/env python3
"""Quick sanity check before full run."""
import numpy as np
if not hasattr(np, 'bool'): np.bool = bool
from gym_multigrid.multigrid import MultiGridEnv
from empo.mcts import MCTSPlanner, MCTSConfig, MinRobotRiskHumanPolicy

env = MultiGridEnv(config_file='multigrid_worlds/puzzles/ali_challenges/mcts_straight.yaml', partial_obs=False)
env.reset()

human_indices = [idx for idx, a in enumerate(env.agents) if a.color != 'grey']
robot_indices = [idx for idx, a in enumerate(env.agents) if a.color == 'grey']

human_prior = MinRobotRiskHumanPolicy(env, human_indices, robot_indices)
config = MCTSConfig(num_simulations=50, max_depth=10, greedy_rollout=True, verbose=False)
planner = MCTSPlanner(env, human_prior, human_indices, robot_indices, config=config)

state = env.get_state()
print('Running quick sanity check...')
result = planner.search_with_result(state)

print(f'Best action: {result.best_action}')
print(f'Action distribution:')
for a, p in sorted(result.action_distribution.items(), key=lambda x: -x[1])[:4]:
    action_name = ['Still', 'Left', 'Right', 'Forward'][a[0]]
    print(f'  {action_name}: {p:.3f}')

if result.best_action == (3,):
    print('✓ Forward chosen - MCTS working correctly!')
    exit(0)
else:
    print('⚠ Forward not chosen - may need adjustments')
    exit(1)
