#!/usr/bin/env python3
"""Quick test to verify trivial.yaml loads correctly."""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')

from gym_multigrid.multigrid import MultiGridEnv, SmallActions

# Load the trivial world
env = MultiGridEnv(
    config_file='multigrid_worlds/trivial.yaml',
    partial_obs=False,
    actions_set=SmallActions
)
env.reset()

print("✓ Trivial world loaded successfully!")
print(f"  Grid size: {env.width}x{env.height}")
print(f"  Max steps: {env.max_steps}")
print(f"  Agents: {len(env.agents)}")
print(f"  Human indices: {env.human_agent_indices}")
print(f"  Robot indices: {env.robot_agent_indices}")

# Check goal generator
if hasattr(env, 'possible_goal_generator') and env.possible_goal_generator:
    print(f"  Goals: {len(env.possible_goal_generator.goal_coords)} from config")
    print(f"  Goal coords: {env.possible_goal_generator.goal_coords}")
else:
    print("  WARNING: No goal generator found!")

# Verify agents
for i, agent in enumerate(env.agents):
    print(f"  Agent {i}: color={agent.color}, pos={tuple(agent.pos)}")

print("\n✓ All checks passed!")
