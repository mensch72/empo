"""Test script to verify indexed goals functionality."""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')
sys.path.insert(0, 'vendor/ai_transport')
sys.path.insert(0, 'multigrid_worlds')

from gym_multigrid.multigrid import MultiGridEnv

# Test 1: Create environment from YAML with possible_goals
print("Test 1: Loading environment with YAML goals...")
env = MultiGridEnv(config_file='multigrid_worlds/basic/two_agents.yaml')

# Check that sampler and generator have indexed=True
assert hasattr(env.possible_goal_sampler, 'indexed'), "Sampler should have indexed attribute"
assert hasattr(env.possible_goal_generator, 'indexed'), "Generator should have indexed attribute"
assert env.possible_goal_sampler.indexed == True, "Sampler.indexed should be True for YAML-loaded goals"
assert env.possible_goal_generator.indexed == True, "Generator.indexed should be True for YAML-loaded goals"
print(f"✓ Sampler.indexed = {env.possible_goal_sampler.indexed}")
print(f"✓ Generator.indexed = {env.possible_goal_generator.indexed}")

# Check that sampler and generator have n_goals attribute
assert hasattr(env.possible_goal_sampler, 'n_goals'), "Sampler should have n_goals attribute"
assert hasattr(env.possible_goal_generator, 'n_goals'), "Generator should have n_goals attribute"
print(f"✓ Sampler.n_goals = {env.possible_goal_sampler.n_goals}")
print(f"✓ Generator.n_goals = {env.possible_goal_generator.n_goals}")

# Test 2: Check that goals have indices set
print("\nTest 2: Checking goal indices...")
goals = list(env.possible_goal_generator._get_goals_for_agent(0))
for i, goal in enumerate(goals):
    assert hasattr(goal, 'index'), f"Goal {i} should have index attribute"
    assert goal.index == i, f"Goal {i} should have index={i}, got {goal.index}"
    print(f"✓ Goal {i}: index={goal.index}, target={goal.target_pos}")

# Test 3: Test with generator iteration
print("\nTest 3: Testing generator iteration...")
state = env.get_state()
for i, (goal, weight) in enumerate(env.possible_goal_generator.generate(state, 0)):
    assert goal.index == i, f"Generated goal {i} should have index={i}, got {goal.index}"
print(f"✓ All {i+1} generated goals have correct indices")

# Test 4: Test with sampler
print("\nTest 4: Testing sampler...")
for _ in range(10):
    goal, weight = env.possible_goal_sampler.sample(state, 0)
    assert hasattr(goal, 'index'), "Sampled goal should have index attribute"
    assert goal.index is not None, "Sampled goal should have non-None index"
    assert 0 <= goal.index < len(goals), f"Goal index {goal.index} should be in range [0, {len(goals)})"
print(f"✓ Sampled goals have valid indices")

print("\n✅ All tests passed!")
