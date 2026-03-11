"""Test that non-YAML goals have indexed=False."""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')
sys.path.insert(0, 'vendor/ai_transport')
sys.path.insert(0, 'multigrid_worlds')

from gym_multigrid.multigrid import MultiGridEnv
from empo.world_specific_helpers.multigrid import ReachCellGoal
from empo.possible_goal import TabularGoalGenerator, TabularGoalSampler

# Test 1: Create environment from YAML  
print("Test 1: Creating environment from YAML (for comparison)...")
env = MultiGridEnv(config_file='multigrid_worlds/basic/tiny_test.yaml')

# Create goals manually
goals = [
    ReachCellGoal(env, 0, (1, 1)),
    ReachCellGoal(env, 0, (2, 2)),
    ReachCellGoal(env, 0, (3, 3)),
]

# Check that goals don't have indices set
print("\nTest 2: Checking manually created goals...")
for goal in goals:
    assert hasattr(goal, 'index'), "Goal should have index attribute"
    assert goal.index is None, f"Goal should have index=None, got {goal.index}"
print(f"✓ All {len(goals)} manually created goals have index=None")

# Test 3: Create generator and sampler without indexed flag
print("\nTest 3: Creating generator and sampler without indexed flag...")
generator = TabularGoalGenerator(goals)
sampler = TabularGoalSampler(goals)

assert hasattr(generator, 'indexed'), "Generator should have indexed attribute"
assert hasattr(sampler, 'indexed'), "Sampler should have indexed attribute"
assert generator.indexed == False, f"Generator.indexed should be False, got {generator.indexed}"
assert sampler.indexed == False, f"Sampler.indexed should be False, got {sampler.indexed}"
print(f"✓ Generator.indexed = {generator.indexed}")
print(f"✓ Sampler.indexed = {sampler.indexed}")

# Tabular generators/samplers should NOT have n_goals attribute
assert not hasattr(generator, 'n_goals'), "Tabular generator should not have n_goals attribute"
assert not hasattr(sampler, 'n_goals'), "Tabular sampler should not have n_goals attribute"
print(f"✓ Tabular generator and sampler don't have n_goals attribute")

# Test 4: Create generator and sampler with indexed=True explicitly
print("\nTest 4: Creating generator and sampler with indexed=True...")
indexed_goals = [
    ReachCellGoal(env, 0, (1, 1), index=0),
    ReachCellGoal(env, 0, (2, 2), index=1),
    ReachCellGoal(env, 0, (3, 3), index=2),
]
generator_indexed = TabularGoalGenerator(indexed_goals, indexed=True)
sampler_indexed = TabularGoalSampler(indexed_goals, indexed=True)

assert generator_indexed.indexed == True, "Generator.indexed should be True"
assert sampler_indexed.indexed == True, "Sampler.indexed should be True"
print(f"✓ Generator.indexed = {generator_indexed.indexed}")
print(f"✓ Sampler.indexed = {sampler_indexed.indexed}")

# Check goal indices
for i, goal in enumerate(indexed_goals):
    assert goal.index == i, f"Goal {i} should have index={i}, got {goal.index}"
print(f"✓ All {len(indexed_goals)} goals have correct indices")

print("\n✅ All tests passed!")
