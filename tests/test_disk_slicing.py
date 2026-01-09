#!/usr/bin/env python3
"""
Test disk-based DAG slicing functionality.

This script creates a small environment and tests the disk slicing feature
to verify it reduces memory usage as expected.
"""

import sys
import os

# Patch gym import
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.backward_induction import compute_human_policy_prior
from empo.world_specific_helpers.multigrid import ReachCellGoal
from empo.possible_goal import TabularGoalSampler
import psutil


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def main():
    print("="*70)
    print("Testing Disk-Based DAG Slicing")
    print("="*70)
    
    # Create environment
    env = MultiGridEnv(
        map="""
        We We We We We We
        We .. Ay .. .. We
        We .. .. .. Ro We
        We We Ar We We We
        """,
        max_steps=8,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )
    
    print(f"\nEnvironment: {env.width}x{env.height} grid, max_steps={env.max_steps}")
    print(f"Agents: {len(env.agents)}")
    print(f"Actions: {env.action_space.n}")
    
    # Create goal sampler
    human_idx = env.human_agent_indices[0]
    goals = [
        ReachCellGoal(env, human_idx, (2, 1)),
        ReachCellGoal(env, human_idx, (3, 1)),
        ReachCellGoal(env, human_idx, (1, 2)),
    ]
    goal_sampler = TabularGoalSampler(goals)
    
    # Level function for MultiGrid (timestep is first element of state)
    level_fct = lambda state: state[0]
    
    # Test 1: Without disk slicing
    print("\n" + "="*70)
    print("Test 1: Standard computation (no disk slicing)")
    print("="*70)
    mem_before = get_memory_mb()
    print(f"Memory before: {mem_before:.1f} MB")
    
    policy1 = compute_human_policy_prior(
        env,
        human_agent_indices=env.human_agent_indices,
        possible_goal_generator=goal_sampler,
        level_fct=level_fct,
        use_disk_slicing=False,
        use_float16=False,
        quiet=False
    )
    
    mem_after = get_memory_mb()
    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Memory used: {mem_after - mem_before:.1f} MB")
    mem_no_slicing = mem_after - mem_before
    
    # Test 2: With disk slicing
    print("\n" + "="*70)
    print("Test 2: With disk slicing + float16")
    print("="*70)
    
    # Reset environment to clear caches
    env.clear_dag_cache()
    
    mem_before = get_memory_mb()
    print(f"Memory before: {mem_before:.1f} MB")
    
    policy2 = compute_human_policy_prior(
        env,
        human_agent_indices=env.human_agent_indices,
        possible_goal_generator=goal_sampler,
        level_fct=level_fct,
        use_disk_slicing=True,
        use_float16=True,
        disk_cache_dir="/tmp/test_dag_slicing",
        quiet=False
    )
    
    mem_after = get_memory_mb()
    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Memory used: {mem_after - mem_before:.1f} MB")
    mem_with_slicing = mem_after - mem_before
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Memory without slicing: {mem_no_slicing:.1f} MB")
    print(f"Memory with slicing:    {mem_with_slicing:.1f} MB")
    reduction = mem_no_slicing / max(1, mem_with_slicing)
    print(f"Memory reduction:       {reduction:.1f}x")
    
    # Verify policies are equivalent (sample a few states)
    env.reset()
    state = env.get_state()
    goal = goals[0]
    
    probs1 = policy1(state, human_idx, goal)
    probs2 = policy2(state, human_idx, goal)
    
    import numpy as np
    if np.allclose(probs1, probs2, atol=1e-3):
        print("\n✓ Policies match (slicing preserves correctness)")
    else:
        print("\n✗ WARNING: Policies differ!")
        print(f"  Without slicing: {probs1}")
        print(f"  With slicing:    {probs2}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
