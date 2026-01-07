#!/usr/bin/env python3
"""
Profile script for transition_probabilities() and get_dag().

This script profiles the main performance-critical functions.

PERFORMANCE RESULTS (with compact state representation)
======================================================

State operations:
- get_state(): ~0.02ms (compact state: step_count, agents, mobile_objects, mutable_objects)
- set_state(): ~0.03ms (syncs grid from compact state)
- transition_probabilities(): ~0.07ms (uses compact state internally)

DAG computation:
- get_dag() for 5000 states: ~5s
- ~52000 transitions cached

The compact state representation only stores mutable/mobile objects,
avoiding serialization of the entire grid (walls are not stored).
"""

import time

from multigrid_worlds.one_or_three_chambers import SmallOneOrTwoChambersMapEnv


def profile_state_operations():
    """Profile get_state, set_state, and transition_probabilities."""
    print("=" * 70)
    print("Profiling State Operations")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    state = env.get_state()
    num_agents = len(env.agents)
    actions_still = [0] * num_agents
    actions_forward = [3] * num_agents
    
    n_iter = 1000
    
    # Profile get_state
    start = time.time()
    for _ in range(n_iter):
        env.get_state()
    get_time = (time.time() - start) / n_iter * 1000
    
    # Profile set_state
    start = time.time()
    for _ in range(n_iter):
        env.set_state(state)
    set_time = (time.time() - start) / n_iter * 1000
    
    # Profile transition_probabilities (still actions - deterministic)
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(state, actions_still)
    trans_still_time = (time.time() - start) / n_iter * 1000
    
    # Profile transition_probabilities (forward actions)
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(state, actions_forward)
    trans_fwd_time = (time.time() - start) / n_iter * 1000
    
    print(f"\nState format: (step_count, agent_states, mobile_objects, mutable_objects)")
    print(f"  step_count: {state[0]}")
    print(f"  num_agents: {len(state[1])}")
    print(f"  num_mobile: {len(state[2])}")
    print(f"  num_mutable: {len(state[3])}")
    
    print(f"\nTiming ({n_iter} iterations):")
    print(f"  get_state(): {get_time:.3f}ms")
    print(f"  set_state(): {set_time:.3f}ms")
    print(f"  transition_probabilities (still): {trans_still_time:.3f}ms")
    print(f"  transition_probabilities (forward): {trans_fwd_time:.3f}ms")
    
    # Verify round-trip
    env.set_state(state)
    state2 = env.get_state()
    print(f"\n  Round-trip test: {'PASS' if state == state2 else 'FAIL'}")


def profile_get_dag(quick_mode=False):
    """Profile the get_dag method."""
    print("\n" + "=" * 70)
    print("Profiling get_dag()")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    if quick_mode:
        env.max_steps = 4  # Reduced from 8 for quick mode
    env.reset()
    
    print(f"\nEnvironment: max_steps={env.max_steps}")
    
    start = time.time()
    states, state_to_idx, successors, transitions = env.get_dag(return_probabilities=True)
    elapsed = time.time() - start
    
    print(f"\nget_dag completed in {elapsed:.2f}s")
    print(f"  States: {len(states)}")
    print(f"  Transitions: {sum(len(t) for t in transitions)}")
    print(f"  Avg transitions per state: {sum(len(t) for t in transitions) / len(states):.1f}")


def verify_correctness():
    """Verify that state operations work correctly."""
    print("\n" + "=" * 70)
    print("Verifying Correctness")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    # Test 1: Round-trip
    state1 = env.get_state()
    env.set_state(state1)
    state2 = env.get_state()
    print(f"\n  1. Round-trip test: {'PASS' if state1 == state2 else 'FAIL'}")
    
    # Test 2: State hashability
    state_set = {state1}
    state_set.add(state2)
    print(f"  2. State hashability: {'PASS' if len(state_set) == 1 else 'FAIL'}")
    
    # Test 3: Transition returns valid states
    num_agents = len(env.agents)
    actions = [0] * num_agents
    result = env.transition_probabilities(state1, actions)
    valid = result is not None and len(result) > 0
    if valid:
        prob_sum = sum(p for p, s in result)
        valid = abs(prob_sum - 1.0) < 0.0001
    print(f"  3. Transition probabilities sum to 1: {'PASS' if valid else 'FAIL'}")
    
    # Test 4: Successor states are hashable
    try:
        succ_set = {s for p, s in result}
        print(f"  4. Successor states hashable: PASS ({len(succ_set)} unique)")
    except Exception as e:
        print(f"  4. Successor states hashable: FAIL ({e})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Profile Transition Operations')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with reduced time steps')
    args = parser.parse_args()
    
    profile_state_operations()
    verify_correctness()
    profile_get_dag(quick_mode=args.quick)
