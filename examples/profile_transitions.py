#!/usr/bin/env python3
"""
Profile script for transition_probabilities() and backward induction.

This script profiles the main performance-critical functions to identify bottlenecks.

PROFILING RESULTS SUMMARY
=========================

Key bottlenecks in get_dag() / transition_probabilities():
1. get_state(): 51% of time - most expensive due to object serialization
2. set_state(): 41% of time - expensive due to object deserialization  
3. _deserialize_object(): Called 6.7M times, 7.7s total
4. sorted(): Called 4.6M times for state tuple creation, 3.1s total

Analysis of return format:
- Current format: List[(prob, successor_state)]
- Alternative format: ([probs], [successor_states])
- Current format is slightly faster for iteration (0.08μs vs 0.13μs)
- Difference is negligible for typical use cases

Optimization opportunities in transition_probabilities():
1. Reduce get_state() calls: Called 2x per transition (backup + result)
2. Avoid set_state() when not needed: Object deserialization is expensive
3. Cache state serialization: sorted() is repeatedly called on same objects
4. Consider memoization: Many states are visited multiple times

Current performance with cached transitions:
- DAG computation (4338 states): ~22s
- Backward induction with cached transitions: 0.2s
- Total speedup vs recomputing transitions: ~3 orders of magnitude
"""

import sys
import os
import time
import cProfile
import pstats
from io import StringIO

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from envs.one_or_three_chambers import SmallOneOrTwoChambersMapEnv


def profile_transition_probabilities():
    """Profile the transition_probabilities method."""
    print("=" * 70)
    print("Profiling transition_probabilities()")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    initial_state = env.get_state()
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    
    print(f"Agents: {num_agents}, Actions per agent: {num_actions}")
    print(f"Total action combinations: {num_actions ** num_agents}")
    
    # Profile calling transition_probabilities for all action combinations
    profiler = cProfile.Profile()
    
    n_calls = 0
    profiler.enable()
    
    start = time.time()
    for combo_idx in range(num_actions ** num_agents):
        actions = []
        temp = combo_idx
        for _ in range(num_agents):
            actions.append(temp % num_actions)
            temp //= num_actions
        
        result = env.transition_probabilities(initial_state, actions)
        n_calls += 1
    
    elapsed = time.time() - start
    profiler.disable()
    
    print(f"\n{n_calls} calls in {elapsed:.4f}s ({elapsed/n_calls*1000:.3f}ms per call)")
    
    # Print profiling stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    print("\nTop functions by cumulative time:")
    print(stream.getvalue())


def profile_get_dag():
    """Profile the get_dag method."""
    print("=" * 70)
    print("Profiling get_dag()")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.time()
    states, state_to_idx, successors, transitions = env.get_dag(return_probabilities=True)
    elapsed = time.time() - start
    
    profiler.disable()
    
    print(f"\nget_dag completed in {elapsed:.2f}s")
    print(f"  States: {len(states)}")
    print(f"  Transitions cached: {sum(len(t) for t in transitions)}")
    
    # Print profiling stats
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    print("\nTop functions by cumulative time:")
    print(stream.getvalue())


def time_individual_operations():
    """Time individual operations within transition_probabilities."""
    print("=" * 70)
    print("Timing individual operations")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    initial_state = env.get_state()
    actions = [0, 0]  # still, still
    
    # Time get_state
    n_iter = 1000
    
    start = time.time()
    for _ in range(n_iter):
        env.get_state()
    get_state_time = (time.time() - start) / n_iter * 1000
    
    # Time set_state
    start = time.time()
    for _ in range(n_iter):
        env.set_state(initial_state)
    set_state_time = (time.time() - start) / n_iter * 1000
    
    # Time transition_probabilities
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(initial_state, actions)
    trans_prob_time = (time.time() - start) / n_iter * 1000
    
    print(f"  get_state: {get_state_time:.3f}ms per call")
    print(f"  set_state: {set_state_time:.3f}ms per call")
    print(f"  transition_probabilities: {trans_prob_time:.3f}ms per call")
    
    # Break down transition_probabilities overhead
    print("\n  transition_probabilities breakdown:")
    
    # Time state dict creation
    start = time.time()
    for _ in range(n_iter):
        state_dict = dict(initial_state)
    dict_time = (time.time() - start) / n_iter * 1000
    print(f"    dict(state): {dict_time:.3f}ms")
    
    # Time with different action types
    actions_forward = [3, 3]  # forward, forward
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(initial_state, actions_forward)
    trans_prob_forward_time = (time.time() - start) / n_iter * 1000
    print(f"    with forward actions: {trans_prob_forward_time:.3f}ms per call")


def analyze_return_format():
    """Analyze the return format of transition_probabilities."""
    print("=" * 70)
    print("Analyzing return format of transition_probabilities")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    initial_state = env.get_state()
    
    # Get some sample transitions
    results = []
    for combo_idx in range(16):
        actions = []
        temp = combo_idx
        for _ in range(2):
            actions.append(temp % 4)
            temp //= 4
        
        result = env.transition_probabilities(initial_state, actions)
        if result:
            results.append((tuple(actions), result))
    
    print(f"\nSample transitions from initial state:")
    for actions, trans in results[:5]:
        print(f"  actions={actions}: {len(trans)} outcomes")
        for prob, succ in trans[:2]:
            print(f"    prob={prob:.4f}, successor_len={len(succ)}")
    
    print("\nCurrent format: List[(prob, successor_state)]")
    print("Alternative format: ([probs], [successor_states])")
    
    # Benchmark both formats
    n_iter = 10000
    
    # Simulate current format processing
    sample_result = results[0][1]
    start = time.time()
    for _ in range(n_iter):
        for prob, succ in sample_result:
            _ = prob * 1.0  # Simulate using prob
    current_time = (time.time() - start) / n_iter * 1000000
    
    # Simulate alternative format processing
    probs = [p for p, s in sample_result]
    succs = [s for p, s in sample_result]
    start = time.time()
    for _ in range(n_iter):
        for i in range(len(probs)):
            _ = probs[i] * 1.0  # Simulate using prob
    alt_time = (time.time() - start) / n_iter * 1000000
    
    print(f"\nProcessing time comparison (n={len(sample_result)} outcomes):")
    print(f"  Current format: {current_time:.2f}μs")
    print(f"  Alternative format: {alt_time:.2f}μs")


if __name__ == "__main__":
    time_individual_operations()
    print()
    analyze_return_format()
    print()
    profile_transition_probabilities()
    print()
    profile_get_dag()
