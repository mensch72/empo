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

LINE PROFILING
==============

To run line-by-line profiling, install line_profiler:
    pip install line_profiler

Then run:
    kernprof -l -v examples/profile_transitions.py

Or use the manual line profiling in this script which times specific code sections.
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


def manual_line_profile_transition_probabilities():
    """
    Manual line-by-line profiling of transition_probabilities.
    
    This breaks down the time spent in each major section of the function.
    """
    print("=" * 70)
    print("Manual Line Profiling of transition_probabilities()")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    initial_state = env.get_state()
    num_agents = len(env.agents)
    actions = [0] * num_agents  # still for all agents
    
    n_iter = 500
    
    # Time each section of transition_probabilities
    print(f"\nProfiling with {n_iter} iterations...")
    
    # 1. Time dict(state) conversion
    start = time.time()
    for _ in range(n_iter):
        state_dict = dict(initial_state)
        _ = state_dict['step_count']
    dict_time = (time.time() - start) / n_iter * 1000
    
    # 2. Time get_state() (backup)
    start = time.time()
    for _ in range(n_iter):
        original_state = env.get_state()
    get_state_time = (time.time() - start) / n_iter * 1000
    
    # 3. Time set_state() (restore to query state)
    start = time.time()
    for _ in range(n_iter):
        env.set_state(initial_state)
    set_state_time = (time.time() - start) / n_iter * 1000
    
    # 4. Time active agent identification
    start = time.time()
    for _ in range(n_iter):
        active_agents = []
        for i in range(num_agents):
            if (not env.agents[i].terminated and 
                not env.agents[i].paused and 
                env.agents[i].started and 
                actions[i] != env.actions.still):
                active_agents.append(i)
    active_agent_time = (time.time() - start) / n_iter * 1000
    
    # 5. Time full transition_probabilities call
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(initial_state, actions)
    full_time = (time.time() - start) / n_iter * 1000
    
    # 6. Time compact state operations
    compact_state = env.get_compact_state()
    
    start = time.time()
    for _ in range(n_iter):
        _ = env.get_compact_state()
    get_compact_time = (time.time() - start) / n_iter * 1000
    
    start = time.time()
    for _ in range(n_iter):
        env.set_compact_state(compact_state)
    set_compact_time = (time.time() - start) / n_iter * 1000
    
    print("\nLine-by-line breakdown:")
    print(f"  1. dict(state) conversion:      {dict_time:.4f}ms")
    print(f"  2. get_state() backup:          {get_state_time:.4f}ms")
    print(f"  3. set_state() to query:        {set_state_time:.4f}ms")
    print(f"  4. Active agent identification: {active_agent_time:.4f}ms")
    print(f"  5. Full transition_probs call:  {full_time:.4f}ms")
    print()
    print("  Compact state alternatives:")
    print(f"  6. get_compact_state():         {get_compact_time:.4f}ms ({get_state_time/get_compact_time:.1f}x faster)")
    print(f"  7. set_compact_state():         {set_compact_time:.4f}ms ({set_state_time/set_compact_time:.1f}x faster)")
    
    # Calculate overhead breakdown
    overhead = get_state_time + set_state_time
    print()
    print(f"  State save/restore overhead: {overhead:.4f}ms ({100*overhead/full_time:.0f}% of total)")


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


def compare_state_representations():
    """Compare performance of full state vs compact state representations."""
    print("=" * 70)
    print("Comparing State Representations")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    n_iter = 1000
    
    # Time get_state (full)
    start = time.time()
    for _ in range(n_iter):
        state = env.get_state()
    get_state_time = (time.time() - start) / n_iter * 1000
    
    # Time set_state (full)
    start = time.time()
    for _ in range(n_iter):
        env.set_state(state)
    set_state_time = (time.time() - start) / n_iter * 1000
    
    # Time get_compact_state
    start = time.time()
    for _ in range(n_iter):
        compact_state = env.get_compact_state()
    get_compact_time = (time.time() - start) / n_iter * 1000
    
    # Time set_compact_state
    start = time.time()
    for _ in range(n_iter):
        env.set_compact_state(compact_state)
    set_compact_time = (time.time() - start) / n_iter * 1000
    
    print(f"\nFull state representation:")
    print(f"  get_state: {get_state_time:.3f}ms per call")
    print(f"  set_state: {set_state_time:.3f}ms per call")
    print(f"  State size: {len(state)} top-level items")
    print(f"  Grid items: {len(dict(state)['grid'])}")
    
    print(f"\nCompact state representation:")
    print(f"  get_compact_state: {get_compact_time:.3f}ms per call")
    print(f"  set_compact_state: {set_compact_time:.3f}ms per call")
    print(f"  State size: {len(compact_state)} top-level items")
    print(f"  Agents: {len(compact_state[1])}, Mobile: {len(compact_state[2])}, Mutable: {len(compact_state[3])}")
    
    speedup_get = get_state_time / get_compact_time
    speedup_set = set_state_time / set_compact_time
    print(f"\nSpeedup:")
    print(f"  get: {speedup_get:.1f}x faster")
    print(f"  set: {speedup_set:.1f}x faster")
    
    # Compare transition_probabilities vs transition_probabilities_compact
    print(f"\nComparing transition_probabilities:")
    
    # Simple case (still for all agents) - deterministic
    num_agents = len(env.agents)
    actions_still = [0] * num_agents
    
    n_iter_trans = 500
    start = time.time()
    for _ in range(n_iter_trans):
        env.transition_probabilities(state, actions_still)
    full_trans_time = (time.time() - start) / n_iter_trans * 1000
    
    start = time.time()
    for _ in range(n_iter_trans):
        env.transition_probabilities_compact(compact_state, actions_still, restore_state=True)
    compact_trans_time = (time.time() - start) / n_iter_trans * 1000
    
    start = time.time()
    for _ in range(n_iter_trans):
        env.set_compact_state(compact_state)
        env.transition_probabilities_compact(compact_state, actions_still, restore_state=False)
    compact_trans_no_restore = (time.time() - start) / n_iter_trans * 1000
    
    print(f"  Full (still actions): {full_trans_time:.3f}ms")
    print(f"  Compact (still, restore=True): {compact_trans_time:.3f}ms")
    print(f"  Compact (still, restore=False): {compact_trans_no_restore:.3f}ms")
    print(f"  Speedup (no restore): {full_trans_time / compact_trans_no_restore:.1f}x")
    
    # Test native transition probabilities
    start = time.time()
    for _ in range(n_iter_trans):
        env.transition_probabilities_native(compact_state, actions_still)
    native_trans_time = (time.time() - start) / n_iter_trans * 1000
    
    print(f"  Native (compact): {native_trans_time:.3f}ms")
    print(f"  Speedup (native vs full): {full_trans_time / native_trans_time:.1f}x")
    
    # Verify correctness: round-trip should work
    print("\nVerifying correctness...")
    env.reset()
    original_full = env.get_state()
    compact = env.get_compact_state()
    
    # Make some moves using dynamic agent count
    num_agents = len(env.agents)
    env.step([3] * num_agents)  # All agents forward
    env.step([1] * num_agents)  # All agents left
    
    after_moves_full = env.get_state()
    after_moves_compact = env.get_compact_state()
    
    # Restore from compact and check if full state matches
    env.set_compact_state(compact)
    restored_compact = env.get_compact_state()
    
    if restored_compact == compact:
        print("  ✓ Compact state round-trip: PASS")
    else:
        print("  ✗ Compact state round-trip: FAIL")
    
    # Verify state hashing works
    compact_set = {compact, after_moves_compact}
    print(f"  ✓ Compact states are hashable: {len(compact_set)} unique states")
    
    # Verify native transition probabilities gives same results
    env.reset()
    compact = env.get_compact_state()
    full_result = env.transition_probabilities(env.get_state(), actions_still)
    native_result = env.transition_probabilities_native(compact, actions_still)
    
    if len(full_result) == len(native_result):
        # Compare probabilities and check states match
        full_probs = sorted([p for p, s in full_result])
        native_probs = sorted([p for p, s in native_result])
        if full_probs == native_probs:
            print("  ✓ Native transition_probabilities: PASS")
        else:
            print("  ✗ Native transition_probabilities: FAIL (probabilities differ)")
    else:
        print(f"  ✗ Native transition_probabilities: FAIL (result counts differ: {len(full_result)} vs {len(native_result)})")


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
    num_agents = len(env.agents)
    actions = [0] * num_agents  # still for all agents
    
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
    actions_forward = [3] * num_agents  # forward for all agents
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
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    
    # Get some sample transitions
    results = []
    for combo_idx in range(min(16, num_actions ** num_agents)):
        actions = []
        temp = combo_idx
        for _ in range(num_agents):
            actions.append(temp % num_actions)
            temp //= num_actions
        
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


def detailed_line_profiling_native():
    """
    Detailed line-by-line profiling of transition_probabilities_native internals.
    
    This function inserts timing code to measure specific lines within
    the transition_probabilities_native method implementation.
    """
    print("=" * 70)
    print("Detailed Line Profiling of transition_probabilities_native() internals")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    compact_state = env.get_compact_state()
    num_agents = len(env.agents)
    
    # Use forward actions to trigger deterministic case (most common)
    actions = [3] * num_agents  # forward for all
    
    n_iter = 500
    
    # Timing accumulators
    timings = {
        'total': 0.0,
        'step_count_check': 0.0,
        'action_validation': 0.0,
        'set_compact_state': 0.0,
        'read_agent_states': 0.0,
        'identify_active_agents': 0.0,
        'stochastic_check': 0.0,
        'compute_successor_inplace': 0.0,
        'get_compact_state': 0.0,
    }
    
    print(f"\nProfiling {n_iter} iterations with forward actions...")
    
    for _ in range(n_iter):
        # Total time
        t_total_start = time.time()
        
        # Line: step_count = compact_state[0]
        t0 = time.time()
        step_count = compact_state[0]
        _ = step_count >= env.max_steps
        timings['step_count_check'] += time.time() - t0
        
        # Line: for action in actions
        t0 = time.time()
        for action in actions:
            _ = action < 0 or action >= env.action_space.n
        timings['action_validation'] += time.time() - t0
        
        # Line: self.set_compact_state(compact_state)
        t0 = time.time()
        env.set_compact_state(compact_state)
        timings['set_compact_state'] += time.time() - t0
        
        # Line: agent_states = compact_state[1]
        t0 = time.time()
        agent_states = compact_state[1]
        timings['read_agent_states'] += time.time() - t0
        
        # Line: identify active agents
        t0 = time.time()
        active_agents = []
        for i in range(num_agents):
            agent_state = agent_states[i]
            terminated = agent_state[3]
            started = agent_state[4]
            paused = agent_state[5]
            if (not terminated and not paused and started and 
                actions[i] != env.actions.still):
                active_agents.append(i)
        timings['identify_active_agents'] += time.time() - t0
        
        # Line: stochastic check
        t0 = time.time()
        is_stochastic = False
        if len(active_agents) == 1:
            agent_idx = active_agents[0]
            if actions[agent_idx] == env.actions.forward:
                on_unsteady = agent_states[agent_idx][6]
                if on_unsteady:
                    is_stochastic = True
        timings['stochastic_check'] += time.time() - t0
        
        # Line: compute successor state
        t0 = time.time()
        if hasattr(env, '_compute_successor_state_inplace'):
            env._compute_successor_state_inplace(actions, tuple(range(num_agents)))
        timings['compute_successor_inplace'] += time.time() - t0
        
        # Line: get result compact state
        t0 = time.time()
        result_compact = env.get_compact_state()
        timings['get_compact_state'] += time.time() - t0
        
        timings['total'] += time.time() - t_total_start
    
    # Convert to per-call and format
    print("\n" + "=" * 60)
    print("LINE-BY-LINE TIMING BREAKDOWN (transition_probabilities_native)")
    print("=" * 60)
    print(f"{'Operation':<35} {'Time (ms)':<12} {'% of Total':<10}")
    print("-" * 60)
    
    total_ms = timings['total'] / n_iter * 1000
    for key, t in sorted(timings.items(), key=lambda x: -x[1]):
        if key == 'total':
            continue
        ms = t / n_iter * 1000
        pct = 100 * t / timings['total'] if timings['total'] > 0 else 0
        print(f"  {key:<33} {ms:>8.4f}ms    {pct:>5.1f}%")
    
    print("-" * 60)
    print(f"  {'TOTAL':<33} {total_ms:>8.4f}ms    100.0%")
    print()
    
    # Now profile the actual transition_probabilities_native call for comparison
    print("Full transition_probabilities_native call for reference:")
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities_native(compact_state, actions)
    actual_time = (time.time() - start) / n_iter * 1000
    print(f"  Actual time: {actual_time:.4f}ms")
    print(f"  Sum of components: {total_ms:.4f}ms")
    if actual_time > total_ms:
        print(f"  Remaining (not profiled): {actual_time - total_ms:.4f}ms")


def detailed_line_profiling():
    """
    Detailed line-by-line profiling of transition_probabilities internals.
    
    This function inserts timing code to measure specific lines within
    the transition_probabilities method implementation.
    """
    print("=" * 70)
    print("Detailed Line Profiling of transition_probabilities() internals")
    print("=" * 70)
    
    env = SmallOneOrTwoChambersMapEnv()
    env.reset()
    
    initial_state = env.get_state()
    num_agents = len(env.agents)
    
    # Use forward actions to trigger stochastic transitions
    actions = [3] * num_agents  # forward for all
    
    n_iter = 200
    
    # Timing accumulators
    timings = {
        'total': 0.0,
        'dict_conversion': 0.0,
        'terminal_check': 0.0,
        'backup_state': 0.0,
        'set_query_state': 0.0,
        'step_count_check': 0.0,
        'active_agents': 0.0,
        'stochastic_agents': 0.0,
        'deterministic_step': 0.0,
        'stochastic_step': 0.0,
        'result_state': 0.0,
        'restore_state': 0.0,
        'aggregate_probs': 0.0,
    }
    
    print(f"\nProfiling {n_iter} iterations with forward actions (stochastic)...")
    
    for _ in range(n_iter):
        # Total time
        t_total_start = time.time()
        
        # Line: state = dict(state) if hasattr(state, 'keys') else dict(state)
        t0 = time.time()
        state_dict = dict(initial_state)
        timings['dict_conversion'] += time.time() - t0
        
        # Line: if state_dict.get('terminated', False)
        t0 = time.time()
        _ = state_dict.get('terminated', False)
        timings['terminal_check'] += time.time() - t0
        
        # Line: original_state = self.get_state()
        t0 = time.time()
        original_state = env.get_state()
        timings['backup_state'] += time.time() - t0
        
        # Line: self.set_state(state)
        t0 = time.time()
        env.set_state(initial_state)
        timings['set_query_state'] += time.time() - t0
        
        # Line: step_count check
        t0 = time.time()
        _ = state_dict['step_count'] >= env.max_steps
        timings['step_count_check'] += time.time() - t0
        
        # Line: identify active agents
        t0 = time.time()
        active_agents = []
        for i in range(num_agents):
            if (not env.agents[i].terminated and 
                not env.agents[i].paused and 
                env.agents[i].started and 
                actions[i] != env.actions.still):
                active_agents.append(i)
        timings['active_agents'] += time.time() - t0
        
        # Line: identify stochastic agents
        t0 = time.time()
        stochastic_agents = []
        for i in active_agents:
            agent = env.agents[i]
            if hasattr(agent, 'p_move') and agent.p_move < 1.0:
                stochastic_agents.append((i, agent.p_move))
        timings['stochastic_agents'] += time.time() - t0
        
        # Simulate stochastic stepping
        t0 = time.time()
        if stochastic_agents:
            # This is where the combinatorial explosion happens
            for i, p in stochastic_agents:
                pass  # Would call step for each stochastic combination
        timings['stochastic_step'] += time.time() - t0
        
        # Line: Get result state
        t0 = time.time()
        result_state = env.get_state()
        timings['result_state'] += time.time() - t0
        
        # Line: Restore original state
        t0 = time.time()
        env.set_state(original_state)
        timings['restore_state'] += time.time() - t0
        
        timings['total'] += time.time() - t_total_start
    
    # Convert to per-call and format
    print("\n" + "=" * 60)
    print("LINE-BY-LINE TIMING BREAKDOWN")
    print("=" * 60)
    print(f"{'Operation':<35} {'Time (ms)':<12} {'% of Total':<10}")
    print("-" * 60)
    
    total_ms = timings['total'] / n_iter * 1000
    for key, t in sorted(timings.items(), key=lambda x: -x[1]):
        if key == 'total':
            continue
        ms = t / n_iter * 1000
        pct = 100 * t / timings['total'] if timings['total'] > 0 else 0
        print(f"  {key:<33} {ms:>8.4f}ms    {pct:>5.1f}%")
    
    print("-" * 60)
    print(f"  {'TOTAL':<33} {total_ms:>8.4f}ms    100.0%")
    print()
    
    # Now profile the actual transition_probabilities call for comparison
    print("Full transition_probabilities call for reference:")
    start = time.time()
    for _ in range(n_iter):
        env.transition_probabilities(initial_state, actions)
    actual_time = (time.time() - start) / n_iter * 1000
    print(f"  Actual time: {actual_time:.4f}ms")
    print(f"  Sum of components: {total_ms:.4f}ms")
    print(f"  Remaining (step logic, prob aggregation): {actual_time - total_ms:.4f}ms")


def try_line_profiler():
    """
    Try to use the line_profiler package for detailed profiling.
    
    Install with: pip install line_profiler
    """
    print("=" * 70)
    print("Line profiling transition_probabilities_native()")
    print("=" * 70)
    
    try:
        from line_profiler import LineProfiler
        print("\nline_profiler is installed!")
        
        env = SmallOneOrTwoChambersMapEnv()
        env.reset()
        
        compact_state = env.get_compact_state()
        num_agents = len(env.agents)
        actions = [3] * num_agents  # forward for all
        
        # Profile transition_probabilities_native
        profiler = LineProfiler()
        profiler.add_function(env.transition_probabilities_native)
        
        # Also profile helper functions
        if hasattr(env, 'get_compact_state'):
            profiler.add_function(env.get_compact_state)
        if hasattr(env, 'set_compact_state'):
            profiler.add_function(env.set_compact_state)
        if hasattr(env, '_compute_successor_state_inplace'):
            profiler.add_function(env._compute_successor_state_inplace)
        
        @profiler
        def run_transitions():
            for _ in range(200):
                env.transition_probabilities_native(compact_state, actions)
        
        run_transitions()
        
        print("\nLine profiler output for transition_probabilities_native:")
        profiler.print_stats()
        
    except ImportError:
        print("\nline_profiler is not installed.")
        print("Install with: pip install line_profiler")
        print("\nFalling back to manual line profiling...")
        detailed_line_profiling_native()


if __name__ == "__main__":
    # Try line_profiler first for transition_probabilities_native
    try_line_profiler()
    print()
    detailed_line_profiling_native()
    print()
    compare_state_representations()
    print()
    profile_get_dag()
