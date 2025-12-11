"""
Benchmark parallel vs sequential DAG computation.

Compares performance of get_dag() vs get_dag_parallel() on environments
of varying complexity to measure parallelization speedup.
"""

import sys
import os
import time

from gym_multigrid.multigrid import MultiGridEnv, World


def benchmark_environment(env_name, env, num_workers_list=[1, 2, 4], num_runs=3):
    """Benchmark an environment with different worker counts."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {env_name}")
    print(f"{'='*70}")
    
    # Get DAG size and environment statistics
    print("Getting environment statistics...")
    states, _, _ = env.get_dag(return_probabilities=False)
    num_states = len(states)
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    total_action_combinations = num_actions ** num_agents
    
    print(f"\nEnvironment statistics:")
    print(f"  States: {num_states}")
    print(f"  Agents: {num_agents}")
    print(f"  Actions per agent: {num_actions}")
    print(f"  Action combinations per state: {total_action_combinations}")
    print(f"  Max steps: {env.max_steps}")
    
    # Benchmark sequential
    print(f"\nSequential (get_dag) - {num_runs} runs:")
    sequential_times = []
    for run in range(num_runs):
        start = time.perf_counter()
        states_seq, _, _ = env.get_dag(return_probabilities=False)
        elapsed = time.perf_counter() - start
        sequential_times.append(elapsed)
        print(f"  Run {run+1}: {elapsed*1000:.1f}ms ({len(states_seq)} states)")
    
    avg_sequential = sum(sequential_times) / len(sequential_times)
    min_sequential = min(sequential_times)
    print(f"  Average: {avg_sequential*1000:.1f}ms, Min: {min_sequential*1000:.1f}ms")
    
    # Benchmark parallel with different worker counts
    results = {'sequential': avg_sequential}
    
    for num_workers in num_workers_list:
        print(f"\nParallel with {num_workers} workers - {num_runs} runs:")
        parallel_times = []
        for run in range(num_runs):
            start = time.perf_counter()
            states_par, _, _ = env.get_dag_parallel(return_probabilities=False, num_workers=num_workers)
            elapsed = time.perf_counter() - start
            parallel_times.append(elapsed)
            print(f"  Run {run+1}: {elapsed*1000:.1f}ms ({len(states_par)} states)")
        
        avg_parallel = sum(parallel_times) / len(parallel_times)
        min_parallel = min(parallel_times)
        speedup = avg_sequential / avg_parallel
        print(f"  Average: {avg_parallel*1000:.1f}ms, Min: {min_parallel*1000:.1f}ms")
        print(f"  Speedup vs sequential: {speedup:.2f}x")
        results[f'{num_workers}_workers'] = {'time': avg_parallel, 'speedup': speedup}
    
    return results


def run_benchmark(quick_mode=False):
    """Run benchmarks, optionally in quick mode."""
    import multiprocessing as mp
    max_workers = mp.cpu_count()
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print(f"Parallel DAG Computation Benchmark")
    print(f"  [{mode_str}]")
    print(f"System has {max_workers} CPU cores")
    print("="*70)
    
    # Test 1: Medium complexity - Unsteady ground environment
    env1 = MultiGridEnv(
        map='''
We We We We We We We
We .. .. .. Un Un We
We .. .. .. Un Un We
We .. .. Ay Un Un We
We .. .. .. Un Un We
We .. .. .. Un Un We
We We We We We We We
''',
        objects_set=World,
        orientations=['n'],
        max_steps=4 if not quick_mode else 2,
        partial_obs=False
    )
    num_runs = 3 if not quick_mode else 1
    benchmark_environment("Unsteady Ground (7x7)", env1,
                         num_workers_list=[1, 2] if quick_mode else [1, 2, 4, max_workers],
                         num_runs=num_runs)
    
    if not quick_mode:
        # Test 2: Higher complexity - Larger grid with more steps (skip in quick mode)
        env2 = MultiGridEnv(
            map='''
We We We We We We We We We
We .. .. .. Un Un .. .. We
We .. .. .. Un Un .. .. We
We .. Ar .. Un Un .. .. We
We .. .. .. Un Un .. .. We
We .. .. .. .. .. .. .. We
We .. .. .. .. .. .. .. We
We We We We We We We We We
''',
            objects_set=World,
            orientations=['e'],
            max_steps=6,
            partial_obs=False
        )
        benchmark_environment("Larger Grid (9x8, max_steps=6)", env2,
                             num_workers_list=[1, max_workers])
        
        # Test 3: Multiple agents (exponentially more action combinations)
        env3 = MultiGridEnv(
            map='''
We We We We We We We
We .. .. .. Un Un We
We Ar .. .. Un Un We
We .. .. Ay Un Un We
We .. .. .. Un Un We
We We We We We We We
''',
            objects_set=World,
            orientations=['e', 'n'],
            max_steps=7,
            partial_obs=False
        )
        benchmark_environment("Multi-Agent (2 agents, max_steps=5)", env3,
                             num_workers_list=[1, max_workers])
    
    print(f"\n{'='*70}")
    print("Benchmark complete!")
    print(f"{'='*70}")
    print("\nAnalysis Notes:")
    print("- Parallelization overhead includes process spawning and environment setup")
    print("- Speedup depends on: state count, action combinations, and worker count")
    print("- Single-agent environments may not benefit due to limited parallelism")
    print("- Multi-agent environments have more action combinations to parallelize")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark Parallel DAG Computation')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with reduced test cases')
    args = parser.parse_args()
    run_benchmark(quick_mode=args.quick)
