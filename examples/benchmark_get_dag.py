#!/usr/bin/env python3
"""
Performance benchmarking script for get_dag function.

This script:
1. Generates progressively larger random DAG environments  
2. Measures runtime of get_dag on each
3. Stops when runtime exceeds 60 seconds
4. Creates a log-log plot of runtime vs transitions with linear regression
5. Tests if slope ≈ 1 (verifying O(|T|) complexity)
"""

import sys
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from empo.env_utils import get_dag


class RandomDAGEnv:
    """
    A random DAG environment generator with efficient construction.
    """
    
    def __init__(self, num_states, avg_out_degree=2.0, seed=None):
        """
        Initialize random DAG environment.
        
        Args:
            num_states: Number of states in the DAG
            avg_out_degree: Average number of outgoing edges per non-terminal state
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.num_states = num_states
        self.current_state = 0
        self.agents = [None]  # Single agent
        self.action_space = MockActionSpace(n=max(2, int(avg_out_degree * 1.5)))
        
        # Generate random DAG structure efficiently
        self._generate_dag_efficient(avg_out_degree)
    
    def _generate_dag_efficient(self, avg_out_degree):
        """Generate random DAG with controlled density."""
        # transitions[state][action] = next_state
        self.transitions = {}
        self.num_transitions = 0
        
        # For efficiency, only connect to a limited set of future states
        for state in range(self.num_states - 1):
            # Decide number of outgoing edges
            # Use Poisson-like distribution but ensure connectivity
            num_edges = max(1, int(np.random.exponential(avg_out_degree)))
            num_edges = min(num_edges, self.action_space.n, self.num_states - state - 1)
            
            # Sample successor states (must be > current state for DAG property)
            # Connect to nearby states for realistic graphs
            max_jump = min(self.num_states - state - 1, max(50, int(np.sqrt(self.num_states))))
            possible_successors = list(range(state + 1, min(state + 1 + max_jump, self.num_states)))
            
            if not possible_successors:
                continue
            
            num_edges = min(num_edges, len(possible_successors))
            successors = random.sample(possible_successors, num_edges)
            
            # Assign to actions
            state_transitions = {}
            for i, successor in enumerate(successors):
                action = i % self.action_space.n
                state_transitions[action] = successor
                self.num_transitions += 1
            
            self.transitions[state] = state_transitions
    
    def reset(self):
        """Reset to initial state."""
        self.current_state = 0
        return self.current_state
    
    def get_state(self):
        """Get current state."""
        return self.current_state
    
    def set_state(self, state):
        """Set current state."""
        self.current_state = state
    
    def transition_probabilities(self, state, actions):
        """Return possible transitions from the given state."""
        if state not in self.transitions:
            return None  # Terminal state
        
        action = actions[0]
        if action in self.transitions[state]:
            next_state = self.transitions[state][action]
            return [(1.0, next_state)]
        else:
            return None  # No transition for this action


class MockActionSpace:
    """Mock action space."""
    def __init__(self, n=4):
        self.n = n


def benchmark_get_dag():
    """Run performance benchmarks on get_dag with increasing DAG sizes."""
    print("=" * 70)
    print("Performance Benchmarking for get_dag")
    print("=" * 70)
    print()
    print("Generating random DAG environments of increasing size...")
    print("Stopping when runtime exceeds 60 seconds or after 50 iterations.")
    print()
    
    results = []
    num_states = 10
    max_runtime = 60.0
    max_iterations = 50
    max_states = 500_000  # Cap to prevent excessive memory/time in env creation
    iteration = 0
    runtime_estimate = 0.001  # Initial estimate
    
    while iteration < max_iterations and num_states <= max_states:
        iteration += 1
        
        # Create random environment with controlled growth
        print(f"[{iteration}/{max_iterations}] Creating env with {num_states:,} states...", end=" ", flush=True)
        try:
            env = RandomDAGEnv(num_states, avg_out_degree=2.0, seed=42 + iteration)
            num_transitions = env.num_transitions
            print(f"{num_transitions:,} transitions...", end=" ", flush=True)
        except Exception as e:
            print(f"✗ Failed to create environment: {e}")
            break
        
        # Measure runtime (multiple runs for better precision)
        num_runs = max(1, min(10, int(1.0 / max(runtime_estimate, 0.001))))  # More runs for fast cases
        runtimes = []
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                states, state_to_idx, successors = get_dag(env)
                runtime = time.time() - start_time
                runtimes.append(runtime)
            except Exception as e:
                print(f"✗ Error running get_dag: {e}")
                return results
        
        # Use median runtime to avoid outliers
        runtime = np.median(runtimes)
        runtime_estimate = runtime  # Update for next iteration
        
        if num_runs > 1:
            print(f"✓ {runtime:.6f}s (median of {num_runs} runs)")
        else:
            print(f"✓ {runtime:.6f}s")
        
        results.append((num_transitions, runtime))
            
        if runtime > max_runtime:
            print()
            print(f"Runtime exceeded {max_runtime}s. Stopping.")
            break
        
        # Adaptive growth based on runtime
        if runtime < 0.001:
            multiplier = 10
        elif runtime < 0.01:
            multiplier = 5
        elif runtime < 0.1:
            multiplier = 3
        elif runtime < 1.0:
            multiplier = 2
        elif runtime < 10.0:
            multiplier = 1.5
        else:
            multiplier = 1.2
        
        num_states = int(num_states * multiplier)
    
    if results:
        print()
        print(f"Collected {len(results)} data points.")
        if results[-1][1] <= max_runtime:
            print(f"Completed {len(results)} iterations without exceeding {max_runtime}s threshold.")
            print(f"Maximum runtime: {results[-1][1]:.4f}s")
            print(f"Largest DAG: {results[-1][0]:,} transitions")
    
    return results


def analyze_and_plot(results):
    """
    Analyze results and create log-log plot with linear regression.
    
    Args:
        results: List of (num_transitions, runtime) tuples
    """
    if len(results) < 3:
        print("Not enough data points for analysis.")
        return
    
    # Extract data
    transitions = np.array([r[0] for r in results])
    runtimes = np.array([r[1] for r in results])
    
    # Log transform
    log_transitions = np.log10(transitions)
    log_runtimes = np.log10(runtimes)
    
    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_transitions, log_runtimes)
    
    print()
    print("=" * 70)
    print("Statistical Analysis")
    print("=" * 70)
    print(f"Number of data points: {len(results)}")
    print(f"Transitions range: {transitions.min()} to {transitions.max()}")
    print(f"Runtime range: {runtimes.min():.4f}s to {runtimes.max():.4f}s")
    print()
    print("Linear Regression in Log-Log Space:")
    print(f"  Slope: {slope:.4f} (expected ≈ 1.0 for O(|T|) complexity)")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Std error: {std_err:.4f}")
    print()
    
    # Test if slope ≈ 1
    # Using 95% confidence interval
    confidence = 0.95
    dof = len(results) - 2  # degrees of freedom
    t_val = stats.t.ppf((1 + confidence) / 2, dof)
    margin_error = t_val * std_err
    ci_lower = slope - margin_error
    ci_upper = slope + margin_error
    
    print(f"95% Confidence Interval for Slope: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print()
    
    # Test if slope ≈ 1 (allowing for small deviations)
    # Check if confidence interval is close to 1.0 (within reasonable bounds)
    if ci_lower <= 1.0 <= ci_upper:
        print("✓ PASS: Slope is statistically consistent with 1.0")
        print("  → Confirms O(|T|) linear time complexity")
    elif 0.9 <= slope <= 1.1 and r_value**2 > 0.95:
        print(f"✓ PASS: Slope is {slope:.3f}, very close to 1.0 with high R²={r_value**2:.4f}")
        print("  → Confirms approximately O(|T|) linear time complexity")
    else:
        print(f"✗ FAIL: Slope={slope:.3f} differs significantly from 1.0")
        print(f"  → Actual complexity appears to be O(|T|^{slope:.2f})")
    
    # Create plot
    print()
    print("Generating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    ax.scatter(transitions, runtimes, alpha=0.6, s=50, label='Measured data')
    
    # Regression line
    log_transitions_fit = np.linspace(log_transitions.min(), log_transitions.max(), 100)
    log_runtimes_fit = slope * log_transitions_fit + intercept
    transitions_fit = 10 ** log_transitions_fit
    runtimes_fit = 10 ** log_runtimes_fit
    ax.plot(transitions_fit, runtimes_fit, 'r--', linewidth=2, 
            label=f'Linear regression (slope={slope:.3f})')
    
    # Reference line with slope=1
    if len(results) > 1:
        ref_intercept = log_runtimes[len(results)//2] - log_transitions[len(results)//2]
        log_runtimes_ref = log_transitions_fit + ref_intercept
        runtimes_ref = 10 ** log_runtimes_ref
        ax.plot(transitions_fit, runtimes_ref, 'g:', linewidth=2, alpha=0.7,
                label='Reference O(|T|) (slope=1.0)')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Transitions (|T|)', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('get_dag Performance: Runtime vs Transitions (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    # Add text box with statistics
    textstr = f'Slope: {slope:.3f}\nR²: {r_value**2:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    output_file = outputs_dir / "get_dag_performance.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # Also save data
    data_file = outputs_dir / "get_dag_performance_data.txt"
    with open(data_file, 'w') as f:
        f.write("# Performance benchmark data for get_dag\n")
        f.write("# Columns: num_transitions runtime_seconds\n")
        for trans, rt in results:
            f.write(f"{trans}\t{rt:.6f}\n")
    print(f"✓ Data saved to: {data_file}")
    
    print()
    print("=" * 70)
    print("Benchmark completed successfully!")
    print("=" * 70)


def main():
    """Main benchmark function."""
    # Run benchmark
    results = benchmark_get_dag()
    
    if not results:
        print("No results collected. Exiting.")
        return
    
    # Analyze and plot
    analyze_and_plot(results)


if __name__ == "__main__":
    main()
