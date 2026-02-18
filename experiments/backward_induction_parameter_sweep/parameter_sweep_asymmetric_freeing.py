#!/usr/bin/env python3
"""
Parameter Sweep for Asymmetric Freeing Simple

This script performs Monte Carlo simulations to study the influence of EMPO parameters
on the probability of freeing the left human first vs the right human in the
asymmetric_freeing_simple.yaml environment.

Parameters varied:
- max_steps (8...14)
- beta_h (5...100)
- gamma_r (0.8...1.0)
- gamma_h (0.8...1.0)
- zeta (1...3)
- eta (1...2)
- xi (1...2)

For each parameter combination, the script:
1. Loads the asymmetric_freeing_simple.yaml world
2. Computes human policy prior (Phase 1)
3. Computes robot policy (Phase 2)
4. Reads P(left) directly from the initial state's robot policy as P(turn right),
   since the south-facing robot turns right to head towards the left (west) human first

The results are saved to a CSV file for subsequent regression analysis.

Usage:
    # Small test run (10 samples)
    python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py --n_samples 10 --output outputs/parameter_sweep/results_test.csv
    
    # Full run (100 samples)
    python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py --n_samples 100 --output outputs/parameter_sweep/results_full.csv
    
    # Parallel with 4 workers
    python experiments/backward_induction_parameter_sweep/parameter_sweep_asymmetric_freeing.py --n_samples 100 --parallel --num_workers 4
"""

import argparse
import csv
import gc
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import fcntl
except ImportError as e:
    raise RuntimeError(
        "This experiment script requires a Unix-like operating system because it uses "
        "the 'fcntl' module for file locking. Please run it on Linux/Unix (or WSL) "
        "instead of native Windows."
    ) from e

# Setup paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')
sys.path.insert(0, 'vendor/ai_transport')

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.backward_induction import compute_human_policy_prior, compute_robot_policy
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy


@dataclass
class PriorBounds:
    """Bounds for prior distributions of parameters."""
    max_steps_min: int = 8
    max_steps_max: int = 14
    beta_h_min: float = 5.0
    beta_h_max: float = 100.0
    gamma_h_min: float = 0.8
    gamma_h_max: float = 1.0
    gamma_r_min: float = 0.8
    gamma_r_max: float = 1.0
    zeta_min: float = 1.0
    zeta_max: float = 3.0
    eta_min: float = 1.0
    eta_max: float = 2.0
    xi_min: float = 1.0
    xi_max: float = 2.0
    beta_r: float = 50.0  # Fixed value


@dataclass
class ParameterSet:
    """A single set of parameters for one Monte Carlo run."""
    max_steps: int
    beta_h: float
    beta_r: float  # Keep fixed at a reasonable value
    gamma_h: float
    gamma_r: float
    zeta: float
    eta: float
    xi: float
    seed: int  # Random seed used for sampling this parameter set
    
    # Results (filled after simulation)
    p_left: Optional[float] = None          # P(robot turns right first) = P(freeing left human first)
    n_states: Optional[int] = None          # Number of states in DAG
    computation_time: Optional[float] = None  # Time for backward induction (seconds)
    
    # Policy metrics at initial state
    pi_r_turn_left: Optional[float] = None   # pi_r(turn left) at initial state
    pi_r_turn_right: Optional[float] = None  # pi_r(turn right) at initial state  
    Q_r_turn_left: Optional[float] = None    # Q_r(turn left) at initial state
    Q_r_turn_right: Optional[float] = None   # Q_r(turn right) at initial state
    V_r: Optional[float] = None              # V_r at initial state
    X_h1: Optional[float] = None             # X_h for human 1 (first in human_agent_indices)
    X_h2: Optional[float] = None             # X_h for human 2 (second in human_agent_indices)
    h1_pos: Optional[str] = None             # Position of human 1 (for reference)
    h2_pos: Optional[str] = None             # Position of human 2 (for reference)


def sample_parameters(seed: Optional[int] = None, bounds: Optional[PriorBounds] = None) -> ParameterSet:
    """
    Sample a random parameter set from suitable priors.
    
    Priors:
    - max_steps: uniform discrete from max_steps_min to max_steps_max
    - beta_h: log-uniform from beta_h_min to beta_h_max
    - gamma_h: uniform from gamma_h_min to gamma_h_max
    - gamma_r: uniform from gamma_r_min to gamma_r_max
    - zeta: uniform from zeta_min to zeta_max
    - eta: uniform from eta_min to eta_max
    - xi: uniform from xi_min to xi_max
    - beta_r: fixed value
    
    Args:
        seed: Random seed for reproducibility
        bounds: Prior bounds for parameters (uses defaults if None)
        
    Returns:
        ParameterSet with sampled values
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if bounds is None:
        bounds = PriorBounds()
    
    # Sample parameters
    max_steps = np.random.randint(bounds.max_steps_min, bounds.max_steps_max + 1)  # inclusive
    beta_h = np.exp(np.random.uniform(np.log(bounds.beta_h_min), np.log(bounds.beta_h_max)))  # log-uniform
    gamma_h = np.random.uniform(bounds.gamma_h_min, bounds.gamma_h_max)
    gamma_r = np.random.uniform(bounds.gamma_r_min, bounds.gamma_r_max)
    zeta = np.random.uniform(bounds.zeta_min, bounds.zeta_max)
    eta = np.random.uniform(bounds.eta_min, bounds.eta_max)
    xi = np.random.uniform(bounds.xi_min, bounds.xi_max)
    beta_r = bounds.beta_r  # Fixed
    
    return ParameterSet(
        max_steps=max_steps,
        beta_h=beta_h,
        beta_r=beta_r,
        gamma_h=gamma_h,
        gamma_r=gamma_r,
        zeta=zeta,
        eta=eta,
        xi=xi,
        seed=seed
    )


def compute_p_left_from_policy(env: MultiGridEnv, robot_policy: RobotPolicy) -> float:
    """
    Compute P(left) = probability that robot turns right first from initial state.
    
    In asymmetric_freeing_simple, the robot starts facing south. Turning right (action=2)
    means it will face west and move towards the left human first.
    
    Args:
        env: Environment instance (reset to initial state)
        robot_policy: Computed robot policy
        
    Returns:
        Probability that robot's first action is "turn right" (going left/west)
    """
    # Get initial state
    env.reset()
    initial_state = env.get_state()
    
    # Get robot policy distribution for initial state
    # robot_policy(state) returns Dict[RobotActionProfile, float]
    # where RobotActionProfile is a tuple of actions (one per robot)
    policy_dist = robot_policy(initial_state)
    
    if not policy_dist:
        # No policy for initial state - this shouldn't happen
        return 0.5  # Return 0.5 as uninformative prior
    
    # Sum probabilities of all action profiles where robot turns right (action=2)
    # SmallActions: still=0, left=1, right=2, forward=3
    TURN_RIGHT = 2
    
    p_left = 0.0
    for action_profile, prob in policy_dist.items():
        # action_profile is a tuple, e.g., (2,) for single robot
        if action_profile[0] == TURN_RIGHT:
            p_left += prob
    
    return p_left


def compute_initial_state_metrics(
    env: MultiGridEnv,
    robot_policy: RobotPolicy,
    human_policy_prior: TabularHumanPolicyPrior,
    Vr_dict: Dict[Any, float],
    Vh_dict: Dict[Any, Dict[int, Dict[Any, float]]],
    params: ParameterSet
) -> Dict[str, Any]:
    """
    Compute Q_r, V_r, and X_h values for the initial state only.
    
    This extracts policy metrics without storing them for all states.
    
    Args:
        env: Environment instance
        robot_policy: Computed robot policy
        human_policy_prior: Human policy prior
        Vr_dict: V_r values for all states
        Vh_dict: V_h^e values for all states (state -> agent_idx -> goal -> float)
        params: Parameters (for zeta, gamma_r, beta_r, xi)
        
    Returns:
        Dict with keys: pi_r_turn_left, pi_r_turn_right, Q_r_turn_left, Q_r_turn_right,
                       V_r, X_h1, X_h2, h1_pos, h2_pos
    """
    # Get initial state
    env.reset()
    initial_state = env.get_state()
    
    # Get policy distribution and extract pi_r values
    policy_dist = robot_policy(initial_state)
    TURN_LEFT = 1
    TURN_RIGHT = 2
    
    pi_r_turn_left = sum(prob for ap, prob in policy_dist.items() if ap[0] == TURN_LEFT)
    pi_r_turn_right = sum(prob for ap, prob in policy_dist.items() if ap[0] == TURN_RIGHT)
    
    # Get V_r at initial state
    V_r = Vr_dict.get(initial_state, None)
    
    # Compute Q_r values for turn_left and turn_right at initial state
    # Q_r(s, a_r) = gamma_r * E_{a_h ~ human_policy, s' ~ T}[V_r(s')]
    human_agent_indices = env.human_agent_indices
    robot_agent_indices = env.robot_agent_indices
    num_agents = len(env.agents)
    num_actions = env.action_space.n
    action_powers = num_actions ** np.arange(num_agents)
    
    # Get transitions for initial state
    states, state_to_idx, successors, transitions = env.get_dag(return_probabilities=True, quiet=True)
    initial_state_idx = state_to_idx[initial_state]
    state_transitions = transitions[initial_state_idx]
    
    action_profile = np.zeros(num_agents, dtype=np.int64)
    
    Q_r_values = {}
    for robot_action in [TURN_LEFT, TURN_RIGHT]:
        action_profile[robot_agent_indices] = [robot_action]  # Single robot
        v = 0.0
        for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution(initial_state):
            action_profile[human_agent_indices] = human_action_profile
            action_profile_index = int((action_profile @ action_powers).item())
            _, next_state_probabilities, next_state_indices = state_transitions[action_profile_index]
            # Look up V_r values for successor states
            vr_successors = np.array([Vr_dict.get(states[i], 0) for i in next_state_indices])
            v += human_action_profile_prob * np.dot(next_state_probabilities, vr_successors)
        Q_r_values[robot_action] = params.gamma_r * v
    
    Q_r_turn_left = Q_r_values[TURN_LEFT]
    Q_r_turn_right = Q_r_values[TURN_RIGHT]
    
    # Compute X_h values from Vh_dict
    # X_h = sum over goals of (weight * V_h^e^zeta)
    # Note: possible_goal_generator provides weights
    X_h_values = {}
    h_positions = {}
    
    Vh_initial = Vh_dict.get(initial_state, {})
    
    for agent_idx in human_agent_indices:
        # Get human position from initial state for reference
        agent = env.agents[agent_idx]
        h_positions[agent_idx] = f"({agent.pos[0]},{agent.pos[1]})"
        
        # Compute X_h = sum over goals of (weight * V_h^e^zeta)
        Vh_agent = Vh_initial.get(agent_idx, {})
        xh = 0.0
        for possible_goal, weight in env.possible_goal_generator.generate(initial_state, agent_idx):
            vh = Vh_agent.get(possible_goal, 0.0)
            xh += weight * (vh ** params.zeta)
        X_h_values[agent_idx] = xh
    
    # Map to h1, h2 (first and second in human_agent_indices)
    h1_idx = human_agent_indices[0]
    h2_idx = human_agent_indices[1] if len(human_agent_indices) > 1 else None
    
    return {
        'pi_r_turn_left': pi_r_turn_left,
        'pi_r_turn_right': pi_r_turn_right,
        'Q_r_turn_left': Q_r_turn_left,
        'Q_r_turn_right': Q_r_turn_right,
        'V_r': V_r,
        'X_h1': X_h_values.get(h1_idx),
        'X_h2': X_h_values.get(h2_idx) if h2_idx is not None else None,
        'h1_pos': h_positions.get(h1_idx),
        'h2_pos': h_positions.get(h2_idx) if h2_idx is not None else None,
    }


def run_single_simulation(params: ParameterSet, 
                         env: MultiGridEnv,
                         parallel: bool = False,
                         quiet: bool = False,
                         use_attainment_cache: bool = True) -> ParameterSet:
    """
    Run a single simulation with the given parameters.
    
    Computes human policy prior (Phase 1) and robot policy (Phase 2),
    then reads P(left) directly from the initial state's robot policy
    as P(turn right) - since the south-facing robot turns right to go west/left.
    
    Args:
        params: Parameter set to use
        env: Pre-initialized environment (DAG will be cached/reused)
        parallel: Whether to use parallel computation for backward induction
        quiet: Suppress output
        
    Returns:
        Updated ParameterSet with results filled in
    """
    start_time = time.time()
    
    try:
        # Reset environment (DAG is already cached from previous runs with same max_steps)
        env.reset()
        
        # Get agent indices
        human_agent_indices = env.human_agent_indices
        robot_agent_indices = env.robot_agent_indices
        
        if not quiet:
            print(f"\n{'='*60}")
            print(f"Running simulation with parameters:")
            print(f"  max_steps={params.max_steps}, beta_h={params.beta_h:.2f}, beta_r={params.beta_r:.2f}")
            print(f"  gamma_h={params.gamma_h:.3f}, gamma_r={params.gamma_r:.3f}")
            print(f"  zeta={params.zeta:.2f}, eta={params.eta:.2f}, xi={params.xi:.2f}")
        
        # Phase 1: Compute human policy prior
        if not quiet:
            print("Phase 1: Computing human policy prior...")
        human_policy_prior = compute_human_policy_prior(
            world_model=env,
            human_agent_indices=human_agent_indices,
            possible_goal_generator=env.possible_goal_generator,
            beta_h=params.beta_h,
            gamma_h=params.gamma_h,
            parallel=parallel,
            quiet=quiet,
            use_attainment_cache=use_attainment_cache
        )
        
        # Phase 2: Compute robot policy
        if not quiet:
            print("Phase 2: Computing robot policy...")
        robot_policy, Vr_dict, Vh_dict = compute_robot_policy(
            world_model=env,
            human_agent_indices=human_agent_indices,
            robot_agent_indices=robot_agent_indices,
            possible_goal_generator=env.possible_goal_generator,
            human_policy_prior=human_policy_prior,
            beta_r=params.beta_r,
            gamma_h=params.gamma_h,
            gamma_r=params.gamma_r,
            zeta=params.zeta,
            xi=params.xi,
            eta=params.eta,
            parallel=parallel,
            quiet=quiet,
            return_values=True
        )
        
        # Get number of states
        params.n_states = len(robot_policy.values)
        
        # Read P(left) directly from initial state's robot policy
        # P(left) = P(turn right) since south-facing robot turns right to go west
        params.p_left = compute_p_left_from_policy(env, robot_policy)
        
        # Compute detailed metrics for initial state only
        metrics = compute_initial_state_metrics(
            env, robot_policy, human_policy_prior, Vr_dict, Vh_dict, params
        )
        params.pi_r_turn_left = metrics['pi_r_turn_left']
        params.pi_r_turn_right = metrics['pi_r_turn_right']
        params.Q_r_turn_left = metrics['Q_r_turn_left']
        params.Q_r_turn_right = metrics['Q_r_turn_right']
        params.V_r = metrics['V_r']
        params.X_h1 = metrics['X_h1']
        params.X_h2 = metrics['X_h2']
        params.h1_pos = metrics['h1_pos']
        params.h2_pos = metrics['h2_pos']
        
        # Free the large dicts to save memory (only needed initial state metrics)
        del Vr_dict, Vh_dict
        
        params.computation_time = time.time() - start_time
        
        if not quiet:
            print(f"Computed policies for {params.n_states} states")
            print(f"Results: P(left) = {params.p_left:.4f}")
            print(f"  pi_r(turn_left)={params.pi_r_turn_left:.4f}, pi_r(turn_right)={params.pi_r_turn_right:.4f}")
            print(f"  Q_r(turn_left)={params.Q_r_turn_left:.6f}, Q_r(turn_right)={params.Q_r_turn_right:.6f}")
            print(f"  V_r={params.V_r:.6f}")
            print(f"  X_h1={params.X_h1:.4f} at {params.h1_pos}, X_h2={params.X_h2:.4f} at {params.h2_pos}")
            print(f"Computation time: {params.computation_time:.1f}s")
        
        return params
        
    except Exception as e:
        if not quiet:
            print(f"ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        params.p_left = None
        params.computation_time = time.time() - start_time
        return params


def save_results_to_csv(results: List[ParameterSet], output_file: str):
    """
    Save simulation results to a CSV file.
    
    Args:
        results: List of ParameterSet results
        output_file: Path to output CSV file
    """
    if not results:
        print("No results to save")
        return
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total samples: {len(results)}")
    valid_results = [r for r in results if r.p_left is not None]
    print(f"Valid samples: {len(valid_results)}")
    if valid_results:
        mean_p_left = np.mean([r.p_left for r in valid_results])
        print(f"Mean P(left): {mean_p_left:.4f}")


def load_existing_results(output_file: str) -> Dict[int, ParameterSet]:
    """
    Load existing results from CSV file for resume capability.
    
    Args:
        output_file: Path to output CSV file
        
    Returns:
        Dictionary mapping sample index to ParameterSet
    """
    output_path = Path(output_file)
    if not output_path.exists():
        return {}
    
    def parse_optional_float(val):
        if val is None or val == '' or val == 'None':
            return None
        return float(val)
    
    def parse_optional_int(val):
        if val is None or val == '' or val == 'None':
            return None
        return int(val)
    
    def parse_optional_str(val):
        if val is None or val == '' or val == 'None':
            return None
        return str(val)
    
    existing = {}
    try:
        with open(output_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Convert row to ParameterSet
                params = ParameterSet(
                    max_steps=int(row['max_steps']),
                    beta_h=float(row['beta_h']),
                    beta_r=float(row['beta_r']),
                    gamma_h=float(row['gamma_h']),
                    gamma_r=float(row['gamma_r']),
                    zeta=float(row['zeta']),
                    eta=float(row['eta']),
                    xi=float(row['xi']),
                    seed=parse_optional_int(row.get('seed')) or 0,
                    p_left=parse_optional_float(row.get('p_left')),
                    n_states=parse_optional_int(row.get('n_states')),
                    computation_time=parse_optional_float(row.get('computation_time')),
                    pi_r_turn_left=parse_optional_float(row.get('pi_r_turn_left')),
                    pi_r_turn_right=parse_optional_float(row.get('pi_r_turn_right')),
                    Q_r_turn_left=parse_optional_float(row.get('Q_r_turn_left')),
                    Q_r_turn_right=parse_optional_float(row.get('Q_r_turn_right')),
                    V_r=parse_optional_float(row.get('V_r')),
                    X_h1=parse_optional_float(row.get('X_h1')),
                    X_h2=parse_optional_float(row.get('X_h2')),
                    h1_pos=parse_optional_str(row.get('h1_pos')),
                    h2_pos=parse_optional_str(row.get('h2_pos')),
                )
                existing[idx] = params
        print(f"Loaded {len(existing)} existing results from {output_file}")
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        existing = {}
    
    return existing


def append_result_to_csv(result: ParameterSet, output_file: str, write_header: bool = False):
    """
    Append a single result to CSV file with file locking for concurrent access.
    
    Args:
        result: ParameterSet to append
        output_file: Path to output CSV file
        write_header: Whether to write header row (for new files)
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use file locking for safe concurrent writes
    with open(output_file, 'a', newline='') as f:
        # Acquire exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check if file is empty (need header)
            f.seek(0, 2)  # Seek to end
            need_header = f.tell() == 0
            
            writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
            if need_header:
                writer.writeheader()
            writer.writerow(asdict(result))
            f.flush()
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for asymmetric_freeing_simple.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of NEW samples to compute (default: 10)')
    parser.add_argument('--output', type=str, default='outputs/parameter_sweep/results.csv',
                       help='Output CSV file (default: outputs/parameter_sweep/results.csv)')
    parser.add_argument('--world', type=str, 
                       default='multigrid_worlds/jobst_challenges/asymmetric_freeing_simple.yaml',
                       help='Path to world YAML file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    # Quick mode for fast testing
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode: sets max_steps=8 and n_samples=3')
    
    # Prior bounds for parameter sampling
    parser.add_argument('--max_steps_min', type=int, default=8,
                       help='Minimum max_steps (default: 8)')
    parser.add_argument('--max_steps_max', type=int, default=14,
                       help='Maximum max_steps (default: 14)')
    parser.add_argument('--beta_h_min', type=float, default=5.0,
                       help='Minimum beta_h, log-uniform (default: 5.0)')
    parser.add_argument('--beta_h_max', type=float, default=100.0,
                       help='Maximum beta_h, log-uniform (default: 100.0)')
    parser.add_argument('--gamma_h_min', type=float, default=0.8,
                       help='Minimum gamma_h (default: 0.8)')
    parser.add_argument('--gamma_h_max', type=float, default=1.0,
                       help='Maximum gamma_h (default: 1.0)')
    parser.add_argument('--gamma_r_min', type=float, default=0.8,
                       help='Minimum gamma_r (default: 0.8)')
    parser.add_argument('--gamma_r_max', type=float, default=1.0,
                       help='Maximum gamma_r (default: 1.0)')
    parser.add_argument('--zeta_min', type=float, default=1.0,
                       help='Minimum zeta (default: 1.0)')
    parser.add_argument('--zeta_max', type=float, default=3.0,
                       help='Maximum zeta (default: 3.0)')
    parser.add_argument('--eta_min', type=float, default=1.0,
                       help='Minimum eta (default: 1.0)')
    parser.add_argument('--eta_max', type=float, default=2.0,
                       help='Maximum eta (default: 2.0)')
    parser.add_argument('--xi_min', type=float, default=1.0,
                       help='Minimum xi (default: 1.0)')
    parser.add_argument('--xi_max', type=float, default=2.0,
                       help='Maximum xi (default: 2.0)')
    parser.add_argument('--beta_r', type=float, default=50.0,
                       help='Fixed beta_r value (default: 50.0)')
    parser.add_argument('--no-attainment-cache', action='store_true',
                       help='Disable attainment cache to save ~3GB memory (default: cache enabled)')
    
    args = parser.parse_args()
    
    # Apply quick mode overrides
    if args.quick:
        args.max_steps_min = 8
        args.max_steps_max = 8
        args.n_samples = 3
    
    # Build prior bounds from arguments
    bounds = PriorBounds(
        max_steps_min=args.max_steps_min,
        max_steps_max=args.max_steps_max,
        beta_h_min=args.beta_h_min,
        beta_h_max=args.beta_h_max,
        gamma_h_min=args.gamma_h_min,
        gamma_h_max=args.gamma_h_max,
        gamma_r_min=args.gamma_r_min,
        gamma_r_max=args.gamma_r_max,
        zeta_min=args.zeta_min,
        zeta_max=args.zeta_max,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        beta_r=args.beta_r
    )
    
    # Create unique seed based on SLURM task ID (if available) or secure random
    # SLURM_PROCID is unique per task in a job step (0, 1, 2, ...)
    # SLURM_ARRAY_TASK_ID is unique per array job task
    slurm_procid = os.environ.get('SLURM_PROCID')
    slurm_array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    
    if slurm_procid is not None:
        # Use SLURM_PROCID * large_prime + job_id for uniqueness across jobs
        job_id = int(slurm_job_id) if slurm_job_id else 0
        base_seed = int(slurm_procid) * 1000003 + (job_id % 1000000)
    elif slurm_array_task_id is not None:
        job_id = int(slurm_job_id) if slurm_job_id else 0
        base_seed = int(slurm_array_task_id) * 1000003 + (job_id % 1000000)
    else:
        # Fallback for non-SLURM: use secure random based on system entropy
        # This is safer for parallel execution than time-based seeds
        base_seed = random.SystemRandom().randint(0, 2**31 - 1)
    
    # Ensure base_seed is positive and within int32 range
    base_seed = abs(base_seed) % (2**31)
    
    print("="*80)
    print(f"Parameter Sweep: Asymmetric Freeing Simple (PID {os.getpid()})")
    print("="*80)
    print(f"Configuration:")
    print(f"  New samples to compute: {args.n_samples}")
    print(f"  Base seed: {base_seed}")
    if slurm_procid is not None:
        print(f"  SLURM_PROCID: {slurm_procid}")
    if slurm_array_task_id is not None:
        print(f"  SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"  World file: {args.world}")
    print(f"  Output file: {args.output}")
    if args.quick:
        print(f"  Quick mode: ENABLED")
    print(f"Prior bounds:")
    print(f"  max_steps: [{bounds.max_steps_min}, {bounds.max_steps_max}]")
    print(f"  beta_h: [{bounds.beta_h_min}, {bounds.beta_h_max}] (log-uniform)")
    print(f"  gamma_h: [{bounds.gamma_h_min}, {bounds.gamma_h_max}]")
    print(f"  gamma_r: [{bounds.gamma_r_min}, {bounds.gamma_r_max}]")
    print(f"  zeta: [{bounds.zeta_min}, {bounds.zeta_max}]")
    print(f"  eta: [{bounds.eta_min}, {bounds.eta_max}]")
    print(f"  xi: [{bounds.xi_min}, {bounds.xi_max}]")
    print(f"  beta_r: {bounds.beta_r} (fixed)")
    print()
    sys.stdout.flush()
    
    # Sample all parameters upfront with unique seeds
    all_params = []
    for i in range(args.n_samples):
        seed = base_seed + i
        params = sample_parameters(seed=seed, bounds=bounds)
        all_params.append((i, params))
    
    # Group by max_steps to reuse DAG
    from collections import defaultdict
    params_by_max_steps: Dict[int, List[Tuple[int, ParameterSet]]] = defaultdict(list)
    for i, params in all_params:
        params_by_max_steps[params.max_steps].append((i, params))
    
    # Sort max_steps values for deterministic ordering
    sorted_max_steps = sorted(params_by_max_steps.keys())
    
    print(f"Grouped {args.n_samples} samples by max_steps: {dict((k, len(v)) for k, v in params_by_max_steps.items())}")
    sys.stdout.flush()
    
    # Run simulations, grouped by max_steps to reuse DAG
    completed_count = 0
    
    for max_steps in sorted_max_steps:
        group = params_by_max_steps[max_steps]
        
        print(f"\nProcessing max_steps={max_steps} ({len(group)} samples)...")
        sys.stdout.flush()
        
        # Create environment once for this max_steps value
        env = MultiGridEnv(
            config_file=args.world,
            partial_obs=False,
            actions_set=SmallActions
        )
        env.max_steps = max_steps
        env.reset()
        
        # Build DAG once (will be cached for all runs in this group)
        print(f"  Building DAG...")
        sys.stdout.flush()
        env.get_dag(return_probabilities=True, quiet=args.quiet)
        
        cache_status = "disabled" if args.no_attainment_cache else "enabled"
        print(f"  Attainment cache: {cache_status}")
        sys.stdout.flush()
        
        # Run all simulations in this group
        for sample_idx, params in group:
            result = run_single_simulation(
                params=params,
                env=env,
                parallel=False,
                quiet=args.quiet,
                use_attainment_cache=not args.no_attainment_cache
            )
            
            # Append result to CSV immediately (with file locking)
            append_result_to_csv(result, args.output)
            completed_count += 1
            
            p_left_str = f"{result.p_left:.4f}" if result.p_left is not None else "None"
            print(f"  Sample {completed_count}/{args.n_samples} done (p_left={p_left_str})")
            sys.stdout.flush()
        
        # Free DAG memory before processing next max_steps group
        del env
        gc.collect()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Complete: {completed_count} samples appended to {args.output}")
    print("="*80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
