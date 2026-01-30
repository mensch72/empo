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
4. Runs rollouts to measure P(left) - probability of freeing left human first

The results are saved to a CSV file for subsequent logistic regression analysis.

Usage:
    # Small test run (10 samples)
    python experiments/parameter_sweep_asymmetric_freeing.py --n_samples 10 --output outputs/parameter_sweep/results_test.csv
    
    # Full run (100 samples)
    python experiments/parameter_sweep_asymmetric_freeing.py --n_samples 100 --output outputs/parameter_sweep/results_full.csv
    
    # Parallel with 4 workers
    python experiments/parameter_sweep_asymmetric_freeing.py --n_samples 100 --parallel --num_workers 4
"""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Setup paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')
sys.path.insert(0, 'vendor/ai_transport')

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.backward_induction import compute_human_policy_prior, compute_robot_policy
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy


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
    
    # Results (filled after simulation)
    left_freed_first: Optional[int] = None  # 1 if left freed first, 0 if right, -1 if neither
    left_freed_step: Optional[int] = None   # Step when left human was freed (-1 if not freed)
    right_freed_step: Optional[int] = None  # Step when right human was freed (-1 if not freed)
    n_states: Optional[int] = None          # Number of states in DAG
    computation_time: Optional[float] = None  # Time for backward induction (seconds)


def sample_parameters(seed: Optional[int] = None) -> ParameterSet:
    """
    Sample a random parameter set from suitable priors.
    
    Priors:
    - max_steps: uniform discrete from 8 to 14
    - beta_h: log-uniform from 5 to 100
    - gamma_h: uniform from 0.8 to 1.0
    - gamma_r: uniform from 0.8 to 1.0
    - zeta: uniform from 1 to 3
    - eta: uniform from 1 to 2
    - xi: uniform from 1 to 2
    - beta_r: fixed at 50.0 (robot policy concentration)
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        ParameterSet with sampled values
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Sample parameters
    max_steps = np.random.randint(8, 15)  # 8 to 14 inclusive
    beta_h = np.exp(np.random.uniform(np.log(5), np.log(100)))  # log-uniform 5 to 100
    gamma_h = np.random.uniform(0.8, 1.0)
    gamma_r = np.random.uniform(0.8, 1.0)
    zeta = np.random.uniform(1.0, 3.0)
    eta = np.random.uniform(1.0, 2.0)
    xi = np.random.uniform(1.0, 2.0)
    beta_r = 50.0  # Fixed
    
    return ParameterSet(
        max_steps=max_steps,
        beta_h=beta_h,
        beta_r=beta_r,
        gamma_h=gamma_h,
        gamma_r=gamma_r,
        zeta=zeta,
        eta=eta,
        xi=xi
    )


def detect_which_human_freed_first(env: MultiGridEnv, 
                                  human_policy_prior: TabularHumanPolicyPrior,
                                  robot_policy: RobotPolicy,
                                  human_agent_indices: List[int],
                                  robot_agent_indices: List[int],
                                  max_steps: int) -> Tuple[int, int, int]:
    """
    Run a single rollout and detect which human was freed first.
    
    A human is "freed" when they can reach at least one goal cell that was previously
    blocked by a rock. This happens when the robot pushes away the rock in front of them.
    
    Args:
        env: Environment instance
        human_policy_prior: Computed human policy prior
        robot_policy: Computed robot policy
        human_agent_indices: List of human agent indices
        robot_agent_indices: List of robot agent indices
        max_steps: Maximum steps to run
        
    Returns:
        Tuple of (left_freed_first, left_freed_step, right_freed_step):
        - left_freed_first: 1 if left human freed first, 0 if right freed first, -1 if neither
        - left_freed_step: step when left was freed (-1 if not freed)
        - right_freed_step: step when right was freed (-1 if not freed)
    """
    # Reset environment
    env.reset()
    
    # Left human is agent 0 at initial position (2, 1)
    # Right human is agent 1 at initial position (5, 1)
    # Rock at (2, 2) blocks left human
    # Rock at (5, 2) blocks right human
    
    # Track which rocks have been moved/removed
    left_rock_pos = (2, 2)
    right_rock_pos = (5, 2)
    
    left_freed_step = -1
    right_freed_step = -1
    
    # Check initial state
    def is_rock_blocking(env, pos):
        """Check if there's a rock at the given position."""
        cell = env.grid.get(*pos)
        return cell is not None and cell.type == 'rock'
    
    for step in range(max_steps):
        # Check if rocks have been moved
        if left_freed_step == -1 and not is_rock_blocking(env, left_rock_pos):
            left_freed_step = step
        if right_freed_step == -1 and not is_rock_blocking(env, right_rock_pos):
            right_freed_step = step
        
        # If both freed, we're done
        if left_freed_step != -1 and right_freed_step != -1:
            break
        
        # Get current state
        current_state = env.get_state()
        
        # Sample actions from policies
        actions = [0] * len(env.agents)
        
        # Sample human actions from human policy prior
        for human_idx in human_agent_indices:
            # Sample a goal for this human
            goal_list = list(env.possible_goal_generator.generate(current_state, human_idx))
            if goal_list and current_state in human_policy_prior.values:
                sampled_goal, weight = goal_list[np.random.randint(len(goal_list))]
                
                # Get human action from policy prior
                if (human_idx in human_policy_prior.values[current_state] and
                    sampled_goal in human_policy_prior.values[current_state][human_idx]):
                    action_dist = human_policy_prior.values[current_state][human_idx][sampled_goal]
                    # action_dist is a numpy array of probabilities
                    actions[human_idx] = np.random.choice(len(action_dist), p=action_dist)
                else:
                    # If state/goal not in policy, use random action
                    actions[human_idx] = env.action_space.sample()
            else:
                # No goals available or state not in policy, use random action
                actions[human_idx] = env.action_space.sample()
        
        # Sample robot actions from robot policy
        # robot_policy(state) returns Dict[RobotActionProfile, float]
        # or we can use robot_policy.sample(state) to directly get a sampled action profile
        sampled_robot_action_profile = robot_policy.sample(current_state)
        
        # Assign robot actions (robot_action_profile is a tuple of actions for each robot)
        for i, robot_idx in enumerate(robot_agent_indices):
            if sampled_robot_action_profile and i < len(sampled_robot_action_profile):
                actions[robot_idx] = sampled_robot_action_profile[i]
            else:
                actions[robot_idx] = env.action_space.sample()
        
        # Take step
        obs, rewards, done, info = env.step(actions)
        
        if done:
            break
    
    # Final check
    if left_freed_step == -1 and not is_rock_blocking(env, left_rock_pos):
        left_freed_step = max_steps - 1
    if right_freed_step == -1 and not is_rock_blocking(env, right_rock_pos):
        right_freed_step = max_steps - 1
    
    # Determine which was freed first
    if left_freed_step != -1 and right_freed_step != -1:
        left_freed_first = 1 if left_freed_step < right_freed_step else 0
    elif left_freed_step != -1:
        left_freed_first = 1
    elif right_freed_step != -1:
        left_freed_first = 0
    else:
        left_freed_first = -1  # Neither freed
    
    return left_freed_first, left_freed_step, right_freed_step


def run_single_simulation(params: ParameterSet, 
                         world_file: str,
                         n_rollouts: int = 10,
                         parallel: bool = False,
                         quiet: bool = False) -> ParameterSet:
    """
    Run a single simulation with the given parameters.
    
    Args:
        params: Parameter set to use
        world_file: Path to YAML world file
        n_rollouts: Number of rollouts to average P(left) over
        parallel: Whether to use parallel computation for backward induction
        quiet: Suppress output
        
    Returns:
        Updated ParameterSet with results filled in
    """
    start_time = time.time()
    
    try:
        # Create environment with specified max_steps
        env = MultiGridEnv(
            config_file=world_file,
            partial_obs=False,
            actions_set=SmallActions
        )
        env.max_steps = params.max_steps
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
            quiet=quiet
        )
        
        # Phase 2: Compute robot policy
        if not quiet:
            print("Phase 2: Computing robot policy...")
        robot_policy = compute_robot_policy(
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
            quiet=quiet
        )
        
        # Get number of states
        params.n_states = len(robot_policy.values)
        
        if not quiet:
            print(f"Computed policies for {params.n_states} states")
            print(f"Running {n_rollouts} rollouts to measure P(left)...")
        
        # Run multiple rollouts to estimate P(left)
        left_freed_count = 0
        total_valid_rollouts = 0
        
        for i in range(n_rollouts):
            left_freed_first, left_step, right_step = detect_which_human_freed_first(
                env, 
                human_policy_prior, 
                robot_policy,
                human_agent_indices,
                robot_agent_indices,
                params.max_steps
            )
            
            # Only count rollouts where at least one human was freed
            if left_freed_first != -1:
                total_valid_rollouts += 1
                if left_freed_first == 1:
                    left_freed_count += 1
            
            # Store results from first rollout as representative
            if i == 0:
                params.left_freed_first = left_freed_first
                params.left_freed_step = left_step
                params.right_freed_step = right_step
        
        # If no valid rollouts, mark as -1
        if total_valid_rollouts == 0:
            params.left_freed_first = -1
        else:
            # Store the proportion as a representative value (0 or 1 based on majority)
            p_left = left_freed_count / total_valid_rollouts
            params.left_freed_first = 1 if p_left >= 0.5 else 0
        
        params.computation_time = time.time() - start_time
        
        if not quiet:
            if total_valid_rollouts > 0:
                print(f"Results: P(left) = {left_freed_count}/{total_valid_rollouts} = {left_freed_count/total_valid_rollouts:.3f}")
            else:
                print("Results: No humans freed in any rollout")
            print(f"Computation time: {params.computation_time:.1f}s")
        
        return params
        
    except Exception as e:
        if not quiet:
            print(f"ERROR in simulation: {e}")
        import traceback
        traceback.print_exc()
        params.left_freed_first = -1
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
    valid_results = [r for r in results if r.left_freed_first != -1]
    print(f"Valid samples (at least one human freed): {len(valid_results)}")
    if valid_results:
        left_freed_count = sum(1 for r in valid_results if r.left_freed_first == 1)
        print(f"Left freed first: {left_freed_count}/{len(valid_results)} = {left_freed_count/len(valid_results):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Parameter sweep for asymmetric_freeing_simple.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of parameter combinations to sample (default: 10)')
    parser.add_argument('--n_rollouts', type=int, default=5,
                       help='Number of rollouts per parameter set to estimate P(left) (default: 5)')
    parser.add_argument('--output', type=str, default='outputs/parameter_sweep/results.csv',
                       help='Output CSV file (default: outputs/parameter_sweep/results.csv)')
    parser.add_argument('--world', type=str, 
                       default='multigrid_worlds/jobst_challenges/asymmetric_freeing_simple.yaml',
                       help='Path to world YAML file')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel computation for backward induction')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Parameter Sweep: Asymmetric Freeing Simple")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Rollouts per sample: {args.n_rollouts}")
    print(f"  World file: {args.world}")
    print(f"  Output file: {args.output}")
    print(f"  Parallel: {args.parallel}")
    if args.parallel and args.num_workers:
        print(f"  Workers: {args.num_workers}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Run simulations
    results = []
    for i in tqdm(range(args.n_samples), desc="Running simulations"):
        # Sample parameters with deterministic seed for reproducibility
        params = sample_parameters(seed=args.seed + i)
        
        # Run simulation
        result = run_single_simulation(
            params=params,
            world_file=args.world,
            n_rollouts=args.n_rollouts,
            parallel=args.parallel,
            quiet=args.quiet
        )
        
        results.append(result)
    
    # Save results
    save_results_to_csv(results, args.output)
    
    print("\n" + "="*80)
    print("Parameter sweep complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Analyze results with logistic regression:")
    print(f"   python experiments/analyze_parameter_sweep.py {args.output}")
    print(f"2. Run on HPC for larger sample sizes:")
    print(f"   sbatch scripts/run_parameter_sweep.sh")


if __name__ == '__main__':
    main()
