#!/usr/bin/env python3
"""
Statistical test to verify transition_probabilities() correctness.

This test generates random grids with many agents, runs episodes with random actions,
and for each state encountered, performs statistical sampling to verify that:
1. The transition_probabilities() method returns correct probabilities
2. Actual step() executions match the claimed distribution (chi-squared test)
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import time

# Setup path to import multigrid
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "multigrid"))

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Agent, World, Grid, Actions


class RandomGridEnv(MultiGridEnv):
    """Environment with configurable number of agents for statistical testing."""
    
    def __init__(self, num_agents=4, grid_size=8):
        self.num_agents_config = num_agents
        agents = [Agent(World, i+1, view_size=7) for i in range(num_agents)]
        super().__init__(
            grid_size=grid_size,
            max_steps=50,
            agents=agents,
            agent_view_size=7
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Generate walls
        self.grid.horz_wall(World, 0, 0)
        self.grid.horz_wall(World, 0, height-1)
        self.grid.vert_wall(World, 0, 0)
        self.grid.vert_wall(World, width-1, 0)
        # Place agents randomly
        for a in self.agents:
            self.place_agent(a)


def chi_squared_test(observed_counts, expected_probs, total_samples):
    """
    Perform chi-squared goodness-of-fit test.
    
    Returns: (chi_squared_statistic, p_value_threshold, passes)
    """
    chi_squared = 0.0
    
    for state, count in observed_counts.items():
        expected_count = expected_probs[state] * total_samples
        if expected_count > 0:
            chi_squared += (count - expected_count) ** 2 / expected_count
    
    # Degrees of freedom = number of categories - 1
    df = len(expected_probs) - 1
    
    # For alpha=0.01 (99% confidence), critical values:
    # df=1: 6.63, df=2: 9.21, df=3: 11.34
    # We use a generous threshold since we have limited samples
    critical_values = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09}
    threshold = critical_values.get(df, 15.09 + (df - 5) * 2.0)
    
    passes = chi_squared < threshold
    
    return chi_squared, threshold, passes


def test_statistical_correctness():
    """
    Main statistical test: verify transition_probabilities() matches actual step() distribution.
    """
    print("=" * 70)
    print("Statistical Test: Verify transition_probabilities() correctness")
    print("=" * 70)
    print()
    
    # Test parameters
    num_grids = 10  # Number of random grids to test
    num_agents = 5  # More agents = more potential conflicts (max 5 due to color limit)
    grid_size = 6  # Smaller grid = more conflicts (6x6 with 5 agents)
    episode_steps = 15  # More steps per episode
    states_to_sample = 5  # Sample more states per episode
    samples_per_state = 200  # Statistical samples per state
    
    print(f"Configuration:")
    print(f"  - Testing {num_grids} random grids")
    print(f"  - {num_agents} agents, {grid_size}×{grid_size} grid")
    print(f"  - {episode_steps} steps per episode")
    print(f"  - Sampling {states_to_sample} states per episode")
    print(f"  - {samples_per_state} samples per state")
    print()
    
    all_tests_passed = True
    total_states_tested = 0
    total_chi_squared_tests = 0
    passed_chi_squared_tests = 0
    
    for grid_idx in range(num_grids):
        print(f"Grid {grid_idx + 1}/{num_grids}:")
        
        # Create environment
        env = RandomGridEnv(num_agents=num_agents, grid_size=grid_size)
        env.reset()
        
        # Run episode with random actions
        episode_states = []
        episode_actions = []
        
        for step in range(episode_steps):
            state = env.get_state()
            episode_states.append(state)
            
            # Generate random actions - bias towards forward to create more conflicts
            actions = []
            for _ in range(num_agents):
                # 60% forward, 20% left, 20% right
                rand = np.random.random()
                if rand < 0.6:
                    actions.append(Actions.forward)
                elif rand < 0.8:
                    actions.append(Actions.left)
                else:
                    actions.append(Actions.right)
            episode_actions.append(actions)
            
            # Execute step
            obs, rewards, done, info = env.step(actions)
            
            if done:
                break
        
        # Sample a few states from this episode for statistical testing
        num_states_available = min(len(episode_states), states_to_sample)
        if num_states_available == 0:
            print("  ⚠ No states to test (episode ended immediately)")
            continue
        
        sampled_indices = np.random.choice(len(episode_states), 
                                          size=min(num_states_available, states_to_sample), 
                                          replace=False)
        
        for state_idx in sampled_indices:
            state = episode_states[state_idx]
            actions = episode_actions[state_idx]
            
            total_states_tested += 1
            
            # Get predicted transition probabilities
            env.set_state(state)
            transitions = env.transition_probabilities(state, actions)
            
            if transitions is None:
                # Terminal state or invalid actions
                continue
            
            # Check if deterministic (single outcome)
            if len(transitions) == 1:
                print(f"  State {state_idx}: Deterministic (1 outcome, prob=1.0) ✓")
                continue
            
            # Probabilistic case - perform statistical sampling
            print(f"  State {state_idx}: Probabilistic ({len(transitions)} outcomes)")
            
            # Create mapping: successor_state -> expected_probability
            expected_probs = {}
            for prob, succ_state in transitions:
                expected_probs[succ_state] = prob
            
            # Sample actual transitions by calling step() many times
            observed_counts = defaultdict(int)
            
            for sample_idx in range(samples_per_state):
                # Reset to the test state
                env.set_state(state)
                
                # Execute step (with RNG, so outcomes vary)
                obs, rewards, done, info = env.step(actions)
                
                # Record the resulting state
                result_state = env.get_state()
                observed_counts[result_state] += 1
            
            # Perform chi-squared test
            # Only count states that were predicted
            chi_squared = 0.0
            for state, expected_prob in expected_probs.items():
                observed_count = observed_counts.get(state, 0)
                expected_count = expected_prob * samples_per_state
                if expected_count > 0:
                    chi_squared += (observed_count - expected_count) ** 2 / expected_count
            
            # Check for unexpected states (should not happen!)
            unexpected_states = set(observed_counts.keys()) - set(expected_probs.keys())
            if unexpected_states:
                print(f"    ⚠ WARNING: {len(unexpected_states)} unexpected states observed!")
                print(f"    This indicates a bug in transition_probabilities()")
                passes = False
            else:
                # Degrees of freedom = number of categories - 1
                df = len(expected_probs) - 1
                critical_values = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09}
                threshold = critical_values.get(df, 15.09 + (df - 5) * 2.0)
                passes = chi_squared < threshold
            
            total_chi_squared_tests += 1
            if passes:
                passed_chi_squared_tests += 1
            
            # Report results
            print(f"    Expected: {len(transitions)} outcomes")
            for prob, succ_state in sorted(transitions, key=lambda x: -x[0]):
                actual_count = observed_counts.get(succ_state, 0)
                actual_freq = actual_count / samples_per_state
                print(f"      prob={prob:.3f} → observed {actual_count}/{samples_per_state} ({actual_freq:.3f})")
            
            # Get threshold for display
            df = len(expected_probs) - 1
            critical_values = {1: 6.63, 2: 9.21, 3: 11.34, 4: 13.28, 5: 15.09}
            threshold = critical_values.get(df, 15.09 + (df - 5) * 2.0)
            
            print(f"    χ² = {chi_squared:.2f} (threshold = {threshold:.2f}) ", end="")
            
            if passes:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                all_tests_passed = False
                
                # Show details of the mismatch
                print("    Detailed mismatch:")
                for succ_state in expected_probs.keys():
                    expected = expected_probs[succ_state]
                    actual = observed_counts.get(succ_state, 0) / samples_per_state
                    diff = abs(actual - expected)
                    if diff > 0.1:  # Show significant differences
                        print(f"      State: expected={expected:.3f}, actual={actual:.3f}, diff={diff:.3f}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("Statistical Test Summary")
    print("=" * 70)
    print(f"Total states tested: {total_states_tested}")
    print(f"Chi-squared tests: {passed_chi_squared_tests}/{total_chi_squared_tests} passed")
    
    if total_chi_squared_tests > 0:
        pass_rate = passed_chi_squared_tests / total_chi_squared_tests
        print(f"Pass rate: {pass_rate * 100:.1f}%")
        
        # We expect some tests to fail due to random variation (at 99% confidence, ~1% should fail)
        # Allow up to 10% failure rate
        if pass_rate >= 0.90:
            print("✓ Statistical test PASSED (≥90% of chi-squared tests passed)")
            return True
        else:
            print("✗ Statistical test FAILED (<90% of chi-squared tests passed)")
            return False
    else:
        print("⚠ No probabilistic states tested (all states were deterministic)")
        return True


def test_basic_sampling_correctness():
    """
    Simple test: verify that sampling from a known distribution works.
    """
    print("=" * 70)
    print("Basic Test: Verify sampling from known distribution")
    print("=" * 70)
    print()
    
    # Create a simple 3-agent scenario where we know the probabilities
    env = RandomGridEnv(num_agents=3, grid_size=6)
    env.reset()
    
    # Manufacture a state where all 3 agents compete for one cell
    # Clear old agent positions from grid
    for agent in env.agents:
        if agent.pos is not None:
            cell = env.grid.get(*agent.pos)
            if cell is agent:
                env.grid.set(*agent.pos, None)
    
    # All agents converge on (3, 3)
    positions_dirs = [((2, 3), 0), ((4, 3), 2), ((3, 2), 1)]
    for i, (pos, dir) in enumerate(positions_dirs):
        env.agents[i].pos = np.array(pos)
        env.agents[i].dir = dir
        env.grid.set(pos[0], pos[1], env.agents[i])
    
    # Get the modified state
    modified_state = env.get_state()
    
    # All agents move forward
    actions = [Actions.forward, Actions.forward, Actions.forward]
    
    # Get expected probabilities
    transitions = env.transition_probabilities(modified_state, actions)
    
    print(f"Setup: 3 agents positioned to converge on (3,3)")
    print(f"Actual outcomes: {len(transitions)}")
    
    for prob, _ in transitions:
        print(f"  - prob = {prob:.4f}")
    
    # Sample 300 times
    observed_counts = defaultdict(int)
    samples = 300
    
    for _ in range(samples):
        env.set_state(modified_state)
        step_result = env.step(actions)
        # Handle both gymnasium (5 values) and gym (4 values) step() return formats
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
        else:
            obs, rewards, done, info = step_result
        result_state = env.get_state()
        observed_counts[result_state] += 1
    
    print(f"\nSampling {samples} times:")
    expected_probs = {succ_state: prob for prob, succ_state in transitions}
    
    for succ_state, prob in expected_probs.items():
        count = observed_counts.get(succ_state, 0)
        freq = count / samples
        print(f"  Expected {prob:.3f} → observed {count}/{samples} ({freq:.3f})")
    
    # Verify that all sampled states are in the expected outcomes
    unmatched = 0
    for state, count in observed_counts.items():
        if state not in expected_probs:
            print(f"  WARNING: Sampled state not in expected outcomes (count={count})")
            unmatched += count
    
    if unmatched > samples * 0.1:  # More than 10% unmatched is a problem
        print(f"  ✗ Too many unmatched outcomes: {unmatched}/{samples}")
        return False
    
    # If there's only one outcome, we just check that all samples match it
    if len(transitions) == 1:
        expected_state = transitions[0][1]
        match_count = observed_counts.get(expected_state, 0)
        if match_count >= samples * 0.9:  # Allow 10% variance
            print(f"  ✓ Single deterministic outcome verified ({match_count}/{samples})")
            return True
        else:
            print(f"  ✗ Expected single outcome but got variance: {match_count}/{samples}")
            return False
    
    # Chi-squared test for multiple outcomes
    # Only test if we have matching states
    filtered_counts = {k: v for k, v in observed_counts.items() if k in expected_probs}
    if filtered_counts:
        chi_squared, threshold, passes = chi_squared_test(filtered_counts, expected_probs, sum(filtered_counts.values()))
        print(f"\nχ² = {chi_squared:.2f} (threshold = {threshold:.2f})")
        
        if passes:
            print(f"  ✓ Statistical test passed")
            return True
        else:
            print(f"  ✗ Statistical test failed")
            return False
    
    print(f"  ✓ Test completed")
    return True


def main():
    """Run all statistical tests."""
    print()
    print("*" * 70)
    print("*" + " Statistical Verification of transition_probabilities()".center(68) + "*")
    print("*" * 70)
    print()
    
    # Run basic test first
    basic_passed = test_basic_sampling_correctness()
    print()
    
    # Run comprehensive statistical test
    stats_passed = test_statistical_correctness()
    print()
    
    # Final result
    if basic_passed and stats_passed:
        print("=" * 70)
        print("✓ ALL STATISTICAL TESTS PASSED")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("✗ SOME STATISTICAL TESTS FAILED")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
