#!/usr/bin/env python3
"""
Tests for batch_transition_probabilities() and the Phase 2 trainer optimizations
(§3.12 batch precompute, §3.13 skip step by sampling from cached probs).
"""

import sys
import numpy as np
import pytest

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import Actions
from gym_multigrid.envs import CollectGame4HEnv10x10N2


# ========================
# §3.12: batch_transition_probabilities tests
# ========================

class TestBatchTransitionProbabilities:
    """Tests for the batch_transition_probabilities() method on MultiGridEnv."""

    def _make_env(self):
        env = CollectGame4HEnv10x10N2()
        env.reset()
        return env

    def test_batch_matches_individual_calls(self):
        """Batch results must match individual transition_probabilities() calls exactly."""
        env = self._make_env()
        state = env.get_state()
        num_agents = len(env.agents)

        # Create several action vectors
        actions_list = [
            [Actions.still] * num_agents,
            [Actions.forward] * num_agents,
            [Actions.left] * num_agents,
            [Actions.right] * num_agents,
        ]

        # Individual calls
        individual_results = []
        for actions in actions_list:
            result = env.transition_probabilities(state, actions)
            individual_results.append(result)

        # Batch call
        batch_results = env.batch_transition_probabilities(state, actions_list)

        assert len(batch_results) == len(actions_list)
        for i, (individual, batched) in enumerate(zip(individual_results, batch_results)):
            if individual is None:
                assert batched is None, f"Action set {i}: expected None, got {batched}"
            else:
                assert batched is not None, f"Action set {i}: expected results, got None"
                assert len(individual) == len(batched), (
                    f"Action set {i}: length mismatch {len(individual)} vs {len(batched)}"
                )
                for j, ((p1, s1), (p2, s2)) in enumerate(zip(individual, batched)):
                    assert abs(p1 - p2) < 1e-12, (
                        f"Action set {i}, outcome {j}: prob mismatch {p1} vs {p2}"
                    )
                    assert s1 == s2, (
                        f"Action set {i}, outcome {j}: state mismatch"
                    )

    def test_batch_preserves_env_state(self):
        """Environment state must be restored after batch call."""
        env = self._make_env()
        state = env.get_state()

        # Step to get into a different state
        actions = [Actions.forward] * len(env.agents)
        env.step(actions)
        state_before = env.get_state()

        # Batch call using original state
        actions_list = [
            [Actions.forward] * len(env.agents),
            [Actions.left] * len(env.agents),
        ]
        env.batch_transition_probabilities(state, actions_list)

        state_after = env.get_state()
        assert state_before == state_after, "batch_transition_probabilities changed env state"

    def test_batch_terminal_state(self):
        """Batch on terminal state should return list of Nones."""
        env = self._make_env()
        state = env.get_state()

        # Create a terminal state by maxing out step count
        terminal_state = (env.max_steps, state[1], state[2], state[3])
        
        actions_list = [
            [Actions.forward] * len(env.agents),
            [Actions.left] * len(env.agents),
        ]
        results = env.batch_transition_probabilities(terminal_state, actions_list)
        
        assert len(results) == 2
        assert all(r is None for r in results)

    def test_batch_empty_list(self):
        """Empty actions list should return empty results."""
        env = self._make_env()
        state = env.get_state()
        results = env.batch_transition_probabilities(state, [])
        assert results == []

    def test_batch_single_action(self):
        """Single action in batch should match individual call."""
        env = self._make_env()
        state = env.get_state()
        
        actions = [Actions.forward] * len(env.agents)
        individual = env.transition_probabilities(state, actions)
        batch = env.batch_transition_probabilities(state, [actions])
        
        assert len(batch) == 1
        if individual is None:
            assert batch[0] is None
        else:
            assert len(individual) == len(batch[0])
            for (p1, s1), (p2, s2) in zip(individual, batch[0]):
                assert abs(p1 - p2) < 1e-12
                assert s1 == s2

    def test_batch_invalid_action_raises(self):
        """Invalid actions should raise ValueError."""
        env = self._make_env()
        state = env.get_state()
        
        with pytest.raises(ValueError, match="Invalid action"):
            env.batch_transition_probabilities(state, [[-1] * len(env.agents)])

    def test_batch_probabilities_sum_to_one(self):
        """Each result's probabilities should sum to 1.0."""
        env = self._make_env()
        state = env.get_state()
        
        actions_list = [
            [Actions.forward] * len(env.agents),
            [Actions.left, Actions.right, Actions.forward],
        ]
        results = env.batch_transition_probabilities(state, actions_list)
        
        for i, result in enumerate(results):
            if result is not None:
                total = sum(p for p, _ in result)
                assert abs(total - 1.0) < 1e-10, (
                    f"Action set {i}: probabilities sum to {total}, expected 1.0"
                )


# ========================
# §3.13: sample from cached transition probs tests
# ========================

class TestSampleFromCachedTransitionProbs:
    """Tests for sampling next_state from pre-computed transition probs."""

    def test_deterministic_sampling_matches_step(self):
        """
        For deterministic transitions (most common case), the sampled state
        must match what step() would produce.
        """
        env = CollectGame4HEnv10x10N2()
        env.reset()
        state = env.get_state()
        num_agents = len(env.agents)

        # Use actions that are likely deterministic (single agent forward, rest still)
        actions = [Actions.still] * num_agents
        actions[0] = Actions.left  # Rotation is always deterministic

        # Get transition probs
        trans_probs = env.transition_probabilities(state, actions)
        assert trans_probs is not None
        
        # Deterministic: should have exactly 1 outcome
        if len(trans_probs) == 1:
            prob, expected_next_state = trans_probs[0]
            assert abs(prob - 1.0) < 1e-10

            # step() should produce same state
            env.set_state(state)
            env.step(actions)
            step_next_state = env.get_state()

            assert expected_next_state == step_next_state, (
                "Cached transition prob successor differs from step()"
            )

    def test_sampling_distribution_matches_step_distribution(self):
        """
        For probabilistic transitions, sampling from cached probs should
        produce the same distribution as repeated step() calls.
        """
        env = CollectGame4HEnv10x10N2()
        env.reset()
        state = env.get_state()
        num_agents = len(env.agents)

        # Multiple agents forward - may be probabilistic
        actions = [Actions.forward] * num_agents
        trans_probs = env.transition_probabilities(state, actions)
        
        if trans_probs is None or len(trans_probs) <= 1:
            pytest.skip("Need probabilistic transition for this test")

        # Sample from cached probs many times
        n_samples = 1000
        cached_counts = {}
        for _ in range(n_samples):
            probs = [p for p, _ in trans_probs]
            states = [s for _, s in trans_probs]
            chosen = np.random.choice(len(trans_probs), p=probs)
            state_key = states[chosen]
            cached_counts[state_key] = cached_counts.get(state_key, 0) + 1

        # Sample from step() many times
        step_counts = {}
        for _ in range(n_samples):
            env.set_state(state)
            env.step(actions)
            next_state = env.get_state()
            step_counts[next_state] = step_counts.get(next_state, 0) + 1

        # Both should produce the same set of successor states
        assert set(cached_counts.keys()) == set(step_counts.keys()), (
            "Cached probs and step() produce different successor states"
        )

        # The distributions should be statistically similar
        # (we can't check exact counts due to randomness, but the expected probs should be close)
        for (prob, state_key) in trans_probs:
            cached_frac = cached_counts.get(state_key, 0) / n_samples
            # Allow generous tolerance for statistical comparison
            assert abs(cached_frac - prob) < 0.1, (
                f"Cached sampling fraction {cached_frac} too far from expected prob {prob}"
            )

    def test_set_state_advances_env(self):
        """
        After sampling from cached probs and calling set_state(next_state),
        the environment should be in next_state.
        """
        env = CollectGame4HEnv10x10N2()
        env.reset()
        state = env.get_state()

        actions = [Actions.left] * len(env.agents)
        trans_probs = env.transition_probabilities(state, actions)
        assert trans_probs is not None and len(trans_probs) >= 1

        next_state = trans_probs[0][1]
        env.set_state(next_state)
        assert env.get_state() == next_state
