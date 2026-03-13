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
        
        # Rotation action should always be deterministic
        assert len(trans_probs) == 1, (
            "Rotation action (Actions.left) is expected to be deterministic, "
            "but transition_probabilities() returned multiple successor states"
        )
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

        # Compare distributions statistically rather than requiring exact key-set
        # equality, since with finite samples low-probability outcomes may be absent
        # from one side. Only check outcomes with expected probability above a threshold.
        min_prob_threshold = 0.01  # Only check outcomes expected to appear
        for (prob, state_key) in trans_probs:
            if prob < min_prob_threshold:
                continue
            cached_frac = cached_counts.get(state_key, 0) / n_samples
            step_frac = step_counts.get(state_key, 0) / n_samples
            # Both sampling methods should be close to the true probability
            assert abs(cached_frac - prob) < 0.1, (
                f"Cached sampling fraction {cached_frac} too far from expected prob {prob}"
            )
            assert abs(step_frac - prob) < 0.1, (
                f"Step sampling fraction {step_frac} too far from expected prob {prob}"
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


# ========================
# §3.15: get_state() consistency tests
# ========================

class TestGetStateConsistency:
    """Tests that get_state() returns correct state after step()."""

    def test_step_then_get_state_matches_recomputed(self):
        """get_state() after step() should return the same state every time."""
        env = CollectGame4HEnv10x10N2()
        env.reset()
        state = env.get_state()

        actions = [Actions.forward] * len(env.agents)
        
        # step() + get_state()
        env.set_state(state)
        env.step(actions)
        result1 = env.get_state()

        # Same again
        env.set_state(state)
        env.step(actions)
        result2 = env.get_state()

        assert result1 == result2, "Repeated step()+get_state() should be consistent"

    def test_consecutive_get_state_matches(self):
        """Two consecutive get_state() calls should return the same state."""
        env = CollectGame4HEnv10x10N2()
        env.reset()

        actions = [Actions.left] * len(env.agents)
        env.step(actions)

        state1 = env.get_state()
        state2 = env.get_state()
        assert state1 == state2, "Second get_state() should match first"


# ========================
# §3.13: _sample_next_state_from_cached_probs integration tests
# ========================

class _MockQR:
    """Minimal mock of q_r network for action_tuple_to_index()."""
    def action_tuple_to_index(self, action_tuple):
        return action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple


class _MockNetworks:
    """Minimal mock of Phase2Networks with just q_r."""
    def __init__(self):
        self.q_r = _MockQR()


class _SampleNextStateHarness:
    """
    Minimal harness that exposes _sample_next_state_from_cached_probs
    with a real environment but without a full Phase 2 trainer.
    """
    def __init__(self, env):
        self.env = env
        self.networks = _MockNetworks()

    # Bind the actual method from the trainer to this harness class, so we test
    # the real implementation without instantiating a full Phase 2 trainer.
    from empo.learning_based.phase2.trainer import BasePhase2Trainer
    _sample_next_state_from_cached_probs = BasePhase2Trainer._sample_next_state_from_cached_probs


class TestSampleNextStateFromCachedProbs:
    """
    Integration tests for _sample_next_state_from_cached_probs() that verify
    the method advances the environment correctly and interacts properly with
    the side-effect clearing in collect_transition().
    """

    def _make_harness(self):
        env = CollectGame4HEnv10x10N2()
        env.reset()
        return _SampleNextStateHarness(env), env

    def test_env_advanced_to_next_state(self):
        """
        After _sample_next_state_from_cached_probs(), env.get_state() must
        return the sampled next_state.
        """
        harness, env = self._make_harness()
        state = env.get_state()
        num_agents = len(env.agents)

        # Use a deterministic action (rotation)
        actions = [Actions.left] + [Actions.still] * (num_agents - 1)
        trans_probs = env.transition_probabilities(state, actions)
        assert trans_probs is not None and len(trans_probs) == 1, (
            "Rotation action should produce a single deterministic outcome"
        )

        # Build transition_probs_by_action dict (action_idx 0 maps to this action)
        transition_probs_by_action = {0: trans_probs}

        # Set env to original state first
        env.set_state(state)

        # Call the method
        next_state = harness._sample_next_state_from_cached_probs(
            state, (0,), transition_probs_by_action
        )

        # Verify env is now at next_state
        assert env.get_state() == next_state, (
            "env.get_state() should match the returned next_state"
        )

    def test_next_state_matches_step(self):
        """
        For deterministic transitions, _sample_next_state_from_cached_probs()
        must produce the same next_state as step().
        """
        harness, env = self._make_harness()
        state = env.get_state()
        num_agents = len(env.agents)

        actions = [Actions.left] + [Actions.still] * (num_agents - 1)
        trans_probs = env.transition_probabilities(state, actions)
        assert trans_probs is not None and len(trans_probs) == 1, (
            "Rotation action should produce a single deterministic outcome"
        )

        transition_probs_by_action = {0: trans_probs}

        # Get next_state via _sample_next_state_from_cached_probs
        env.set_state(state)
        sampled_next = harness._sample_next_state_from_cached_probs(
            state, (0,), transition_probs_by_action
        )

        # Get next_state via step()
        env.set_state(state)
        env.step(actions)
        step_next = env.get_state()

        assert sampled_next == step_next, (
            "Sampled next_state should match step() for deterministic transitions"
        )

    def test_missing_action_idx_raises_key_error(self):
        """
        _sample_next_state_from_cached_probs() must raise KeyError when
        the action_idx is not in transition_probs_by_action.
        """
        harness, env = self._make_harness()
        state = env.get_state()

        # Empty transition_probs_by_action — action_idx 0 is missing
        with pytest.raises(KeyError, match="action_idx 0"):
            harness._sample_next_state_from_cached_probs(
                state, (0,), {}
            )

    def test_empty_trans_probs_returns_current_state(self):
        """
        When trans_probs is empty (terminal/no-transition), the method
        should return the current state without calling set_state().
        """
        harness, env = self._make_harness()
        state = env.get_state()

        # action_idx 0 maps to empty list (terminal)
        transition_probs_by_action = {0: []}

        result = harness._sample_next_state_from_cached_probs(
            state, (0,), transition_probs_by_action
        )

        assert result is state, "Should return the same state object for empty trans_probs"

    def test_accumulator_clearing_pattern(self):
        """
        Verify the accumulator clearing pattern used by collect_transition()
        works correctly: after _sample_next_state_from_cached_probs, the
        accumulators should be clearable via the same loop used in the trainer.

        Note: This tests the clearing mechanism in isolation, not via
        collect_transition() itself (which requires full trainer setup).
        The clearing is needed because the cached-prob path never executes
        the transition, so accumulators may contain stale data from precompute.
        """
        harness, env = self._make_harness()
        state = env.get_state()
        num_agents = len(env.agents)

        # Simulate accumulators having stale data from precompute
        env.stumbled_cells = {(1, 1), (2, 2)}
        env.magic_wall_entered_cells = {(3, 3)}

        # Build transition_probs for a deterministic action
        actions = [Actions.left] + [Actions.still] * (num_agents - 1)
        trans_probs = env.transition_probabilities(state, actions)
        assert trans_probs is not None
        transition_probs_by_action = {0: trans_probs}

        # Call _sample_next_state_from_cached_probs (this doesn't clear accumulators)
        env.set_state(state)
        harness._sample_next_state_from_cached_probs(
            state, (0,), transition_probs_by_action
        )

        # Apply the same clearing pattern used by collect_transition()
        for attr_name in ("stumbled_cells", "magic_wall_entered_cells"):
            acc = getattr(env, attr_name, None)
            if acc is not None and hasattr(acc, "clear"):
                acc.clear()

        assert len(env.stumbled_cells) == 0, "stumbled_cells should be cleared"
        assert len(env.magic_wall_entered_cells) == 0, "magic_wall_entered_cells should be cleared"

    def test_accumulators_grow_without_clearing(self):
        """
        Verify that transition_probabilities() can add to accumulators, so
        without explicit clearing they would grow unbounded.
        """
        env = CollectGame4HEnv10x10N2()
        env.reset()
        state = env.get_state()
        num_agents = len(env.agents)

        # Set some initial data in accumulators
        env.stumbled_cells = {(99, 99)}

        # Call transition_probabilities (which may modify env internally)
        actions = [Actions.forward] * num_agents
        env.transition_probabilities(state, actions)

        # The set should still have our sentinel value since tp restores state
        # but the point is it's not cleared to empty by tp itself
        # After a cached-prob step without clearing, old data persists
        assert (99, 99) in env.stumbled_cells, (
            "Accumulators should retain old data when not explicitly cleared"
        )
