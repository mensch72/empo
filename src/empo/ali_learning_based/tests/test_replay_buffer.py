"""
Tests for the replay buffer.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_replay_buffer.py -v
"""

import pytest
import torch

from empo.ali_learning_based.replay_buffer import ReplayBuffer, Batch


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

STATE_DIM = 10
GOAL_DIM = 4


def make_transition(i: int):
    """Create a transition where all values are derived from `i` for easy checking."""
    return dict(
        state=torch.full((STATE_DIM,), float(i)),
        goal=torch.full((GOAL_DIM,), float(i * 10)),
        action=i % 6,
        reward=float(i) / 100.0,
        next_state=torch.full((STATE_DIM,), float(i + 1)),
        done=(i % 5 == 0),
        goal_reward=1.0 if i % 3 == 0 else 0.0,
    )


# -----------------------------------------------------------------------
# Basic operations
# -----------------------------------------------------------------------

class TestBasicOps:

    def test_empty_buffer(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        assert len(buf) == 0

    def test_push_increases_size(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        t = make_transition(0)
        buf.push(**t)
        assert len(buf) == 1

    def test_push_multiple(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(10):
            buf.push(**make_transition(i))
        assert len(buf) == 10

    def test_push_to_capacity(self):
        buf = ReplayBuffer(capacity=5, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(5):
            buf.push(**make_transition(i))
        assert len(buf) == 5

    def test_push_wraps_around(self):
        """After exceeding capacity, size stays at capacity."""
        buf = ReplayBuffer(capacity=5, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(8):
            buf.push(**make_transition(i))
        assert len(buf) == 5

    def test_wraparound_overwrites_oldest(self):
        """After wrapping, the oldest data is gone and newest is present."""
        buf = ReplayBuffer(capacity=3, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(5):
            buf.push(**make_transition(i))
        # Buffer should contain transitions 2, 3, 4 (oldest 0, 1 overwritten)
        # Position wraps: 5 % 3 = 2, so buffer has [3, 4, 2] at indices [0, 1, 2]
        stored_values = {buf.states[j, 0].item() for j in range(3)}
        assert stored_values == {2.0, 3.0, 4.0}

    def test_clear(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(10):
            buf.push(**make_transition(i))
        buf.clear()
        assert len(buf) == 0


# -----------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------

class TestSampling:

    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(20):
            buf.push(**make_transition(i))

        batch = buf.sample(batch_size=8)
        assert batch.states.shape == (8, STATE_DIM)
        assert batch.goals.shape == (8, GOAL_DIM)
        assert batch.actions.shape == (8,)
        assert batch.rewards.shape == (8,)
        assert batch.next_states.shape == (8, STATE_DIM)
        assert batch.dones.shape == (8,)
        assert batch.goal_rewards.shape == (8,)

    def test_sample_dtypes(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        for i in range(10):
            buf.push(**make_transition(i))

        batch = buf.sample(4)
        assert batch.states.dtype == torch.float32
        assert batch.goals.dtype == torch.float32
        assert batch.actions.dtype == torch.int64
        assert batch.rewards.dtype == torch.float32
        assert batch.dones.dtype == torch.bool
        assert batch.goal_rewards.dtype == torch.float32

    def test_sample_values_correct(self):
        """Push one transition and sample it — values should match exactly."""
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        t = make_transition(7)
        buf.push(**t)

        batch = buf.sample(1)
        assert torch.equal(batch.states[0], t["state"])
        assert torch.equal(batch.goals[0], t["goal"])
        assert batch.actions[0].item() == t["action"]
        assert abs(batch.rewards[0].item() - t["reward"]) < 1e-6
        assert torch.equal(batch.next_states[0], t["next_state"])
        assert batch.dones[0].item() == t["done"]
        assert abs(batch.goal_rewards[0].item() - t["goal_reward"]) < 1e-6

    def test_sample_only_from_stored(self):
        """Sampled indices must be within [0, size)."""
        buf = ReplayBuffer(capacity=1000, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        # Push only 3 transitions with distinct state values
        for i in range(3):
            buf.push(**make_transition(i))

        # Sample many times — all sampled states should be from {0, 1, 2}
        batch = buf.sample(100)
        unique_vals = batch.states[:, 0].unique()
        assert all(v.item() in {0.0, 1.0, 2.0} for v in unique_vals)

    def test_sample_empty_raises(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        with pytest.raises(ValueError):
            buf.sample(1)

    def test_sample_larger_than_buffer(self):
        """Sampling more than size should work (with replacement)."""
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        buf.push(**make_transition(0))
        batch = buf.sample(50)
        assert batch.states.shape == (50, STATE_DIM)

    def test_batch_is_namedtuple(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)
        buf.push(**make_transition(0))
        batch = buf.sample(1)
        assert isinstance(batch, Batch)
        # Access by name
        _ = batch.states
        _ = batch.rewards


# -----------------------------------------------------------------------
# Batch push
# -----------------------------------------------------------------------

class TestBatchPush:

    def test_push_batch_basic(self):
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)

        n = 5
        states = torch.arange(n).unsqueeze(1).expand(n, STATE_DIM).float()
        goals = torch.zeros(n, GOAL_DIM)
        actions = torch.arange(n)
        rewards = torch.arange(n).float()
        next_states = torch.arange(1, n + 1).unsqueeze(1).expand(n, STATE_DIM).float()
        dones = torch.zeros(n, dtype=torch.bool)
        goal_rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])

        buf.push_batch(states, goals, actions, rewards, next_states, dones, goal_rewards)
        assert len(buf) == 5

    def test_push_batch_values(self):
        """Values stored via push_batch should be retrievable."""
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)

        n = 3
        states = torch.arange(n).unsqueeze(1).expand(n, STATE_DIM).float()
        goals = torch.ones(n, GOAL_DIM) * 99
        actions = torch.tensor([0, 1, 2])
        rewards = torch.tensor([0.1, 0.2, 0.3])
        next_states = states + 1
        dones = torch.tensor([False, False, True])
        goal_rewards = torch.tensor([0.0, 0.0, 1.0])

        buf.push_batch(states, goals, actions, rewards, next_states, dones, goal_rewards)

        # Check index 1
        assert buf.states[1, 0].item() == 1.0
        assert buf.rewards[1].item() == pytest.approx(0.2)
        assert buf.dones[2].item() == True
        assert buf.goal_rewards[2].item() == pytest.approx(1.0)
        assert buf.goal_rewards[0].item() == pytest.approx(0.0)

    def test_push_batch_wrap(self):
        """push_batch that wraps around the ring boundary."""
        buf = ReplayBuffer(capacity=5, state_dim=STATE_DIM, goal_dim=GOAL_DIM)

        # Fill 3 slots
        for i in range(3):
            buf.push(**make_transition(i))
        assert len(buf) == 3

        # Push 4 more (3 + 4 = 7 > capacity 5, must wrap)
        n = 4
        states = torch.arange(10, 10 + n).unsqueeze(1).expand(n, STATE_DIM).float()
        goals = torch.zeros(n, GOAL_DIM)
        actions = torch.zeros(n, dtype=torch.long)
        rewards = torch.zeros(n)
        next_states = torch.zeros(n, STATE_DIM)
        dones = torch.zeros(n, dtype=torch.bool)

        buf.push_batch(states, goals, actions, rewards, next_states, dones)
        assert len(buf) == 5

        # Buffer should contain the latest 5 transitions
        stored = {buf.states[j, 0].item() for j in range(5)}
        # Last 5 are: transition 2 (from push), then 10, 11, 12, 13 (from batch)
        assert stored == {2.0, 10.0, 11.0, 12.0, 13.0}

    def test_push_batch_larger_than_capacity(self):
        """Pushing more items than capacity keeps only the last `capacity`."""
        buf = ReplayBuffer(capacity=3, state_dim=STATE_DIM, goal_dim=GOAL_DIM)

        n = 7
        states = torch.arange(n).unsqueeze(1).expand(n, STATE_DIM).float()
        goals = torch.zeros(n, GOAL_DIM)
        actions = torch.zeros(n, dtype=torch.long)
        rewards = torch.zeros(n)
        next_states = torch.zeros(n, STATE_DIM)
        dones = torch.zeros(n, dtype=torch.bool)

        buf.push_batch(states, goals, actions, rewards, next_states, dones)
        assert len(buf) == 3

        stored = {buf.states[j, 0].item() for j in range(3)}
        assert stored == {4.0, 5.0, 6.0}

    def test_push_batch_then_sample(self):
        """End-to-end: push_batch followed by sample."""
        buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, goal_dim=GOAL_DIM)

        n = 20
        states = torch.randn(n, STATE_DIM)
        goals = torch.randn(n, GOAL_DIM)
        actions = torch.randint(0, 6, (n,))
        rewards = torch.randn(n)
        next_states = torch.randn(n, STATE_DIM)
        dones = torch.randint(0, 2, (n,)).bool()

        buf.push_batch(states, goals, actions, rewards, next_states, dones)
        batch = buf.sample(8)
        assert batch.states.shape == (8, STATE_DIM)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
