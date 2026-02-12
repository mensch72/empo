"""
Replay buffer for storing and sampling transitions.

Stores transitions as pre-allocated tensors in a fixed-capacity ring buffer.
Used by both Phase 1 (human Q-learning) and Phase 2 (robot Q-learning).

Design choices:
    - Pre-allocated tensors: avoids repeated memory allocation during training.
      You specify state_dim and goal_dim at creation, and the buffer allocates
      all memory upfront.
    - Ring buffer: when full, oldest transitions get overwritten. No need to
      manually manage eviction.
    - Uniform sampling: each stored transition is equally likely to be sampled.
      Prioritized replay can be added later if needed.
    - Batch is a NamedTuple: fields are accessed by name (batch.states,
      batch.actions, etc.) for readability.
"""

import torch
from typing import NamedTuple


class Batch(NamedTuple):
    """A batch of transitions sampled from the replay buffer."""
    states: torch.Tensor       # (batch_size, state_dim)
    goals: torch.Tensor        # (batch_size, goal_dim)
    actions: torch.Tensor      # (batch_size,) int64
    rewards: torch.Tensor      # (batch_size,) float32 — shaped reward (may include bonuses)
    next_states: torch.Tensor  # (batch_size, state_dim)
    dones: torch.Tensor        # (batch_size,) bool
    goal_rewards: torch.Tensor # (batch_size,) float32 — pure goal achievement (0/1)


class ReplayBuffer:
    """
    Fixed-capacity ring buffer storing transitions as tensors.

    Each transition has 6 fields:
        state      (state_dim,)    — encoded state
        goal       (goal_dim,)     — encoded goal
        action     scalar int      — action taken
        reward     scalar float    — immediate reward
        next_state (state_dim,)    — encoded next state
        done       scalar bool     — episode ended?

    Example:
        buf = ReplayBuffer(capacity=10000, state_dim=291, goal_dim=4)

        # During rollout:
        buf.push(state_enc, goal_enc, action=2, reward=0.0, next_state_enc, done=False)

        # During training:
        batch = buf.sample(batch_size=64)
        loss = compute_loss(batch.states, batch.actions, batch.rewards, ...)

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimension of encoded state vectors.
        goal_dim: Dimension of encoded goal vectors.
    """

    def __init__(self, capacity: int, state_dim: int, goal_dim: int,
                 device: torch.device = None):
        self.capacity = capacity
        self._size = 0        # number of transitions currently stored
        self._pos = 0         # next write index (wraps around)
        self.device = device or torch.device("cpu")

        # Pre-allocate all storage on CPU (data comes from CPU encoders)
        self.states = torch.zeros(capacity, state_dim)
        self.goals = torch.zeros(capacity, goal_dim)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        self.goal_rewards = torch.zeros(capacity, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def push(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        goal_reward: float = 0.0,
    ) -> None:
        """
        Store a single transition.

        If the buffer is full, the oldest transition is overwritten.
        """
        i = self._pos
        self.states[i] = state
        self.goals[i] = goal
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.goal_rewards[i] = goal_reward

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push_batch(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        goal_rewards: torch.Tensor = None,
    ) -> None:
        """
        Store multiple transitions at once.

        More efficient than calling push() in a loop because it uses
        tensor slicing instead of per-element Python overhead.

        All arguments must have the same first dimension (batch size).
        """
        n = states.shape[0]
        if goal_rewards is None:
            goal_rewards = rewards  # fallback: treat rewards as goal rewards

        if n >= self.capacity:
            # More data than capacity: keep only the last `capacity` items
            start = n - self.capacity
            self.states[:] = states[start:]
            self.goals[:] = goals[start:]
            self.actions[:] = actions[start:]
            self.rewards[:] = rewards[start:]
            self.next_states[:] = next_states[start:]
            self.dones[:] = dones[start:]
            self.goal_rewards[:] = goal_rewards[start:]
            self._pos = 0
            self._size = self.capacity
            return

        # How much fits before we wrap around?
        space_at_end = self.capacity - self._pos

        if n <= space_at_end:
            # Everything fits without wrapping
            s = self._pos
            self.states[s : s + n] = states
            self.goals[s : s + n] = goals
            self.actions[s : s + n] = actions
            self.rewards[s : s + n] = rewards
            self.next_states[s : s + n] = next_states
            self.dones[s : s + n] = dones
            self.goal_rewards[s : s + n] = goal_rewards
        else:
            # Split across the wrap boundary
            first = space_at_end
            second = n - first
            s = self._pos

            self.states[s:] = states[:first]
            self.states[:second] = states[first:]

            self.goals[s:] = goals[:first]
            self.goals[:second] = goals[first:]

            self.actions[s:] = actions[:first]
            self.actions[:second] = actions[first:]

            self.rewards[s:] = rewards[:first]
            self.rewards[:second] = rewards[first:]

            self.next_states[s:] = next_states[:first]
            self.next_states[:second] = next_states[first:]

            self.dones[s:] = dones[:first]
            self.dones[:second] = dones[first:]

            self.goal_rewards[s:] = goal_rewards[:first]
            self.goal_rewards[:second] = goal_rewards[first:]

        self._pos = (self._pos + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """
        Sample a random batch of transitions (uniform, with replacement).

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            A Batch namedtuple with all fields as tensors.

        Raises:
            ValueError: If the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        indices = torch.randint(0, self._size, (batch_size,))
        return Batch(
            states=self.states[indices].to(self.device),
            goals=self.goals[indices].to(self.device),
            actions=self.actions[indices].to(self.device),
            rewards=self.rewards[indices].to(self.device),
            next_states=self.next_states[indices].to(self.device),
            dones=self.dones[indices].to(self.device),
            goal_rewards=self.goal_rewards[indices].to(self.device),
        )

    def clear(self) -> None:
        """Remove all transitions (memory stays allocated)."""
        self._size = 0
        self._pos = 0
