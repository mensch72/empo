"""
Replay buffer for Phase 2 training.

Stores transitions that include robot actions, human actions, and goal profiles.
Includes both uniform and prioritised replay buffer implementations.

Prioritised replay (PER) samples transitions proportional to their priority,
giving more frequent training on informative transitions (e.g., where X_h
changed significantly). See Schaul et al. (2016) for the general PER approach.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class Phase2Transition:
    """
    A transition for Phase 2 training.
    
    Attributes:
        state: The current state s.
        robot_action: Tuple of actions, one per robot (a_r).
        goals: Dict mapping human index to their goal {h: g_h}.
        goal_weights: Dict mapping human index to their goal's weight {h: w_h}.
            Used for X_h target computation: X_h = E[weight * V_h^e].
        human_actions: List of human actions (a_H).
        next_state: The successor state s'.
        transition_probs_by_action: Optional pre-computed transition probabilities
            for model-based targets. Maps robot_action_index -> [(prob, next_state), ...].
            When provided, avoids re-computing transition_probabilities during training.
        compact_features: Optional pre-computed compact features for current state.
            Tuple of (global_features, agent_features, interactive_features, compressed_grid).
            The compressed_grid is a (H, W) int32 tensor that encodes all grid information.
            When provided, avoids expensive tensorization during training.
        next_compact_features: Optional pre-computed compact features for next state.
        terminal: Whether this transition ends the episode (next_state has no continuation).
            When True, the V_h^e TD target should not bootstrap from next_state.
    """
    state: Any
    robot_action: Tuple[int, ...]
    goals: Dict[int, Any]
    goal_weights: Dict[int, float]
    human_actions: List[int]
    next_state: Any
    transition_probs_by_action: Optional[Dict[int, List[Tuple[float, Any]]]] = None
    compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    next_compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
    terminal: bool = False


class Phase2ReplayBuffer:
    """
    Replay buffer for Phase 2 experience replay.
    
    Stores Phase2Transition objects containing:
    - state s
    - robot action tuple a_r
    - goal profile g = {h: g_h}
    - human actions a_H
    - next state s'
    
    This is more complex than Phase 1's buffer because Phase 2 requires
    the full joint action and goal profile for training the various networks.
    
    Args:
        capacity: Maximum number of transitions to store.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Phase2Transition] = []
        self.position = 0
    
    def push(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        goals: Dict[int, Any],
        goal_weights: Dict[int, float],
        human_actions: List[int],
        next_state: Any,
        transition_probs_by_action: Optional[Dict[int, List[Tuple[float, Any]]]] = None,
        compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        next_compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        terminal: bool = False
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state.
            robot_action: Tuple of robot actions.
            goals: Dict mapping human index to goal.
            goal_weights: Dict mapping human index to goal weight.
            human_actions: List of human actions.
            next_state: Next state.
            transition_probs_by_action: Optional pre-computed transition probabilities.
            compact_features: Optional pre-computed (global, agent, interactive, compressed_grid) tensors for state.
            next_compact_features: Optional pre-computed (global, agent, interactive, compressed_grid) tensors for next_state.
            terminal: Whether this transition ends the episode.
        """
        transition = Phase2Transition(
            state=state,
            robot_action=robot_action,
            goals=goals,
            goal_weights=goal_weights,
            human_actions=human_actions,
            next_state=next_state,
            transition_probs_by_action=transition_probs_by_action,
            compact_features=compact_features,
            next_compact_features=next_compact_features,
            terminal=terminal
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Phase2Transition]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            List of Phase2Transition objects.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer = []
        self.position = 0


class PrioritizedBatch(NamedTuple):
    """
    Result of sampling from a prioritised replay buffer.
    
    Attributes:
        transitions: List of sampled Phase2Transition objects.
        indices: List of buffer indices for each sampled transition.
        is_weights: Importance sampling weights to correct for non-uniform sampling.
            Shape: numpy array of length len(transitions), values in (0, 1].
    """
    transitions: List[Phase2Transition]
    indices: List[int]
    is_weights: np.ndarray


class SumTree:
    """
    Sum tree for O(log n) proportional priority sampling.
    
    A binary tree where each leaf stores a priority value and internal nodes
    store the sum of their children. This allows efficient:
    - Update of a single priority: O(log n)
    - Proportional sampling: O(log n)
    - Total priority sum: O(1)
    
    Args:
        capacity: Maximum number of leaf nodes (transitions).
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree has 2*capacity - 1 nodes (internal + leaves)
        # Leaves are at indices [capacity-1, 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._min_cache_valid = False
        self._min_cache_value = float('inf')
    
    def _propagate(self, idx: int, change: float) -> None:
        """Propagate a priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def update(self, leaf_idx: int, priority: float) -> None:
        """
        Update the priority of a leaf node.
        
        Args:
            leaf_idx: Index of the leaf (0 to capacity-1).
            priority: New priority value (must be > 0).
        """
        tree_idx = leaf_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self._min_cache_valid = False
    
    def get(self, leaf_idx: int) -> float:
        """Get the priority of a leaf node."""
        return self.tree[leaf_idx + self.capacity - 1]
    
    def total(self) -> float:
        """Return the total sum of all priorities (root node)."""
        return self.tree[0]
    
    def min(self, size: int) -> float:
        """
        Return the minimum priority among the first `size` leaves.
        
        Args:
            size: Number of active leaves to consider.
        """
        # Cache assumes min() is always called with the same size between
        # update() calls.  In practice this holds because sample() always
        # passes self.size, but callers must not mix different size values
        # without an intervening update().
        if self._min_cache_valid:
            return self._min_cache_value
        if size == 0:
            return 0.0
        leaves = self.tree[self.capacity - 1:self.capacity - 1 + size]
        self._min_cache_value = float(np.min(leaves))
        self._min_cache_valid = True
        return self._min_cache_value
    
    def sample(self, value: float) -> int:
        """
        Sample a leaf index proportional to priorities.
        
        Given a value in [0, total()), traverse the tree to find the leaf
        whose cumulative priority range contains this value.
        
        Args:
            value: Random value in [0, total()).
            
        Returns:
            Leaf index (0 to capacity-1).
        """
        idx = 0  # Start at root
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # Reached a leaf
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx
    
    def clear(self) -> None:
        """Reset all priorities to zero."""
        self.tree[:] = 0.0
        self._min_cache_valid = False
        self._min_cache_value = float('inf')


class PrioritizedPhase2ReplayBuffer:
    """
    Prioritised replay buffer for Phase 2 experience replay.
    
    Samples transitions proportionally to their priority using a sum tree.
    Transitions where X_h changed significantly get higher priority,
    as they are most informative for Q_r (and potentially other networks).
    
    New transitions receive max priority to ensure they are sampled at least once.
    After training, priorities are updated based on TD errors via update_priorities().
    
    Uses importance sampling (IS) weights to correct for the non-uniform sampling
    distribution, following Schaul et al. (2016) "Prioritized Experience Replay".
    
    Args:
        capacity: Maximum number of transitions to store.
        alpha: Priority exponent. 0 = uniform sampling, 1 = full prioritization.
        beta_start: Initial IS correction exponent. 0 = no correction, 1 = full correction.
        beta_end: Final IS correction exponent (beta is annealed to beta_end over training).
        epsilon: Small constant added to priorities to prevent zero probability.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        
        self.buffer: List[Optional[Phase2Transition]] = [None] * capacity
        self.tree = SumTree(capacity)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
    
    def push(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        goals: Dict[int, Any],
        goal_weights: Dict[int, float],
        human_actions: List[int],
        next_state: Any,
        transition_probs_by_action: Optional[Dict[int, List[Tuple[float, Any]]]] = None,
        compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        next_compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        terminal: bool = False,
        priority: Optional[float] = None
    ) -> None:
        """
        Add a transition to the buffer with priority.
        
        New transitions receive max_priority by default to ensure they are
        sampled at least once before their priority is updated.
        
        Args:
            state: Current state.
            robot_action: Tuple of robot actions.
            goals: Dict mapping human index to goal.
            goal_weights: Dict mapping human index to goal weight.
            human_actions: List of human actions.
            next_state: Next state.
            transition_probs_by_action: Optional pre-computed transition probabilities.
            compact_features: Optional pre-computed tensors for state.
            next_compact_features: Optional pre-computed tensors for next_state.
            terminal: Whether this transition ends the episode.
            priority: Optional explicit priority. If None, uses max_priority.
        """
        transition = Phase2Transition(
            state=state,
            robot_action=robot_action,
            goals=goals,
            goal_weights=goal_weights,
            human_actions=human_actions,
            next_state=next_state,
            transition_probs_by_action=transition_probs_by_action,
            compact_features=compact_features,
            next_compact_features=next_compact_features,
            terminal=terminal
        )
        
        # Use provided priority or max priority for new transitions.
        # abs() is defensive in case callers provide signed values.
        if priority is not None:
            p = (abs(priority) + self.epsilon) ** self.alpha
            # Keep max_priority consistent with explicitly provided priorities.
            self.max_priority = max(self.max_priority, p)
        else:
            # max_priority is already stored in alpha-scaled priority space.
            # New transitions should receive this value directly (PER convention).
            p = self.max_priority
        
        self.buffer[self.position] = transition
        self.tree.update(self.position, p)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None
    ) -> PrioritizedBatch:
        """
        Sample a batch of transitions proportional to their priorities.
        
        Args:
            batch_size: Number of transitions to sample.
            beta: IS correction exponent. If None, uses beta_start.
                Caller should anneal beta from beta_start to beta_end over training.
        
        Returns:
            PrioritizedBatch with transitions, indices, and IS weights.
        """
        if beta is None:
            beta = self.beta_start
        
        actual_batch = min(batch_size, self.size)
        indices = []
        transitions = []
        priorities = np.zeros(actual_batch, dtype=np.float64)
        
        # Stratified sampling: divide [0, total) into batch_size segments
        total = self.tree.total()
        if total <= 0:
            # Fallback to uniform if all priorities are zero
            valid_indices = list(range(self.size))
            chosen = random.sample(valid_indices, actual_batch)
            return PrioritizedBatch(
                transitions=[self.buffer[i] for i in chosen],
                indices=chosen,
                is_weights=np.ones(actual_batch, dtype=np.float32)
            )
        
        segment_size = total / actual_batch
        
        for i in range(actual_batch):
            low = segment_size * i
            high = segment_size * (i + 1)
            value = random.uniform(low, high)
            leaf_idx = self.tree.sample(value)
            
            # Ensure valid index
            leaf_idx = min(leaf_idx, self.size - 1)
            
            indices.append(leaf_idx)
            transitions.append(self.buffer[leaf_idx])
            priorities[i] = self.tree.get(leaf_idx)
        
        # Compute importance sampling weights
        # w_i = (N * P(i))^{-beta} / max_w
        # P(i) = p_i / sum(p)
        probs = priorities / total
        probs = np.clip(probs, 1e-10, None)  # Prevent division by zero
        
        # Compute IS weights normalized by max weight in batch
        min_prob = self.tree.min(self.size) / total
        min_prob = max(min_prob, 1e-10)
        max_weight = (self.size * min_prob) ** (-beta)
        
        is_weights = (self.size * probs) ** (-beta) / max_weight
        is_weights = is_weights.astype(np.float32)
        
        return PrioritizedBatch(
            transitions=transitions,
            indices=indices,
            is_weights=is_weights
        )
    
    def update_priorities(
        self, indices: Sequence[int], priorities: Union[Sequence[float], np.ndarray]
    ) -> None:
        """
        Update priorities for sampled transitions.
        
        Typically called after computing TD errors during training.
        
        Args:
            indices: Buffer indices (from PrioritizedBatch.indices).
            priorities: New priority values (e.g., |TD_error| for each transition).
                Accepts lists, tuples, or numpy arrays. Values are passed through
                abs() defensively in case callers provide signed errors.
        """
        for idx, priority in zip(indices, priorities):
            p = (abs(float(priority)) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)
    
    def get_beta(self, step: int, total_steps: int) -> float:
        """
        Compute annealed beta value for importance sampling correction.
        
        Beta is linearly annealed from beta_start to beta_end over training.
        
        Args:
            step: Current training step.
            total_steps: Total number of training steps.
            
        Returns:
            Current beta value.
        """
        if total_steps <= 0:
            return self.beta_end
        fraction = min(step / total_steps, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return self.size
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer = [None] * self.capacity
        self.tree.clear()
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
