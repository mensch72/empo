"""
Replay buffer for Phase 2 training.

Stores transitions that include robot actions, human actions, and goal profiles.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Phase2Transition:
    """
    A transition for Phase 2 training.
    
    Attributes:
        state: The current state s.
        robot_action: Tuple of actions, one per robot (a_r).
        goals: Dict mapping human index to their goal {h: g_h}.
        goal_weights: Dict mapping human index to their goal's importance weight {h: w_h}.
            Used for proper X_h target computation with importance sampling.
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
        next_compact_features: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None
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
            next_compact_features=next_compact_features
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
