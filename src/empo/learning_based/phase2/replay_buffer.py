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
        episode_id: Identifier for the rollout episode this transition belongs to.
        env_step_index: Position of this transition within its replay-linked rollout segment.
        search_policy: Optional root visit-count distribution from MCTS acting.
        search_value: Optional root state-value estimate from MCTS acting.
        search_action_value: Optional per-action root value estimates from MCTS acting.
        insertion_training_step: Learner training_step when replay stored the transition.
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
    episode_id: Optional[Any] = None
    env_step_index: Optional[int] = None
    search_policy: Optional[Tuple[float, ...]] = None
    search_value: Optional[float] = None
    search_action_value: Optional[Tuple[float, ...]] = None
    insertion_training_step: Optional[int] = None


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
        self._episode_transitions: Dict[Any, Dict[int, Phase2Transition]] = {}
        self._episode_terminal_indices: Dict[Any, int] = {}
    
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
        episode_id: Optional[Any] = None,
        env_step_index: Optional[int] = None,
        search_policy: Optional[Tuple[float, ...]] = None,
        search_value: Optional[float] = None,
        search_action_value: Optional[Tuple[float, ...]] = None,
        insertion_training_step: Optional[int] = None,
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
            episode_id: Optional rollout episode identifier.
            env_step_index: Optional env_step position within the replay-linked rollout segment.
            search_policy: Optional MCTS root policy stored with the transition.
            search_value: Optional MCTS root value estimate stored with the transition.
            search_action_value: Optional MCTS root per-action values stored with the transition.
            insertion_training_step: Optional learner training_step when replay stored
                the transition.
        """
        if (
            (episode_id is None and env_step_index is not None)
            or (episode_id is not None and env_step_index is None)
        ):
            raise ValueError(
                "episode_id and env_step_index must both be provided or both be None. "
                f"Got episode_id={episode_id!r}, env_step_index={env_step_index!r}."
            )

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
            terminal=terminal,
            episode_id=episode_id,
            env_step_index=env_step_index,
            search_policy=search_policy,
            search_value=search_value,
            search_action_value=search_action_value,
            insertion_training_step=insertion_training_step,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self._remove_episode_reference(self.buffer[self.position])
            self.buffer[self.position] = transition

        self._index_episode_transition(transition)
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
        *,
        max_age_training_steps: Optional[int] = None,
        current_training_step: Optional[int] = None,
    ) -> List[Phase2Transition]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            max_age_training_steps: Optional replay-age cutoff measured in learner
                training_steps.
            current_training_step: Current learner training_step for computing age.
        
        Returns:
            List of Phase2Transition objects.
        """
        population = self.buffer
        sample_size = min(batch_size, len(population))

        if max_age_training_steps is not None:
            if current_training_step is None:
                raise ValueError(
                    "current_training_step is required when max_age_training_steps "
                    "is provided."
                )

            fresh_population = [
                transition
                for transition in population
                if transition.insertion_training_step is not None
                and current_training_step - transition.insertion_training_step
                <= max_age_training_steps
            ]
            if len(fresh_population) >= sample_size:
                population = fresh_population
                sample_size = min(batch_size, len(population))

        return random.sample(population, sample_size)
    
    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.buffer)

    def get_episode_transition(
        self,
        episode_id: Any,
        env_step_index: int,
    ) -> Optional[Phase2Transition]:
        """Get a specific transition from an episode-aware replay record."""
        return self._episode_transitions.get(episode_id, {}).get(env_step_index)

    def get_episode_terminal_index(self, episode_id: Any) -> Optional[int]:
        """Get the cached terminal env_step_index for an episode, if present."""
        return self._episode_terminal_indices.get(episode_id)

    def get_episode_suffix(
        self,
        transition: Phase2Transition,
        horizon: Optional[int] = None,
    ) -> List[Phase2Transition]:
        """
        Recover an ordered suffix starting at the given transition.

        Args:
            transition: Transition whose episode suffix should be recovered.
            horizon: Optional maximum number of future env_steps to include beyond
                the current transition. None returns the remainder of the stored
                episode segment. A value of 0 returns only the current transition.
        """
        if transition.episode_id is None or transition.env_step_index is None:
            return [transition]
        if horizon is not None and horizon < 0:
            raise ValueError(f"horizon must be >= 0 or None, got {horizon}.")

        episode = self._episode_transitions.get(transition.episode_id)
        if not episode:
            return [transition]

        terminal_index = self._episode_terminal_indices.get(
            transition.episode_id,
            max(episode.keys()),
        )
        if horizon is None:
            end_index = terminal_index
        else:
            end_index = min(transition.env_step_index + horizon, terminal_index)

        suffix: List[Phase2Transition] = []
        for step_index in range(transition.env_step_index, end_index + 1):
            suffix_transition = episode.get(step_index)
            if suffix_transition is None:
                break
            suffix.append(suffix_transition)
            if suffix_transition.terminal:
                break
        return suffix or [transition]
    
    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self.buffer = []
        self.position = 0
        self._episode_transitions = {}
        self._episode_terminal_indices = {}

    def _index_episode_transition(self, transition: Phase2Transition) -> None:
        """Index a transition for episode-aware suffix lookup."""
        if transition.episode_id is None or transition.env_step_index is None:
            return

        episode = self._episode_transitions.setdefault(transition.episode_id, {})
        episode[transition.env_step_index] = transition
        if transition.terminal:
            self._episode_terminal_indices[transition.episode_id] = transition.env_step_index

    def _remove_episode_reference(self, transition: Phase2Transition) -> None:
        """Remove a transition from the episode index when the ring buffer overwrites it."""
        if transition.episode_id is None or transition.env_step_index is None:
            return

        episode = self._episode_transitions.get(transition.episode_id)
        if not episode:
            return

        episode.pop(transition.env_step_index, None)
        if not episode:
            self._episode_transitions.pop(transition.episode_id, None)
            self._episode_terminal_indices.pop(transition.episode_id, None)
            return

        if self._episode_terminal_indices.get(transition.episode_id) == transition.env_step_index:
            self._episode_terminal_indices.pop(transition.episode_id, None)
            remaining_terminal_indices = [
                step_index
                for step_index, remaining_transition in episode.items()
                if remaining_transition.terminal
            ]
            if remaining_terminal_indices:
                self._episode_terminal_indices[transition.episode_id] = max(remaining_terminal_indices)
