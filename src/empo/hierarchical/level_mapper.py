"""
Level Mapper abstract base class for hierarchical world models.

A LevelMapper connects adjacent levels in a hierarchical world model,
providing state aggregation, agent grouping, action feasibility checks,
and control transfer logic between a coarser level l and a finer level l+1.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from empo.world_model import WorldModel


class LevelMapper(ABC):
    """Maps between adjacent levels l (coarser) and l+1 (finer) in a hierarchy.

    Connects a coarser world model M^l to a finer world model M^{l+1} by providing
    state aggregation, agent grouping, action feasibility, and control transfer logic.

    Attributes:
        coarse_model: The coarser (macro) world model M^l.
        fine_model: The finer (micro) world model M^{l+1}.
    """

    def __init__(self, coarse_model: WorldModel, fine_model: WorldModel):
        if not isinstance(coarse_model, WorldModel):
            raise TypeError(
                f"coarse_model must be a WorldModel instance, got {type(coarse_model)}"
            )
        if not isinstance(fine_model, WorldModel):
            raise TypeError(
                f"fine_model must be a WorldModel instance, got {type(fine_model)}"
            )
        self.coarse_model = coarse_model
        self.fine_model = fine_model

    @abstractmethod
    def super_state(self, fine_state: Any) -> Any:
        """Map a fine-level state s^{l+1} to the coarse-level state s^l containing it.

        This defines the partition: s^{l+1} ∈ s^l iff super_state(s^{l+1}) == s^l.

        Args:
            fine_state: A state from M^{l+1}.

        Returns:
            The corresponding state from M^l.
        """

    @abstractmethod
    def super_agent(self, fine_agent_index: int) -> int:
        """Map a fine-level agent index to the coarse-level agent (group) index.

        Defines the partition of agents into groups: agent i^{l+1} belongs to
        group j^l iff super_agent(i) == j.

        Args:
            fine_agent_index: An agent index in M^{l+1}.

        Returns:
            The corresponding agent (group) index in M^l.
        """

    @abstractmethod
    def is_feasible(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...]
    ) -> bool:
        """Check if a fine-level action profile is compatible with the current coarse-level one.

        An action profile a^{l+1} is feasible if it does not contradict the plan specified
        by a^l. For example, if a^l says "walk to cell X", then a^{l+1} should not
        walk in the opposite direction.

        Args:
            coarse_action_profile: The current action profile from M^l.
            fine_state: The current state in M^{l+1}.
            fine_action_profile: The proposed action profile in M^{l+1}.

        Returns:
            True if fine_action_profile is compatible with coarse_action_profile in fine_state.
        """

    @abstractmethod
    def is_abort(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...]
    ) -> bool:
        """Check if a fine-level action profile constitutes aborting the coarse-level plan.

        Aborting means the fine-level agent explicitly chooses to stop pursuing
        the coarse-level action profile (e.g., by passing/staying still).

        Args:
            coarse_action_profile: The current action profile from M^l.
            fine_state: The current state in M^{l+1}.
            fine_action_profile: The proposed action profile in M^{l+1}.

        Returns:
            True if fine_action_profile is considered aborting coarse_action_profile.
        """

    @abstractmethod
    def return_control(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
        fine_successor_state: Any
    ) -> bool:
        """Check if control should return to the coarse level after a fine-level transition.

        Control returns when:
        - The coarse-level action profile is achieved (agent reached target cell, object toggled, etc.)
        - The coarse-level action profile becomes unachievable (target moved away, path blocked, etc.)
        - The fine-level action profile was an abort

        Args:
            coarse_action_profile: The current action profile from M^l.
            fine_state: The state in M^{l+1} before the transition.
            fine_action_profile: The action profile taken in M^{l+1}.
            fine_successor_state: The state in M^{l+1} after the transition.

        Returns:
            True if control should be returned to level l.
        """
