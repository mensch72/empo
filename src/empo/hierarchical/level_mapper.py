"""
Level Mapper abstract base class for hierarchical world models.

A LevelMapper connects adjacent levels in a hierarchical world model,
providing state aggregation, agent grouping, action feasibility checks,
and control transfer logic between a coarser level ℓ and a finer level ℓ+1.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from empo.world_model import WorldModel


class LevelMapper(ABC):
    """Maps between adjacent levels ℓ (coarser) and ℓ+1 (finer) in a hierarchy.

    Connects a coarser world model M^ℓ to a finer world model M^{ℓ+1} by providing
    state aggregation, agent grouping, action feasibility, and control transfer logic.

    Attributes:
        coarse_model: The coarser (macro) world model M^ℓ.
        fine_model: The finer (micro) world model M^{ℓ+1}.
    """

    def __init__(self, coarse_model: WorldModel, fine_model: WorldModel):
        self.coarse_model = coarse_model
        self.fine_model = fine_model

    @abstractmethod
    def super_state(self, fine_state: Any) -> Any:
        """Map a fine-level state s^{ℓ+1} to the coarse-level state s^ℓ containing it.

        This defines the partition: s^{ℓ+1} ∈ s^ℓ iff super_state(s^{ℓ+1}) == s^ℓ.

        Args:
            fine_state: A state from M^{ℓ+1}.

        Returns:
            The corresponding state from M^ℓ.
        """

    @abstractmethod
    def super_agent(self, fine_agent_index: int) -> int:
        """Map a fine-level agent index to the coarse-level agent (group) index.

        Defines the partition of agents into groups: agent i^{ℓ+1} belongs to
        group j^ℓ iff super_agent(i) == j.

        Args:
            fine_agent_index: An agent index in M^{ℓ+1}.

        Returns:
            The corresponding agent (group) index in M^ℓ.
        """

    @abstractmethod
    def is_feasible(
        self,
        coarse_action: Tuple[int, ...],
        fine_state: Any,
        fine_action: Tuple[int, ...]
    ) -> bool:
        """Check if a fine-level action is compatible with the current coarse-level action.

        An action a^{ℓ+1} is feasible if it does not contradict the plan specified
        by a^ℓ. For example, if a^ℓ says "walk to cell X", then a^{ℓ+1} should not
        walk in the opposite direction.

        Args:
            coarse_action: The current action profile from M^ℓ.
            fine_state: The current state in M^{ℓ+1}.
            fine_action: The proposed action profile in M^{ℓ+1}.

        Returns:
            True if fine_action is compatible with coarse_action in fine_state.
        """

    @abstractmethod
    def is_abort(
        self,
        coarse_action: Tuple[int, ...],
        fine_state: Any,
        fine_action: Tuple[int, ...]
    ) -> bool:
        """Check if a fine-level action constitutes aborting the coarse-level plan.

        Aborting means the fine-level agent explicitly chooses to stop pursuing
        the coarse-level action (e.g., by passing/staying still).

        Args:
            coarse_action: The current action profile from M^ℓ.
            fine_state: The current state in M^{ℓ+1}.
            fine_action: The proposed action profile in M^{ℓ+1}.

        Returns:
            True if fine_action is considered aborting coarse_action.
        """

    @abstractmethod
    def return_control(
        self,
        coarse_action: Tuple[int, ...],
        fine_state: Any,
        fine_action: Tuple[int, ...],
        fine_successor_state: Any
    ) -> bool:
        """Check if control should return to the coarse level after a fine-level transition.

        Control returns when:
        - The coarse-level action is achieved (agent reached target cell, object toggled, etc.)
        - The coarse-level action becomes unachievable (target moved away, path blocked, etc.)
        - The fine-level action was an abort

        Args:
            coarse_action: The current action profile from M^ℓ.
            fine_state: The state in M^{ℓ+1} before the transition.
            fine_action: The action taken in M^{ℓ+1}.
            fine_successor_state: The state in M^{ℓ+1} after the transition.

        Returns:
            True if control should be returned to level ℓ.
        """
