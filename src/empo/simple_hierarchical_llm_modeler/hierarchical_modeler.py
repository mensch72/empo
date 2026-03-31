"""
Hierarchical modeler: extends the tree builder with higher-level context
awareness, success/failure detection, and consequence matching.

This module produces :class:`~empo.hierarchical.HierarchicalWorldModel`
instances by building two (or more) levels of NL world models and connecting
them with an :class:`NLLevelMapper`.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.level_mapper import LevelMapper
from empo.simple_hierarchical_llm_modeler.llm_connector import LLMConnector
from empo.simple_hierarchical_llm_modeler.nl_world_model import NLWorldModel
from empo.simple_hierarchical_llm_modeler.prompts import (
    hierarchical_status_prompt,
    match_consequence_prompt,
)
from empo.simple_hierarchical_llm_modeler.tree_builder import (
    _parse_json_object,
    build_tree,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchical status check
# ---------------------------------------------------------------------------


def check_hierarchical_status(
    llm: LLMConnector,
    higher_level_context: str,
    higher_level_action: str,
    initial_state: str,
    history: List[str],
) -> str:
    """Ask the LLM whether the current state is success / failure / in-progress.

    Returns:
        One of ``"success"``, ``"failure"``, or ``"still in progress"``.
    """
    prompt = hierarchical_status_prompt(
        higher_level_context, higher_level_action, initial_state, history
    )
    raw = llm.query(prompt)
    try:
        data = _parse_json_object(raw)
        status = data.get("status", "still in progress")
    except ValueError:
        status = "still in progress"

    normalised = status.strip().lower()
    if normalised in ("success", "failure", "still in progress"):
        return normalised
    return "still in progress"


def match_consequence(
    llm: LLMConnector,
    higher_level_context: str,
    higher_level_action: str,
    known_consequences: List[str],
    status: str,
    initial_state: str,
    history: List[str],
) -> Tuple[Optional[int], Optional[str]]:
    """Match current outcome to a known higher-level consequence.

    Returns:
        ``(match_index, new_description)`` where *match_index* is a 0-based
        index into *known_consequences* if there is a match (else ``None``),
        and *new_description* is a novel consequence string if no match
        (else ``None``).
    """
    prompt = match_consequence_prompt(
        higher_level_context,
        higher_level_action,
        known_consequences,
        status,
        initial_state,
        history,
    )
    raw = llm.query(prompt)
    try:
        data = _parse_json_object(raw)
    except ValueError:
        return None, f"Unmatched {status}"

    match_idx = data.get("match")
    new_cons = data.get("new_consequence")
    if match_idx is not None:
        # Convert 1-based to 0-based
        return int(match_idx) - 1, None
    return None, new_cons or f"Novel {status} outcome"


# ---------------------------------------------------------------------------
# NLLevelMapper
# ---------------------------------------------------------------------------


class NLLevelMapper(LevelMapper):
    """Minimal LevelMapper connecting two NL world models.

    Since NL world model states are history tuples, the coarse state is
    derived by extracting the prefix up to the matching depth.
    """

    def __init__(
        self,
        coarse_model: NLWorldModel,
        fine_model: NLWorldModel,
        coarse_action_desc: str = "",
    ) -> None:
        super().__init__(coarse_model, fine_model)
        self._coarse_action_desc = coarse_action_desc

    def super_state(self, fine_state: Any) -> Any:
        """Map fine state back to coarse state (empty tuple = coarse root)."""
        return ()

    def super_agent(self, fine_agent_index: int) -> int:
        return fine_agent_index

    def is_feasible(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        return True

    def is_abort(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        return False

    def return_control(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
        fine_successor_state: Any,
    ) -> bool:
        return False


# ---------------------------------------------------------------------------
# Two-level builder
# ---------------------------------------------------------------------------


def build_two_level_model(
    llm: LLMConnector,
    initial_state_description: str,
    *,
    coarse_n_steps: int = 1,
    fine_n_steps: int = 2,
    n_robotactions: int = 3,
    n_humansreactions: int = 3,
    n_consequences: int = 2,
) -> HierarchicalWorldModel:
    """Build a two-level :class:`HierarchicalWorldModel` from NL descriptions.

    1. Build a **coarse** (level-0) tree with ``coarse_n_steps`` cycles.
    2. For each coarse-level robot action at the root, build a **fine**
       (level-1) tree using the coarse action as ``higher_level_context``.
    3. Wrap both in :class:`NLWorldModel` instances and connect them with an
       :class:`NLLevelMapper`.

    Args:
        llm: LLM connector.
        initial_state_description: Starting situation in natural language.
        coarse_n_steps: Depth of the coarse tree.
        fine_n_steps: Depth of each fine tree.
        n_robotactions: Robot actions per expansion.
        n_humansreactions: Human reactions per expansion.
        n_consequences: Consequences per expansion.

    Returns:
        A :class:`HierarchicalWorldModel` with two levels.
    """
    # --- Level 0 (coarse) ---
    coarse_tree = build_tree(
        llm=llm,
        initial_state_description=initial_state_description,
        n_steps=coarse_n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
    )
    coarse_model = NLWorldModel.from_tree(coarse_tree, initial_state_description)

    # --- Level 1 (fine) – one per coarse robot action ---
    # For simplicity, we build the fine model for the *first* coarse robot
    # action at the root. A full implementation would build one per action and
    # merge them, but that is left for future work.
    coarse_ra_labels = coarse_model.robot_action_labels(coarse_model._initial_state)
    if coarse_ra_labels:
        first_ra = coarse_ra_labels[0]
    else:
        first_ra = "proceed with the plan"

    higher_ctx = (
        f"{initial_state_description}. "
        f"Higher-level decision: The robot has chosen to: {first_ra}"
    )
    fine_tree = build_tree(
        llm=llm,
        initial_state_description=initial_state_description,
        n_steps=fine_n_steps,
        n_robotactions=n_robotactions,
        n_humansreactions=n_humansreactions,
        n_consequences=n_consequences,
        higher_level_context=higher_ctx,
    )
    fine_model = NLWorldModel.from_tree(fine_tree, initial_state_description)

    mapper = NLLevelMapper(coarse_model, fine_model, coarse_action_desc=first_ra)
    return HierarchicalWorldModel(levels=[coarse_model, fine_model], mappers=[mapper])
