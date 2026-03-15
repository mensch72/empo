"""Level mapper connecting MacroGridEnv (M^0) to MultiGridEnv (M^1).

Defines state aggregation (micro position → macro cell), agent mapping
(identity), action feasibility (fine actions compatible with coarse plan),
and control transfer (macro action achieved, failed, or aborted).
"""

from typing import Any, Tuple

from empo.hierarchical.level_mapper import LevelMapper
from empo.hierarchical.macro_grid_env import MACRO_PASS

# Direction vectors matching MultiGridEnv: 0=right, 1=down, 2=left, 3=up
_DIR_TO_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


class MultiGridLevelMapper(LevelMapper):
    """Level mapper connecting MacroGridEnv (M^0) to MultiGridEnv (M^1).

    Provides:

    - ``super_state()``: maps M^1 grid state to M^0 macro-state by
      abstracting agent positions to macro-cell indices and computing
      passage flags from the grid.
    - ``super_agent()``: identity mapping (one macro-agent per micro-agent).
    - ``is_feasible()``: rejects M^1 forward actions that would enter
      a macro-cell different from the coarse WALK target.
    - ``is_abort()``: treats M^1 'still' (action 0) as aborting the M^0
      plan for agents with non-PASS coarse actions.
    - ``return_control()``: returns control when an agent reaches the
      target macro-cell, or when the fine action was an abort.

    Attributes:
        macro_env: The MacroGridEnv (M^0) serving as the coarse model.
    """

    def __init__(self, macro_env: Any, micro_env: Any):
        """Construct mapper between macro and micro environments.

        Args:
            macro_env: The MacroGridEnv (coarse/M^0).
            micro_env: The MultiGridEnv (fine/M^1).
        """
        super().__init__(macro_env, micro_env)
        self.macro_env = macro_env

    def super_state(self, fine_state: Any) -> Any:
        """Map a micro-level state to the corresponding macro-level state.

        Sets the micro_env to *fine_state* so that passage flags can be
        computed from the grid, then delegates to
        ``MacroGridEnv._micro_to_macro_state()``.
        """
        self.fine_model.set_state(fine_state)
        return self.macro_env._micro_to_macro_state(fine_state)

    def super_agent(self, fine_agent_index: int) -> int:
        """Identity mapping (no agent grouping in two-level MultiGrid)."""
        return fine_agent_index

    def is_feasible(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        """Check that fine actions don't contradict coarse actions.

        For each agent with a WALK(j) coarse action, rejects fine-level
        'forward' actions that would move the agent into a macro-cell
        other than the target cell j.  All other fine actions (turn,
        still, pickup, etc.) are allowed.

        Args:
            coarse_action_profile: Current M^0 action profile.
            fine_state: Current M^1 state tuple.
            fine_action_profile: Proposed M^1 action profile.

        Returns:
            True if fine_action_profile is compatible with
            coarse_action_profile.
        """
        _, micro_agents, _, _ = fine_state
        partition = self.macro_env.partition

        for i, (coarse_a, fine_a) in enumerate(
            zip(coarse_action_profile, fine_action_profile)
        ):
            if coarse_a == MACRO_PASS:
                continue  # PASS at macro level → any fine action OK

            target_cell = coarse_a - 1  # WALK target

            # Only check 'forward' actions (action index 3 in standard
            # MultiGrid actions).  Turns and non-movement actions are
            # always compatible with WALK.
            fine_actions = getattr(self.fine_model, 'actions', None)
            forward_idx = getattr(fine_actions, 'forward', 3)
            if fine_a != forward_idx:
                continue

            # Compute where forward would take the agent
            ax, ay, adir = (
                micro_agents[i][0],
                micro_agents[i][1],
                micro_agents[i][2],
            )
            if ax is None or adir is None:
                continue  # Terminated agent

            dx, dy = _DIR_TO_VEC[adir]
            fwd_x, fwd_y = ax + dx, ay + dy

            try:
                fwd_cell = partition.cell_of(fwd_x, fwd_y)
            except KeyError:
                continue  # Forward into wall / off grid → no cell change

            current_cell = partition.cell_of(ax, ay)
            if fwd_cell != current_cell and fwd_cell != target_cell:
                return False  # Moving to wrong cell

        return True

    def is_abort(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
    ) -> bool:
        """Check if any agent is aborting its macro-level action.

        An agent aborts when it performs 'still' (action 0) while having
        a non-PASS coarse action.

        Args:
            coarse_action_profile: Current M^0 action profile.
            fine_state: Current M^1 state tuple (unused).
            fine_action_profile: Proposed M^1 action profile.

        Returns:
            True if any agent is aborting its coarse action.
        """
        fine_actions = getattr(self.fine_model, 'actions', None)
        still_idx = getattr(fine_actions, 'still', 0)

        for coarse_a, fine_a in zip(
            coarse_action_profile, fine_action_profile
        ):
            if coarse_a != MACRO_PASS and fine_a == still_idx:
                return True
        return False

    def return_control(
        self,
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...],
        fine_successor_state: Any,
    ) -> bool:
        """Check if control should return to the macro level.

        Returns True when:
        - Any agent with a WALK(j) action has reached target cell j.
        - The fine-level action was an abort (``is_abort`` returned True).

        Args:
            coarse_action_profile: Current M^0 action profile.
            fine_state: M^1 state before the transition.
            fine_action_profile: M^1 action profile taken.
            fine_successor_state: M^1 state after the transition.

        Returns:
            True if control should return to M^0.
        """
        # Check abort first
        if self.is_abort(
            coarse_action_profile, fine_state, fine_action_profile
        ):
            return True

        _, succ_agents, _, _ = fine_successor_state
        partition = self.macro_env.partition

        for i, coarse_a in enumerate(coarse_action_profile):
            if coarse_a == MACRO_PASS:
                continue

            target_cell = coarse_a - 1  # WALK target
            sx, sy = succ_agents[i][0], succ_agents[i][1]
            if sx is None:
                continue  # Terminated agent

            try:
                succ_cell = partition.cell_of(sx, sy)
            except KeyError:
                continue

            if succ_cell == target_cell:
                return True  # Agent reached target cell

        return False
