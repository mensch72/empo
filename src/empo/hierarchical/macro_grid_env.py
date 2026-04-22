"""Macro-level world model for hierarchical MultiGrid environments.

Provides a coarse spatial abstraction of a MultiGridEnv by partitioning the
grid into rectangular macro-cells and defining macro-level transitions based
on cell-to-cell movement with heuristic duration estimates.

The macro-level state encodes:
  (remaining_time, passage_flags, agent_states, object_states)

The macro-level action space is:
  0: PASS (do nothing)
  j+1: WALK(j) — walk to macro-cell j
"""

from typing import Any, Dict, List, Optional, Tuple

from gymnasium import spaces
import math

from empo.world_model import WorldModel
from empo.hierarchical.cell_partition import CellPartition


# Macro action constants
MACRO_PASS = 0


class _MacroAgent:
    """Lightweight stand-in for agent objects at the macro level.

    The backward induction algorithm accesses ``len(world_model.agents)``
    and may iterate over agents.  These proxies satisfy that interface
    without carrying micro-level state.
    """
    pass


def macro_walk(cell_index: int) -> int:
    """Return the macro-action index for WALK(cell_index)."""
    return cell_index + 1


def decode_macro_action(action: int) -> Tuple[str, int]:
    """Decode a macro-action index into (action_type, target).

    Returns:
        ('PASS', -1) for action 0.
        ('WALK', cell_index) for action >= 1.
    """
    if action == MACRO_PASS:
        return ('PASS', -1)
    return ('WALK', action - 1)


class MacroGridEnv(WorldModel):
    """Macro-level world model abstracting a MultiGridEnv into macro-cells.

    Partitions the grid into rectangular macro-cells using CellPartition,
    then defines macro-level states (agent cell locations, passage flags,
    remaining time) and transitions (WALK between adjacent cells with
    heuristic duration estimates).

    This serves as M^0 in a two-level hierarchical model where the
    MultiGridEnv is M^1 (the micro/fine level).

    State encoding::

        (remaining_time, passage_flags, agent_states, object_states)

    where:

    - remaining_time: int — expected remaining M^1 steps
    - passage_flags: tuple of bool — passage connectivity between adjacent
      cell pairs, indexed by sorted (i, j) with i < j
    - agent_states: tuple of (cell, carry_type, carry_color, terminated,
      started, paused) per agent
    - object_states: (mobile_objects, mutable_objects) from M^1

    Action encoding::

        0: PASS (do nothing)
        j+1: WALK(j) — walk to macro-cell j

    Attributes:
        micro_env: The underlying MultiGridEnv (M^1).
        partition: The CellPartition used for macro-cells.
        num_cells: Number of macro-cells.
        adj_pairs: Sorted list of adjacent cell pairs (i, j) with i < j.
    """

    def __init__(self, micro_env: Any, *, seed: Optional[int] = None):
        """Construct a macro-level environment from a MultiGridEnv.

        Args:
            micro_env: The MultiGridEnv serving as M^1.
            seed: Random seed for cell partition tie-breaking.
        """
        super().__init__()

        self.micro_env = micro_env
        self._num_agents = len(micro_env.agents)
        self._partition_seed = seed

        # Build partition and derived structures from the current grid
        self._build_macro_structures()

        # Observation space is Discrete(1) because MacroGridEnv is a
        # WorldModel used for planning via get_state()/set_state().
        # reset() and step() return 0 as the observation; callers
        # should use get_state() for the full macro-state tuple.
        self.observation_space = spaces.Discrete(1)

        # Agent proxy objects — backward induction accesses len(agents)
        # and iterates over them.  These are lightweight stand-ins.
        self.agents = [_MacroAgent() for _ in range(self._num_agents)]

        # Copy agent role indices from micro_env
        self._human_agent_indices = list(micro_env.human_agent_indices)
        self._robot_agent_indices = list(micro_env.robot_agent_indices)

        # Upper bound on macro steps
        self.max_steps = micro_env.max_steps

        # Compute initial macro state from current micro state
        self._state = self._micro_to_macro_state(micro_env.get_state())

    def _build_macro_structures(self) -> None:
        """(Re)build partition, adjacency index, and action space.

        Called from ``__init__`` and ``reset()`` to keep the macro
        abstraction consistent with the micro environment's grid.
        """
        self._partition = CellPartition.from_grid(
            self.micro_env.grid,
            self.micro_env.width,
            self.micro_env.height,
            seed=self._partition_seed,
        )

        # Sorted adjacent pairs (i < j) for canonical passage flag ordering
        self._adj_pairs: List[Tuple[int, int]] = sorted(
            {(min(i, j), max(i, j))
             for i in range(self._partition.num_cells)
             for j in self._partition.adjacency.get(i, frozenset())}
        )
        self._passage_pair_idx: Dict[Tuple[int, int], int] = {
            pair: idx for idx, pair in enumerate(self._adj_pairs)
        }

        # Action space: PASS + WALK(j) for each cell index.
        # Uses Discrete (per-agent) to match the convention expected by
        # WorldModel.get_dag() and backward_induction, which read
        # action_space.n.  Multi-agent action profiles are constructed
        # externally from len(self.agents) × action_space.n.
        self._num_actions = self._partition.num_cells + 1
        self.action_space = spaces.Discrete(self._num_actions)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_cells(self) -> int:
        """Number of macro-cells in the partition."""
        return self._partition.num_cells

    @property
    def partition(self) -> CellPartition:
        """The cell partition used by this macro environment."""
        return self._partition

    @property
    def adj_pairs(self) -> List[Tuple[int, int]]:
        """Sorted list of adjacent cell pairs (i, j) with i < j."""
        return list(self._adj_pairs)

    @property
    def human_agent_indices(self) -> List[int]:
        return self._human_agent_indices

    @human_agent_indices.setter
    def human_agent_indices(self, value: List[int]) -> None:
        self._human_agent_indices = list(value)

    @property
    def robot_agent_indices(self) -> List[int]:
        return self._robot_agent_indices

    @robot_agent_indices.setter
    def robot_agent_indices(self, value: List[int]) -> None:
        self._robot_agent_indices = list(value)

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Return the current macro-level state (hashable tuple)."""
        return self._state

    def set_state(self, state: Any) -> None:
        """Set the macro-level state."""
        self._state = state

    def transition_probabilities(
        self, state: Any, actions: List[int],
    ) -> Optional[List[Tuple[float, Any]]]:
        """Compute macro-level transitions.

        For WALK(j) actions: deterministically moves the agent to cell j
        if the passage is open and j is adjacent, decrementing remaining
        time by the estimated inter-cell distance.  Otherwise (closed
        passage, non-adjacent, invalid target), the agent stays in place
        and 1 unit of time elapses (same as PASS).

        Returns None for terminal states (remaining_time <= 0 or all
        agents terminated).
        """
        remaining_time, passage_flags, agent_states, object_states = state

        # Terminal check
        if remaining_time <= 0:
            return None
        if all(a[3] for a in agent_states):  # all terminated
            return None

        new_agents = list(agent_states)
        max_duration = 1  # baseline: at least 1 step (PASS duration)

        for i, action in enumerate(actions):
            cell, carry_t, carry_c, term, started, paused = agent_states[i]

            # Skip inactive agents
            if term or paused or not started:
                continue

            if action == MACRO_PASS:
                continue

            target_cell = action - 1  # WALK(target_cell)

            # Validate target
            if target_cell < 0 or target_cell >= self._partition.num_cells:
                continue  # Invalid action → treated as PASS
            if target_cell == cell:
                continue  # Walking to same cell → PASS
            if target_cell not in self._partition.adjacency.get(
                cell, frozenset()
            ):
                continue  # Not adjacent → PASS

            # Check passage connectivity
            key = (min(cell, target_cell), max(cell, target_cell))
            pidx = self._passage_pair_idx.get(key)
            if pidx is not None and passage_flags[pidx]:
                # Passage open → agent moves to target cell
                new_agents[i] = (
                    target_cell, carry_t, carry_c, term, started, paused,
                )
                distance = self._partition.estimated_distance(cell, target_cell)
                dur = max(1, math.ceil(distance))
                max_duration = max(max_duration, dur)
            # Else: passage closed → agent stays, unit-step duration only

        new_remaining = max(0, remaining_time - max_duration)
        new_state = (
            new_remaining, passage_flags,
            tuple(new_agents), object_states,
        )
        return [(1.0, new_state)]

    def transition_durations(
        self,
        state: Any,
        actions: List[int],
        transitions: List[Tuple[float, Any]],
    ) -> List[float]:
        """Return the duration for each transition outcome.

        Duration is computed as the change in remaining_time from the
        state to each successor state.
        """
        if not transitions:
            return []
        remaining_time = state[0]
        durations = []
        for _, succ in transitions:
            succ_remaining = succ[0]
            durations.append(float(max(1, remaining_time - succ_remaining)))
        return durations

    def terminal_duration(self, state: Any) -> float:
        """Return terminal state duration (always 1.0)."""
        return 1.0

    # ------------------------------------------------------------------
    # Macro-specific helpers
    # ------------------------------------------------------------------

    def passage_open(self, state: Any, cell_i: int, cell_j: int) -> bool:
        """Check if passage between adjacent cells i and j is open."""
        _, passage_flags, _, _ = state
        key = (min(cell_i, cell_j), max(cell_i, cell_j))
        pidx = self._passage_pair_idx.get(key)
        if pidx is None:
            return False  # Not adjacent
        return passage_flags[pidx]

    def available_actions(self, state: Any, agent_index: int) -> List[int]:
        """Return list of valid macro-actions for an agent in the given state.

        Always includes PASS (0). Adds WALK(j) = j+1 for each adjacent cell.
        """
        _, _, agent_states, _ = state
        cell = agent_states[agent_index][0]

        actions = [MACRO_PASS]
        for adj_cell in sorted(
            self._partition.adjacency.get(cell, frozenset())
        ):
            actions.append(macro_walk(adj_cell))
        return actions

    def macro_cell_of(self, x: int, y: int) -> int:
        """Return the macro-cell index containing grid position (x, y)."""
        return self._partition.cell_of(x, y)

    # ------------------------------------------------------------------
    # State conversion
    # ------------------------------------------------------------------

    def _micro_to_macro_state(self, micro_state: Any) -> Any:
        """Convert a micro-level state to a macro-level state.

        The micro_env's grid must reflect *micro_state* (call
        ``micro_env.set_state(micro_state)`` first if needed) because
        passage flags are computed from the grid.
        """
        remaining_time, micro_agents, mobile_objects, mutable_objects = \
            micro_state

        passage_flags = self._compute_passage_flags()

        agent_states = tuple(
            (
                self._partition.cell_of(a[0], a[1])
                if a[0] is not None else -1,
                a[6],  # carrying_type
                a[7],  # carrying_color
                a[3],  # terminated
                a[4],  # started
                a[5],  # paused
            )
            for a in micro_agents
        )

        object_states = (mobile_objects, mutable_objects)
        return (remaining_time, passage_flags, agent_states, object_states)

    def _compute_passage_flags(self) -> Tuple[bool, ...]:
        """Compute passage flags from the micro_env's current grid state.

        For each adjacent cell pair (i, j), the passage is open if at
        least one border pair has both positions passable (empty or
        ``can_overlap``).  Agent positions are treated as passable.
        """
        flags = []
        for i, j in self._adj_pairs:
            pairs = self._partition.border_pairs(i, j)
            is_open = False
            for pos1, pos2 in pairs:
                if (self._is_passable(pos1[0], pos1[1])
                        and self._is_passable(pos2[0], pos2[1])):
                    is_open = True
                    break
            flags.append(is_open)
        return tuple(flags)

    def _is_passable(self, x: int, y: int) -> bool:
        """Check if a grid position allows passage (ignoring agents)."""
        obj = self.micro_env.grid.get(x, y)
        if obj is None:
            return True
        if getattr(obj, 'type', None) == 'agent':
            return True  # Agents don't block passages at macro level
        return getattr(obj, 'can_overlap', lambda: False)()

    # ------------------------------------------------------------------
    # gym.Env interface (minimal)
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset by resetting the micro environment and recomputing state.

        Rebuilds the cell partition and derived structures (adjacency,
        action space) after resetting the micro environment, since
        ``MultiGridEnv.reset()`` may regenerate the grid layout.

        Note: ``seed`` and ``options`` are accepted for Gymnasium
        compatibility but are not forwarded to ``micro_env.reset()``,
        because ``MultiGridEnv.reset()`` (in ``gym_multigrid``) is
        defined with no keyword arguments.  To seed the micro
        environment, call ``micro_env.seed(s)`` before resetting.
        """
        self.micro_env.reset()
        self._build_macro_structures()
        # Rebuilding macro structures changes the macro dynamics and
        # action_space.n, so any cached DAG based on the previous
        # configuration must be invalidated.
        if hasattr(self, "clear_dag_cache"):
            self.clear_dag_cache()
        self._state = self._micro_to_macro_state(
            self.micro_env.get_state()
        )
        return 0, {}

    def step(self, actions):
        """Apply a macro action and return (obs, reward, done, trunc, info).

        The observation is always 0 (matching ``observation_space =
        Discrete(1)``).  Use ``get_state()`` for the full macro state.
        """
        transitions = self.transition_probabilities(self._state, actions)
        if transitions is None:
            return 0, 0.0, True, False, {}

        _, new_state = transitions[0]
        self._state = new_state

        done = (
            new_state[0] <= 0
            or all(a[3] for a in new_state[2])
        )
        return 0, 0.0, done, False, {}
