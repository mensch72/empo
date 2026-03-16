"""Macro-level goals and goal generator for hierarchical MultiGrid environments.

Provides spatial objectives at the macro level:

- ``MacroCellGoal``: A specific agent is in a specific macro-cell.
- ``MacroProximityGoal``: Two agents are in the same (or different) macro-cell.
- ``MacroGoalGenerator``: Enumerates all macro-level goals for a human agent.

All goals follow the ``PossibleGoal`` immutability protocol (``_freeze()``
pattern) and are hashable for use as dictionary keys in backward induction.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from empo.possible_goal import PossibleGoal, PossibleGoalGenerator


class MacroCellGoal(PossibleGoal):
    """Goal: a specific agent is in a specific macro-cell.

    Represents goals like 'agent 0 is in macro-cell 3'.

    Attributes:
        agent_index: Index of the agent this goal applies to.
        target_cell: Target macro-cell index.
    """

    def __init__(
        self,
        env: Any,
        agent_index: int,
        target_cell: int,
        index: Optional[int] = None,
    ):
        """Initialise a MacroCellGoal.

        Args:
            env: The MacroGridEnv this goal applies to.
            agent_index: Index of the agent.
            target_cell: Target macro-cell index.
            index: Optional goal index (for YAML loader compatibility).
        """
        super().__init__(env, index=index)
        self.agent_index = agent_index
        self.target_cell = target_cell
        self._hash = hash(('MacroCellGoal', self.agent_index, self.target_cell))
        super()._freeze()

    def is_achieved(self, state) -> int:
        """Return 1 if the agent is in the target macro-cell, else 0.

        Args:
            state: Macro-state tuple
                ``(remaining_time, passage_flags, agent_states, object_states)``
                where ``agent_states[i][0]`` is agent *i*'s macro-cell index.
        """
        agent_cell = state[2][self.agent_index][0]
        return 1 if agent_cell == self.target_cell else 0

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (isinstance(other, MacroCellGoal)
                and self.agent_index == other.agent_index
                and self.target_cell == other.target_cell)

    def __repr__(self) -> str:
        return f"MacroCellGoal(agent={self.agent_index}, cell={self.target_cell})"


class MacroProximityGoal(PossibleGoal):
    """Goal: two agents are in the same (or different) macro-cell.

    Represents goals like 'agent 0 and agent 1 are in the same macro-cell'
    or 'agent 0 and agent 1 are in different macro-cells'.

    Attributes:
        agent_index: Primary agent index.
        other_agent_index: Secondary agent index.
        same_cell: If True, goal is achieved when agents share a cell;
            if False, when they are in different cells.
    """

    def __init__(
        self,
        env: Any,
        agent_index: int,
        other_agent_index: int,
        same_cell: bool = True,
        index: Optional[int] = None,
    ):
        """Initialise a MacroProximityGoal.

        Args:
            env: The MacroGridEnv this goal applies to.
            agent_index: Primary agent index.
            other_agent_index: Secondary agent index.
            same_cell: True to require same cell, False for different cells.
            index: Optional goal index (for YAML loader compatibility).
        """
        super().__init__(env, index=index)
        self.agent_index = agent_index
        self.other_agent_index = other_agent_index
        self.same_cell = same_cell
        self._hash = hash((
            'MacroProximityGoal',
            self.agent_index,
            self.other_agent_index,
            self.same_cell,
        ))
        super()._freeze()

    def is_achieved(self, state) -> int:
        """Return 1 if the proximity condition is satisfied, else 0.

        Args:
            state: Macro-state tuple
                ``(remaining_time, passage_flags, agent_states, object_states)``.
        """
        cell_a = state[2][self.agent_index][0]
        cell_b = state[2][self.other_agent_index][0]
        if self.same_cell:
            return 1 if cell_a == cell_b else 0
        else:
            return 1 if cell_a != cell_b else 0

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return (isinstance(other, MacroProximityGoal)
                and self.agent_index == other.agent_index
                and self.other_agent_index == other.other_agent_index
                and self.same_cell == other.same_cell)

    def __repr__(self) -> str:
        mode = "same" if self.same_cell else "diff"
        return (f"MacroProximityGoal(agent={self.agent_index}, "
                f"other={self.other_agent_index}, {mode})")


class MacroGoalGenerator(PossibleGoalGenerator):
    """Generate all macro-level goals for a MacroGridEnv.

    For each human agent generates:

    - ``MacroCellGoal`` for each macro-cell (weight 1.0)
    - ``MacroProximityGoal(same_cell=True)`` for each other agent (weight 1.0)
    - ``MacroProximityGoal(same_cell=False)`` for each other agent (weight 1.0)

    All goals are equally weighted.
    """

    def __init__(self, env: Any, indexed: bool = False):
        """Initialise the generator.

        Args:
            env: A MacroGridEnv instance.
            indexed: Whether goals have indices.
        """
        super().__init__(env, indexed=indexed)
        self._goals_cache: Dict[int, List[Tuple[PossibleGoal, float]]] = {}

    def set_world_model(self, world_model: Any) -> None:
        """Set or update the world model reference.

        Clears the cached goal lists so they are rebuilt with the new env.
        """
        self.env = self.world_model = world_model
        self._goals_cache.clear()

    def __getstate__(self):
        """Exclude env/world_model from pickling."""
        state = self.__dict__.copy()
        state['env'] = None
        state['world_model'] = None
        state['_goals_cache'] = {}
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)

    def _get_goals_for_agent(
        self, human_agent_index: int,
    ) -> List[Tuple[PossibleGoal, float]]:
        """Return cached goal list for *human_agent_index*."""
        if human_agent_index not in self._goals_cache:
            num_cells = self.env.num_cells
            num_agents = len(self.env.agents)
            goals = []
            for cell_idx in range(num_cells):
                goals.append(
                    (MacroCellGoal(self.env, human_agent_index, cell_idx), 1.0)
                )
            for other_idx in range(num_agents):
                if other_idx != human_agent_index:
                    goals.append(
                        (MacroProximityGoal(
                            self.env, human_agent_index, other_idx, True,
                        ), 1.0)
                    )
                    goals.append(
                        (MacroProximityGoal(
                            self.env, human_agent_index, other_idx, False,
                        ), 1.0)
                    )
            self._goals_cache[human_agent_index] = goals
        return self._goals_cache[human_agent_index]

    def generate(
        self, state, human_agent_index: int,
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        """Yield ``(goal, weight)`` pairs for *human_agent_index*.

        Goals are cached per *human_agent_index* to avoid creating fresh
        objects on every call (important for backward induction where
        ``generate()`` is invoked for every state).

        Args:
            state: Current macro-state (unused; goals are state-independent).
            human_agent_index: Index of the human agent.

        Yields:
            ``(PossibleGoal, float)`` pairs with weight 1.0 each.
        """
        yield from self._get_goals_for_agent(human_agent_index)
