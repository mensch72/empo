# Implementation Plan: Hierarchical World Models

**Status:** In Progress (Tasks 1–8 complete)  
**Date:** 2026-03-14

## 1. Overview

This document outlines a plan to extend the EMPO framework to support **hierarchical world models** and **hierarchical decision making** as described in the [theory gist](https://gist.github.com/mensch72/0da58048dee6b0bccd8e41bd7a5fcaac). The key idea is to introduce multiple levels of abstraction: a coarse macro-level model $M^0$ and a fine micro-level model $M^1$, where macro-actions correspond to sub-problems solved at the micro level. This enables tractable empowerment computation for larger environments by decomposing the problem hierarchically.

### 1.1 Core Concepts

- **Multi-level world models**: A sequence of WorldModels $M^0, \dots, M^{L-1}$ from coarsest to finest, connected by `LevelMapper`s that translate states, agents, and actions between adjacent levels.
- **Variable step duration**: Each transition at each level has an associated expected duration $D$, used for continuous-time discounting via $\rho := -\ln\gamma$.
- **Control transfer**: At any point, control resides at exactly one level. Selecting a macro-action transfers control to the next finer level; control returns when the macro-action completes, fails, or is aborted.
- **Hierarchical backward induction**: The macro-level policy is computed exactly via the existing algorithm. Micro-level policies are computed on-demand as partial sub-problems defined by the macro-action context.

### 1.2 Scope

The initial implementation targets a **two-level hierarchy** for MultiGrid environments. The abstract interfaces are designed for $L$ levels but only $L = 2$ will be implemented and tested.

### 1.3 Testing on the Command Line

**Quick test (Tasks 1–8):**
```bash
# No Docker required — just pip install and run
pip install -r setup/requirements.txt pytest
make test-hierarchical
```

This runs 144 tests covering:
- `tests/test_world_model_duration.py` — WorldModel duration API (9 tests)
- `tests/test_duration_discounting.py` — Phase 1/2 duration-aware discounting (6 tests)
- `tests/test_hierarchical_base.py` — HierarchicalWorldModel & LevelMapper ABCs (15 tests)
- `tests/test_cell_partition.py` — Macro-cell partitioning (31 tests)
- `tests/test_macro_grid_env.py` — MacroGridEnv M^0 world model (45 tests)
- `tests/test_multigrid_level_mapper.py` — MultiGridLevelMapper (16 tests)
- `tests/test_two_level_multigrid.py` — TwoLevelMultigrid end-to-end (22 tests)

**Full local test suite (no Docker):**
```bash
make test-local
```

**Run a single test file:**
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
  python -m pytest tests/test_hierarchical_base.py -v
```

**Run a single test:**
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
  python -m pytest tests/test_hierarchical_base.py::test_construct_two_level -v
```

**With Docker (full environment):**
```bash
make up          # start dev container
make shell       # enter container
python -m pytest tests/test_hierarchical_base.py tests/test_duration_discounting.py tests/test_world_model_duration.py -v
```

## 2. Extending WorldModel with Step Duration

### 2.1 Motivation

The current `WorldModel` assumes uniform step durations (implicitly $D = 1$). The theory requires transitions to have potentially different durations $D(s, a, s')$, used for:
- Continuous-time discounting: $e^{-\rho D}$ replaces the constant discount factor $\gamma$
- Duration-weighted power aggregation: $\frac{1 - e^{-\rho D}}{\rho} K(s)^\eta$ replaces $D \cdot K(s)^\eta$

### 2.2 API Extension

Add an optional method to `WorldModel`:

```python
class WorldModel(gym.Env):
    # ... existing methods ...

    def transition_durations(
        self, state: Any, actions: List[int], transitions: List[Tuple[float, Any]]
    ) -> List[float]:
        """Return the expected duration for each transition outcome.

        Given a state, an action profile, and the list of (probability, successor_state)
        outcomes from transition_probabilities(), return a list of durations D(s, a, s')
        of the same length.

        The duration D(s, a, s') represents the certainty-equivalent expected real-time
        elapsed between taking the action and regaining control in successor state s',
        defined as $D = -(1/\rho) \ln E[e^{-\rho d}]$ (eq. from theory gist). For deterministic
        durations, this reduces to the actual elapsed time.

        The default implementation returns [1.0, ...] (unit duration for every transition),
        preserving backward compatibility with the existing uniform-step assumption.

        Args:
            state: The current state.
            actions: The joint action profile.
            transitions: The list of (probability, successor_state) tuples as returned
                by transition_probabilities().

        Returns:
            List of floats, one per transition outcome. Must have len == len(transitions).
        """
        return [1.0] * len(transitions)

    def terminal_duration(self, state: Any) -> float:
        """Return the expected duration of a terminal state before the episode ends.

        Used for terminal-state power aggregation: D(s) for terminal s.
        Default returns 1.0.

        Args:
            state: A terminal state.

        Returns:
            Duration as a float.
        """
        return 1.0
```

### 2.3 Changes to Backward Induction

The Phase 1 and Phase 2 backward induction algorithms must be updated to incorporate durations:

**Phase 1 (human policy prior):**
- Replace the constant discount $\gamma_h$ with per-transition discount $e^{-\rho_h D(s, a, s')}$ where $\rho_h = -\ln\gamma_h$
- For $\gamma_h = 1.0$ (default, no discounting), $\rho_h = 0$ and $e^{-\rho_h D} = 1$ regardless of $D$, so existing behavior is preserved

**Phase 2 (robot policy):**
- Replace constant $\gamma_r$ with per-transition discount $e^{-\rho_r D(s, a, s')}$
- Update the $M(s)$ computation to use duration-weighted reward: $\frac{1 - e^{-\rho_r D}}{\rho_r} K(s)^\eta$
- For $\gamma_r = 1.0$ (default), this reduces to $D \cdot K(s)^\eta$

### 2.4 Backward Compatibility

- The default `transition_durations()` returns `[1.0, ...]`, so existing environments are unaffected
- When $\rho = 0$ (i.e., $\gamma = 1.0$), the discounting formulas reduce to the current non-duration-aware equations
- When $\rho > 0$ and $D = 1$ uniformly, we get $e^{-\rho} = \gamma$, recovering the current per-step discounting

## 3. HierarchicalWorldModel

### 3.1 Class Design

```python
class HierarchicalWorldModel:
    """A hierarchical model consisting of L world models connected by level mappers.

    Contains a sequence of WorldModels M^0, ..., M^{L-1} from coarsest (level 0) to
    finest (level L-1), and L-1 LevelMappers F^0, ..., F^{L-2} connecting adjacent levels.

    The LevelMapper F^ℓ connects M^ℓ (the coarser) to M^{ℓ+1} (the finer).

    Attributes:
        levels: List[WorldModel] — the world models [M^0, ..., M^{L-1}]
        mappers: List[LevelMapper] — the level mappers [F^0, ..., F^{L-2}]
    """

    def __init__(
        self, levels: List[WorldModel], mappers: List['LevelMapper']
    ):
        assert len(mappers) == len(levels) - 1
        self.levels = levels
        self.mappers = mappers

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def coarsest(self) -> WorldModel:
        """Return the coarsest (macro) world model M^0."""
        return self.levels[0]

    def finest(self) -> WorldModel:
        """Return the finest (micro) world model M^{L-1}."""
        return self.levels[-1]
```

### 3.2 LevelMapper

```python
class LevelMapper(ABC):
    """Maps between adjacent levels ℓ (coarser) and ℓ+1 (finer) in a hierarchy.

    Connects a coarser world model M^ℓ to a finer world model M^{ℓ+1} by providing
    state aggregation, agent grouping, action feasibility, and control transfer logic.
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
        coarse_action_profile: Tuple[int, ...],
        fine_state: Any,
        fine_action_profile: Tuple[int, ...]
    ) -> bool:
        """Check if a fine-level action profile is compatible with the current coarse-level one.

        An action profile a^{ℓ+1} is feasible if it does not contradict the plan specified
        by a^ℓ. For example, if a^ℓ says "walk to cell X", then a^{ℓ+1} should not
        walk in the opposite direction.

        Args:
            coarse_action_profile: The current action profile from M^ℓ.
            fine_state: The current state in M^{ℓ+1}.
            fine_action_profile: The proposed action profile in M^{ℓ+1}.

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
            coarse_action_profile: The current action profile from M^ℓ.
            fine_state: The current state in M^{ℓ+1}.
            fine_action_profile: The proposed action profile in M^{ℓ+1}.

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
            coarse_action_profile: The current action profile from M^ℓ.
            fine_state: The state in M^{ℓ+1} before the transition.
            fine_action_profile: The action profile taken in M^{ℓ+1}.
            fine_successor_state: The state in M^{ℓ+1} after the transition.

        Returns:
            True if control should be returned to level ℓ.
        """
```

### 3.3 File Location

- `src/empo/hierarchical/hierarchical_world_model.py` — `HierarchicalWorldModel`
- `src/empo/hierarchical/level_mapper.py` — `LevelMapper` ABC

## 4. TwoLevelMultigrid

### 4.1 Overview

`TwoLevelMultigrid` extends a given `MultiGridEnv` (as $M^1$) by constructing a higher-level view $M^0$ that operates over rectangular "cells" (regions of the grid) instead of individual grid positions.

```python
class TwoLevelMultigrid(HierarchicalWorldModel):
    """A two-level hierarchical model for MultiGrid environments.

    Given a MultiGridEnv as M^1 (the micro/fine level), constructs M^0 (the macro/coarse
    level) whose "cells" are rectangular blocks of walkable M^1 cells, and a LevelMapper
    connecting the two.

    Args:
        micro_env: The MultiGridEnv serving as M^1.
    """

    def __init__(self, micro_env: 'MultiGridEnv'):
        macro_env = MacroGridEnv(micro_env)
        mapper = MultiGridLevelMapper(macro_env, micro_env)
        super().__init__(levels=[macro_env, micro_env], mappers=[mapper])
```

### 4.2 Macro-Level World Model ($M^0$): `MacroGridEnv`

#### 4.2.1 Cell Partitioning

The $M^0$ cells are formed by an **agglomerative hierarchical clustering** of the walkable cells of $M^1$ into rectangles:
1. Start with each walkable cell (non-wall, non-lava) in $M^1$ as its own singleton block
2. Iteratively merge that pair of adjacent blocks for which the merged block is still rectangular and has minimal area among the mergeable pairs (resolve ties at random)
3. Stop when no further rectangular merges are possible
4. Each resulting rectangle becomes one $M^0$ cell

**Algorithm sketch:**
- Initialize each walkable cell as a 1×1 block
- Maintain a priority queue of mergeable adjacent block pairs, keyed by merged area
- At each step, pop the minimal-area merge, verify it is still valid (both blocks still exist and merge is rectangular), merge, and update the queue with new adjacencies
- Each partition block is identified by an index $i \in \{0, \dots, N_{\text{cells}}-1\}$

#### 4.2.2 Adjacency

Two $M^0$ cells $i$ and $j$ are **adjacent** if there exists at least one pair of $M^1$ cells $(k \in i, m \in j)$ that are grid-neighbors (horizontally or vertically adjacent in $M^1$).

#### 4.2.3 State Space

An $M^0$ state encodes:

```python
macro_state = (
    remaining_time,        # float: expected remaining M^1 steps of the episode,
                           #   computed as initial max_steps minus the sum of expected
                           #   durations of all macro-actions taken so far
    passage_flags,         # tuple of bools, indexed by the sorted adjacency list:
                           #   for each adjacent cell pair (i,j) with i < j, ordered
                           #   lexicographically, whether at least one pair of adjacent
                           #   M^1 border cells (k∈i, m∈j) are both empty/passable
                           #   (open door, no blocking object)
    agent_states,          # tuple: for each agent:
                           #   (macro_cell_index, carrying, terminated, started, paused)
                           #   — same as M^1 agent state but position replaced by M^0 cell
    object_states,         # tuple: for each tracked object (doors, keys, blocks, etc.):
                           #   same state info as in M^1 (type, position, is_open, etc.)
)
```

**Notes:**
- `remaining_time` is the **expected** remaining $M^1$ step count: successor state's remaining time = previous state's remaining time minus expected action duration. This avoids state-space blow-up from tracking exact step counts.
- Passage flags encode whether movement between adjacent macro-cells is currently possible
- Agent positions are abstracted to their containing macro-cell index
- Object states are preserved in full (tracking all objects the micro-state tracks) since they affect passage flags and goal achievement

#### 4.2.4 Action Space

Each agent's macro-action is one of:

| Action | Description | Duration estimate |
|--------|-------------|-------------------|
| `PASS` | Do nothing (wait for control return) | 1 |
| `WALK(j)` | Walk into adjacent macro-cell $j$ | avg shortest distance from current cell to closest position in $j$ |
| `CLEAR_PASSAGE(j)` | Clear/open the passage to adjacent cell $j$ (push blocks, open door, step aside) | estimated steps to clear |
| `BLOCK_PASSAGE(j)` | Block the passage to adjacent cell $j$ (push blocks into passage, close door, stand in the way) | estimated steps to block |
| `APPROACH(agent_k)` | Walk towards agent $k$ in the same macro-cell | estimated steps to reach agent |
| `TOGGLE(obj)` | Toggle/switch/button-push object in the same macro-cell | estimated steps to reach and toggle |
| `PICKUP(obj)` | Pick up an object in the same macro-cell | estimated steps to reach and pick up |
| `DROP(obj)` | Drop a carried object in the same macro-cell | 1 |

The available actions depend on the current macro-state (which cells are adjacent, which objects are present, etc.).

#### 4.2.5 Transition Dynamics

$M^0$ transitions use **heuristic estimates** since they summarize the outcomes of multi-step $M^1$ sub-problems. The macro model need not be perfectly accurate — the micro-level sub-problem solve provides the actual behavioral fidelity.

For `WALK(j)`:
- Success probability estimated from passage flags and agent positions
- Successor state: agent's macro-cell changes to $j$, passage flags updated, remaining_time decremented by expected duration
- Duration: average shortest path length from current positions to $j$

For `TOGGLE(obj)`, `PICKUP(obj)`, etc.:
- Success probability estimated from reachability within the macro-cell
- Successor state: object state changed accordingly, remaining_time decremented by expected duration
- Duration: estimated steps to navigate to object and perform action

#### 4.2.6 Duration Estimates

Duration $D^0(s^0, a^0, s'^0)$ is a rough estimate of the real-time steps the corresponding $M^1$ sub-problem will take:

- `WALK(j)`: average of shortest Manhattan distance from any position in the agent's current macro-cell to the nearest position in target cell $j$
- `TOGGLE(obj)` / `PICKUP(obj)`: shortest distance from agent position to object position (within the same macro-cell), plus 1 for the action
- `PASS`: duration 1
- `CLEAR_PASSAGE(j)` / `BLOCK_PASSAGE(j)`: heuristic estimate based on obstacle type (e.g., door toggle = 2 steps, block push = estimated push distance + walk distance)

### 4.3 `MacroGridEnv` Class

```python
class MacroGridEnv(WorldModel):
    """The macro-level world model for a two-level MultiGrid hierarchy.

    Constructed from a MultiGridEnv, abstracting the grid into rectangular
    macro-cells with passage connectivity.

    Implements the WorldModel interface (get_state, set_state, transition_probabilities)
    plus transition_durations() for variable step durations.
    """

    def __init__(self, micro_env: 'MultiGridEnv'):
        self.micro_env = micro_env
        self.cells = self._compute_cell_partition()
        self.adjacency = self._compute_adjacency()
        self._action_space = self._build_action_space()
        # ... (initialize from micro_env state)

    def _compute_cell_partition(self) -> List[Tuple[int, int, int, int]]:
        """Partition walkable area into maximal rectangles.

        Returns:
            List of (x_min, y_min, x_max, y_max) tuples, one per macro-cell.
        """

    def _compute_adjacency(self) -> Dict[int, List[int]]:
        """Compute adjacency between macro-cells.

        Returns:
            Dict mapping cell index to list of adjacent cell indices.
        """

    # WorldModel interface
    def get_state(self) -> Any: ...
    def set_state(self, state: Any) -> None: ...
    def transition_probabilities(self, state, actions) -> Optional[List[Tuple[float, Any]]]: ...
    def transition_durations(self, state, actions, transitions) -> List[float]: ...
    def terminal_duration(self, state) -> float: ...

    @property
    def human_agent_indices(self) -> List[int]: ...
    @property
    def robot_agent_indices(self) -> List[int]: ...

    # Macro-specific helpers
    def macro_cell_of(self, x: int, y: int) -> int:
        """Return the macro-cell index containing grid position (x, y)."""

    def passage_open(self, state, cell_i: int, cell_j: int) -> bool:
        """Check if passage between adjacent cells i and j is open."""

    def available_actions(self, state, agent_index: int) -> List[int]:
        """Return the list of valid macro-actions for an agent in the given state."""
```

### 4.4 MultiGridLevelMapper

```python
class MultiGridLevelMapper(LevelMapper):
    """Level mapper connecting MacroGridEnv (M^0) to MultiGridEnv (M^1).

    Defines:
    - super_state: maps M^1 grid position to M^0 macro-cell
    - super_agent: identity mapping (one agent per agent, no grouping for now)
    - is_feasible: rejects M^1 actions that contradict the M^0 plan
    - is_abort: treats M^1 'still' action as aborting the M^0 plan
    - return_control: returns control when M^0 action achieved, unachievable, or aborted
    """

    def super_state(self, fine_state: Any) -> Any:
        """Map an M^1 state to the corresponding M^0 state.

        Extracts agent positions, maps them to macro-cells, computes passage flags
        from the M^1 grid state, and assembles the macro-state tuple.
        """

    def super_agent(self, fine_agent_index: int) -> int:
        """Identity mapping (no agent grouping in two-level MultiGrid)."""
        return fine_agent_index

    def is_feasible(self, coarse_action_profile, fine_state, fine_action_profile) -> bool:
        """Check that fine_action_profile doesn't contradict coarse_action_profile.

        Rejects fine-level actions that:
        - Enter a different macro-cell than the one specified by WALK(j)
        - Operate on a different object than the one specified by TOGGLE/PICKUP/DROP
        - Drop an object when the plan says PICKUP
        - Clear a passage when the plan says BLOCK_PASSAGE (and vice versa)
        - Walk away from the target when APPROACH(agent) is active
        """

    def is_abort(self, coarse_action_profile, fine_state, fine_action_profile) -> bool:
        """M^1 'still' (action 0) is treated as aborting the M^0 plan."""

    def return_control(self, coarse_action_profile, fine_state, fine_action_profile, fine_successor) -> bool:
        """Return control when:
        - The agent successfully completed the M^0 action (reached target cell,
          toggled target object, picked up target object, etc.)
        - The M^0 action became unachievable (target agent/object left the cell,
          path became blocked, etc.)
        - The fine-level action was an abort (is_abort returned True)
        """
```

### 4.5 File Location

- `src/empo/hierarchical/two_level_multigrid.py` — `TwoLevelMultigrid`, `MacroGridEnv`
- `src/empo/hierarchical/multigrid_level_mapper.py` — `MultiGridLevelMapper`

## 5. Goal Sampler/Generator for $M^0$

### 5.1 Macro-Level Goals

Goals at the macro level represent high-level spatial objectives:

```python
class MacroCellGoal(PossibleGoal):
    """Goal: a specific agent is in a specific M^0 cell.

    Represents goals like 'agent 0 is in macro-cell 3'.
    """

    def __init__(self, env: 'MacroGridEnv', agent_index: int, target_cell: int,
                 index: Optional[int] = None):
        self.agent_index = agent_index
        self.target_cell = target_cell
        super().__init__(env, index=index)
        self._freeze()

    def is_achieved(self, state) -> int:
        agent_cell = state[2][self.agent_index][0]  # macro_cell_index from agent_states
        return 1 if agent_cell == self.target_cell else 0

    def __hash__(self) -> int:
        return hash(('MacroCellGoal', self.agent_index, self.target_cell))

    def __eq__(self, other) -> bool:
        return (isinstance(other, MacroCellGoal)
                and self.agent_index == other.agent_index
                and self.target_cell == other.target_cell)


class MacroProximityGoal(PossibleGoal):
    """Goal: two agents are in the same (or different) M^0 cell.

    Represents goals like 'agent 0 and agent 1 are in the same macro-cell'
    or 'agent 0 and agent 1 are in different macro-cells'.
    """

    def __init__(self, env: 'MacroGridEnv', agent_index: int, other_agent_index: int,
                 same_cell: bool = True, index: Optional[int] = None):
        self.agent_index = agent_index
        self.other_agent_index = other_agent_index
        self.same_cell = same_cell
        super().__init__(env, index=index)
        self._freeze()

    def is_achieved(self, state) -> int:
        cell_a = state[2][self.agent_index][0]
        cell_b = state[2][self.other_agent_index][0]
        if self.same_cell:
            return 1 if cell_a == cell_b else 0
        else:
            return 1 if cell_a != cell_b else 0

    def __hash__(self) -> int:
        return hash(('MacroProximityGoal', self.agent_index, self.other_agent_index,
                      self.same_cell))

    def __eq__(self, other) -> bool:
        return (isinstance(other, MacroProximityGoal)
                and self.agent_index == other.agent_index
                and self.other_agent_index == other.other_agent_index
                and self.same_cell == other.same_cell)
```

### 5.2 Goal Generator

```python
class MacroGoalGenerator(PossibleGoalGenerator):
    """Generate all macro-level goals for a MacroGridEnv.

    For each human agent, generates:
    - MacroCellGoal for each reachable M^0 cell
    - MacroProximityGoal (same cell) for each other agent
    - MacroProximityGoal (different cell) for each other agent

    All goals are equally weighted.
    """

    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        # Yield MacroCellGoal for each cell
        for cell_index in range(self.env.num_cells):
            yield MacroCellGoal(self.env, human_agent_index, cell_index), 1.0
        # Yield MacroProximityGoal for each other agent
        for other_idx in range(len(self.env.agents)):
            if other_idx != human_agent_index:
                yield MacroProximityGoal(self.env, human_agent_index, other_idx, True), 1.0
                yield MacroProximityGoal(self.env, human_agent_index, other_idx, False), 1.0
```

### 5.3 File Location

- `src/empo/hierarchical/macro_goals.py` — `MacroCellGoal`, `MacroProximityGoal`, `MacroGoalGenerator`

## 6. Heuristic Human Policy for $M^0$

### 6.1 Design

A heuristic policy for the macro level that makes reasonable movement decisions:

```python
class MacroHeuristicPolicy(HumanPolicyPrior):
    """Heuristic human policy prior for the macro-level world model.

    For each macro-goal, computes a Boltzmann distribution over macro-actions
    based on a potential function (estimated distance to goal achievement).

    Strategy:
    - For MacroCellGoal: prefer WALK actions that move closer to the target cell
      (using shortest path in the macro-cell adjacency graph, accounting for
      passage connectivity)
    - For MacroProximityGoal (same cell): prefer WALK actions toward the other
      agent's macro-cell
    - For MacroProximityGoal (different cell): prefer WALK actions away from the
      other agent's macro-cell
    - When in the same cell as the target: prefer APPROACH, TOGGLE, PICKUP, etc.
      as appropriate for the goal
    """

    def __init__(self, world_model: 'MacroGridEnv',
                 human_agent_indices: List[int],
                 possible_goal_generator: PossibleGoalGenerator):
        super().__init__(world_model, human_agent_indices)
        self.possible_goal_generator = possible_goal_generator

    def __call__(self, state, human_agent_index: int,
                 possible_goal: Optional[PossibleGoal] = None) -> np.ndarray:
        """Compute action distribution.

        Uses shortest-path distance in the passage-weighted macro-cell graph
        as a potential function to produce Boltzmann-rational action probabilities.
        """
```

### 6.2 File Location

- `src/empo/hierarchical/macro_heuristic_policy.py` — `MacroHeuristicPolicy`

## 7. Hierarchical Backward Induction

### 7.1 Algorithm Overview

The hierarchical algorithm computes policies top-down:

1. **Macro-level ($M^0$):** Compute the full DAG and robot policy using the existing `compute_robot_policy()` algorithm, but with duration-aware discounting. This produces $\pi^0_r(s^0)(a^0_r)$.

2. **Micro-level ($M^1$) sub-problems:** For each macro-state $s^0$ and macro-action profile $a^0$ encountered during rollout, compute a **partial** micro-level robot policy on demand:
   - The sub-problem starts from the current micro-state $s^1_0$
   - Only micro-actions satisfying `is_feasible(a^0, s^1, a^1)` are considered (action space filtering)
   - States where `return_control(...)` is True are treated as terminal
   - At these terminal states, $M(s^1) = M(\sigma^0(s^1))$ (delegates to macro-level value)
   - No caching of sub-problem solutions is performed — each sub-problem is solved fresh since it is unlikely to encounter the same sub-problem twice in practice

### 7.2 API

```python
def compute_hierarchical_robot_policy(
    hierarchical_model: HierarchicalWorldModel,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    possible_goal_generators: List[PossibleGoalGenerator],
    human_policy_priors: Optional[List[HumanPolicyPrior]] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: float = 1.0,
    gamma_r: float = 1.0,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    quiet: bool = False,
    # ... other params from compute_robot_policy ...
) -> 'HierarchicalRobotPolicy':
    """Compute a hierarchical robot policy via top-down backward induction.

    1. Computes the macro-level (M^0) robot policy fully via compute_robot_policy().
    2. Returns a HierarchicalRobotPolicy that computes micro-level sub-problem
       policies on demand during rollouts (no caching).

    Args:
        hierarchical_model: The hierarchical world model.
        human_agent_indices: Indices of human agents (same across levels for now).
        robot_agent_indices: Indices of robot agents.
        possible_goal_generators: One PossibleGoalGenerator per level.
        human_policy_priors: Optional precomputed policy priors per level.
        beta_r, gamma_h, gamma_r, zeta, xi, eta: Theory parameters.

    Returns:
        A HierarchicalRobotPolicy.
    """
```

### 7.3 HierarchicalRobotPolicy

```python
class HierarchicalRobotPolicy(RobotPolicy):
    """A robot policy that operates across multiple levels of a hierarchical model.

    Maintains the current control level and delegates to the appropriate level's
    policy. Computes micro-level sub-problem policies on demand (no caching).

    Usage:
        policy = compute_hierarchical_robot_policy(...)
        policy.reset(hierarchical_model)
        while not done:
            action = policy.sample(micro_state)
            # action is a micro-level action profile
    """

    def __init__(
        self,
        hierarchical_model: HierarchicalWorldModel,
        macro_policy: TabularRobotPolicy,
        macro_Vr: Dict[Any, float],
        macro_Xh: Dict,
        # ... parameters for computing sub-problems ...
    ):
        self.hierarchical_model = hierarchical_model
        self.macro_policy = macro_policy
        self.macro_Vr = macro_Vr
        self.macro_Xh = macro_Xh
        self._current_coarse_action_profile: Optional[Tuple[int, ...]] = None
        self._current_coarse_state: Optional[Any] = None

    def sample(self, state: Any) -> Any:
        """Sample a micro-level action.

        Logic:
        1. If control is at macro level (no active coarse action profile):
           a. Compute macro-state from micro-state via super_state()
           b. Sample macro-action from macro_policy
           c. Store as current coarse action profile, transfer control to micro level
        2. If control is at micro level:
           a. Compute the sub-problem policy for (coarse_state, coarse_action_profile, micro_state)
           b. Sample micro-action from sub-problem policy
           c. Check return_control() — if True, return control to macro level
        """

    def reset(self, world_model: Any) -> None:
        """Reset control state at the start of an episode."""
        self._current_coarse_action_profile = None
        self._current_coarse_state = None

    def _compute_sub_policy(
        self, coarse_state, coarse_action_profile, micro_state
    ) -> TabularRobotPolicy:
        """Compute a micro-level sub-problem policy on demand.

        No caching is performed — each sub-problem is solved fresh since it is
        unlikely to encounter the same sub-problem twice in practice.

        The sub-problem is defined by:
        - Initial state: micro_state
        - Feasible actions: those satisfying is_feasible(coarse_action_profile, ...)
        - Terminal states: those where return_control(...) is True
        - Terminal values: M(σ^0(s^1)) from the macro-level computation
        """
```

### 7.4 Sub-Problem DAG Construction

The sub-problem DAG for a given $(s^0, a^0, s^1_0)$ context is built as follows:

1. **BFS from $s^1_0$**: Explore reachable states, but:
   - Only expand action profiles where `is_feasible(a^0, s^1, a^1)` is True for all agents
   - Mark states where `return_control(a^0, s^1, a^1, s'^1)` is True as terminal
2. **Backward induction on the sub-DAG**: Same as the standard Phase 2 algorithm, but:
   - Terminal states get $M(s^1) = M(\sigma^0(s^1))$ from the pre-computed macro values
   - The GAC computation includes both local micro-goals and macro-goals (lifted via $\sigma^0$)
   - Discounting uses per-transition durations from $M^1$

### 7.5 File Location

- `src/empo/hierarchical/hierarchical_backward_induction.py` — `compute_hierarchical_robot_policy()`
- `src/empo/hierarchical/hierarchical_robot_policy.py` — `HierarchicalRobotPolicy`

## 8. Demonstration Script

### 8.1 Overview

A script that demonstrates the two-level treatment of a MultiGrid environment:

```
examples/hierarchical/two_level_multigrid_demo.py
```

### 8.2 Environment

Use a larger version of an existing gridworld (e.g., a 15×15 or 20×20 grid with multiple rooms, doors, and agents) that benefits from hierarchical decomposition:
- Multiple rooms connected by doors → natural macro-cells
- At least one human agent and one robot agent
- Objects (keys, blocks) that affect passage between rooms

### 8.3 Visualization

The demo produces a video (MP4/GIF) showing:
1. The micro-level grid world with agents moving
2. An overlay showing the macro-cell partition (colored regions)
3. **Curved arrows** indicating the currently active $M^0$ action for each agent:
   - Arrow pointing to the target macro-cell for `WALK(j)`
   - Arrow pointing to the passage for `CLEAR_PASSAGE(j)` / `BLOCK_PASSAGE(j)`
   - Arrow pointing to the other agent for `APPROACH(agent_k)`
   - Arrow pointing to the object for `TOGGLE(obj)` / `PICKUP(obj)` / `DROP(obj)`
4. A text panel showing the current macro-state and action

### 8.4 Script Structure

```python
#!/usr/bin/env python3
"""Two-level hierarchical MultiGrid demonstration.

Computes and visualizes hierarchical robot policy in a multi-room grid world.
Produces an animated video showing macro-level plans as curved arrow overlays.

Usage:
    python examples/hierarchical/two_level_multigrid_demo.py [--quick]
"""

def main():
    # 1. Create micro-level environment
    micro_env = MultiGridEnv(config_file='multigrid_worlds/hierarchical/two_level_demo.yaml', max_steps=200)

    # 2. Build two-level hierarchy
    hierarchy = TwoLevelMultigrid(micro_env)

    # 3. Compute macro-level goals and policy prior
    macro_goal_gen = MacroGoalGenerator(hierarchy.coarsest())
    macro_policy_prior = MacroHeuristicPolicy(hierarchy.coarsest(), ...)

    # 4. Compute hierarchical robot policy
    h_policy = compute_hierarchical_robot_policy(
        hierarchy,
        human_agent_indices=[0],
        robot_agent_indices=[1],
        possible_goal_generators=[macro_goal_gen, micro_goal_gen],
        human_policy_priors=[macro_policy_prior, micro_policy_prior],
        beta_r=10.0, gamma_r=0.99,
    )

    # 5. Run rollout with visualization
    micro_env.start_video_recording("hierarchical_demo")
    state = micro_env.reset()
    h_policy.reset(hierarchy)

    for step in range(max_steps):
        action = h_policy.sample(state)
        state, reward, done, truncated, info = micro_env.step(action)
        # Render with macro-action overlay arrows
        micro_env.render(annotation_text=..., goal_overlays=...)
        if done:
            break

    micro_env.save_video("hierarchical_demo.mp4")
```

### 8.5 File Location

- `examples/hierarchical/two_level_multigrid_demo.py`
- `multigrid_worlds/hierarchical/` — YAML config for the demo environment

## 9. Neural Network Encoders for $M^0$ (Low Priority)

### 9.1 Overview

Eventually, the macro-level state and goal representations should be encodable by neural networks for the learning-based approach. This requires:

- **Macro state encoder**: Encodes the variable-size macro-state (passage flags, agent cells, object states) into a fixed-size vector. Could use a graph neural network (GNN) over the macro-cell adjacency graph, or a simpler approach that concatenates per-cell features.

- **Macro goal encoder**: Encodes macro-goals (`MacroCellGoal`, `MacroProximityGoal`) into vectors compatible with the state encoder.

### 9.2 Design Sketch

```python
class MacroStateEncoder(nn.Module):
    """Encode macro-level state for neural network based Phase 1/2.

    Input: macro-state tuple (remaining_time, passage_flags, agent_states, object_states)
    Output: fixed-size feature vector
    """

class MacroGoalEncoder(nn.Module):
    """Encode macro-level goals for neural network based Phase 1.

    Input: (goal_type, target_cell/agent/object)
    Output: fixed-size feature vector
    """
```

### 9.3 File Location

- `src/empo/learning_based/multigrid/macro_encoders.py`
- `src/empo/learning_based/multigrid/constants.py` — updated with macro-level constants

## 10. Implementation Tasks

The following is a numbered sequence of coding agent tasks, ordered by dependencies. Each task is self-contained and produces testable output.

### Task 1: Extend WorldModel with Step Duration

**Files to modify:**
- `src/empo/world_model.py`

**Work:**
1. Add `transition_durations(state, actions, transitions) -> List[float]` method with default returning `[1.0, ...]`
2. Add `terminal_duration(state) -> float` method with default returning `1.0`
3. Add unit tests verifying defaults and that existing environments are unaffected

**Tests:**
- `tests/test_world_model_duration.py` — verify default durations, subclass override

---

### Task 2: Add Duration-Aware Discounting to Phase 1 Backward Induction

**Files to modify:**
- `src/empo/backward_induction/phase1.py`

**Work:**
1. Compute $\rho_h = -\ln(\gamma_h)$ (handle $\gamma_h = 1.0$ as $\rho_h = 0$ special case)
2. In the backward induction loop, replace constant discount $\gamma_h$ with per-transition $e^{-\rho_h D(s, a, s')}$
3. Query `world_model.transition_durations()` once per state alongside `transition_probabilities()`
4. When $\rho_h = 0$, skip duration queries and use discount factor 1.0 (preserving current behavior exactly)

**Tests:**
- Verify existing Phase 1 tests still pass (no behavioral change for $D = 1$, $\gamma_h = 1.0$)
- Add test with a custom WorldModel that returns non-uniform durations

---

### Task 3: Add Duration-Aware Discounting to Phase 2 Backward Induction

**Files to modify:**
- `src/empo/backward_induction/phase2.py`

**Work:**
1. Compute $\rho_r = -\ln(\gamma_r)$ (handle $\gamma_r = 1.0$ as $\rho_r = 0$ special case)
2. Replace constant $\gamma_r$ with per-transition $e^{-\rho_r D}$
3. Update $M(s)$ computation to use $(1 - e^{-\rho_r D}) / \rho_r \cdot K(s)^\eta$ for the duration-weighted reward
4. For terminal states, use $D(s)$ via `terminal_duration()`
5. When $\rho_r = 0$, use the limit form $D \cdot K(s)^\eta$ for the reward term

**Tests:**
- Verify existing Phase 2 tests still pass
- Add test with non-uniform durations verifying correct discounting

---

### Task 4: Implement HierarchicalWorldModel and LevelMapper ABCs

**Files to create:**
- `src/empo/hierarchical/hierarchical_world_model.py`
- `src/empo/hierarchical/level_mapper.py`
- Update `src/empo/hierarchical/__init__.py` with exports

**Work:**
1. Implement `HierarchicalWorldModel` class as specified in Section 3.1
2. Implement `LevelMapper` ABC as specified in Section 3.2
3. Add basic validation (level count matches mapper count, etc.)

**Tests:**
- `tests/test_hierarchical_base.py` — test construction, validation, property access

---

### Task 5: Implement Macro-Cell Partitioning

**Files to create:**
- `src/empo/hierarchical/cell_partition.py`

**Work:**
1. Implement the **agglomerative hierarchical clustering** algorithm: start with single-walkable-cell blocks, iteratively merge that pair of adjacent blocks for which the merged block is still rectangular and has minimal area among the mergeable pairs (ties broken at random), stop when no further rectangular merges are possible
2. Compute adjacency between macro-cells
3. Compute border cells between adjacent macro-cells (for passage flag computation)
4. Compute estimated distances between macro-cells (for duration estimates)

**Tests:**
- `tests/test_cell_partition.py` — test on known grid layouts (single room, two rooms with door, L-shaped room, etc.); verify that the partitioning produces valid non-overlapping rectangular blocks covering all walkable cells

---

### Task 6: Implement MacroGridEnv ($M^0$ WorldModel)

**Files to create:**
- `src/empo/hierarchical/macro_grid_env.py`

**Work:**
1. Implement `MacroGridEnv(WorldModel)` with state/action spaces as specified in Section 4.2–4.3
2. Implement `get_state()`, `set_state()`, `transition_probabilities()`, `transition_durations()`
3. Implement `is_terminal()`, `human_agent_indices`, `robot_agent_indices`
4. Implement macro-action availability logic
5. Implement approximate transition dynamics and duration estimates

**Tests:**
- `tests/test_macro_grid_env.py` — test state encoding/decoding, transitions, durations, action availability

---

### Task 7: Implement MultiGridLevelMapper

**Files to create:**
- `src/empo/hierarchical/multigrid_level_mapper.py`

**Work:**
1. Implement `super_state()` — map micro-state to macro-state
2. Implement `super_agent()` — identity mapping
3. Implement `is_feasible()` — check micro-action compatibility with macro-action
4. Implement `is_abort()` — detect still/pass as abort
5. Implement `return_control()` — detect completion, failure, and abort conditions

**Tests:**
- `tests/test_multigrid_level_mapper.py` — test each method on specific scenarios

---

### Task 8: Implement TwoLevelMultigrid

**Files to create:**
- `src/empo/hierarchical/two_level_multigrid.py`

**Work:**
1. Implement `TwoLevelMultigrid(HierarchicalWorldModel)` as specified in Section 4.1
2. Wire up `MacroGridEnv` construction from a `MultiGridEnv`
3. Wire up `MultiGridLevelMapper` construction

**Tests:**
- `tests/test_two_level_multigrid.py` — test end-to-end construction, super_state consistency

---

### Task 9: Implement Macro-Level Goals and Goal Generator

**Files to create:**
- `src/empo/hierarchical/macro_goals.py`

**Work:**
1. Implement `MacroCellGoal(PossibleGoal)`
2. Implement `MacroProximityGoal(PossibleGoal)`
3. Implement `MacroGoalGenerator(PossibleGoalGenerator)`
4. Ensure all goals are properly hashable and immutable (follow `_freeze()` pattern)

**Tests:**
- `tests/test_macro_goals.py` — test goal achievement, hashing, equality, generator output

---

### Task 10: Implement Macro-Level Heuristic Human Policy

**Files to create:**
- `src/empo/hierarchical/macro_heuristic_policy.py`

**Work:**
1. Implement `MacroHeuristicPolicy(HumanPolicyPrior)`
2. Implement shortest-path computation on macro-cell adjacency graph (accounting for passage connectivity)
3. Implement potential-based Boltzmann policy for each goal type
4. Handle goal-conditioned and unconditional (marginal) distributions

**Tests:**
- `tests/test_macro_heuristic_policy.py` — test policy outputs for simple macro-cell layouts

---

### Task 11: Implement Hierarchical Backward Induction

**Files to create:**
- `src/empo/hierarchical/hierarchical_backward_induction.py`
- `src/empo/hierarchical/hierarchical_robot_policy.py`

**Work:**
1. Implement `compute_hierarchical_robot_policy()` — computes macro-level policy via `compute_robot_policy()`
2. Implement `HierarchicalRobotPolicy(RobotPolicy)` with on-demand sub-problem computation (no caching — each sub-problem is solved fresh)
3. Implement sub-problem DAG construction with feasibility filtering (only feasible action profiles are expanded) and terminal state handling
4. Implement control transfer logic in `sample()`

**Tests:**
- `tests/test_hierarchical_backward_induction.py` — test on small two-room environment
- Verify that macro-level policy computation produces valid distributions
- Verify that sub-problem policies respect feasibility constraints
- Verify control transfer at macro-action boundaries

---

### Task 12: Create Demo Environment Config

**Files to create:**
- `multigrid_worlds/hierarchical/two_level_demo.yaml`

**Work:**
1. Design a 15×15 or 20×20 grid with 4–6 rooms connected by doors
2. Place one human agent and one robot agent in different rooms
3. Add objects (keys, blocks) that affect inter-room passage
4. Verify the environment loads and runs with the existing MultiGridEnv

**Tests:**
- Manual verification that the environment loads, renders, and has the expected macro-cell structure

---

### Task 13: Create Visualization Utilities for Macro-Action Overlays

**Files to create/modify:**
- `src/empo/hierarchical/visualization.py`
- Potentially modify `vendor/multigrid/gym_multigrid/rendering.py` if needed

**Work:**
1. Implement curved arrow rendering for macro-actions
2. Implement macro-cell boundary overlay (colored regions)
3. Implement text annotation for current macro-state/action
4. Integration with MultiGridEnv's `render()` via `goal_overlays` or custom overlay mechanism

**Tests:**
- Manual visual verification

---

### Task 14: Create Two-Level Demo Script

**Files to create:**
- `examples/hierarchical/two_level_multigrid_demo.py`

**Work:**
1. Set up the demo environment and hierarchy
2. Compute macro-level goals, policy prior, and robot policy
3. Run a rollout with the hierarchical robot policy
4. Render each frame with macro-action overlays
5. Save as MP4/GIF video
6. Support `--quick` flag for fast testing

**Tests:**
- Manual: run `python examples/hierarchical/two_level_multigrid_demo.py --quick` and verify video output

---

### Task 15 (Low Priority): Implement Macro-Level Neural Network Encoders

**Files to create:**
- `src/empo/learning_based/multigrid/macro_encoders.py`
- Update `src/empo/learning_based/multigrid/constants.py`

**Work:**
1. Implement `MacroStateEncoder(nn.Module)`
2. Implement `MacroGoalEncoder(nn.Module)`
3. Integrate with the learning-based Phase 1/2 pipeline
4. Update `docs/ENCODER_ARCHITECTURE.md`

**Tests:**
- `tests/test_macro_encoders.py` — test forward pass shapes, gradient flow

## 11. Resolved Questions

The following questions were raised in the initial plan draft and have been resolved via [discussion on issue #125](https://github.com/mensch72/empo/issues/125#issuecomment-4060647060).

### Q1: Cell Partition Algorithm — RESOLVED

**Question:** What precisely defines the "coarsest possible partition" of walkable cells into rectangles?

**Answer:** Use an **agglomerative hierarchical clustering** algorithm: start with single-walkable-cell blocks and iteratively merge that pair of adjacent blocks for which the merged block is still rectangular and has a minimal area among the mergeable pairs (resolving ties at random). Stop when no further rectangular merges are possible.

### Q2: Macro-Level Transition Probabilities — RESOLVED

**Question:** How should $M^0$ transition probabilities be computed?

**Answer:** Use **heuristic estimates** based on path connectivity (is a passage open?), distance estimates (Manhattan distance on the macro-cell graph), and simple success/failure probabilities (passage open → high success, passage closed → 0). The macro model need not be perfectly accurate — the micro-level sub-problem solve provides the actual behavioral fidelity.

### Q3: Macro-Level "Remaining Time" Granularity — RESOLVED

**Question:** Should `remaining_time` in the macro-state be the exact remaining $M^1$ step count, or a coarser approximation?

**Answer:** It should be the **expected** remaining $M^1$ step count, based on the initial `max_steps` minus the expected durations of actions taken. In other words: successor state's remaining time = previous state's remaining time minus expected action duration. This avoids the state-space blow-up from tracking exact step counts while maintaining meaningful time information.

### Q4: Agent Grouping for $L > 2$ — RESOLVED

**Question:** Should we support non-trivial agent grouping for the initial two-level MultiGrid implementation?

**Answer:** Start with **identity mapping** (each micro-agent is its own macro-agent). Agent grouping can be added later when the framework is extended to $L > 2$ or to environments with natural team structure.

### Q5: Sub-Problem Caching — RESOLVED

**Question:** What should the cache key be for micro-level sub-problem policies?

**Answer:** **Skip caching for now.** It is highly unlikely to encounter the same sub-problem twice in practice, so the complexity of a caching mechanism is not justified at this stage.

### Q6: Macro-Level Object Tracking — RESOLVED

**Question:** Should all $M^1$ objects be tracked in the $M^0$ state, or only those that affect passage connectivity and goal achievement?

**Answer:** **Track all objects** that the micro-state tracks. This preserves full information and avoids the need to define "relevance." Optimization can be considered later if the state space becomes too large.

### Q7: Interaction Between `is_feasible` and Action Space — RESOLVED

**Question:** Should `is_feasible` filter out infeasible actions from the micro-level action space, or should infeasible actions simply be assigned zero probability in the policy?

**Answer:** **Filter the action space.** Only pass feasible action profiles to the sub-problem DAG construction. This reduces the sub-problem state space and is conceptually cleaner (the sub-problem is defined over a restricted action space).

## 12. Dependencies and Order

```
Task 1 (WorldModel duration)
  └──→ Task 2 (Phase 1 duration discounting)
  └──→ Task 3 (Phase 2 duration discounting)
  └──→ Task 4 (Hierarchical + LevelMapper ABCs)
         └──→ Task 5 (Cell partitioning)
               └──→ Task 6 (MacroGridEnv)
               └──→ Task 7 (LevelMapper impl)
                     └──→ Task 8 (TwoLevelMultigrid)
                           └──→ Task 9 (Macro goals)
                           └──→ Task 10 (Macro heuristic policy)
                           └──→ Task 11 (Hierarchical backward induction)
                                 └──→ Task 12 (Demo env config)
                                       └──→ Task 13 (Visualization)
                                             └──→ Task 14 (Demo script)
                                                   └──→ Task 15 (Neural encoders, low priority)
```

Tasks 2, 3, and 4 can proceed in parallel after Task 1.  
Tasks 5, 6, and 7 can largely proceed in parallel after Task 4.  
Tasks 9 and 10 can proceed in parallel after Task 8.

## 13. References

- **Theory gist:** https://gist.github.com/mensch72/0da58048dee6b0bccd8e41bd7a5fcaac
- **WorldModel interface:** `src/empo/world_model.py`
- **PossibleGoal interface:** `src/empo/possible_goal.py`
- **HumanPolicyPrior interface:** `src/empo/human_policy_prior.py`
- **RobotPolicy interface:** `src/empo/robot_policy.py`
- **Phase 1 backward induction:** `src/empo/backward_induction/phase1.py`
- **Phase 2 backward induction:** `src/empo/backward_induction/phase2.py`
- **MultiGridEnv implementation:** `vendor/multigrid/gym_multigrid/multigrid.py`
- **Existing plan documents:** `docs/plans/parameterized_goal_sampler.md`, `docs/plans/curiosity.md`

## 14. Summary

This plan extends the EMPO framework with hierarchical world models through:

1. **Duration-aware WorldModel** (Task 1–3): Adding `transition_durations()` and continuous-time discounting ($e^{-\rho D}$) to the base class and backward induction algorithms, with full backward compatibility when durations are uniform.

2. **Abstract hierarchy** (Task 4): `HierarchicalWorldModel` and `LevelMapper` ABCs that support $L$ levels of abstraction with state mapping, agent grouping, action feasibility, and control transfer.

3. **Two-level MultiGrid** (Tasks 5–8): A concrete `TwoLevelMultigrid` that partitions a MultiGrid into rectangular macro-cells with passage connectivity, approximate transition dynamics, and variable step durations.

4. **Macro-level goals and policy** (Tasks 9–10): `MacroCellGoal`, `MacroProximityGoal` goals and a `MacroHeuristicPolicy` that uses shortest-path potentials on the macro-cell graph.

5. **Hierarchical backward induction** (Task 11): Top-down policy computation — exact macro-level solve followed by on-demand micro-level sub-problem solves with feasibility-filtered action spaces and control transfer.

6. **Demonstration** (Tasks 12–14): An animated visualization of hierarchical decision making with curved-arrow macro-action overlays in a multi-room gridworld.

7. **Neural encoders** (Task 15, low priority): Macro-level state and goal encoders for eventual integration with the learning-based approach.
