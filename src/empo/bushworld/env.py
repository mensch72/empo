"""
BushWorld: a simple, efficient grid world model for EMPO.

BushWorld is a deliberately minimal :class:`~empo.world_model.WorldModel` used to
study human-empowerment robot policies on a small, exactly-solvable environment.

World description
-----------------
- A 2D rectangular grid of size ``width`` x ``height``. Every cell is, in
  principle, walkable.
- Each cell holds a *bush* of mutable integer density ``0, 1, ..., B``.
- One or more **robot** bodies (steered collectively by the AI) and one or more
  **humans** live on the grid. Robots have the lowest agent ids, humans follow.
  No two players may occupy the same cell.
- Players act simultaneously. Each player picks one of five actions:
  ``N, W, S, E`` (move) or ``pass``.
- A **human** that moves into a cell of bush density ``d`` succeeds with
  probability ``1 - d / B``; otherwise the move *falls back to* ``pass`` and the
  human stays put. Humans never change bush densities.
- A **robot** can always move into a bush. When it moves into a cell, that
  target cell's density decreases by one. When it ``pass``es, the density of the
  cell it stands on decreases by one. Densities never go below zero.
- **Conflicts** (several players trying to enter the same cell) are resolved in
  the fixed order of agent id (robots first, then humans). Only the first in
  line attempts the move, so every conflict has at most two outcomes: either the
  first in line succeeds and moves in, or it fails and nobody moves in.

State
-----
A BushWorld state is the hashable tuple ``(step_count, positions, densities)``
where

- ``step_count`` is an ``int`` (number of elapsed env steps),
- ``positions`` is a tuple of ``(x, y)`` integer cell coordinates, one per
  player, in agent-id order (robots first, then humans),
- ``densities`` is a flat tuple of ``width * height`` integers, row-major, i.e.
  the density of cell ``(x, y)`` is ``densities[y * width + x]``.

Movement model details (simplifications)
----------------------------------------
To keep transitions exact and cheap to enumerate, a player may only move into a
cell that is *empty at the start of the step*. This means players cannot form a
"train" (immediately follow a player vacating a cell) and cannot swap places in
a single step; such moves are simply blocked. Off-grid moves are treated as a
blocked move (no movement, no density change). Only the explicit ``pass`` action
decreases a robot's current-cell density.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # gymnasium is an optional-at-import-time dependency of WorldModel
    from gymnasium import spaces
except Exception:  # pragma: no cover - gymnasium is required in practice
    spaces = None  # type: ignore

from empo.world_model import WorldModel


Position = Tuple[int, int]
State = Tuple[int, Tuple[Position, ...], Tuple[int, ...]]


class Actions(IntEnum):
    """Action set for BushWorld players (order matches the issue: N, W, S, E, pass)."""

    north = 0
    west = 1
    south = 2
    east = 3
    pass_ = 4


# Movement deltas in (dx, dy) grid coordinates (x = column, y = row, y grows down).
ACTION_DELTAS: Dict[int, Position] = {
    int(Actions.north): (0, -1),
    int(Actions.west): (-1, 0),
    int(Actions.south): (0, 1),
    int(Actions.east): (1, 0),
    int(Actions.pass_): (0, 0),
}

ACTION_NAMES = ["N", "W", "S", "E", "pass"]


class _SimpleAgent:
    """Lightweight agent record so generic helpers (e.g. ``len(env.agents)``) work."""

    __slots__ = ("index", "is_robot", "color")

    def __init__(self, index: int, is_robot: bool):
        self.index = index
        self.is_robot = is_robot
        self.color = "grey" if is_robot else "yellow"


class BushWorld(WorldModel):
    """A simple, efficient grid world with mutable bush densities.

    Args:
        width: Number of columns.
        height: Number of rows.
        num_robots: Number of robot bodies (agent ids ``0 .. num_robots-1``).
        num_humans: Number of humans (agent ids after the robots).
        max_steps: Episode horizon ``T`` (number of env steps before termination).
        B: Maximum bush density. Must be ``>= 1``.
        robot_positions: Initial robot ``(x, y)`` positions (one per robot).
        human_positions: Initial human ``(x, y)`` positions (one per human).
        initial_densities: Optional explicit initial densities. Either a flat
            sequence of ``width * height`` ints (row-major) or a 2D array-like of
            shape ``(height, width)``. If ``None``, densities are sampled
            uniformly from ``0 .. B`` using ``seed``.
        seed: Random seed used both for sampling initial densities (when
            ``initial_densities is None``) and for ``step`` sampling.
        possible_goals: Optional list of goal coordinate specs used to build the
            default ``possible_goal_generator``. Each entry is either ``(x, y)``
            for a cell goal or ``(x1, y1, x2, y2)`` for a rectangle goal. If
            ``None``, a generator covering every single cell is created lazily.
        render_tile_size: Pixel size of a grid cell for rendering.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        width: int = 5,
        height: int = 1,
        num_robots: int = 1,
        num_humans: int = 1,
        max_steps: int = 10,
        B: int = 1,
        robot_positions: Optional[Sequence[Position]] = None,
        human_positions: Optional[Sequence[Position]] = None,
        initial_densities: Optional[Any] = None,
        seed: Optional[int] = None,
        possible_goals: Optional[List[Sequence[int]]] = None,
        render_tile_size: int = 48,
    ):
        super().__init__()

        if width < 1 or height < 1:
            raise ValueError("width and height must be >= 1")
        if B < 1:
            raise ValueError("B (maximum bush density) must be >= 1")
        if num_robots < 0 or num_humans < 0:
            raise ValueError("num_robots and num_humans must be non-negative")

        self.width = int(width)
        self.height = int(height)
        self.num_robots = int(num_robots)
        self.num_humans = int(num_humans)
        self.num_players = self.num_robots + self.num_humans
        if self.num_players < 1:
            raise ValueError("BushWorld requires at least one player")
        self.max_steps = int(max_steps)
        self.B = int(B)
        self.render_tile_size = int(render_tile_size)

        # Resolve initial positions.
        self._init_robot_positions = self._default_positions(
            robot_positions, self.num_robots, start=0
        )
        self._init_human_positions = self._default_positions(
            human_positions, self.num_humans, start=self.num_robots
        )
        self._validate_positions(
            list(self._init_robot_positions) + list(self._init_human_positions)
        )

        self._init_densities_arg = initial_densities
        self._init_seed = seed
        self._possible_goals_spec = possible_goals

        # Agents list (used by generic base-class helpers).
        self.agents = [
            _SimpleAgent(i, is_robot=(i < self.num_robots))
            for i in range(self.num_players)
        ]

        # Gymnasium spaces.
        if spaces is not None:
            self.action_space = spaces.Discrete(len(Actions))
            self.observation_space = spaces.Box(
                low=0,
                high=max(self.B, self.width, self.height),
                shape=(self.num_players * 2 + self.width * self.height,),
                dtype=np.int64,
            )

        self._rng = np.random.default_rng(seed)

        # Mutable runtime state.
        self._positions: List[Position] = []
        self._densities: List[int] = []
        self._step_count = 0

        # Lazily-created default goal generator.
        self._possible_goal_generator = None

        self.reset()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _default_positions(
        self, positions: Optional[Sequence[Position]], count: int, start: int
    ) -> Tuple[Position, ...]:
        """Return explicit positions, or place players left-to-right, top-to-bottom."""
        if positions is not None:
            if len(positions) != count:
                raise ValueError(
                    f"Expected {count} positions, got {len(positions)}"
                )
            return tuple((int(x), int(y)) for (x, y) in positions)
        result = []
        for k in range(count):
            idx = start + k
            x = idx % self.width
            y = (idx // self.width) % self.height
            result.append((x, y))
        return tuple(result)

    def _validate_positions(self, positions: Sequence[Position]) -> None:
        seen = set()
        for (x, y) in positions:
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"Position {(x, y)} is outside the grid")
            if (x, y) in seen:
                raise ValueError(f"Two players share the cell {(x, y)}")
            seen.add((x, y))

    def _sample_initial_densities(self) -> List[int]:
        if self._init_densities_arg is None:
            return list(
                self._rng.integers(0, self.B + 1, size=self.width * self.height)
            )
        arr = np.asarray(self._init_densities_arg)
        if arr.ndim == 2:
            if arr.shape != (self.height, self.width):
                raise ValueError(
                    f"initial_densities 2D shape {arr.shape} != "
                    f"(height, width) = {(self.height, self.width)}"
                )
            arr = arr.reshape(-1)
        elif arr.ndim == 1:
            if arr.shape[0] != self.width * self.height:
                raise ValueError(
                    "initial_densities length "
                    f"{arr.shape[0]} != width*height = {self.width * self.height}"
                )
        else:
            raise ValueError("initial_densities must be 1D or 2D")
        clipped = np.clip(arr.astype(int), 0, self.B)
        return [int(v) for v in clipped]

    # ------------------------------------------------------------------ #
    # Indexing helpers
    # ------------------------------------------------------------------ #
    def cell_index(self, x: int, y: int) -> int:
        """Return the flat density index for cell ``(x, y)``."""
        return y * self.width + x

    def density_at(self, densities: Sequence[int], x: int, y: int) -> int:
        return densities[self.cell_index(x, y)]

    # ------------------------------------------------------------------ #
    # Gym / WorldModel API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._positions = [
            tuple(p) for p in (self._init_robot_positions + self._init_human_positions)
        ]
        self._densities = self._sample_initial_densities()
        self._step_count = 0
        return self._get_observation(), {}

    def get_state(self) -> State:
        return (
            int(self._step_count),
            tuple((int(x), int(y)) for (x, y) in self._positions),
            tuple(int(d) for d in self._densities),
        )

    def set_state(self, state: State) -> None:
        step_count, positions, densities = state
        self._step_count = int(step_count)
        self._positions = [(int(x), int(y)) for (x, y) in positions]
        self._densities = [int(d) for d in densities]

    def initial_state(self) -> State:
        saved = self.get_state()
        self.reset(seed=self._init_seed)
        init = self.get_state()
        self.set_state(saved)
        return init

    def _get_observation(self) -> np.ndarray:
        flat_positions = [c for pos in self._positions for c in pos]
        return np.array(flat_positions + list(self._densities), dtype=np.int64)

    @property
    def human_agent_indices(self) -> List[int]:
        return list(range(self.num_robots, self.num_players))

    @property
    def robot_agent_indices(self) -> List[int]:
        return list(range(self.num_robots))

    def is_terminal(self, state: Optional[State] = None) -> bool:
        if state is None:
            state = self.get_state()
        return int(state[0]) >= self.max_steps

    # ------------------------------------------------------------------ #
    # Transition dynamics
    # ------------------------------------------------------------------ #
    def transition_probabilities(
        self, state: State, actions: Sequence[int]
    ) -> Optional[List[Tuple[float, State]]]:
        """Return the exact ``[(probability, successor_state), ...]`` distribution.

        Returns ``None`` if ``state`` is terminal.
        """
        step_count, positions, densities = state
        if step_count >= self.max_steps:
            return None
        if len(actions) != self.num_players:
            raise ValueError(
                f"Expected {self.num_players} actions, got {len(actions)}"
            )

        positions = [(int(x), int(y)) for (x, y) in positions]
        densities = list(int(d) for d in densities)
        actions = [int(a) for a in actions]

        occupied_now = set(positions)

        # 1. Compute each player's intended target and pass flag.
        targets: List[Position] = []
        is_pass: List[bool] = []
        for i, (x, y) in enumerate(positions):
            a = actions[i]
            dx, dy = ACTION_DELTAS[a]
            if a == int(Actions.pass_):
                targets.append((x, y))
                is_pass.append(True)
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                targets.append((nx, ny))
            else:
                # Off-grid move: blocked no-op (not a pass).
                targets.append((x, y))
            is_pass.append(False)

        # 2. Determine attempters: only the lowest-id mover into a currently-empty
        #    target cell gets to attempt the move.
        movers_by_target: Dict[Position, List[int]] = {}
        for i in range(self.num_players):
            if targets[i] != positions[i]:  # genuine move attempt
                movers_by_target.setdefault(targets[i], []).append(i)

        attempters: List[int] = []
        for cell, movers in movers_by_target.items():
            if cell in occupied_now:
                # Empty-target rule: cannot move into a currently-occupied cell.
                continue
            attempters.append(min(movers))  # lowest agent id = first in line

        # 3. Apply deterministic density changes (robot moves / robot passes).
        new_densities = list(densities)
        for i in attempters:
            if i < self.num_robots:
                idx = self.cell_index(*targets[i])
                new_densities[idx] = max(0, new_densities[idx] - 1)
        for i in range(self.num_players):
            if is_pass[i] and i < self.num_robots:
                idx = self.cell_index(*positions[i])
                new_densities[idx] = max(0, new_densities[idx] - 1)

        # 4. Classify attempter moves into deterministic vs stochastic.
        deterministic_moves: List[int] = []  # always succeed
        stochastic_humans: List[Tuple[int, float]] = []  # (agent, success_prob)
        for i in attempters:
            if i < self.num_robots:
                deterministic_moves.append(i)  # robots always succeed
                continue
            d = densities[self.cell_index(*targets[i])]
            p_success = 1.0 - d / self.B
            if p_success >= 1.0:
                deterministic_moves.append(i)
            elif p_success <= 0.0:
                pass  # always fails: human stays
            else:
                stochastic_humans.append((i, p_success))

        # 5. Build successor states by enumerating stochastic outcomes.
        base_positions = list(positions)
        for i in deterministic_moves:
            base_positions[i] = targets[i]

        density_tuple = tuple(new_densities)
        next_step = step_count + 1

        outcomes: Dict[State, float] = {}
        k = len(stochastic_humans)
        for mask in range(1 << k):
            prob = 1.0
            out_positions = list(base_positions)
            for bit, (agent, p_success) in enumerate(stochastic_humans):
                if mask & (1 << bit):
                    prob *= p_success
                    out_positions[agent] = targets[agent]
                else:
                    prob *= 1.0 - p_success
            successor: State = (
                next_step,
                tuple(out_positions),
                density_tuple,
            )
            outcomes[successor] = outcomes.get(successor, 0.0) + prob

        return [(prob, succ) for succ, prob in outcomes.items()]

    def step(self, actions: Sequence[int]):
        state = self.get_state()
        transitions = self.transition_probabilities(state, list(actions))
        if transitions is None:
            return self._get_observation(), 0.0, True, False, {}
        probs = [p for p, _ in transitions]
        successors = [s for _, s in transitions]
        idx = self._rng.choice(len(transitions), p=probs)
        self.set_state(successors[idx])
        terminated = self.is_terminal(self.get_state())
        return self._get_observation(), 0.0, terminated, False, {}

    # ------------------------------------------------------------------ #
    # Goal generator
    # ------------------------------------------------------------------ #
    @property
    def possible_goal_generator(self):
        """Default possible-goal generator (built lazily)."""
        if self._possible_goal_generator is None:
            from empo.bushworld.goals import (
                BushWorldConfigGoalGenerator,
                all_cell_goal_coords,
            )

            if self._possible_goals_spec is not None:
                coords = [tuple(c) for c in self._possible_goals_spec]
            else:
                coords = all_cell_goal_coords(self.width, self.height)
            self._possible_goal_generator = BushWorldConfigGoalGenerator(self, coords)
        return self._possible_goal_generator

    @possible_goal_generator.setter
    def possible_goal_generator(self, value):
        self._possible_goal_generator = value

    # ------------------------------------------------------------------ #
    # Reconstruction support (parallel DAG workers)
    # ------------------------------------------------------------------ #
    def _get_construction_kwargs(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "num_robots": self.num_robots,
            "num_humans": self.num_humans,
            "max_steps": self.max_steps,
            "B": self.B,
            "robot_positions": list(self._init_robot_positions),
            "human_positions": list(self._init_human_positions),
            "initial_densities": self._init_densities_arg,
            "seed": self._init_seed,
            "possible_goals": self._possible_goals_spec,
            "render_tile_size": self.render_tile_size,
        }

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(
        self,
        mode: str = "rgb_array",
        tile_size: Optional[int] = None,
        annotation_text=None,
        annotation_panel_width: int = 260,
        annotation_font_size: int = 11,
        goal_overlays: Optional[Dict[int, Any]] = None,
    ):
        from empo.bushworld.rendering import render_frame

        return render_frame(
            self,
            tile_size=tile_size or self.render_tile_size,
            annotation_text=annotation_text,
            annotation_panel_width=annotation_panel_width,
            annotation_font_size=annotation_font_size,
            goal_overlays=goal_overlays,
        )

    def __repr__(self) -> str:
        return (
            f"BushWorld(width={self.width}, height={self.height}, "
            f"num_robots={self.num_robots}, num_humans={self.num_humans}, "
            f"B={self.B}, max_steps={self.max_steps})"
        )
