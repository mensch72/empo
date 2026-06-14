"""
Possible goals for BushWorld.

Mirrors the multigrid goal design: a goal is a 0/1 reward over states, achieved
when a specific human is at a particular cell (``ReachCellGoal``) or anywhere in
a rectangle (``ReachRectangleGoal``). A ``BushWorldConfigGoalGenerator`` builds
such goals on the fly for any human agent index, exactly like multigrid's
``ConfigGoalGenerator``.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from empo.possible_goal import (
    PossibleGoal,
    PossibleGoalGenerator,
    PossibleGoalSampler,
)

if TYPE_CHECKING:  # pragma: no cover
    from empo.bushworld.env import BushWorld


def _human_position(state, human_agent_index: int) -> Tuple[int, int]:
    _, positions, _ = state
    x, y = positions[human_agent_index]
    return int(x), int(y)


class ReachCellGoal(PossibleGoal):
    """Goal: the given human reaches a specific cell ``(x, y)``."""

    def __init__(
        self,
        world_model: "BushWorld",
        human_agent_index: int,
        target_pos: Tuple[int, int],
        index: Optional[int] = None,
    ):
        super().__init__(world_model, index=index)
        self.human_agent_index = int(human_agent_index)
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self.target_rect = (
            self.target_pos[0],
            self.target_pos[1],
            self.target_pos[0],
            self.target_pos[1],
        )
        self._hash = hash((self.human_agent_index, self.target_pos))
        super()._freeze()

    def is_achieved(self, state) -> int:
        x, y = _human_position(state, self.human_agent_index)
        return 1 if (x, y) == self.target_pos else 0

    def __str__(self):
        return f"ReachCell(h{self.human_agent_index}, {self.target_pos})"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, ReachCellGoal)
            and self.human_agent_index == other.human_agent_index
            and self.target_pos == other.target_pos
        )


class ReachRectangleGoal(PossibleGoal):
    """Goal: the given human reaches any cell in rectangle ``(x1, y1, x2, y2)``."""

    def __init__(
        self,
        world_model: "BushWorld",
        human_agent_index: int,
        target_rect: Tuple[int, int, int, int],
        index: Optional[int] = None,
    ):
        super().__init__(world_model, index=index)
        self.human_agent_index = int(human_agent_index)
        x1, y1, x2, y2 = target_rect
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        self.target_rect = (int(x1), int(y1), int(x2), int(y2))
        self.target_pos = ((x1 + x2) // 2, (y1 + y2) // 2)
        self._hash = hash((self.human_agent_index, self.target_rect))
        super()._freeze()

    def is_achieved(self, state) -> int:
        x, y = _human_position(state, self.human_agent_index)
        x1, y1, x2, y2 = self.target_rect
        return 1 if (x1 <= x <= x2 and y1 <= y <= y2) else 0

    def __str__(self):
        return f"ReachRect(h{self.human_agent_index}, {self.target_rect})"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, ReachRectangleGoal)
            and self.human_agent_index == other.human_agent_index
            and self.target_rect == other.target_rect
        )


def all_cell_goal_coords(width: int, height: int) -> List[Tuple[int, int]]:
    """Return ``(x, y)`` specs for every cell in a ``width`` x ``height`` grid."""
    return [(x, y) for y in range(height) for x in range(width)]


def _make_goal(world_model, human_agent_index, coords, index=None):
    coords = tuple(int(c) for c in coords)
    if len(coords) == 2:
        return ReachCellGoal(world_model, human_agent_index, coords, index=index)
    if len(coords) == 4:
        return ReachRectangleGoal(world_model, human_agent_index, coords, index=index)
    raise ValueError(f"Goal spec must have length 2 or 4, got {coords!r}")


class BushWorldConfigGoalGenerator(PossibleGoalGenerator):
    """Deterministic goal generator built from a list of coordinate specs.

    Args:
        world_model: The BushWorld instance.
        goal_coords: List of ``(x, y)`` or ``(x1, y1, x2, y2)`` specs.
        weights: Optional per-goal weights. Defaults to uniform ``1/n``.
    """

    def __init__(
        self,
        world_model: "BushWorld",
        goal_coords: Sequence[Sequence[int]],
        weights: Optional[Sequence[float]] = None,
        indexed: bool = False,
    ):
        super().__init__(world_model, indexed=indexed)
        self.world_model = self.env = world_model
        self.goal_coords = [tuple(int(c) for c in spec) for spec in goal_coords]
        n = len(self.goal_coords)
        if n == 0:
            raise ValueError("goal_coords must be non-empty")
        self.weights = list(weights) if weights is not None else [1.0 / n] * n
        self.indexed = indexed
        self.n_goals = n
        self._goals_cache: dict = {}

    def _goals_for_agent(self, human_agent_index: int):
        if human_agent_index not in self._goals_cache:
            goals = [
                _make_goal(
                    self.env,
                    human_agent_index,
                    coords,
                    index=(idx if self.indexed else None),
                )
                for idx, coords in enumerate(self.goal_coords)
            ]
            self._goals_cache[human_agent_index] = goals
        return self._goals_cache[human_agent_index]

    @property
    def goals(self):
        return self._goals_for_agent(0)

    def set_world_model(self, world_model):
        self.world_model = self.env = world_model
        self._goals_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        state["world_model"] = None
        state["_goals_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        goals = self._goals_for_agent(human_agent_index)
        for goal, weight in zip(goals, self.weights):
            yield goal, weight

    def get_sampler(self) -> "BushWorldConfigGoalSampler":
        return BushWorldConfigGoalSampler(
            self.env, self.goal_coords, probabilities=self.weights, indexed=self.indexed
        )


class BushWorldConfigGoalSampler(PossibleGoalSampler):
    """Stochastic counterpart of :class:`BushWorldConfigGoalGenerator`."""

    def __init__(
        self,
        world_model: "BushWorld",
        goal_coords: Sequence[Sequence[int]],
        probabilities: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        indexed: bool = False,
    ):
        super().__init__(world_model, indexed=indexed)
        self.world_model = self.env = world_model
        self.goal_coords = [tuple(int(c) for c in spec) for spec in goal_coords]
        n = len(self.goal_coords)
        if probabilities is None:
            probabilities = [1.0 / n] * n
        probs = np.asarray(probabilities, dtype=np.float64)
        self._probs = probs / probs.sum()
        self.indexed = indexed
        self._rng = np.random.default_rng(seed)
        self._goals_cache: dict = {}

    def _goals_for_agent(self, human_agent_index: int):
        if human_agent_index not in self._goals_cache:
            goals = [
                _make_goal(
                    self.env,
                    human_agent_index,
                    coords,
                    index=(idx if self.indexed else None),
                )
                for idx, coords in enumerate(self.goal_coords)
            ]
            self._goals_cache[human_agent_index] = goals
        return self._goals_cache[human_agent_index]

    def set_world_model(self, world_model):
        self.world_model = self.env = world_model
        self._goals_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        state["world_model"] = None
        state["_goals_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        goals = self._goals_for_agent(human_agent_index)
        idx = int(self._rng.choice(len(goals), p=self._probs))
        return goals[idx], 1.0
