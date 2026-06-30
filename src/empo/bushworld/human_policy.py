"""
Heuristic human policy prior for BushWorld.

The human "moves along the shortest path" toward its goal, *ignoring bush
densities* (and other players). Because every cell is walkable and the grid is
4-connected, a shortest path simply reduces the Manhattan distance to the
nearest goal cell. The resulting prior puts uniform probability on the set of
actions that strictly minimize that distance; if the human is already on a goal
cell it ``pass``es.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

import numpy as np

from empo.bushworld.env import ACTION_DELTAS, Actions
from empo.human_policy_prior import HumanPolicyPrior

if TYPE_CHECKING:  # pragma: no cover
    from empo.bushworld.env import BushWorld
    from empo.possible_goal import PossibleGoal, PossibleGoalGenerator


def _goal_bounding_box(goal) -> tuple:
    if hasattr(goal, "target_rect"):
        return tuple(int(c) for c in goal.target_rect)
    if hasattr(goal, "target_pos"):
        x, y = goal.target_pos
        return (int(x), int(y), int(x), int(y))
    raise ValueError(f"Unsupported goal type for heuristic policy: {goal!r}")


class ShortestPathHumanPolicyPrior(HumanPolicyPrior):
    """Shortest-path (density-agnostic) human policy prior.

    Args:
        world_model: The BushWorld instance.
        human_agent_indices: Indices of human agents.
        possible_goal_generator: Generator used to compute the marginal policy
            (averaging over goals). Defaults to ``world_model.possible_goal_generator``.
        beta_h: If finite, blend the uniform shortest-path distribution toward a
            Boltzmann distribution over negative distance. By default
            (``inf``) the policy is uniform over the strictly-distance-reducing
            actions (pure shortest path).
    """

    def __init__(
        self,
        world_model: "BushWorld",
        human_agent_indices: List[int],
        possible_goal_generator: Optional["PossibleGoalGenerator"] = None,
        beta_h: float = float("inf"),
    ):
        super().__init__(world_model, human_agent_indices)
        self._generator = possible_goal_generator
        self.beta_h = beta_h
        self.num_actions = len(Actions)

    @property
    def generator(self):
        if self._generator is not None:
            return self._generator
        return self.world_model.possible_goal_generator

    def set_world_model(self, world_model: "BushWorld") -> None:
        super().set_world_model(world_model)
        if self._generator is not None and hasattr(self._generator, "set_world_model"):
            self._generator.set_world_model(world_model)

    # ------------------------------------------------------------------ #
    def _distance_to_goal(self, pos, goal) -> int:
        x, y = pos
        x1, y1, x2, y2 = _goal_bounding_box(goal)
        cx = min(max(x, x1), x2)
        cy = min(max(y, y1), y2)
        return abs(x - cx) + abs(y - cy)

    def _goal_conditioned_distribution(self, state, human_agent_index, goal) -> np.ndarray:
        _, positions, _ = state
        x, y = positions[human_agent_index]
        width = self.world_model.width
        height = self.world_model.height

        distances = np.empty(self.num_actions, dtype=np.float64)
        for a in range(self.num_actions):
            dx, dy = ACTION_DELTAS[a]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                nx, ny = x, y  # off-grid move keeps you in place
            distances[a] = self._distance_to_goal((nx, ny), goal)

        if np.isinf(self.beta_h):
            min_d = distances.min()
            probs = (distances == min_d).astype(np.float64)
        else:
            logits = -self.beta_h * distances
            logits -= logits.max()
            probs = np.exp(logits)
        total = probs.sum()
        if total <= 0:
            return np.ones(self.num_actions) / self.num_actions
        return probs / total

    def __call__(
        self,
        state,
        human_agent_index: int,
        possible_goal: Optional["PossibleGoal"] = None,
    ) -> np.ndarray:
        if possible_goal is not None:
            return self._goal_conditioned_distribution(
                state, human_agent_index, possible_goal
            )

        # Marginal: weighted average over goals from the generator.
        marginal = np.zeros(self.num_actions, dtype=np.float64)
        total_weight = 0.0
        for goal, weight in self.generator.generate(state, human_agent_index):
            marginal += weight * self._goal_conditioned_distribution(
                state, human_agent_index, goal
            )
            total_weight += weight
        if total_weight <= 0:
            return np.ones(self.num_actions) / self.num_actions
        return marginal / total_weight
