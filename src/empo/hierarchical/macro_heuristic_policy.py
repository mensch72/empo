"""Heuristic human policy prior for the macro-level world model.

Provides ``MacroHeuristicPolicy``, a ``HumanPolicyPrior`` that computes
Boltzmann-rational action distributions over macro-actions based on a
potential function (shortest-path distance to goal achievement in the
macro-cell adjacency graph, respecting passage connectivity).

Strategy per goal type:

- **MacroCellGoal**: prefer WALK actions that reduce the shortest-path
  distance to the target cell.
- **MacroProximityGoal (same_cell=True)**: prefer WALK toward the other
  agent's cell.
- **MacroProximityGoal (same_cell=False)**: prefer WALK away from the
  other agent's cell.
- **Marginal (no goal)**: average over all goals from the generator.
"""

from typing import Any, Dict, List, Optional
import heapq

import numpy as np

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.hierarchical.macro_goals import MacroCellGoal, MacroProximityGoal
from empo.hierarchical.macro_grid_env import MACRO_PASS, MacroGridEnv


class MacroHeuristicPolicy(HumanPolicyPrior):
    """Heuristic human policy prior for a ``MacroGridEnv``.

    For each macro-goal, computes a Boltzmann distribution over macro-actions
    based on a potential function (estimated distance to goal achievement).

    The Boltzmann temperature is controlled by ``beta`` (higher β → more
    deterministic, 0 → uniform).

    Attributes:
        possible_goal_generator: Generator providing all macro-level goals.
        beta: Boltzmann rationality parameter (default 5.0).
    """

    def __init__(
        self,
        world_model: MacroGridEnv,
        human_agent_indices: List[int],
        possible_goal_generator: PossibleGoalGenerator,
        *,
        beta: float = 5.0,
    ):
        """Initialise the heuristic policy.

        Args:
            world_model: A ``MacroGridEnv`` instance.
            human_agent_indices: Indices of human agents.
            possible_goal_generator: Generator enumerating macro-level goals.
            beta: Boltzmann rationality parameter (default 5.0).
        """
        super().__init__(world_model, human_agent_indices)
        self.possible_goal_generator = possible_goal_generator
        self.beta = beta

    def __call__(
        self,
        state: Any,
        human_agent_index: int,
        possible_goal: Optional[PossibleGoal] = None,
    ) -> np.ndarray:
        """Compute action distribution for *human_agent_index*.

        Args:
            state: Macro-state tuple.
            human_agent_index: Index of the human agent.
            possible_goal: If provided, condition on this goal.  If ``None``,
                return the marginal distribution (average over all goals).

        Returns:
            ``np.ndarray`` of shape ``(action_space.n,)`` summing to 1.
        """
        num_actions = self.world_model.action_space.n

        if possible_goal is not None:
            return self._goal_conditioned(state, human_agent_index,
                                          possible_goal, num_actions)

        # Marginal: weighted average over all goals from the generator
        total_dist = np.zeros(num_actions, dtype=np.float64)
        total_weight = 0.0
        for goal, weight in self.possible_goal_generator.generate(
            state, human_agent_index
        ):
            dist = self._goal_conditioned(state, human_agent_index,
                                          goal, num_actions)
            total_dist += weight * dist
            total_weight += weight

        if total_weight > 0:
            total_dist /= total_weight

        # Normalise to handle floating-point drift
        s = total_dist.sum()
        if s > 0:
            total_dist /= s
        else:
            total_dist = np.ones(num_actions, dtype=np.float64) / num_actions
        return total_dist

    # ------------------------------------------------------------------
    # Goal-conditioned distributions
    # ------------------------------------------------------------------

    def _goal_conditioned(
        self,
        state: Any,
        human_agent_index: int,
        goal: PossibleGoal,
        num_actions: int,
    ) -> np.ndarray:
        """Compute Boltzmann action distribution conditioned on *goal*."""
        available = self.world_model.available_actions(state, human_agent_index)
        current_cell = state[2][human_agent_index][0]

        advantages = self._compute_advantages(
            state, current_cell, human_agent_index, goal, available,
        )

        return self._boltzmann(advantages, available, num_actions)

    def _compute_advantages(
        self,
        state: Any,
        current_cell: int,
        human_agent_index: int,
        goal: PossibleGoal,
        available: List[int],
    ) -> Dict[int, float]:
        """Map each available action to an advantage value.

        Higher advantage → more preferred.  The advantage is defined as
        the negative change in distance-to-goal caused by the action.
        """
        if isinstance(goal, MacroCellGoal):
            return self._advantages_cell_goal(
                state, current_cell, goal.target_cell, available,
            )
        if isinstance(goal, MacroProximityGoal):
            other_cell = state[2][goal.other_agent_index][0]
            if goal.same_cell:
                return self._advantages_cell_goal(
                    state, current_cell, other_cell, available,
                )
            else:
                return self._advantages_away(
                    state, current_cell, other_cell, available,
                )
        # Fallback: uniform
        return {a: 0.0 for a in available}

    def _advantages_cell_goal(
        self,
        state: Any,
        current_cell: int,
        target_cell: int,
        available: List[int],
    ) -> Dict[int, float]:
        """Advantages for moving *toward* target_cell."""
        macro_env: MacroGridEnv = self.world_model  # type: ignore[assignment]
        dist_from_current = self._shortest_path_distance(
            current_cell, target_cell, state,
        )
        advantages: Dict[int, float] = {}
        for action in available:
            if action == MACRO_PASS:
                advantages[action] = 0.0
            else:
                dest_cell = action - 1  # WALK(dest_cell)
                # A closed passage makes WALK equivalent to PASS (no move)
                if not macro_env.passage_open(state, current_cell, dest_cell):
                    advantages[action] = 0.0
                else:
                    dist_from_dest = self._shortest_path_distance(
                        dest_cell, target_cell, state,
                    )
                    # Advantage = how much closer we get
                    advantages[action] = dist_from_current - dist_from_dest
        return advantages

    def _advantages_away(
        self,
        state: Any,
        current_cell: int,
        other_cell: int,
        available: List[int],
    ) -> Dict[int, float]:
        """Advantages for moving *away from* other_cell."""
        macro_env: MacroGridEnv = self.world_model  # type: ignore[assignment]
        dist_from_current = self._shortest_path_distance(
            current_cell, other_cell, state,
        )
        advantages: Dict[int, float] = {}
        for action in available:
            if action == MACRO_PASS:
                advantages[action] = 0.0
            else:
                dest_cell = action - 1
                # A closed passage makes WALK equivalent to PASS (no move)
                if not macro_env.passage_open(state, current_cell, dest_cell):
                    advantages[action] = 0.0
                else:
                    dist_from_dest = self._shortest_path_distance(
                        dest_cell, other_cell, state,
                    )
                    # Advantage = how much further away we get
                    advantages[action] = dist_from_dest - dist_from_current
        return advantages

    # ------------------------------------------------------------------
    # Shortest-path on macro-cell graph
    # ------------------------------------------------------------------

    def _shortest_path_distance(
        self,
        from_cell: int,
        to_cell: int,
        state: Any,
    ) -> float:
        """Dijkstra shortest-path distance on the passage-weighted adjacency graph.

        Uses ``partition.estimated_distance`` as edge weights, restricted
        to edges whose passage is currently open.

        Returns ``float('inf')`` if *to_cell* is unreachable from *from_cell*.
        """
        if from_cell == to_cell:
            return 0.0

        macro_env: MacroGridEnv = self.world_model  # type: ignore[assignment]
        partition = macro_env.partition

        # Dijkstra with estimated_distance as edge weight
        dist: Dict[int, float] = {from_cell: 0.0}
        heap = [(0.0, from_cell)]

        while heap:
            d, node = heapq.heappop(heap)
            if node == to_cell:
                return d
            if d > dist.get(node, float('inf')):
                continue
            for neighbour in partition.adjacency.get(node, frozenset()):
                if not macro_env.passage_open(state, node, neighbour):
                    continue
                edge_weight = partition.estimated_distance(node, neighbour)
                new_dist = d + edge_weight
                if new_dist < dist.get(neighbour, float('inf')):
                    dist[neighbour] = new_dist
                    heapq.heappush(heap, (new_dist, neighbour))

        return float('inf')

    # ------------------------------------------------------------------
    # Boltzmann conversion
    # ------------------------------------------------------------------

    def _boltzmann(
        self,
        advantages: Dict[int, float],
        available: List[int],
        num_actions: int,
    ) -> np.ndarray:
        """Convert advantages to a Boltzmann probability distribution.

        Actions not in *available* get probability 0.
        """
        probs = np.zeros(num_actions, dtype=np.float64)
        if not available:
            return probs

        vals = np.array([advantages.get(a, 0.0) for a in available],
                        dtype=np.float64)

        # Replace non-finite advantages (inf/-inf/NaN from unreachable
        # targets) with explicit finite caps so the softmax stays
        # well-defined.  np.clip alone does not handle NaN, so we
        # overwrite non-finite entries directly.
        finite_mask = np.isfinite(vals)
        if not finite_mask.all():
            max_finite = (np.abs(vals[finite_mask]).max()
                          if finite_mask.any() else 0.0)
            cap = max(max_finite + 1.0, 100.0)
            vals[~finite_mask & (vals > 0)] = cap
            vals[~finite_mask & (vals < 0)] = -cap
            # NaN (neither >0 nor <0) → treat as neutral
            vals[np.isnan(vals)] = 0.0

        # Numerically stable softmax
        vals *= self.beta
        vals -= vals.max()
        exp_vals = np.exp(vals)
        total = exp_vals.sum()

        if total > 0:
            exp_vals /= total
        else:
            exp_vals = np.ones_like(exp_vals) / len(exp_vals)

        for i, action in enumerate(available):
            probs[action] = exp_vals[i]

        return probs
