"""
Exploration policies for BushWorld Phase 2 training.

Mirrors the multigrid exploration-policy module but is much simpler: BushWorld
has a small discrete action space, so a configurable categorical policy over the
joint robot action space is sufficient for epsilon exploration. When no
exploration policy is supplied to the trainer, a uniform random policy is used
(handled by the base trainer), so this class is optional.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from empo.robot_policy import RobotPolicy


class BushWorldRobotExplorationPolicy(RobotPolicy):
    """Per-robot categorical exploration policy for BushWorld.

    Args:
        num_actions: Number of actions available to each robot.
        action_probs: Optional probabilities over a single robot's actions. If
            ``None``, a uniform distribution is used.
        robot_agent_indices: Robot indices; auto-detected on ``reset`` if None.
    """

    def __init__(
        self,
        num_actions: int,
        action_probs: Optional[List[float]] = None,
        robot_agent_indices: Optional[List[int]] = None,
    ):
        self.num_actions = num_actions
        if action_probs is None:
            action_probs = [1.0 / num_actions] * num_actions
        if len(action_probs) != num_actions:
            raise ValueError(
                f"action_probs must have {num_actions} elements, got {len(action_probs)}"
            )
        total = sum(action_probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"action_probs must sum to 1.0, got {total}")
        self.action_probs = np.array(action_probs, dtype=np.float64)
        self._robot_agent_indices = robot_agent_indices
        self._world_model = None

    def reset(self, world_model: Any) -> None:
        self._world_model = world_model
        if self._robot_agent_indices is None and hasattr(world_model, "robot_agent_indices"):
            self._robot_agent_indices = world_model.robot_agent_indices

    @property
    def robot_agent_indices(self) -> List[int]:
        if self._robot_agent_indices is None:
            return [0]
        return self._robot_agent_indices

    def sample(self, state: Any) -> Tuple[int, ...]:
        return tuple(
            int(np.random.choice(self.num_actions, p=self.action_probs))
            for _ in self.robot_agent_indices
        )
