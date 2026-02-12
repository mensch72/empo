from __future__ import annotations

from typing import Any, List, Optional, Tuple
import math

import numpy as np

from empo.human_policy_prior import HumanPolicyPrior


class MinRobotRiskHumanPolicy(HumanPolicyPrior):
    """
    Human policy prior that:
    1. Escapes if currently in danger of being hit by a rock
    2. Otherwise stays still

    This creates simple, predictable human behavior that avoids rocks but
    doesn't introduce complexity by moving unnecessarily.
    """

    DIR_TO_VEC = [
        (1, 0),   # east (0)
        (0, 1),   # south (1)
        (-1, 0),  # west (2)
        (0, -1),  # north (3)
    ]

    ACTION_STILL = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_FORWARD = 3

    def __init__(
        self,
        world_model: Any,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        beta: float = 5.0,
        danger_penalty: float = 10.0,
    ):
        super().__init__(world_model, human_agent_indices)
        self.robot_agent_indices = robot_agent_indices
        self.beta = beta
        self.danger_penalty = danger_penalty

    def __call__(
        self,
        state: Any,
        human_agent_index: int,
        possible_goal: Optional[Any] = None,
    ) -> np.ndarray:
        num_actions = self.world_model.action_space.n
        if num_actions < 4:
            return np.ones(num_actions) / num_actions

        saved_state = self.world_model.get_state()
        self.world_model.set_state(state)

        agent_state = state[1][human_agent_index]
        hx, hy, hdir = int(agent_state[0]), int(agent_state[1]), int(agent_state[2])
        current_pos = (hx, hy)

        # Check if human is currently in danger (rock can be pushed into current cell)
        currently_in_danger = self._robot_can_push_rock_into(current_pos)

        scores = np.zeros(num_actions, dtype=np.float32)
        for action in range(num_actions):
            if action > self.ACTION_FORWARD:
                # Ignore pickup, drop, toggle, done for simplicity
                scores[action] = -100.0
                continue

            next_pos, next_dir = self._simulate_human_move(hx, hy, hdir, action)
            next_danger = self._robot_can_push_rock_into(next_pos)

            if currently_in_danger:
                # In danger: prioritize escaping to safe positions
                if next_pos != current_pos and not next_danger:
                    # Moving to a safe cell - very good
                    scores[action] = self.danger_penalty
                elif next_danger:
                    # Check if this is a turn action that would face an escape route
                    # If so, give it a bonus (it's a necessary intermediate step)
                    if next_pos == current_pos:  # Turn action (didn't move)
                        # Check if moving forward from the new direction would be safe
                        dx, dy = self.DIR_TO_VEC[next_dir]
                        future_pos = (current_pos[0] + dx, current_pos[1] + dy)
                        future_cell = self.world_model.grid.get(future_pos[0], future_pos[1])
                        can_move = future_cell is None or (hasattr(future_cell, "can_overlap") and future_cell.can_overlap())
                        future_danger = self._robot_can_push_rock_into(future_pos)

                        if can_move and not future_danger:
                            # This turn faces toward safety - good!
                            scores[action] = self.danger_penalty * 0.8
                        else:
                            # Turn toward another dangerous or blocked cell
                            scores[action] = -self.danger_penalty
                    else:
                        # Moved into another dangerous cell - bad
                        scores[action] = -self.danger_penalty
                else:
                    # Staying still while in danger - bad
                    scores[action] = -self.danger_penalty * 0.5
            else:
                # Not in danger: strongly prefer staying still
                if action == self.ACTION_STILL:
                    scores[action] = 10.0  # Very high bonus for staying still
                elif next_danger:
                    # Moving into danger - very bad
                    scores[action] = -self.danger_penalty
                else:
                    # Any movement when safe - penalize
                    scores[action] = -1.0

        self.world_model.set_state(saved_state)

        # Softmax with temperature beta
        max_s = scores.max()
        exp_s = np.exp(self.beta * (scores - max_s))
        probs = exp_s / exp_s.sum()
        return probs

    def best_action(
        self,
        state: Any,
        human_agent_index: int,
        possible_goal: Optional[Any] = None,
    ) -> int:
        probs = self(state, human_agent_index, possible_goal)
        return int(np.argmax(probs))

    def _simulate_human_move(
        self, x: int, y: int, direction: int, action: int
    ) -> Tuple[Tuple[int, int], int]:
        if action == self.ACTION_LEFT:
            return (x, y), (direction - 1) % 4
        if action == self.ACTION_RIGHT:
            return (x, y), (direction + 1) % 4
        if action == self.ACTION_STILL:
            return (x, y), direction

        # forward
        dx, dy = self.DIR_TO_VEC[direction]
        nx, ny = x + dx, y + dy
        cell = self.world_model.grid.get(nx, ny)
        if cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap()):
            return (nx, ny), direction
        return (x, y), direction

    def _robot_can_push_rock_into(self, target_pos: Tuple[int, int]) -> bool:
        """
        Check if any robot can push a rock into target_pos soon (within 3-4 turns).

        Strategy:
        1. Find all rocks in the grid
        2. For each rock, check if pushing it would hit target_pos
        3. If yes, check if robot can reach the push position within 3-4 moves
        """
        tx, ty = target_pos

        # Find all rocks in the grid
        for rock_x in range(self.world_model.width):
            for rock_y in range(self.world_model.height):
                cell = self.world_model.grid.get(rock_x, rock_y)
                if cell is None or getattr(cell, "type", None) != "rock":
                    continue

                # For each direction, check if pushing the rock would hit the target
                for direction in range(4):
                    dx, dy = self.DIR_TO_VEC[direction]
                    dest_x, dest_y = rock_x + dx, rock_y + dy

                    if (dest_x, dest_y) != (tx, ty):
                        continue

                    # This rock WOULD hit the target if pushed from this direction
                    # The robot needs to be at (rock_x - dx, rock_y - dy) facing direction
                    robot_push_x = rock_x - dx
                    robot_push_y = rock_y - dy

                    # Check if any robot can reach (robot_push_x, robot_push_y) soon
                    for ridx in self.robot_agent_indices:
                        agent = self.world_model.agents[ridx]
                        if not hasattr(agent, "pos") or agent.pos is None:
                            continue
                        if hasattr(agent, "can_push_rocks") and not agent.can_push_rocks:
                            continue

                        rx, ry = int(agent.pos[0]), int(agent.pos[1])

                        # Manhattan distance to the push position
                        dist = abs(rx - robot_push_x) + abs(ry - robot_push_y)

                        # If robot can reach within 3-4 moves, it's dangerous
                        # (being conservative to give humans time to escape)
                        if dist <= 4:
                            return True

        return False
