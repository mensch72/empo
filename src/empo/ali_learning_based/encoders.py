"""
State and goal encoders for multigrid environments.

These are pure data transforms (not nn.Modules). They convert raw
multigrid state tuples and goal objects into fixed-size tensors
suitable for neural network input.

Design choices:
    - StateEncoder caches the static grid layout (walls, goals) at init
      time since these never change during an episode.
    - The grid is encoded as a multi-channel spatial tensor, then flattened.
      This preserves spatial structure in the data while keeping the encoder
      simple (no CNNs here — the network can learn spatial patterns from
      the flat vector).
    - GoalEncoder normalizes coordinates to [0, 1] so the network doesn't
      need to know grid dimensions.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Grid channel indices
# ---------------------------------------------------------------------------
CH_WALL = 0       # 1.0 if wall
CH_DOOR = 1       # 0.33=open, 0.67=closed, 1.0=locked
CH_KEY = 2        # 1.0 if key on the ground
CH_GOAL = 3       # 1.0 if goal cell
CH_ROCK = 4       # 1.0 if rock
CH_BLOCK = 5      # 1.0 if block
CH_ROBOT = 6      # 1.0 if robot agent here
CH_HUMAN = 7      # 1.0 if any human agent here
NUM_GRID_CHANNELS = 8

# Per-agent feature size: 4 (direction one-hot) + 1 (carrying key)
AGENT_FEATURE_SIZE = 5


class StateEncoder:
    """
    Converts a multigrid state tuple into a flat tensor.

    The output tensor has two concatenated parts:

        [spatial grid (flattened) | agent features | global features]

    Spatial grid — shape (NUM_GRID_CHANNELS, H, W), then flattened:
        ch0 wall:  1.0 where walls are (static, cached at init)
        ch1 door:  encodes door state (0.33 open / 0.67 closed / 1.0 locked)
        ch2 key:   1.0 where a key sits on the ground (removed if carried)
        ch3 goal:  1.0 where goal cells are (static, cached at init)
        ch4 rock:  1.0 where rocks are (from mobile_objects)
        ch5 block: 1.0 where blocks are (from mobile_objects)
        ch6 robot: 1.0 at the robot agent's position
        ch7 human: 1.0 at each human agent's position

    Agent features — per agent, 5 values:
        [dir_east, dir_south, dir_west, dir_north, carrying_key]
        Direction is one-hot (4 values). carrying_key is binary.

    Global features — 1 value:
        [time_remaining]   normalized to [0, 1]

    Args:
        world_model: A multigrid WorldModel (must have .grid, .height,
            .width, .max_steps, .agents attributes).
        robot_agent_index: Index of the robot agent (default 0).
        human_agent_indices: Indices of human agents. If None, all agents
            except the robot are treated as humans.
    """

    def __init__(
        self,
        world_model: Any,
        robot_agent_index: int = 0,
        human_agent_indices: Optional[List[int]] = None,
    ):
        self.height: int = world_model.height
        self.width: int = world_model.width
        self.max_steps: int = world_model.max_steps
        self.robot_agent_index = robot_agent_index

        num_agents = len(world_model.agents)
        if human_agent_indices is not None:
            self.human_agent_indices = list(human_agent_indices)
        else:
            self.human_agent_indices = [
                i for i in range(num_agents) if i != robot_agent_index
            ]
        self.num_agents = num_agents

        # --- cache static grid features (don't change within an episode) ---
        self._wall_grid = torch.zeros(self.height, self.width)
        self._goal_grid = torch.zeros(self.height, self.width)
        self._key_positions: Dict[str, Tuple[int, int]] = {}   # color -> (x, y)
        self._door_positions: Set[Tuple[int, int]] = set()

        for y in range(self.height):
            for x in range(self.width):
                cell = world_model.grid.get(x, y)
                if cell is None:
                    continue
                cell_type = getattr(cell, "type", None)
                if cell_type == "wall":
                    self._wall_grid[y, x] = 1.0
                elif cell_type == "goal":
                    self._goal_grid[y, x] = 1.0
                elif cell_type == "key":
                    color = getattr(cell, "color", "red")
                    self._key_positions[color] = (x, y)
                elif cell_type == "door":
                    self._door_positions.add((x, y))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Total output dimension of encode()."""
        grid_dim = NUM_GRID_CHANNELS * self.height * self.width
        agent_dim = AGENT_FEATURE_SIZE * self.num_agents
        global_dim = 1  # time_remaining
        return grid_dim + agent_dim + global_dim

    def encode(self, state: Tuple) -> torch.Tensor:
        """
        Encode a single state into a flat tensor.

        Args:
            state: The 4-tuple (step_count, agent_states,
                   mobile_objects, mutable_objects).

        Returns:
            A 1-D float tensor of shape (self.dim,).
        """
        step_count, agent_states, mobile_objects, mutable_objects = state

        grid = torch.zeros(NUM_GRID_CHANNELS, self.height, self.width)

        # ch0: walls (static)
        grid[CH_WALL] = self._wall_grid

        # ch1: door states (dynamic, from mutable_objects)
        self._encode_doors(grid, mutable_objects)

        # ch2: keys on the ground (static positions minus carried ones)
        self._encode_keys(grid, agent_states)

        # ch3: goals (static)
        grid[CH_GOAL] = self._goal_grid

        # ch4, ch5: rocks and blocks (from mobile_objects)
        self._encode_mobile(grid, mobile_objects)

        # ch6: robot position
        self._encode_agent_position(grid, agent_states, self.robot_agent_index, CH_ROBOT)

        # ch7: human positions
        for h_idx in self.human_agent_indices:
            self._encode_agent_position(grid, agent_states, h_idx, CH_HUMAN)

        # --- agent features ---
        agent_features = self._encode_agent_features(agent_states)

        # --- global features ---
        time_remaining = (self.max_steps - step_count) / self.max_steps
        global_features = torch.tensor([time_remaining], dtype=torch.float32)

        return torch.cat([grid.flatten(), agent_features, global_features])

    def encode_batch(self, states: List[Tuple]) -> torch.Tensor:
        """Encode a list of states into a (batch, dim) tensor."""
        return torch.stack([self.encode(s) for s in states])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_doors(
        self, grid: torch.Tensor, mutable_objects: Tuple
    ) -> None:
        """Write door states into grid channel CH_DOOR."""
        for obj in mutable_objects:
            if obj[0] != "door":
                continue
            x, y = int(obj[1]), int(obj[2])
            is_open, is_locked = obj[3], obj[4]
            if is_open:
                grid[CH_DOOR, y, x] = 0.33
            elif is_locked:
                grid[CH_DOOR, y, x] = 1.0
            else:  # closed but unlocked
                grid[CH_DOOR, y, x] = 0.67

    def _encode_keys(
        self, grid: torch.Tensor, agent_states: Tuple
    ) -> None:
        """Write key positions into grid channel CH_KEY.

        A key is shown on the grid only if no agent is currently
        carrying a key of that color.
        """
        # Collect colors of keys that are being carried
        carried_colors: Set[str] = set()
        for agent_state in agent_states:
            if agent_state[6] == "key":  # carrying_type
                carried_colors.add(agent_state[7])  # carrying_color

        for color, (x, y) in self._key_positions.items():
            if color not in carried_colors:
                grid[CH_KEY, y, x] = 1.0

    def _encode_mobile(
        self, grid: torch.Tensor, mobile_objects: Tuple
    ) -> None:
        """Write rock/block positions into grid channels CH_ROCK / CH_BLOCK."""
        for obj in mobile_objects:
            obj_type, x, y = obj[0], int(obj[1]), int(obj[2])
            if 0 <= x < self.width and 0 <= y < self.height:
                if obj_type == "rock":
                    grid[CH_ROCK, y, x] = 1.0
                elif obj_type == "block":
                    grid[CH_BLOCK, y, x] = 1.0

    @staticmethod
    def _encode_agent_position(
        grid: torch.Tensor,
        agent_states: Tuple,
        agent_idx: int,
        channel: int,
    ) -> None:
        """Set a 1.0 in `channel` at the agent's position (if alive)."""
        if agent_idx >= len(agent_states):
            return
        agent = agent_states[agent_idx]
        if agent[0] is None:  # position is None → not on the grid
            return
        x, y = int(agent[0]), int(agent[1])
        grid[channel, y, x] = 1.0

    def _encode_agent_features(self, agent_states: Tuple) -> torch.Tensor:
        """Build a flat vector of per-agent features.

        For each agent (in index order):
            [dir_east, dir_south, dir_west, dir_north, carrying_key]
        """
        features = []
        for i in range(self.num_agents):
            agent = agent_states[i]

            # Direction one-hot (4 values)
            direction = agent[2]  # 0=E, 1=S, 2=W, 3=N
            dir_onehot = [0.0, 0.0, 0.0, 0.0]
            if direction is not None and 0 <= direction <= 3:
                dir_onehot[direction] = 1.0

            # Carrying key (binary)
            carrying_key = 1.0 if agent[6] == "key" else 0.0

            features.extend(dir_onehot)
            features.append(carrying_key)

        return torch.tensor(features, dtype=torch.float32)


class GoalEncoder:
    """
    Converts a multigrid goal into a normalized coordinate tensor.

    Output is always 4 values: [x1/W, y1/H, x2/W, y2/H].

    For point goals (ReachCellGoal at position (x,y)):
        → [x/W, y/H, x/W, y/H]

    For rectangle goals (ReachRectangleGoal with rect (x1,y1,x2,y2)):
        → [x1/W, y1/H, x2/W, y2/H]

    Normalizing to [0, 1] means the network doesn't need to know
    absolute grid dimensions.

    Args:
        world_model: A multigrid WorldModel (for .height, .width).
    """

    GOAL_DIM = 4

    def __init__(self, world_model: Any):
        self.height: int = world_model.height
        self.width: int = world_model.width

    @property
    def dim(self) -> int:
        """Output dimension of encode()."""
        return self.GOAL_DIM

    def encode(self, goal: Any) -> torch.Tensor:
        """
        Encode a single goal into a 1-D tensor of shape (4,).

        Args:
            goal: A PossibleGoal with .target_rect or .target_pos.

        Returns:
            Tensor [x1/W, y1/H, x2/W, y2/H].
        """
        if hasattr(goal, "target_rect"):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, "target_pos"):
            x, y = goal.target_pos
            x1, y1, x2, y2 = x, y, x, y
        else:
            raise ValueError(
                f"Goal must have target_rect or target_pos, got {type(goal)}"
            )

        return torch.tensor(
            [
                float(x1) / self.width,
                float(y1) / self.height,
                float(x2) / self.width,
                float(y2) / self.height,
            ],
            dtype=torch.float32,
        )

    def encode_batch(self, goals: List[Any]) -> torch.Tensor:
        """Encode a list of goals into a (batch, 4) tensor."""
        return torch.stack([self.encode(g) for g in goals])
