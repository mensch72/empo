import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional

State = Any

from gym_multigrid.multigrid import MultiGridEnv, Key

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

@dataclass
class StateSamplerConfig:
    """Base configuration for state sampling."""
    seed: Optional[int] = None

    sample_time: bool = False
    time_min: int = 0
    time_max: Optional[int] = None  # If None, use env.max_steps

    max_distance_from_current: Optional[float] = None 


class PlacementStrategy(ABC):
    """Abstract base for object placement strategies."""

    @abstractmethod
    def select_positions(
        self,
        num_positions: int,
        valid_cells: List[Tuple[int, int]],
        current_positions: Optional[List[Tuple[int, int]]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """
        Select positions for objects.

        Args:
            num_positions: Number of positions to select
            valid_cells: List of all valid (x, y) positions
            current_positions: Current positions of objects
            constraints: Additional constraints for placement

        Returns:
            List of (x, y) positions
        """
        pass


class ProximityPlacement(PlacementStrategy):
    """Place objects near their current positions."""

    def __init__(self, max_distance: float):
        """
        Initialize proximity placement.

        Args:
            max_distance: Maximum Manhattan distance from current position
        """
        self.max_distance = max_distance

    def select_positions(
        self,
        num_positions: int,
        valid_cells: List[Tuple[int, int]],
        current_positions: Optional[List[Tuple[int, int]]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """Select positions near current positions."""
        if current_positions is None or len(current_positions) == 0:
            # Fallback to uniform if no current positions
            return random.sample(valid_cells, num_positions)

        new_positions = []

        for i in range(num_positions):
            if i < len(current_positions):
                current_pos = current_positions[i]
            else:
                # More objects than current positions - pick any available cell
                available = [c for c in valid_cells if c not in new_positions]
                if available:
                    new_positions.append(random.choice(available))
                else:
                    raise ValueError(
                        f"Cannot place object {i+1}/{num_positions}: "
                        f"all {len(valid_cells)} valid cells already occupied"
                    )
                continue

            nearby_cells = [
                cell for cell in valid_cells
                if manhattan_distance(cell, current_pos) <= self.max_distance
            ]

            available_nearby = [c for c in nearby_cells if c not in new_positions]

            if available_nearby:
                new_positions.append(random.choice(available_nearby))
            else:
                # Fallback to any valid cell not yet selected
                available_all = [c for c in valid_cells if c not in new_positions]
                if available_all:
                    new_positions.append(random.choice(available_all))
                else:
                    raise ValueError(
                        f"Cannot place {num_positions} objects: only {len(valid_cells)} valid cells available"
                    )

        return new_positions


class UniformPlacement(PlacementStrategy):
    """Uniformly random placement strategy."""

    def select_positions(
        self,
        num_positions: int,
        valid_cells: List[Tuple[int, int]],
        current_positions: Optional[List[Tuple[int, int]]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, int]]:
        """Select positions uniformly at random."""
        if num_positions > len(valid_cells):
            raise ValueError(
                f"Not enough valid cells ({len(valid_cells)}) for "
                f"{num_positions} positions"
            )

        return random.sample(valid_cells, num_positions)


class StateSampler(ABC):
    """Base class for state sampling."""
    
    def __init__(
        self,
        world_model: Any,
        config: Optional[StateSamplerConfig] = None
    ):
        """
        Initialize state sampler.

        Args:
            world_model: The environment (must have get_state/set_state methods)
            config: Configuration for sampling behavior
        """
        self.world_model = world_model
        self.config = config or StateSamplerConfig()

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

    @abstractmethod
    def sample(self) -> None:
        """Sample a random state and apply to world_model."""
        pass

    def _sample_time(self) -> int:
        """Sample a random time step."""
        if not self.config.sample_time:
            return self.world_model.get_state()[0]

        max_time = self.config.time_max
        if max_time is None:
            max_time = getattr(self.world_model, 'max_steps', 100)

        return random.randint(self.config.time_min, max_time)

    @abstractmethod
    def _get_valid_cells(self) -> Dict[str, List[Tuple[int, int]]]:
        """Get valid cells categorized by type."""
        pass


@dataclass
class MultigridStateSamplerConfig(StateSamplerConfig):
    """Configuration specific to MultiGrid environments."""

    # Position sampling probabilities (should sum to <= 1.0)
    p_agent_on_lava: float = 0.0
    p_agent_on_unsteady_ground: float = 0.0

    # Inventory sampling
    p_agent_carries_key: float = 0.0

    # Mutable objects sampling
    sample_mutable_objects: bool = False
    p_door_open: float = 0.5
    p_door_locked: float = 0.5
    p_magic_wall_active: float = 0.5
    p_button_enabled: float = 0.5

    # Agent state
    sample_agent_direction: bool = False
    sample_agent_paused: bool = False
    p_agent_paused: float = 0.0

    # Validation
    validate_state: bool = True  # Check if sampled state is valid


class MultigridStateSampler(StateSampler):
    """State sampler for MultiGrid environments."""

    def __init__(
        self,
        world_model: MultiGridEnv,
        config: Optional[MultigridStateSamplerConfig] = None,
        placement_strategy: Optional[PlacementStrategy] = None
    ):
        """
        Initialize MultiGrid state sampler.

        Args:
            world_model: MultiGridEnv instance
            config: Configuration for sampling
            placement_strategy: Strategy for placing objects
        """
        super().__init__(world_model, config or MultigridStateSamplerConfig())
        self.config: MultigridStateSamplerConfig

        # Default to uniform placement if none provided
        if placement_strategy is None:
            if self.config.max_distance_from_current is not None:
                placement_strategy = ProximityPlacement(
                    self.config.max_distance_from_current
                )
            else:
                placement_strategy = UniformPlacement()

        self.placement_strategy = placement_strategy

    def sample(self) -> None:
        """Sample a random state and apply to environment."""
        env = self.world_model
        current_state = env.get_state()

        new_time = self._sample_time()
        valid_cells_by_type = self._get_valid_cells()
        new_mobile_objects, keys_for_agents = self._sample_mobile_objects(
            current_state, valid_cells_by_type
        )

        occupied_by_objects = {(x, y) for _, x, y in new_mobile_objects}
        occupied_by_keys, grid_keys_for_agents = self._relocate_grid_keys(
            valid_cells_by_type, occupied_by_objects
        )
        keys_for_agents = keys_for_agents + grid_keys_for_agents
        occupied_by_objects |= occupied_by_keys

        valid_cells_for_agents = {
            terrain_type: [cell for cell in cells if cell not in occupied_by_objects]
            for terrain_type, cells in valid_cells_by_type.items()
        }

        new_agent_states = self._sample_agent_states(
            current_state, valid_cells_for_agents, keys_for_agents
        )

        new_mutable_objects = self._sample_mutable_objects(current_state)

        new_state = (
            new_time,
            new_agent_states,
            new_mobile_objects,
            new_mutable_objects
        )

        if self.config.validate_state:
            self._validate_state(new_state)

        env.set_state(new_state)

    def _sample_mobile_objects(
        self,
        current_state: Tuple,
        valid_cells_by_type: Dict[str, List[Tuple[int, int]]]
    ) -> Tuple[Tuple, List[str]]:
        """
        Sample positions for blocks, rocks, and keys.

        Returns:
            Tuple of (new_mobile_objects, keys_for_agents)
            where keys_for_agents is list of key colors to distribute to agents
        """
        mobile_objects = list(current_state[2])

        blocks = []
        rocks = []

        for obj_type, x, y in mobile_objects:
            if obj_type == 'block':
                blocks.append(obj_type)
            elif obj_type == 'rock':
                rocks.append(obj_type)

        current_positions = [(x, y) for _, x, y in mobile_objects]
        all_valid = self._combine_valid_cells(valid_cells_by_type)
        num_objects = len(blocks) + len(rocks)

        if num_objects > 0:
            new_positions = self.placement_strategy.select_positions(
                num_positions=num_objects,
                valid_cells=all_valid,
                current_positions=current_positions[:num_objects] if current_positions else None
            )
        else:
            new_positions = []

        new_mobile = []
        idx = 0

        for block_type in blocks:
            x, y = new_positions[idx]
            new_mobile.append((block_type, x, y))
            idx += 1

        for rock_type in rocks:
            x, y = new_positions[idx]
            new_mobile.append((rock_type, x, y))
            idx += 1

        return tuple(new_mobile), []

    def _sample_agent_states(
        self,
        current_state: Tuple,
        valid_cells_by_type: Dict[str, List[Tuple[int, int]]],
        keys_for_agents: List[str]
    ) -> Tuple:
        """
        Sample agent positions, directions, and inventories.

        Args:
            current_state: Current state tuple
            valid_cells_by_type: Valid cells categorized by terrain
            keys_for_agents: Keys to distribute to agents
        """
        agent_states = list(current_state[1])
        current_positions = [(state[0], state[1]) for state in agent_states]
        new_agent_states = []
        placed_agent_positions = []  # Track where we've already placed agents

        for agent_idx, agent_state in enumerate(agent_states):
            # Sample position based on terrain probabilities
            terrain_type = self._sample_terrain_type()
            agent_cells = valid_cells_by_type.get(terrain_type, valid_cells_by_type['empty'])

            if not agent_cells:
                agent_cells = self._combine_valid_cells(valid_cells_by_type)

            # Filter out already-placed agent positions to avoid overlaps
            available_agent_cells = [c for c in agent_cells if c not in placed_agent_positions]

            if not available_agent_cells:
                # All preferred cells taken - try all valid cells
                all_cells = self._combine_valid_cells(valid_cells_by_type)
                available_agent_cells = [c for c in all_cells if c not in placed_agent_positions]

            if available_agent_cells:
                # Use placement strategy for position
                if self.config.max_distance_from_current is not None and current_positions:
                    positions = self.placement_strategy.select_positions(
                        num_positions=1,
                        valid_cells=available_agent_cells,
                        current_positions=[current_positions[agent_idx]]
                    )
                    new_x, new_y = positions[0]
                else:
                    new_x, new_y = random.choice(available_agent_cells)

                # Track this position to avoid overlap with next agents
                placed_agent_positions.append((new_x, new_y))
            else:
                # No available cells - this shouldn't happen if validate_state is working
                raise ValueError(
                    f"Cannot place agent {agent_idx}: no available cells "
                    f"(already placed {len(placed_agent_positions)} agents)"
                )

            # Sample direction
            if self.config.sample_agent_direction:
                direction = random.randint(0, 3)
            else:
                direction = agent_state[2]

            # Sample paused state
            if self.config.sample_agent_paused:
                paused = random.random() < self.config.p_agent_paused
            else:
                paused = agent_state[5]

            # Build agent state (without key)
            new_agent_states.append((
                new_x,
                new_y,
                direction,
                agent_state[3],  # terminated
                agent_state[4],  # started
                paused,
                None,  # carrying_type
                None,  # carrying_color
                agent_state[8]   # forced_action
            ))

        # Distribute keys to agents
        new_agent_states = self._distribute_keys_to_agents(
            new_agent_states, keys_for_agents
        )

        return tuple(new_agent_states)

    def _distribute_keys_to_agents(
        self,
        agent_states: List[Tuple],
        keys: List[str]
    ) -> List[Tuple]:
        """
        Distribute keys to agents.

        Args:
            agent_states: List of agent state tuples
            keys: List of key colors to distribute

        Returns:
            Updated agent states with keys
        """
        if not keys:
            return agent_states

        new_agent_states = list(agent_states)

        # Shuffle keys for random distribution
        keys_copy = keys.copy()
        random.shuffle(keys_copy)

        # Assign keys to random agents
        for key_color in keys_copy:
            # Find agents without keys
            available_agents = [
                i for i, state in enumerate(new_agent_states)
                if state[6] is None  # No key
            ]

            if not available_agents:
                # All agents have keys
                continue

            # Pick random agent from available ones
            agent_idx = random.choice(available_agents)

            old_state = new_agent_states[agent_idx]
            new_agent_states[agent_idx] = (
                old_state[0],  # pos_x
                old_state[1],  # pos_y
                old_state[2],  # dir
                old_state[3],  # terminated
                old_state[4],  # started
                old_state[5],  # paused
                'key',         # carrying_type
                key_color,     # carrying_color
                old_state[8]   # forced_action
            )

        return new_agent_states

    def _sample_mutable_objects(self, current_state: Tuple) -> Tuple:
        """Sample states for mutable objects (doors, buttons, etc.)."""
        if not self.config.sample_mutable_objects:
            return current_state[3]

        mutable_objects = list(current_state[3])
        new_mutable = []

        for obj in mutable_objects:
            obj_type = obj[0]

            if obj_type == 'door':
                _, x, y, _, _ = obj  # Ignore old is_open, is_locked
                new_is_open = random.random() < self.config.p_door_open
                new_is_locked = random.random() < self.config.p_door_locked
                new_mutable.append(('door', x, y, new_is_open, new_is_locked))

            elif obj_type == 'magicwall':
                _, x, y, _ = obj  # Ignore old active
                new_active = random.random() < self.config.p_magic_wall_active
                new_mutable.append(('magicwall', x, y, new_active))

            elif obj_type == 'killbutton':
                # (type, x, y, enabled)
                _, x, y, _ = obj  # Ignore old enabled
                new_enabled = random.random() < self.config.p_button_enabled
                new_mutable.append(('killbutton', x, y, new_enabled))

            elif obj_type == 'pauseswitch':
                # (type, x, y, is_on, enabled)
                _, x, y, _, _ = obj  # Ignore old is_on, enabled
                new_enabled = random.random() < self.config.p_button_enabled
                new_is_on = random.random() < 0.5
                new_mutable.append(('pauseswitch', x, y, new_is_on, new_enabled))

            elif obj_type == 'controlbutton':
                # (type, x, y, enabled, controlled_agent_idx, triggered_action, awaiting_action)
                new_obj = list(obj)
                new_obj[3] = random.random() < self.config.p_button_enabled
                new_mutable.append(tuple(new_obj))

            else:
                new_mutable.append(obj)

        return tuple(new_mutable)

    def _relocate_grid_keys(
        self,
        valid_cells_by_type: Dict[str, List[Tuple[int, int]]],
        already_occupied: set
    ) -> Tuple[set, List[str]]:
        """
        Find keys sitting on the grid, clear them, and either place each at a new
        position or assign it to an agent (based on p_agent_carries_key).

        Returns:
            (new_key_positions, keys_for_agents) where new_key_positions is the set of
            (x, y) cells now occupied by keys, and keys_for_agents is a list of key
            colors to be distributed to agents.
        """
        env = self.world_model
        num_agents = len(env.agents)
        current_state = env.get_state()

        grid_keys = []
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell is not None and cell.type == 'key':
                    grid_keys.append((x, y, cell))

        # Also collect keys currently carried by agents so they get redistributed
        carried_key_colors = []
        for agent_state in current_state[1]:
            if agent_state[6] == 'key':
                carried_key_colors.append(agent_state[7])

        if not grid_keys and not carried_key_colors:
            return set(), []

        # Clear all keys from the grid first so their old cells are free
        for x, y, key_obj in grid_keys:
            env.grid.set(x, y, None)

        all_valid = self._combine_valid_cells(valid_cells_by_type)
        new_key_positions = set()
        keys_for_agents = []

        for old_x, old_y, key_obj in grid_keys:
            if random.random() < self.config.p_agent_carries_key and num_agents > 0:
                # Assign to an agent — key stays off the grid
                keys_for_agents.append(key_obj.color)
                continue

            available = [c for c in all_valid if c not in already_occupied and c not in new_key_positions]

            if not available:
                # No room — put key back in original position
                env.grid.set(old_x, old_y, key_obj)
                new_key_positions.add((old_x, old_y))
                continue

            if self.config.max_distance_from_current is not None:
                positions = self.placement_strategy.select_positions(
                    num_positions=1,
                    valid_cells=available,
                    current_positions=[(old_x, old_y)]
                )
                new_x, new_y = positions[0]
            else:
                new_x, new_y = random.choice(available)

            env.grid.set(new_x, new_y, key_obj)
            new_key_positions.add((new_x, new_y))

        # Redistribute carried keys: re-apply p_agent_carries_key for each
        for color in carried_key_colors:
            if random.random() < self.config.p_agent_carries_key and num_agents > 0:
                keys_for_agents.append(color)
            else:
                available = [c for c in all_valid if c not in already_occupied and c not in new_key_positions]
                if available:
                    new_x, new_y = random.choice(available)
                    env.grid.set(new_x, new_y, Key(env.objects, color=color))
                    new_key_positions.add((new_x, new_y))
                else:
                    # No room on grid — keep it with an agent
                    keys_for_agents.append(color)

        return new_key_positions, keys_for_agents

    def _sample_terrain_type(self) -> str:
        """Sample terrain type based on configured probabilities."""
        rand = random.random()

        if rand < self.config.p_agent_on_lava:
            return 'lava'
        elif rand < self.config.p_agent_on_lava + self.config.p_agent_on_unsteady_ground:
            return 'unsteady'
        else:
            return 'empty'

    def _get_valid_cells(self) -> Dict[str, List[Tuple[int, int]]]:
        """Get valid cells categorized by terrain type."""
        env = self.world_model

        cells = {
            'empty': [],
            'lava': [],
            'unsteady': []
        }

        for x in range(env.width):
            for y in range(env.height):
                terrain_cell = env.terrain_grid.get(x, y)
                cell = env.grid.get(x, y)

                # Check terrain first
                if terrain_cell and hasattr(terrain_cell, 'type'):
                    if terrain_cell.type == 'lava':
                        cells['lava'].append((x, y))
                        continue
                    elif terrain_cell.type == 'unsteadyground':
                        cells['unsteady'].append((x, y))
                        continue

                # Check main grid
                if cell is None:
                    cells['empty'].append((x, y))
                elif hasattr(cell, 'type'):
                    if cell.type in ('agent', 'block', 'rock', 'key'):
                        # Mobile objects - count as empty
                        cells['empty'].append((x, y))
                    # elif hasattr(cell, 'can_overlap') and cell.can_overlap():
                    #    cells['empty'].append((x, y))

        return cells

    def _combine_valid_cells(
        self, valid_cells_by_type: Dict[str, List[Tuple[int, int]]]
    ) -> List[Tuple[int, int]]:
        """Combine all valid cells into a single list."""
        all_cells = []
        for cell_list in valid_cells_by_type.values():
            all_cells.extend(cell_list)
        return all_cells

    def _validate_state(self, state: Tuple) -> None:
        """Validate that the sampled state is legal."""
        env = self.world_model

        # Validate time is in range
        time = state[0]
        if time < 0:
            raise ValueError(f"Invalid time: {time} < 0")
        if hasattr(env, 'max_steps') and time > env.max_steps:
            raise ValueError(f"Invalid time: {time} > {env.max_steps}")

        # Validate agent positions are in bounds
        agent_states = state[1]
        for i, agent_state in enumerate(agent_states):
            x, y = agent_state[0], agent_state[1]
            if x is not None and y is not None:
                if not (0 <= x < env.width and 0 <= y < env.height):
                    raise ValueError(f"Agent {i} position ({x}, {y}) out of bounds")

        # Validate mobile object positions are in bounds
        mobile_objects = state[2]
        for obj_type, x, y in mobile_objects:
            if not (0 <= x < env.width and 0 <= y < env.height):
                raise ValueError(f"{obj_type} position ({x}, {y}) out of bounds")
