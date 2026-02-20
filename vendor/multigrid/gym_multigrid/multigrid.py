import math
import json
from abc import ABC
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from .rendering import *
from .window import Window
import numpy as np
from itertools import product

# Optional imports for ControlButton text rendering
try:
    import matplotlib
    # Only set backend if not already configured
    if matplotlib.get_backend() == 'module://backend_interagg':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO
    from scipy.ndimage import zoom
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError):
    MATPLOTLIB_AVAILABLE = False

# Constants for ControlButton text rendering
_CONTROLBUTTON_TEXT_ALPHA_THRESHOLD = 0.1  # Alpha threshold for text compositing
_CONTROLBUTTON_TEXT_DPI = 150  # DPI for text rendering (higher = better quality)
_CONTROLBUTTON_TEXT_FONTSIZE_SHORT = (12, 18, 2.5)  # (min, max, divisor) for short text (≤4 chars)
_CONTROLBUTTON_TEXT_FONTSIZE_LONG = (10, 14, 3)     # (min, max, divisor) for longer text
_CONTROLBUTTON_CORNER_RADIUS = 0.1  # Corner radius for rounded button shape
_BLOCK_SIZE_RATIO = 0.15  # Inset ratio for blocks (0.15 = 70% size)

"""
MAINTAINER NOTE - Encoder Synchronization
=========================================
When adding new object types, agent features, or modifying transition-relevant
attributes in this file, the neural network encoders in `empo.nn_based` may need
to be updated accordingly.

The encoders must encode ALL attributes that:
1. Can change during a transition (state before -> state after)
2. Can influence which transitions are possible or their probabilities

Currently encoded attributes are documented in:
- src/empo/nn_based/neural_policy_prior.py (OBJECT_TYPE_TO_CHANNEL, feature sizes)
- docs/ENCODER_ARCHITECTURE.md (comprehensive encoder documentation)

Examples of attributes that MUST be encoded:
- Agent: position, direction, color, terminated, paused, forced_next_action,
         can_enter_magic_walls, can_push_rocks, carrying
- Door: is_open, is_locked, color
- MagicWall: magic_side, active, entry_probability, solidify_probability
- ControlButton: enabled, trigger_color, target_color, triggered_action, _awaiting_action (in state)
- UnsteadyGround: stumble_probability

If you add a new object type or feature, update the encoders and documentation!
"""

# Import WorldModel base class from empo
import sys
from pathlib import Path
# Add src directory to path if not already there
_src_path = str(Path(__file__).parent.parent.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

from empo.world_model import WorldModel

# Lazy imports for goal types (loaded on demand to avoid circular imports)
_ReachCellGoal = None
_ReachRectangleGoal = None
_TabularGoalGenerator = None
_TabularGoalSampler = None
_PossibleGoalGenerator = None
_PossibleGoalSampler = None

def _load_goal_classes():
    """Lazy load goal classes to avoid circular imports."""
    global _ReachCellGoal, _ReachRectangleGoal, _TabularGoalGenerator, _TabularGoalSampler
    global _PossibleGoalGenerator, _PossibleGoalSampler
    if _ReachCellGoal is None:
        from empo.world_specific_helpers.multigrid import ReachCellGoal, ReachRectangleGoal
        from empo.possible_goal import (
            TabularGoalGenerator, TabularGoalSampler,
            PossibleGoalGenerator, PossibleGoalSampler
        )
        _ReachCellGoal = ReachCellGoal
        _ReachRectangleGoal = ReachRectangleGoal
        _TabularGoalGenerator = TabularGoalGenerator
        _TabularGoalSampler = TabularGoalSampler
        _PossibleGoalGenerator = PossibleGoalGenerator
        _PossibleGoalSampler = PossibleGoalSampler


def parse_goal_specs(goal_specs: list, env: 'MultiGridEnv', human_agent_index: int = 0):
    """
    Parse goal specifications from config file into goal objects.
    
    Goal format examples:
      - "1,1, 3,3"  -> Rectangle goal from (1,1) to (3,3)
      - "3,2"       -> Single cell goal at (3,2)
      - [1, 1, 3, 3] -> Rectangle goal from (1,1) to (3,3)
      - [3, 2]      -> Single cell goal at (3,2)
    
    Args:
        goal_specs: List of goal specifications (strings or lists of ints)
        env: The MultiGridEnv instance
        human_agent_index: Index of the human agent for these goals
        
    Returns:
        List of PossibleGoal instances (ReachCellGoal or ReachRectangleGoal)
    """
    _load_goal_classes()
    
    goals = []
    for spec in goal_specs:
        # Parse spec into coordinates
        if isinstance(spec, str):
            # String format: "x1,y1, x2,y2" or "x,y"
            parts = [int(p.strip()) for p in spec.replace(' ', '').split(',')]
        elif isinstance(spec, (list, tuple)):
            parts = [int(p) for p in spec]
        else:
            raise ValueError(f"Invalid goal spec: {spec}. Must be string or list.")
        
        if len(parts) == 2:
            # Single cell goal
            x, y = parts
            goals.append(_ReachCellGoal(env, human_agent_index, (x, y)))
        elif len(parts) == 4:
            # Rectangle goal
            x1, y1, x2, y2 = parts
            goals.append(_ReachRectangleGoal(env, human_agent_index, (x1, y1, x2, y2)))
        else:
            raise ValueError(f"Invalid goal spec: {spec}. Must have 2 or 4 coordinates.")
    
    return goals


def _parse_goal_spec_to_coords(spec) -> tuple:
    """
    Parse a single goal specification into coordinates.
    
    Supported formats:
        - "x,y" or "x, y" - cell goal (string with 2 coords)
        - "x1,y1,x2,y2" or "x1,y1, x2,y2" - rectangle goal (string with 4 coords)
        - [x, y] - cell goal (flat list with 2 elements)
        - [x1, y1, x2, y2] - rectangle goal (flat list with 4 elements)
        - [[x1, y1], [x2, y2]] - rectangle goal (nested list with 2 pairs)
    
    Returns:
        Tuple of (x1, y1) for cell goal or (x1, y1, x2, y2) for rectangle goal.
    """
    if isinstance(spec, str):
        # String format: "x,y" or "x1,y1,x2,y2" (with optional spaces)
        parts = [int(p.strip()) for p in spec.replace(' ', '').split(',')]
    elif isinstance(spec, (list, tuple)):
        # Check if nested list: [[x1, y1], [x2, y2]]
        if len(spec) == 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in spec):
            # Nested format: [[x1, y1], [x2, y2]] -> (x1, y1, x2, y2)
            parts = [int(spec[0][0]), int(spec[0][1]), int(spec[1][0]), int(spec[1][1])]
        else:
            # Flat list: [x, y] or [x1, y1, x2, y2]
            parts = [int(p) for p in spec]
    else:
        raise ValueError(f"Invalid goal spec: {spec}. Must be string or list.")
    
    if len(parts) == 2:
        return tuple(parts)  # (x, y)
    elif len(parts) == 4:
        return tuple(parts)  # (x1, y1, x2, y2)
    else:
        raise ValueError(f"Invalid goal spec: {spec}. Must have 2 or 4 coordinates.")


class ConfigGoalGenerator:
    """
    Goal generator that creates goals from config specs for any human agent.
    
    This class creates goal objects on-the-fly from coordinate specs. When
    generate() is called with a human_agent_index, it creates goals about
    THAT human (e.g., human h reaches cell X).
    
    Args:
        env: The MultiGridEnv instance
        goal_coords: List of goal coordinate tuples - either (x, y) for cell goals
                    or (x1, y1, x2, y2) for rectangle goals
        weights: Optional list of weights. If None, uses uniform 1/n weights.
    """
    
    def __init__(self, env: 'MultiGridEnv', goal_coords: list, weights: list = None, indexed: bool = False):
        _load_goal_classes()
        # Initialize parent class with indexed parameter
        if _PossibleGoalGenerator is not None:
            _PossibleGoalGenerator.__init__(self, env, indexed=indexed)
        self.env = self.world_model = env
        self.goal_coords = goal_coords
        n = len(goal_coords)
        self.weights = weights if weights is not None else [1.0 / n] * n
        self.indexed = indexed
        self.n_goals = n  # Number of goals
        self._goals_cache = {}  # Cache goals by human_agent_index
    
    @property
    def goals(self):
        """Return list of goals for agent index 0 (for compatibility)."""
        return self._get_goals_for_agent(0)
    
    def _get_goals_for_agent(self, human_agent_index: int):
        """Get or create goals for a specific agent index."""
        if human_agent_index not in self._goals_cache:
            _load_goal_classes()
            goals = []
            for idx, coords in enumerate(self.goal_coords):
                if len(coords) == 2:
                    goal = _ReachCellGoal(self.env, human_agent_index, coords, index=idx if self.indexed else None)
                else:
                    goal = _ReachRectangleGoal(self.env, human_agent_index, coords, index=idx if self.indexed else None)
                goals.append(goal)
            self._goals_cache[human_agent_index] = goals
        return self._goals_cache[human_agent_index]
    
    def set_world_model(self, world_model):
        """Set or update the world model reference."""
        self.env = self.world_model = world_model
        self._goals_cache.clear()  # Clear cache when world model changes
    
    def __getstate__(self):
        """Exclude world_model/env from pickling."""
        state = self.__dict__.copy()
        state['env'] = None
        state['world_model'] = None
        state['_goals_cache'] = {}
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
    
    def generate(self, state, human_agent_index: int):
        """
        Generate goals for the specified human agent.
        
        Creates goal objects on-the-fly about the specified human
        (e.g., human h reaches cell X).
        
        Args:
            state: Current state (unused)
            human_agent_index: Index of the human agent these goals are ABOUT
            
        Yields:
            Tuples of (goal, weight)
        """
        goals = self._get_goals_for_agent(human_agent_index)
        for goal, weight in zip(goals, self.weights):
            yield goal, weight
    
    def get_sampler(self) -> 'ConfigGoalSampler':
        """
        Create a ConfigGoalSampler from this generator.
        
        The sampler's probabilities are set to the generator's weights (normalized),
        and the sampler's weights are set to 1.0 for all goals.
        
        Returns:
            ConfigGoalSampler with probs = weights (normalized), weights = 1.0.
        """
        return ConfigGoalSampler(self.env, self.goal_coords, probabilities=self.weights, weights=None, indexed=self.indexed)
    
    def validate_coverage(self, raise_on_error: bool = True) -> list:
        """
        Validate that goals cover all walkable cells in the grid.
        
        A cell is considered walkable if an agent could potentially stand on it.
        Cells containing immovable non-overlappable objects (wall, lava, magicwall,
        pauseswitch, disablingswitch, controlbutton) are excluded since agents
        can never occupy them.
        
        Args:
            raise_on_error: If True, raises ValueError listing uncovered cells.
                           If False, returns list of uncovered cells.
        
        Returns:
            List of (x, y) tuples of uncovered walkable cells.
            Empty list if all walkable cells are covered.
        
        Raises:
            ValueError: If raise_on_error=True and there are uncovered cells.
        """
        if self.env is None:
            raise ValueError("Cannot validate coverage: env is None")
        
        # Build set of all cells covered by goals
        covered_cells = set()
        for coords in self.goal_coords:
            if len(coords) == 2:
                # Cell goal: (x, y)
                covered_cells.add((coords[0], coords[1]))
            else:
                # Rectangle goal: (x1, y1, x2, y2)
                x1, y1, x2, y2 = coords
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        covered_cells.add((x, y))
        
        # Find all walkable cells (not containing immovable non-overlappable objects)
        # Agents can never stand on these cell types:
        non_walkable_types = {
            'wall', 'lava', 'magicwall',
            'killbutton', 'pauseswitch', 'disablingswitch', 'controlbutton',
        }
        uncovered = []
        
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = self.env.grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell else None
                
                # Skip cells with non-walkable objects
                if cell_type in non_walkable_types:
                    continue
                
                # Check if this walkable cell is covered
                if (x, y) not in covered_cells:
                    uncovered.append((x, y))
        
        if uncovered and raise_on_error:
            raise ValueError(
                f"Goals do not cover all walkable cells. Uncovered cells: {uncovered}\n"
                f"Goal coords: {self.goal_coords}\n"
                f"Add goals covering these cells or use a rectangle goal like "
                f"[0, 0, {self.env.width-1}, {self.env.height-1}] to cover the entire grid."
            )
        
        return uncovered


class ConfigGoalSampler:
    """
    Goal sampler that creates goals from config specs for any human agent.
    
    This class creates goal objects on-the-fly from coordinate specs. When
    sample() is called with a human_agent_index, it creates and samples goals
    about THAT human (e.g., human h reaches cell X).
    
    Args:
        env: The MultiGridEnv instance
        goal_coords: List of goal coordinate tuples - either (x, y) for cell goals
                    or (x1, y1, x2, y2) for rectangle goals
        probabilities: Optional list of sampling probabilities. If None, uses uniform 1/n.
        weights: Optional list of weights for X_h computation. If None, uses 1.0 for all.
    """
    
    def __init__(self, env: 'MultiGridEnv', goal_coords: list, 
                 probabilities: list = None, weights: list = None, indexed: bool = False):
        _load_goal_classes()
        # Initialize parent class with indexed parameter
        if _PossibleGoalSampler is not None:
            _PossibleGoalSampler.__init__(self, env, indexed=indexed)
        self.env = self.world_model = env
        self.goal_coords = goal_coords
        n = len(goal_coords)
        
        if probabilities is None:
            self.probs = [1.0 / n] * n
        else:
            total = sum(probabilities)
            self.probs = [p / total for p in probabilities]
        
        self.weights = weights if weights is not None else [1.0] * n
        self.indexed = indexed
        self.n_goals = n  # Number of goals
        self._goals_cache = {}  # Cache goals by human_agent_index
    
    @property
    def goals(self):
        """Return list of goals for agent index 0 (for compatibility)."""
        return self._get_goals_for_agent(0)
    
    def _get_goals_for_agent(self, human_agent_index: int):
        """Get or create goals for a specific agent index."""
        if human_agent_index not in self._goals_cache:
            _load_goal_classes()
            goals = []
            for idx, coords in enumerate(self.goal_coords):
                if len(coords) == 2:
                    goal = _ReachCellGoal(self.env, human_agent_index, coords, index=idx if self.indexed else None)
                else:
                    goal = _ReachRectangleGoal(self.env, human_agent_index, coords, index=idx if self.indexed else None)
                goals.append(goal)
            self._goals_cache[human_agent_index] = goals
        return self._goals_cache[human_agent_index]
    
    def set_world_model(self, world_model):
        """Set or update the world model reference."""
        self.env = self.world_model = world_model
        self._goals_cache.clear()  # Clear cache when world model changes
    
    def __getstate__(self):
        """Exclude world_model/env from pickling."""
        state = self.__dict__.copy()
        state['env'] = None
        state['world_model'] = None
        state['_goals_cache'] = {}
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
    
    def sample(self, state, human_agent_index: int):
        """
        Sample a goal for the specified human agent.
        
        Creates a goal object on-the-fly about the specified human
        (e.g., human h reaches cell X).
        
        Args:
            state: Current state (unused)
            human_agent_index: Index of the human agent these goals are ABOUT
            
        Returns:
            Tuple of (goal, weight)
        """
        import random
        goals = self._get_goals_for_agent(human_agent_index)
        idx = random.choices(range(len(goals)), weights=self.probs, k=1)[0]
        return goals[idx], self.weights[idx]
    
    def get_generator(self) -> 'ConfigGoalGenerator':
        """
        Create a ConfigGoalGenerator from this sampler.
        
        The generator's weights are computed as probs * weights from the sampler,
        which represents the expected contribution of each goal to X_h computation.
        
        Returns:
            ConfigGoalGenerator with weights = probs * weights.
        """
        combined_weights = [p * w for p, w in zip(self.probs, self.weights)]
        return ConfigGoalGenerator(self.env, self.goal_coords, weights=combined_weights, indexed=self.indexed)
    
    def validate_coverage(self, raise_on_error: bool = True) -> list:
        """
        Validate that goals cover all walkable cells in the grid.
        
        A cell is considered walkable if an agent could potentially stand on it.
        Cells containing immovable non-overlappable objects (wall, lava, magicwall,
        pauseswitch, disablingswitch, controlbutton) are excluded since agents
        can never occupy them.
        
        Args:
            raise_on_error: If True, raises ValueError listing uncovered cells.
                           If False, returns list of uncovered cells.
        
        Returns:
            List of (x, y) tuples of uncovered walkable cells.
            Empty list if all walkable cells are covered.
        
        Raises:
            ValueError: If raise_on_error=True and there are uncovered cells.
        """
        if self.env is None:
            raise ValueError("Cannot validate coverage: env is None")
        
        # Build set of all cells covered by goals
        covered_cells = set()
        for coords in self.goal_coords:
            if len(coords) == 2:
                # Cell goal: (x, y)
                covered_cells.add((coords[0], coords[1]))
            else:
                # Rectangle goal: (x1, y1, x2, y2)
                x1, y1, x2, y2 = coords
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        covered_cells.add((x, y))
        
        # Find all walkable cells (not containing immovable non-overlappable objects)
        # Agents can never stand on these cell types:
        non_walkable_types = {
            'wall', 'lava', 'magicwall',
            'killbutton', 'pauseswitch', 'disablingswitch', 'controlbutton',
        }
        uncovered = []
        
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = self.env.grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell else None
                
                # Skip cells with non-walkable objects
                if cell_type in non_walkable_types:
                    continue
                
                # Check if this walkable cell is covered
                if (x, y) not in covered_cells:
                    uncovered.append((x, y))
        
        if uncovered and raise_on_error:
            raise ValueError(
                f"Goals do not cover all walkable cells. Uncovered cells: {uncovered}\n"
                f"Goal coords: {self.goal_coords}\n"
                f"Add goals covering these cells or use a rectangle goal like "
                f"[0, 0, {self.env.width-1}, {self.env.height-1}] to cover the entire grid."
            )
        
        return uncovered


def create_goal_sampler_and_generator(goals: list, env: 'MultiGridEnv'):
    """
    Create TabularGoalSampler and TabularGoalGenerator from a list of goals.
    
    Uses uniform weights (1/n) for the generator. The sampler is derived from
    the generator using get_sampler(), which uses the weights as probabilities
    and sets sampler weights to 1.0.
    
    Args:
        goals: List of PossibleGoal instances
        env: The MultiGridEnv instance (for setting world_model reference)
        
    Returns:
        Tuple of (TabularGoalSampler, TabularGoalGenerator)
    """
    _load_goal_classes()
    
    n = len(goals)
    if n == 0:
        return None, None
    
    # Generator: uniform weights = 1/n (for exact integration)
    generator = _TabularGoalGenerator(goals, weights=[1.0 / n] * n)
    
    # Sampler: derived from generator (uses weights as probabilities, sets weights to 1)
    sampler = generator.get_sampler()
    
    return sampler, generator


def create_config_goal_sampler_and_generator(goal_specs: list, env: 'MultiGridEnv'):
    """
    Create ConfigGoalSampler and ConfigGoalGenerator from goal specs.
    
    These create goals on-the-fly for any human_agent_index. When generate()
    or sample() is called with a human index, they create goals about THAT
    human (e.g., human h reaching cell X).
    
    Uses uniform weights (1/n) for the generator. The sampler is derived from
    the generator using get_sampler(), which uses the weights as probabilities
    and sets sampler weights to 1.0.
    
    Args:
        goal_specs: List of goal specifications from config file
        env: The MultiGridEnv instance
        
    Returns:
        Tuple of (ConfigGoalSampler, ConfigGoalGenerator)
    
    Raises:
        ValueError: If the goals don't cover all walkable cells in the grid.
    """
    if not goal_specs:
        return None, None
    
    # Parse specs to coordinates
    goal_coords = [_parse_goal_spec_to_coords(spec) for spec in goal_specs]
    n = len(goal_coords)
    
    # Generator: uniform weights = 1/n (for exact integration), indexed=True for YAML-loaded goals
    generator = ConfigGoalGenerator(env, goal_coords, weights=[1.0 / n] * n, indexed=True)
    
    # Validate that goals cover all walkable cells
    generator.validate_coverage(raise_on_error=True)
    
    # Sampler: derived from generator (uses weights as probabilities, sets weights to 1)
    sampler = generator.get_sampler()
    
    return sampler, generator


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'brown': np.array([139, 90, 43])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

class World:

    encode_dim = 6

    normalize_obs = 1

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5,
        'brown': 6
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'agent': 10,
        'objgoal': 11,
        'switch': 12,
        'block': 13,
        'rock': 14,
        'unsteadyground': 15,
        'magicwall': 16,
        'killbutton': 17,
        'pauseswitch': 18,
        'disablingswitch': 19,
        'controlbutton': 20
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class SmallWorld:

    encode_dim = 3

    normalize_obs = 1/3

    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'grey': 3
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'agent': 3
    }

    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos, agent_idx=None):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)
        else:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, 0, 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(self, world, index, target_type='ball', reward=1, color=None):
        if color is None:
            super().__init__(world, 'objgoal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'objgoal', world.IDX_TO_COLOR[color])
        self.target_type = target_type
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, world, index, reward=1, color=None):
        if color is None:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Switch(WorldObj):
    def __init__(self, world):
        super().__init__(world, 'switch', world.IDX_TO_COLOR[0])

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class KillButton(WorldObj):
    """
    A non-overlappable switch that permanently kills agents when toggled.
    
    Agents of trigger_color can use the "toggle" action on this switch when facing it.
    When toggled (and enabled), all agents of target_color become permanently
    "killed" (terminated, can only use "still" action).
    
    Attributes:
        trigger_color: Color of agents that can activate the button (default: 'yellow')
        target_color: Color of agents that will be killed (default: 'grey')
        enabled: Whether the button is active (default: True). When disabled, toggling has no effect.
    """
    
    def __init__(self, world, trigger_color='yellow', target_color='grey', enabled=True):
        """
        Args:
            world: World object defining the environment
            trigger_color: Color of agents that trigger the kill effect
            target_color: Color of agents that will be killed
            enabled: Whether the button is active
        """
        # Use red color to distinguish kill button visually
        super().__init__(world, 'killbutton', 'red')
        self.trigger_color = trigger_color
        self.target_color = target_color
        self.enabled = enabled
    
    def can_overlap(self):
        return False
    
    def see_behind(self):
        return True
    
    def toggle(self, env, pos, agent_idx=None):
        """Toggle the kill effect if enabled and toggled by correct color agent."""
        if not self.enabled:
            return False
        
        # Get the toggling agent
        if agent_idx is not None:
            toggler_agent = env.agents[agent_idx]
        else:
            # Fallback: find agent facing this position
            toggler_agent = None
            for agent in env.agents:
                if agent.pos is not None and np.array_equal(agent.front_pos, pos):
                    toggler_agent = agent
                    break
        
        if toggler_agent is None or toggler_agent.color != self.trigger_color:
            return False
        
        # Kill all agents of the target color
        for agent in env.agents:
            if agent.color == self.target_color:
                agent.terminated = True
        
        return True
    
    def encode(self, world, current_agent=False):
        """Encode the kill button with all its attributes for observations."""
        if world.encode_dim == 3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 
                    1 if self.enabled else 0)
        else:
            # Encode all attributes: trigger_color, target_color, and enabled state
            trigger_idx = world.COLOR_TO_IDX.get(self.trigger_color, 0)
            target_idx = world.COLOR_TO_IDX.get(self.target_color, 0)
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color],
                   trigger_idx, target_idx, 1 if self.enabled else 0, 0)
    
    def render(self, img):
        """Render the kill button as a red floor tile with a skull/X pattern.
        - Enabled: dark red background with darker red X
        - Disabled (by DisablingSwitch): grey background with faded X"""
        if self.enabled:
            # Red background for enabled kill button (not yet triggered)
            c = COLORS['red']
            fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c / 2)
            # Draw an X pattern in darker red
            darker_red = np.array([150, 0, 0])
            fill_coords(img, point_in_line(0.15, 0.15, 0.85, 0.85, r=0.05), darker_red)
            fill_coords(img, point_in_line(0.15, 0.85, 0.85, 0.15, r=0.05), darker_red)
        else:
            # Grey background for disabled kill button, with faded X still visible
            c = COLORS['grey']
            fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c / 3)
            faded = np.array([80, 80, 80])
            fill_coords(img, point_in_line(0.15, 0.15, 0.85, 0.85, r=0.05), faded)
            fill_coords(img, point_in_line(0.15, 0.85, 0.85, 0.15, r=0.05), faded)


class PauseSwitch(WorldObj):
    """
    A non-overlappable switch that pauses agents when toggled on.
    
    Agents of toggle_color can use the "toggle" action on this switch when facing it.
    While "on", all agents of target_color can only use the "still" action.
    
    Attributes:
        toggle_color: Color of agents that can toggle the switch (default: 'yellow')
        target_color: Color of agents that will be paused (default: 'grey')
        is_on: Whether the switch is currently on (default: False)
        enabled: Whether the switch can be toggled (default: True). When disabled, retains on/off state.
    """
    
    def __init__(self, world, toggle_color='yellow', target_color='grey', is_on=False, enabled=True):
        """
        Args:
            world: World object defining the environment
            toggle_color: Color of agents that can toggle the switch
            target_color: Color of agents that will be paused
            is_on: Initial on/off state
            enabled: Whether the switch can be toggled
        """
        # Use blue color to distinguish pause switch visually
        super().__init__(world, 'pauseswitch', 'blue')
        self.toggle_color = toggle_color
        self.target_color = target_color
        self.is_on = is_on
        self.enabled = enabled
    
    def can_overlap(self):
        return False
    
    def see_behind(self):
        return True
    
    def toggle(self, env, pos, agent_idx=None):
        """Toggle the switch on/off if enabled and toggled by correct color agent."""
        if not self.enabled:
            return False
        
        # Get the toggling agent - use agent_idx if provided, otherwise search
        if agent_idx is not None:
            toggler_agent = env.agents[agent_idx]
        else:
            # Fallback: find agent facing this position
            toggler_agent = None
            for agent in env.agents:
                if agent.pos is not None and np.array_equal(agent.front_pos, pos):
                    toggler_agent = agent
                    break
        
        if toggler_agent is None or toggler_agent.color != self.toggle_color:
            return False
        
        self.is_on = not self.is_on
        
        # Update paused state of all target color agents
        for agent in env.agents:
            if agent.color == self.target_color:
                agent.paused = self.is_on
        
        return True
    
    def encode(self, world, current_agent=False):
        """Encode the pause switch with its state."""
        if world.encode_dim == 3:
            # Encode is_on and enabled in state field (2 bits)
            state = (1 if self.is_on else 0) | ((1 if self.enabled else 0) << 1)
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], state)
        else:
            toggle_idx = world.COLOR_TO_IDX.get(self.toggle_color, 0)
            target_idx = world.COLOR_TO_IDX.get(self.target_color, 0)
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color],
                   toggle_idx, target_idx, 
                   1 if self.is_on else 0, 
                   1 if self.enabled else 0)
    
    def render(self, img):
        """Render the pause switch with state indication.
        When disabled, the switch is greyed out but the pause/play icon is still faintly visible."""
        if self.enabled:
            if self.is_on:
                # Bright blue when on
                c = COLORS['blue']
                fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c)
                # Draw pause symbol (two vertical bars)
                fill_coords(img, point_in_rect(0.3, 0.4, 0.25, 0.75), np.array([255, 255, 255]))
                fill_coords(img, point_in_rect(0.6, 0.7, 0.25, 0.75), np.array([255, 255, 255]))
            else:
                # Darker blue when off
                c = COLORS['blue'] / 2
                fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c)
                # Draw play symbol (triangle pointing right)
                fill_coords(img, point_in_triangle((0.3, 0.25), (0.3, 0.75), (0.7, 0.5)), 
                           np.array([200, 200, 200]))
        else:
            # Grey background when disabled, with faded icon still visible
            c = COLORS['grey']
            fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c / 3)
            faded = np.array([80, 80, 80])
            if self.is_on:
                # Show faded pause bars
                fill_coords(img, point_in_rect(0.3, 0.4, 0.25, 0.75), faded)
                fill_coords(img, point_in_rect(0.6, 0.7, 0.25, 0.75), faded)
            else:
                # Show faded play triangle
                fill_coords(img, point_in_triangle((0.3, 0.25), (0.3, 0.75), (0.7, 0.5)), faded)


class DisablingSwitch(WorldObj):
    """
    A non-overlappable switch that toggles enabled/disabled state of other objects.
    
    Agents of toggle_color can use the "toggle" action on this switch when facing it.
    This toggles the enabled/disabled state of all objects of target_type.
    
    Attributes:
        toggle_color: Color of agents that can toggle the switch (default: 'grey')
        target_type: Type of objects to enable/disable ('killbutton', 'pauseswitch', or 'controlbutton')
    """
    
    def __init__(self, world, toggle_color='grey', target_type='killbutton'):
        """
        Args:
            world: World object defining the environment
            toggle_color: Color of agents that can toggle the switch
            target_type: Type of objects to enable/disable ('killbutton', 'pauseswitch', or 'controlbutton')
        """
        # Use purple color to distinguish disabling switch visually
        super().__init__(world, 'disablingswitch', 'purple')
        self.toggle_color = toggle_color
        self.target_type = target_type
    
    def can_overlap(self):
        return False
    
    def see_behind(self):
        return True
    
    def toggle(self, env, pos, agent_idx=None):
        """Toggle enabled state of all target objects if toggled by correct color agent."""
        # Get the toggling agent - use agent_idx if provided, otherwise search
        if agent_idx is not None:
            toggler_agent = env.agents[agent_idx]
        else:
            toggler_agent = None
            for agent in env.agents:
                if agent.pos is not None and np.array_equal(agent.front_pos, pos):
                    toggler_agent = agent
                    break
        
        if toggler_agent is None or toggler_agent.color != self.toggle_color:
            return False
        
        # Toggle enabled state of all objects of target_type in the grid
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                cell = env.grid.get(i, j)
                if cell is not None and cell.type == self.target_type:
                    cell.enabled = not cell.enabled
                    
                    # If target is PauseSwitch and it's being disabled while ON,
                    # we should unpause the affected agents
                    if self.target_type == 'pauseswitch' and not cell.enabled and cell.is_on:
                        # Unpause agents since the switch can't affect them while disabled
                        for agent in env.agents:
                            if agent.color == cell.target_color:
                                agent.paused = False
        
        # Also check terrain_grid for objects that might be stored there
        if hasattr(env, 'terrain_grid') and env.terrain_grid is not None:
            for j in range(env.terrain_grid.height):
                for i in range(env.terrain_grid.width):
                    cell = env.terrain_grid.get(i, j)
                    if cell is not None and cell.type == self.target_type:
                        cell.enabled = not cell.enabled
        
        return True
    
    def encode(self, world, current_agent=False):
        """Encode the disabling switch."""
        if world.encode_dim == 3:
            # Encode target_type as index
            target_idx = 0  # killbutton
            if self.target_type == 'pauseswitch':
                target_idx = 1
            elif self.target_type == 'controlbutton':
                target_idx = 2
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], target_idx)
        else:
            toggle_idx = world.COLOR_TO_IDX.get(self.toggle_color, 0)
            target_idx = 0  # killbutton
            if self.target_type == 'pauseswitch':
                target_idx = 1
            elif self.target_type == 'controlbutton':
                target_idx = 2
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color],
                   toggle_idx, target_idx, 0, 0)
    
    def render(self, img):
        """Render the disabling switch with target-type-specific visuals.

        dK (target_type='killbutton'):  reddish-purple background + faded X from KillButton + ⊘ overlay
        dP (target_type='pauseswitch'): bluish-purple background + faded pause bars from PauseSwitch + ⊘ overlay
        Other targets:                  plain purple + ⊘ overlay
        """
        purple = COLORS['purple']

        if self.target_type == 'killbutton':
            # Reddish-purple tint that visually links to the red KillButton
            bg = np.clip(purple * 0.5 + COLORS['red'] * 0.25, 0, 255).astype(np.uint8)
        elif self.target_type == 'pauseswitch':
            # Bluish-purple tint that visually links to the blue PauseSwitch
            bg = np.clip(purple * 0.5 + COLORS['blue'] * 0.25, 0, 255).astype(np.uint8)
        else:
            bg = purple

        fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), bg)

        # Draw the target's characteristic symbol as a faint hint
        hint = (bg * 0.6).astype(np.uint8)  # slightly darker than background
        if self.target_type == 'killbutton':
            # Small X echoing KillButton
            fill_coords(img, point_in_line(0.2, 0.2, 0.8, 0.8, r=0.04), hint)
            fill_coords(img, point_in_line(0.2, 0.8, 0.8, 0.2, r=0.04), hint)
        elif self.target_type == 'pauseswitch':
            # Small pause bars echoing PauseSwitch
            fill_coords(img, point_in_rect(0.3, 0.4, 0.3, 0.7), hint)
            fill_coords(img, point_in_rect(0.6, 0.7, 0.3, 0.7), hint)

        # Overlay the ⊘ (circle-slash) "disable" symbol in white
        white = np.array([255, 255, 255])
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), white)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), bg)
        fill_coords(img, point_in_line(0.25, 0.75, 0.75, 0.25, r=0.04), white)


class ControlButton(WorldObj):
    """
    A non-overlappable control button that allows programming and triggering agent actions.
    
    This button enables a two-step interaction:
    1. Programming phase: An agent of controlled_color toggles it and then performs an action,
       which gets memorized in the button.
    2. Triggering phase: An agent of trigger_color toggles it, which sets forced_next_action
       on the controlled_agent so their next action is replaced by the triggered_action.
    
    Attributes:
        trigger_color: Color of agents that can trigger programmed actions (default: 'yellow')
        controlled_color: Color of agents that can program the button (default: 'grey')
        enabled: Whether the button is active (default: True). When disabled, cannot be used.
        controlled_agent: Index of the agent that programmed this button (None initially)
        triggered_action: The action that was programmed (None initially)
    """
    
    def __init__(self, world, trigger_color='yellow', controlled_color='grey', enabled=True, actions_set=None):
        """
        Args:
            world: World object defining the environment
            trigger_color: Color of agents that trigger the programmed action
            controlled_color: Color of agents that can program the button
            enabled: Whether the button is active
            actions_set: Actions class to use for action labels (optional, defaults to Actions).
                        Must have an 'available' attribute that is a list of action names.
        """
        assert trigger_color != controlled_color, "trigger_color and controlled_color must be different"
        # Use green color to distinguish control button visually
        super().__init__(world, 'controlbutton', 'green')
        self.trigger_color = trigger_color
        self.controlled_color = controlled_color
        self.enabled = enabled
        self.controlled_agent = None  # Agent index that programmed this button
        self.triggered_action = None  # Action that was programmed
        self._awaiting_action = False  # Internal: waiting for action after toggle by controlled_color
        self._just_activated = False   # Internal: True on the step when programming mode was just activated
        # Store actions_set for getting action labels, default to Actions if not provided
        self.actions_set = actions_set if actions_set is not None else Actions
        # Validate that actions_set has the required 'available' attribute
        if not hasattr(self.actions_set, 'available'):
            raise ValueError(f"actions_set must have an 'available' attribute with action names")
    
    def can_overlap(self):
        return False
    
    def see_behind(self):
        return True
    
    def toggle(self, env, pos, agent_idx=None):
        """Handle toggle action on this control button."""
        if not self.enabled:
            return False
        
        # Get the toggling agent - use agent_idx if provided, otherwise search
        if agent_idx is not None:
            toggler_agent = env.agents[agent_idx]
            toggler_idx = agent_idx
        else:
            toggler_agent = None
            toggler_idx = None
            for i, agent in enumerate(env.agents):
                if agent.pos is not None and np.array_equal(agent.front_pos, pos):
                    toggler_agent = agent
                    toggler_idx = i
                    break
        
        if toggler_agent is None:
            return False
        
        # Check if controlled_color agent is programming the button
        if toggler_agent.color == self.controlled_color:
            # If already awaiting action from this agent, this toggle IS the action to record
            # (robot wants to program a toggle action for controlling other switches)
            if self._awaiting_action and self.controlled_agent == toggler_idx and not self._just_activated:
                # Don't restart programming - let the toggle be recorded as the action
                # The record_action will be called by step() after this toggle executes
                return True
            
            # Start waiting for the next action from this agent
            self._awaiting_action = True
            self._just_activated = True  # Mark that we just entered programming mode
            self.controlled_agent = toggler_idx
            return True
        
        # Check if trigger_color agent is triggering the button
        if toggler_agent.color == self.trigger_color:
            # Only trigger if button has been programmed
            if self.controlled_agent is not None and self.triggered_action is not None:
                # Set the forced_next_action on the controlled agent
                # This will be applied on the NEXT call to step()
                controlled_agent = env.agents[self.controlled_agent]
                controlled_agent.forced_next_action = self.triggered_action
                return True
        
        return False
    
    def record_action(self, action):
        """Record the action taken by the controlled agent after toggling."""
        if self._awaiting_action:
            self.triggered_action = action
            self._awaiting_action = False
            return True
        return False
    
    def encode(self, world, current_agent=False):
        """Encode the control button with all its attributes for observations."""
        if world.encode_dim == 3:
            # Encode enabled and whether programmed in state field
            state = (1 if self.enabled else 0) | ((1 if self.triggered_action is not None else 0) << 1)
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], state)
        else:
            trigger_idx = world.COLOR_TO_IDX.get(self.trigger_color, 0)
            controlled_idx = world.COLOR_TO_IDX.get(self.controlled_color, 0)
            # Encode: trigger_color, controlled_color, enabled, triggered_action (or -1 if None)
            action_val = self.triggered_action if self.triggered_action is not None else -1
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color],
                   trigger_idx, controlled_idx, 
                   1 if self.enabled else 0, 
                   action_val + 1)  # +1 so None (-1) becomes 0
    
    def render(self, img):
        """Render the control button with state indication and action text label."""
        if self.enabled:
            if self.triggered_action is not None:
                # Muted green when programmed (less bright for readable labels)
                c = np.array([50, 120, 50])  # Darker green for better label contrast
                fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c)
                
                # Get the action label text from the action space
                action = self.triggered_action
                # Use the action name from actions_set.available if it's a valid index
                if 0 <= action < len(self.actions_set.available):
                    action_name = self.actions_set.available[action]
                    self._draw_text_label(img, action_name)
                else:
                    # Fallback to action number for invalid/unknown actions
                    self._draw_text_label(img, str(action))
            else:
                # Darker green when not programmed
                c = COLORS['green'] / 2
                fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c)
                # Draw empty circle
                white = np.array([200, 200, 200])
                fill_coords(img, point_in_circle(0.5, 0.5, 0.25), white)
                fill_coords(img, point_in_circle(0.5, 0.5, 0.15), c)
        else:
            # Grey when disabled
            c = COLORS['grey']
            fill_coords(img, point_in_rounded_rect(0.05, 0.95, 0.05, 0.95, _CONTROLBUTTON_CORNER_RADIUS), c / 2)
    
    def _draw_text_label(self, img, text):
        """Draw a text label on the button using matplotlib rendering."""
        if not MATPLOTLIB_AVAILABLE:
            # Fallback: if matplotlib is not available, draw simple centered line
            h, w = img.shape[:2]
            white = np.array([255, 255, 255])
            center_y = h // 2
            for y in range(max(0, center_y - 1), min(h, center_y + 2)):
                for x in range(w // 4, 3 * w // 4):
                    img[y, x] = white
            return
        
        try:
            h, w = img.shape[:2]
            
            # Create a figure for text rendering with higher DPI for better quality
            dpi = _CONTROLBUTTON_TEXT_DPI
            fig_width = w / dpi
            fig_height = h / dpi
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='none')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.patch.set_alpha(0)
            
            # Larger font sizes to make labels legible
            # Scale based on button size and text length
            if len(text) <= 4:
                min_fs, max_fs, divisor = _CONTROLBUTTON_TEXT_FONTSIZE_SHORT
                fontsize = max(min_fs, min(max_fs, h // divisor))
            else:
                min_fs, max_fs, divisor = _CONTROLBUTTON_TEXT_FONTSIZE_LONG
                fontsize = max(min_fs, min(max_fs, h // divisor))
            
            # Draw text with minimal padding to use full button width
            ax.text(0.5, 0.5, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='black', 
                            alpha=0.4, edgecolor='none'))
            
            # Render to buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                       pad_inches=0, transparent=True)
            buf.seek(0)
            
            # Read image from buffer
            text_img = mpimg.imread(buf)
            buf.close()
            plt.close(fig)
            
            # Handle RGBA format
            if len(text_img.shape) == 3 and text_img.shape[2] == 4:
                # Get RGB and alpha channels
                alpha = text_img[:, :, 3]
                rgb = text_img[:, :, :3]
                
                # Convert to uint8
                text_rgb = (rgb * 255).astype(np.uint8)
                
                # Resize if needed to match button dimensions
                if text_img.shape[0] != h or text_img.shape[1] != w:
                    zoom_h = h / text_img.shape[0]
                    zoom_w = w / text_img.shape[1]
                    text_rgb = zoom(text_rgb, (zoom_h, zoom_w, 1), order=1).astype(np.uint8)
                    alpha = zoom(alpha, (zoom_h, zoom_w), order=1)
                
                # Composite text onto button using alpha channel
                # Only apply where alpha is significant (text is visible)
                mask = alpha > _CONTROLBUTTON_TEXT_ALPHA_THRESHOLD
                
                # Ensure dimensions match
                if text_rgb.shape[0] == h and text_rgb.shape[1] == w:
                    # Vectorized alpha blending for all channels
                    alpha_expanded = alpha[..., np.newaxis]
                    img[:] = np.where(mask[..., np.newaxis], 
                                     (alpha_expanded * text_rgb + (1 - alpha_expanded) * img).astype(np.uint8),
                                     img)
            
        except Exception as e:
            # Fallback: if rendering fails, draw simple centered line
            # Log the error for debugging if needed
            import sys
            print(f"Warning: ControlButton text rendering failed: {e}", file=sys.stderr)
            
            h, w = img.shape[:2]
            white = np.array([255, 255, 255])
            center_y = h // 2
            for y in range(max(0, center_y - 1), min(h, center_y + 2)):
                for x in range(w // 4, 3 * w // 4):
                    img[y, x] = white


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world, color='blue'):
        super().__init__(world, 'floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c / 2)


class Lava(WorldObj):
    def __init__(self, world):
        super().__init__(world, 'lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class UnsteadyGround(WorldObj):
    """
    Unsteady ground tile that agents can walk over but may stumble on.
    
    When an agent attempts a forward action on unsteady ground, there is a 
    probability that the agent stumbles, causing the forward action to be 
    replaced by left+forward or right+forward.
    """
    
    def __init__(self, world, stumble_probability=0.5, color='brown'):
        """
        Args:
            world: World object defining the environment
            stumble_probability: Probability that an agent stumbles when moving forward (0.0 to 1.0)
            color: Color of the tile for rendering
        """
        super().__init__(world, 'unsteadyground', color)
        self.stumble_probability = stumble_probability
    
    def can_overlap(self):
        return True
    
    def encode(self, world, current_agent=False):
        """Encode the unsteady ground with its stumble probability."""
        if world.encode_dim == 3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)
        else:
            # Encode stumble_probability in the state field (scaled to 0-255)
            stumble_encoded = int(self.stumble_probability * 255)
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 
                   stumble_encoded, 0, 0, 0)
    
    def render(self, img):
        """Render unsteady ground with a distinctive pattern (diagonal lines)."""
        c = COLORS[self.color]
        # Base color - slightly darker than floor
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c / 2.5)
        
        # Add diagonal lines to indicate unsteadiness
        # Create diagonal stripes across the tile
        line_color = np.array([50, 50, 50])
        for i in range(-1, 2):  # Create 3 diagonal lines
            # Diagonal line from bottom-left to top-right
            fill_coords(img, point_in_line(i * 0.3, 1 + i * 0.3, 1 + i * 0.3, i * 0.3, 0.02), line_color)
    
    def render_with_stumble(self, img):
        """Render unsteady ground with a highlight indicating a stumble occurred."""
        c = COLORS[self.color]
        # Brighter base color to indicate stumbling
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), c / 1.5)
        
        # Add diagonal lines (more visible)
        line_color = np.array([255, 200, 0])  # Yellow/orange for emphasis
        for i in range(-1, 2):  # Create 3 diagonal lines
            # Diagonal line from bottom-left to top-right
            fill_coords(img, point_in_line(i * 0.3, 1 + i * 0.3, 1 + i * 0.3, i * 0.3, 0.03), line_color)


class Wall(WorldObj):
    def __init__(self, world, color='grey'):
        super().__init__(world, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class MagicWall(WorldObj):
    """
    Magic wall that can be entered by certain agents with a certain probability 
    from one specific direction (or all directions).
    
    Attributes:
        magic_side: Direction from which the wall can be entered (0=right, 1=down, 2=left, 3=up, 4=all) [immutable]
        entry_probability: Probability (0.0 to 1.0) that an authorized agent successfully enters [immutable]
        solidify_probability: Probability (0.0 to 1.0) that a failed entry attempt deactivates this wall [immutable]
        active: Whether this wall still functions as a magic wall (True) or has solidified into a regular wall (False) [mutable]
    """
    
    def __init__(self, world, magic_side, entry_probability, solidify_probability=0.0, color='grey'):
        """
        Args:
            world: World object defining the environment
            magic_side: Direction from which agents can attempt to enter (0-4: 0=right, 1=down, 2=left, 3=up, 4=all)
            entry_probability: Probability of successful entry (0.0 to 1.0)
            solidify_probability: Probability that a failed entry turns this into a normal wall (0.0 to 1.0)
            color: Color of the wall for rendering
        """
        super().__init__(world, 'magicwall', color)
        assert 0 <= magic_side <= 4, "magic_side must be 0 (right), 1 (down), 2 (left), 3 (up), or 4 (all)"
        assert 0.0 <= entry_probability <= 1.0, "entry_probability must be between 0.0 and 1.0"
        assert 0.0 <= solidify_probability <= 1.0, "solidify_probability must be between 0.0 and 1.0"
        self.magic_side = magic_side
        self.entry_probability = entry_probability
        self.solidify_probability = solidify_probability
        self.active = True  # Mutable: whether wall is still magic (True) or solidified (False)
    
    def see_behind(self):
        return False
    
    def can_overlap(self):
        """
        Magic walls cannot be overlapped in normal movement.
        Agents can only step OFF magic walls (if they're already on one).
        Entry is only through the special magic wall processing.
        """
        return False
    
    def encode(self, world, current_agent=False):
        """Encode the magic wall with all its attributes (immutable and mutable)."""
        if world.encode_dim == 3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], self.magic_side)
        else:
            # Encode all attributes: magic_side, entry_probability, solidify_probability, and active state
            entry_prob_encoded = int(self.entry_probability * 255)
            solidify_prob_encoded = int(self.solidify_probability * 255)
            active_encoded = 1 if self.active else 0
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 
                   self.magic_side, entry_prob_encoded, solidify_prob_encoded, active_encoded)
    
    def render(self, img):
        """Render magic wall like a normal wall with a dashed blue line near its magic side.
        If not active (solidified), render as a plain wall without the blue line."""
        # Render base wall
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        
        # Only add blue line if wall is still active/magic
        if not self.active:
            return
        
        # Add dashed blue line parallel to magic side
        blue_color = COLORS['blue']
        line_width = 0.04
        dash_length = 0.15
        gap_length = 0.10
        offset = 0.15  # Distance from the edge
        
        # Create dashed line based on magic_side direction
        if self.magic_side == 4:  # All sides - draw lines on all four edges
            # Right edge
            x_pos = 1 - offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), blue_color)
            # Bottom edge
            y_pos = 1 - offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), blue_color)
            # Left edge
            x_pos = offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), blue_color)
            # Top edge
            y_pos = offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), blue_color)
        elif self.magic_side == 0:  # Right - vertical dashed line near right edge
            x_pos = 1 - offset
            y_start = 0.05
            y_end = 0.95
            for y in np.arange(y_start, y_end, dash_length + gap_length):
                dash_end = min(y + dash_length, y_end)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), blue_color)
        elif self.magic_side == 1:  # Down - horizontal dashed line near bottom edge
            y_pos = 1 - offset
            x_start = 0.05
            x_end = 0.95
            for x in np.arange(x_start, x_end, dash_length + gap_length):
                dash_end = min(x + dash_length, x_end)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), blue_color)
        elif self.magic_side == 2:  # Left - vertical dashed line near left edge
            x_pos = offset
            y_start = 0.05
            y_end = 0.95
            for y in np.arange(y_start, y_end, dash_length + gap_length):
                dash_end = min(y + dash_length, y_end)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), blue_color)
        elif self.magic_side == 3:  # Up - horizontal dashed line near top edge
            y_pos = offset
            x_start = 0.05
            x_end = 0.95
            for x in np.arange(x_start, x_end, dash_length + gap_length):
                dash_end = min(x + dash_length, x_end)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), blue_color)
    
    def render_with_magic_entry(self, img):
        """Render magic wall with a bright highlight indicating successful entry."""
        # Brighter base color to indicate magic entry
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), np.clip(c * 1.5, 0, 255).astype(np.uint8))
        
        # Add bright cyan/magenta dashed line to show magic activation
        magic_color = np.array([255, 0, 255])  # Magenta for visibility
        line_width = 0.06  # Thicker line
        dash_length = 0.15
        gap_length = 0.10
        offset = 0.15
        
        # Create dashed line based on magic_side direction (same positions, different color)
        if self.magic_side == 4:  # All sides - draw lines on all four edges
            # Right edge
            x_pos = 1 - offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), magic_color)
            # Bottom edge
            y_pos = 1 - offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), magic_color)
            # Left edge
            x_pos = offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), magic_color)
            # Top edge
            y_pos = offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), magic_color)
        elif self.magic_side == 0:  # Right
            x_pos = 1 - offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), magic_color)
        elif self.magic_side == 1:  # Down
            y_pos = 1 - offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), magic_color)
        elif self.magic_side == 2:  # Left
            x_pos = offset
            for y in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(y + dash_length, 0.95)
                fill_coords(img, point_in_rect(x_pos - line_width/2, x_pos + line_width/2, y, dash_end), magic_color)
        elif self.magic_side == 3:  # Up
            y_pos = offset
            for x in np.arange(0.05, 0.95, dash_length + gap_length):
                dash_end = min(x + dash_length, 0.95)
                fill_coords(img, point_in_rect(x, dash_end, y_pos - line_width/2, y_pos + line_width/2), magic_color)



class Door(WorldObj):
    def __init__(self, world, color, is_open=False, is_locked=False):
        super().__init__(world, 'door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos, agent_idx=None):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], state, 0, 0, 0)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, world, color='blue'):
        super(Key, self).__init__(world, 'key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, world, index=0, reward=1):
        super(Ball, self).__init__(world, 'ball', world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world, color, contains=None):
        super(Box, self).__init__(world, 'box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos, agent_idx=None):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Block(WorldObj):
    def __init__(self, world):
        super(Block, self).__init__(world, 'block', 'brown')

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def render(self, img):
        # Render as a smaller square to show floor underneath
        # Block takes up approximately 70% of the cell, centered
        c = COLORS[self.color]
        inset = _BLOCK_SIZE_RATIO
        fill_coords(img, point_in_rect(inset, 1-inset, inset, 1-inset), c)


class Rock(WorldObj):
    def __init__(self, world):
        super(Rock, self).__init__(world, 'rock', 'grey')

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False
    
    def can_be_pushed_by(self, agent):
        """
        Check if this rock can be pushed by the given agent.
        
        Args:
            agent: The Agent object attempting to push this rock
            
        Returns:
            bool: True if the agent has can_push_rocks=True, False otherwise
        """
        return getattr(agent, 'can_push_rocks', False)

    def render(self, img):
        # Medium grey irregular rock shape
        c = COLORS[self.color]
        # Create an irregular rock-like shape using multiple overlapping shapes
        # Base rock body
        fill_coords(img, point_in_circle(0.45, 0.5, 0.35), c)
        fill_coords(img, point_in_circle(0.55, 0.45, 0.30), c)
        fill_coords(img, point_in_circle(0.50, 0.60, 0.28), c)
        # Add some texture/detail with darker grey
        darker_grey = np.array([70, 70, 70])  # Darker shade for texture
        fill_coords(img, point_in_circle(0.35, 0.40, 0.12), darker_grey)
        fill_coords(img, point_in_circle(0.65, 0.55, 0.10), darker_grey)


class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7, can_enter_magic_walls=False, can_push_rocks=False):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False
        self.on_unsteady_ground = False  # Track if agent is on unsteady ground
        self.can_enter_magic_walls = can_enter_magic_walls  # Can attempt to enter magic walls
        self.can_push_rocks = can_push_rocks  # Can push rocks (immutable)
        self.forced_next_action = None  # If set, overrides the agent's next action (used by ControlButton)

    def render(self, img):
        c = COLORS[self.color]
        if self.terminated:
            # Dim the agent color for terminated agents
            c = c // 3
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)
        
        if self.terminated:
            # Draw a black dagger (†) overlay on the agent
            black = np.array([0, 0, 0])
            # Vertical blade (top to bottom)
            fill_coords(img, point_in_line(0.50, 0.15, 0.50, 0.85, r=0.04), black)
            # Horizontal crossguard
            fill_coords(img, point_in_line(0.35, 0.38, 0.65, 0.38, r=0.04), black)

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], self.dir)
        elif self.carrying:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 0)

        else:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 0)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        # Removed assertion checks for performance - bounds checking done at higher level
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        # Removed assertion checks for performance - bounds checking done at higher level
        return self.grid[j * self.width + i]
    
    def get_unsafe(self, i, j):
        """Fast grid access without bounds checking. Use only when bounds are guaranteed."""
        return self.grid[j * self.width + i]
    
    def set_unsafe(self, i, j, v):
        """Fast grid set without bounds checking. Use only when bounds are guaranteed."""
        self.grid[j * self.width + i] = v

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            world,
            obj,
            terrain=None,
            highlights=[],
            tile_size=TILE_PIXELS,
            subdivs=3,
            stumbled=False,
            magic_entered=False
    ):
        """
        Render a tile and cache the result
        """

        # Include terrain, stumbled, and magic_entered state in cache key
        # Convert highlights to tuple for hashing
        highlights_tuple = tuple(highlights) if highlights else ()
        key = (*highlights_tuple, tile_size, stumbled, magic_entered)
        key = obj.encode(world) + key if obj else key
        # Include terminated/paused state for agents (affects rendering but not encode())
        if obj is not None and obj.type == 'agent':
            key = key + (getattr(obj, 'terminated', False), getattr(obj, 'paused', False))
        if terrain:
            key = terrain.encode(world) + key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Render terrain first (if present)
        if terrain != None:
            # Pass stumbled state to terrain if it's unsteady ground
            if hasattr(terrain, 'render_with_stumble') and stumbled:
                terrain.render_with_stumble(img)
            # Pass magic_entered state to terrain if it's magic wall
            elif hasattr(terrain, 'render_with_magic_entry') and magic_entered:
                terrain.render_with_magic_entry(img)
            else:
                terrain.render(img)
        
        # Render object on top
        if obj != None:
            obj.render(img)

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(img, color=COLORS[world.IDX_TO_COLOR[h%len(world.IDX_TO_COLOR)]])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            world,
            tile_size,
            terrain_grid=None,
            highlight_masks=None,
            stumbled_cells=None,
            magic_wall_entered_cells=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        :param terrain_grid: optional terrain grid to render under objects
        :param stumbled_cells: optional set of (x, y) positions where stumbling occurred
        :param magic_wall_entered_cells: optional set of (x, y) positions where magic wall entry succeeded
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                terrain = terrain_grid.get(i, j) if terrain_grid else None
                stumbled_here = bool(stumbled_cells and (i, j) in stumbled_cells)
                magic_entered_here = bool(magic_wall_entered_cells and (i, j) in magic_wall_entered_cells)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    terrain=terrain,
                    highlights=[] if highlight_masks is None else highlight_masks[i, j],
                    tile_size=tile_size,
                    stumbled=stumbled_here,
                    magic_entered=magic_entered_here
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world)

        return array

    def encode_for_agents(self, world, agent_pos, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world, current_agent=np.array_equal(agent_pos, (i, j)))

        return array

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class ActionsBase(ABC):
    """
    Abstract base class for action sets.
    
    All action classes must define:
    - available: list of action names that are valid for this action set
    - Class attributes mapping action names to integer codes
    
    Action codes should be consecutive integers starting from 0.
    Actions not available in a subclass should be set to None.
    """
    available: list  # List of available action names
    
    # Common action attributes - subclasses should define these as int or None
    still: int = None  # Stay in place (no-op)
    left: int = None   # Turn left
    right: int = None  # Turn right
    forward: int = None  # Move forward


class Actions(ActionsBase):
    """Full action set with all available actions."""
    available = ['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3

    # Pick up an object
    pickup = 4
    # Drop an object
    drop = 5
    # Toggle/activate an object
    toggle = 6

    # Done completing task
    done = 7

FullActions = Actions  # Alias for backward compatibility
class ObjectActions(ActionsBase):
    """Standard action set with all available actions except 'done'."""
    available = ['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3

    # Pick up an object
    pickup = 4
    # Drop an object
    drop = 5
    # Toggle/activate an object
    toggle = 6


class SmallActions(ActionsBase):
    """Reduced action set: still, left, right, forward."""
    available = ['still', 'left', 'right', 'forward']

    # Turn left, turn right, move forward
    still = 0
    left = 1
    right = 2
    forward = 3


class MinimalActions(ActionsBase):
    """
    Minimal action set: forward and left only.
    
    This is useful for environments where agents only need to move forward
    and turn left (turning right can be achieved by three left turns).
    
    Note: There is no 'still' action - agents must always move or turn.
    Action codes are consecutive starting from 0.
    """
    available = ['forward', 'left']
    
    still = None  # No still action available
    forward = 0
    left = 1
    right = None  # No right action available


class MineActions(ActionsBase):
    """Action set with building capability."""
    available = ['still', 'left', 'right', 'forward', 'build']

    still = 0
    left = 1
    right = 2
    forward = 3
    build = 4


def get_actions_class(name: str) -> type:
    """
    Look up an action class by name in this module.
    
    The class must derive from ActionsBase.
    
    Args:
        name: Name of the action class (e.g., 'SmallActions', 'MinimalActions')
        
    Returns:
        The action class
        
    Raises:
        ValueError: If the class is not found or doesn't derive from ActionsBase
    """
    # Look up in the current module's globals
    cls = globals().get(name)
    
    if cls is None:
        raise ValueError(f"Unknown action class '{name}'. "
                        f"Class not found in multigrid module.")
    
    if not isinstance(cls, type) or not issubclass(cls, ActionsBase):
        raise ValueError(f"'{name}' is not a valid action class. "
                        f"Must be a class deriving from ActionsBase.")
    
    return cls

# Object types that require pickup/drop/toggle actions
# If a map contains any of these, ObjectActions should be used instead of SmallActions
OBJECTS_REQUIRING_INTERACTION = {
    # Pickable objects
    'key', 'ball', 'box',
    # Toggleable objects  
    'door', 'pauseswitch', 'disablingswitch', 'controlbutton',
}


# Map encoding constants
MAP_COLOR_CODES = {
    'r': 'red',
    'g': 'green',
    'b': 'blue',
    'p': 'purple',
    'y': 'yellow',
    'e': 'grey',
}

# Magic wall direction constants (matching DIR_TO_VEC indices)
MAGIC_SIDE_EAST = 0   # Entry from east (approaching from west)
MAGIC_SIDE_SOUTH = 1  # Entry from south (approaching from north)
MAGIC_SIDE_WEST = 2   # Entry from west (approaching from east)
MAGIC_SIDE_NORTH = 3  # Entry from north (approaching from south)
MAGIC_SIDE_ALL = 4    # Entry from all sides


def parse_map_string(map_spec, objects_set=World):
    """
    Parse a map specification string into cell specifications.
    
    The map can be specified as:
    - A single string with newlines separating rows
    - A list of strings (one per row)
    - A list of lists of two-character strings (one per cell)
    
    Cell encoding (two characters each):
    - .. : empty cell
    - Wc : wall of color c
    - Bl : block
    - Ro : rock  
    - Lc : locked door of color c
    - Cc : closed door of color c
    - Oc : open door of color c
    - Kc : key of color c
    - Bc : ball of color c
    - Xc : box of color c
    - Gc : goal of color c
    - La : lava
    - Sw : switch
    - Un : unsteady ground
    - Mn : magic wall with north magic side
    - Ms : magic wall with south magic side
    - Mw : magic wall with west magic side
    - Me : magic wall with east magic side
    - Kb or Ki : KillButton (yellow triggers, grey killed)
    - Ps or Pa : PauseSwitch (yellow toggles, grey paused)
    - Dk or dK : DisablingSwitch for KillButtons (grey toggles)
    - Dp or dP : DisablingSwitch for PauseSwitches (grey toggles)
    - DC or dC : DisablingSwitch for ControlButtons (grey toggles)
    - CB : ControlButton (yellow triggers, grey controlled)
    - Ac : agent of color c
    
    Color codes (c):
    - r : red
    - g : green
    - b : blue
    - p : purple
    - y : yellow
    - e : grey
    
    Whitespace between cells is allowed and will be stripped.
    
    Args:
        map_spec: The map specification (string, list of strings, or list of lists)
        objects_set: The World class to use for object creation
        
    Returns:
        tuple: (width, height, cells, agents, has_interactive_objects) where:
               - width: The grid width in cells
               - height: The grid height in cells
               - cells: A 2D list of cell specs, each is a tuple (type, params_dict) or None for empty
               - agents: A list of (x, y, params_dict) tuples for each agent found
               - has_interactive_objects: True if map contains objects that need pickup/toggle actions
    """
    # Normalize to list of strings (one per row)
    if isinstance(map_spec, str):
        # Single string with newlines
        rows = map_spec.strip().split('\n')
    elif isinstance(map_spec, list):
        if len(map_spec) > 0 and isinstance(map_spec[0], list):
            # List of lists - each inner list is a row of two-char strings
            # Join them back to strings for uniform processing
            rows = [''.join(row) for row in map_spec]
        else:
            # List of strings
            rows = map_spec
    else:
        raise ValueError(f"map_spec must be a string, list of strings, or list of lists, got {type(map_spec)}")
    
    # Strip whitespace from each row
    rows = [''.join(row.split()) for row in rows]
    
    # Validate all rows have the same length and even number of chars
    if len(rows) == 0:
        raise ValueError("Map specification is empty")
    
    # Each cell is 2 characters, so row length must be even
    for i, row in enumerate(rows):
        if len(row) % 2 != 0:
            raise ValueError(f"Row {i} has odd length {len(row)}, expected even (2 chars per cell)")
    
    width = len(rows[0]) // 2
    for i, row in enumerate(rows):
        row_width = len(row) // 2
        if row_width != width:
            raise ValueError(f"Row {i} has width {row_width}, expected {width}")
    
    height = len(rows)
    
    # Parse each cell
    cells = []
    agents = []  # Track agent positions and colors for later creation
    has_interactive_objects = False  # Track if map has objects needing pickup/toggle
    
    for y, row in enumerate(rows):
        cell_row = []
        for x in range(width):
            cell_str = row[x*2:x*2+2]
            cell_spec = _parse_cell(cell_str, objects_set)
            cell_row.append(cell_spec)
            
            # Track agents for later
            if cell_spec and cell_spec[0] == 'agent':
                agents.append((x, y, cell_spec[1]))
            
            # Check if this object requires interaction actions
            if cell_spec and cell_spec[0] in OBJECTS_REQUIRING_INTERACTION:
                has_interactive_objects = True
        
        cells.append(cell_row)
    
    return width, height, cells, agents, has_interactive_objects


def _parse_cell(cell_str, objects_set):
    """
    Parse a two-character cell specification.
    
    Args:
        cell_str: Two-character cell specification
        objects_set: The World class to use
        
    Returns:
        tuple: (type, params_dict) or None for empty cell
    """
    if cell_str == '..':
        return None
    
    full_cell_code = cell_str[0:2]
    
    # Handle cells without color codes
    if full_cell_code == 'Bl':
        return ('block', {})
    elif full_cell_code == 'Ro':
        return ('rock', {})
    elif full_cell_code == 'La':
        return ('lava', {})
    elif full_cell_code == 'Sw':
        return ('switch', {})
    elif full_cell_code == 'Un':
        return ('unsteady', {})
    elif full_cell_code in ('Kb', 'Ki'):
        # KillButton with default colors (yellow triggers, grey killed)
        return ('killbutton', {})
    elif full_cell_code in ('Ps', 'Pa'):
        # PauseSwitch with default colors (yellow toggles, grey paused)
        return ('pauseswitch', {})
    elif full_cell_code in ('Dk', 'dK'):
        # DisablingSwitch for killbuttons (grey toggles)
        return ('disablingswitch', {'target_type': 'killbutton'})
    elif full_cell_code in ('Dp', 'dP'):
        # DisablingSwitch for pauseswitches (grey toggles)
        return ('disablingswitch', {'target_type': 'pauseswitch'})
    elif full_cell_code in ('dC', 'DC'):
        # DisablingSwitch for controlbuttons (grey toggles)
        return ('disablingswitch', {'target_type': 'controlbutton'})
    elif full_cell_code == 'CB':
        # ControlButton with default colors (yellow triggers, grey controlled)
        return ('controlbutton', {})
    
    # Handle magic walls
    if cell_str[0] == 'M':
        direction = cell_str[1]
        magic_side_map = {
            'n': MAGIC_SIDE_NORTH,
            's': MAGIC_SIDE_SOUTH,
            'w': MAGIC_SIDE_WEST,
            'e': MAGIC_SIDE_EAST,
            'a': MAGIC_SIDE_ALL
        }
        if direction not in magic_side_map:
            raise ValueError(f"Invalid magic wall direction: {direction}")
        return ('magicwall', {'magic_side': magic_side_map[direction]})
    
    # Handle cells with color codes
    obj_code = cell_str[0]
    color_code = cell_str[1]
    
    if color_code not in MAP_COLOR_CODES:
        raise ValueError(f"Invalid color code '{color_code}' in cell '{cell_str}'")
    
    color = MAP_COLOR_CODES[color_code]
    
    if obj_code == 'W':
        return ('wall', {'color': color})
    elif obj_code == 'L':
        return ('door', {'color': color, 'is_locked': True, 'is_open': False})
    elif obj_code == 'C':
        return ('door', {'color': color, 'is_locked': False, 'is_open': False})
    elif obj_code == 'O':
        return ('door', {'color': color, 'is_locked': False, 'is_open': True})
    elif obj_code == 'K':
        return ('key', {'color': color})
    elif obj_code == 'B':
        return ('ball', {'color': color})
    elif obj_code == 'X':
        return ('box', {'color': color})
    elif obj_code == 'G':
        return ('goal', {'color': color})
    elif obj_code == 'A':
        return ('agent', {'color': color})
    else:
        raise ValueError(f"Unknown cell type: {cell_str}")


def create_object_from_spec(cell_spec, objects_set, actions_set=None, stumble_probability=0.5, solidify_probability=0.1):
    """
    Create a WorldObj from a cell specification.
    
    Args:
        cell_spec: Tuple (type, params_dict) from _parse_cell
        objects_set: The World class to use
        actions_set: The Actions class to use (optional, needed for ControlButton)
        stumble_probability: Default stumble probability for UnsteadyGround (0.0 to 1.0)
        solidify_probability: Default solidify probability for MagicWall (0.0 to 1.0)
        
    Returns:
        WorldObj or None for empty cells
    """
    if cell_spec is None:
        return None
    
    obj_type, params = cell_spec
    
    if obj_type == 'wall':
        return Wall(objects_set, params.get('color', 'grey'))
    elif obj_type == 'block':
        return Block(objects_set)
    elif obj_type == 'rock':
        return Rock(objects_set)
    elif obj_type == 'lava':
        return Lava(objects_set)
    elif obj_type == 'switch':
        return Switch(objects_set)
    elif obj_type == 'unsteady':
        return UnsteadyGround(objects_set, stumble_probability=stumble_probability)
    elif obj_type == 'magicwall':
        return MagicWall(objects_set, 
                        magic_side=params.get('magic_side', 0),
                        entry_probability=params.get('entry_probability', 0.5),
                        solidify_probability=solidify_probability)
    elif obj_type == 'killbutton':
        return KillButton(objects_set,
                         trigger_color=params.get('trigger_color', 'yellow'),
                         target_color=params.get('target_color', 'grey'),
                         enabled=params.get('enabled', True))
    elif obj_type == 'pauseswitch':
        return PauseSwitch(objects_set,
                          toggle_color=params.get('toggle_color', 'yellow'),
                          target_color=params.get('target_color', 'grey'),
                          is_on=params.get('is_on', False),
                          enabled=params.get('enabled', True))
    elif obj_type == 'disablingswitch':
        return DisablingSwitch(objects_set,
                              toggle_color=params.get('toggle_color', 'grey'),
                              target_type=params.get('target_type', 'killbutton'))
    elif obj_type == 'controlbutton':
        return ControlButton(objects_set,
                            trigger_color=params.get('trigger_color', 'yellow'),
                            controlled_color=params.get('controlled_color', 'grey'),
                            enabled=params.get('enabled', True),
                            actions_set=actions_set)
    elif obj_type == 'door':
        return Door(objects_set, 
                   params.get('color', 'blue'),
                   is_open=params.get('is_open', False),
                   is_locked=params.get('is_locked', False))
    elif obj_type == 'key':
        return Key(objects_set, params.get('color', 'blue'))
    elif obj_type == 'ball':
        color = params.get('color', 'red')
        color_idx = objects_set.COLOR_TO_IDX.get(color, 0)
        return Ball(objects_set, index=color_idx)
    elif obj_type == 'box':
        return Box(objects_set, params.get('color', 'blue'))
    elif obj_type == 'goal':
        color = params.get('color', 'green')
        color_idx = objects_set.COLOR_TO_IDX.get(color, 1)
        return Goal(objects_set, index=color_idx, color=color_idx)
    elif obj_type == 'agent':
        # Agents are handled separately in the environment
        return None
    else:
        raise ValueError(f"Unknown object type: {obj_type}")


class MultiGridEnv(WorldModel):
    """
    2D grid world game environment
    
    Inherits from WorldModel which provides:
    - get_state(): Get a hashable representation of the environment state
    - set_state(): Restore the environment to a specific state
    - transition_probabilities(): Compute exact transition probabilities
    - get_dag(): Compute the DAG structure of the environment
    - plot_dag(): Visualize the DAG structure
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions

    # Orientation codes mapping to direction indices
    ORIENTATION_TO_DIR = {
        'e': 0,  # east/right
        's': 1,  # south/down
        'w': 2,  # west/left
        'n': 3,  # north/up
    }

    def __init__(
            self,
            grid_size=None,
            width=None,
            height=None,
            max_steps=100,
            see_through_walls=False,
            seed=2,
            agents=None,
            partial_obs=True,
            agent_view_size=7,
            actions_set=Actions,
            objects_set = World,
            map=None,
            orientations=0,
            can_push_rocks='e',
            config_file=None,
            config=None,
            stumble_probability=0.5,
            solidify_probability=0.1
    ):
        """
        Initialize a MultiGridEnv.
        
        Args:
            grid_size: Size of the square grid (mutually exclusive with width/height)
            width: Width of the grid
            height: Height of the grid
            max_steps: Maximum number of steps per episode
            see_through_walls: Whether agents can see through walls
            seed: Random seed
            agents: List of Agent objects (auto-created from map if not provided)
            partial_obs: Whether to use partial observations
            agent_view_size: Size of agent's partial observation view
            actions_set: Set of available actions (default: Actions)
            objects_set: Set of available objects (default: World)
            map: ASCII map string defining the grid layout
            orientations: Agent orientations - can be:
                - A single int (0-3) or code ('n','s','e','w') applied to all agents
                - A list of ints or codes, one per agent
                Default is 0 (east/right) for all agents.
            can_push_rocks: Color codes of agents that can push rocks (default: 'e' for grey)
            config_file: Path to JSON or YAML config file containing all init parameters.
                        If provided, loads map and other parameters from the file.
                        YAML files (.yaml, .yml) use PyYAML; JSON files use standard json.
                        Parameters passed explicitly to __init__ override config file values.
                        Supports 'action_class' key to specify the action set by name
                        (e.g., 'SmallActions', 'MinimalActions', 'Actions', 'MineActions').
            config: Dict containing config parameters (same format as YAML/JSON files).
                   Alternative to config_file for programmatic configuration.
                   If both config and config_file are provided, config_file takes precedence.
            stumble_probability: Default probability of stumbling on UnsteadyGround (0.0 to 1.0)
            solidify_probability: Default probability of MagicWall solidifying on failed entry (0.0 to 1.0)
        """
        # Load config from config file or dict if provided
        if config_file is not None:
            config = self._load_config_file(config_file)
        # If no config_file but config dict provided, use it directly
        # (config parameter is already set from function argument)
        
        if config is not None:
            # Apply config values as defaults (explicit params override config)
            # Note: We check against the default values to determine if a parameter
            # was explicitly passed. This ensures backward compatibility with existing
            # code that relies on default values. If the signature defaults change,
            # update these checks accordingly.
            if map is None and 'map' in config:
                map = config['map']
            if max_steps == 100 and 'max_steps' in config:  # default: 100
                max_steps = config['max_steps']
            if seed == 2 and 'seed' in config:  # default: 2
                seed = config['seed']
            if partial_obs is True and 'partial_obs' in config:  # default: True
                partial_obs = config['partial_obs']
            if agent_view_size == 7 and 'agent_view_size' in config:  # default: 7
                agent_view_size = config['agent_view_size']
            if see_through_walls is False and 'see_through_walls' in config:  # default: False
                see_through_walls = config['see_through_walls']
            if orientations == 0 and 'orientations' in config:  # default: 0
                orientations = config['orientations']
            if can_push_rocks == 'e' and 'can_push_rocks' in config:  # default: 'e'
                can_push_rocks = config['can_push_rocks']
            if grid_size is None and 'grid_size' in config:
                grid_size = config['grid_size']
            if width is None and 'width' in config:
                width = config['width']
            if height is None and 'height' in config:
                height = config['height']
            if stumble_probability == 0.5 and 'stumble_probability' in config:  # default: 0.5
                stumble_probability = config['stumble_probability']
            if solidify_probability == 0.1 and 'solidify_probability' in config:  # default: 0.1
                solidify_probability = config['solidify_probability']
            # Handle action_class from config
            if actions_set is Actions and 'action_class' in config:
                action_class_name = config['action_class']
                actions_set = get_actions_class(action_class_name)
            
            # Store possible_goals specs for later parsing (after reset)
            self._possible_goals_specs = config.get('possible_goals', None)
            
            # Store config file path for reference
            self._config_file = config_file
            self._config_metadata = config.get('metadata', {})
        else:
            # No config provided (neither file nor dict)
            self._config_file = None
            self._config_metadata = {}
            self._possible_goals_specs = None
        
        # Store map specification for use in _gen_grid
        self._map_spec = map
        self._map_parsed = None
        
        # Store original parameters for environment reconstruction
        self._init_grid_size = grid_size
        self._init_seed = seed
        self._init_orientations = orientations
        self._init_can_push_rocks = can_push_rocks
        self._init_stumble_probability = stumble_probability
        self._init_solidify_probability = solidify_probability
        
        # Store stumble_probability for use by UnsteadyGround objects
        self.stumble_probability = stumble_probability
        
        # Store solidify_probability for use by MagicWall objects
        self.solidify_probability = solidify_probability
        
        # Initialize RNG early so we can use it for random orientations
        # This is done before reset() to allow random orientations to be drawn in __init__
        self.np_random, _ = seeding.np_random(seed)
        
        # Parse can_push_rocks color codes to color names
        if can_push_rocks is not None:
            self._can_push_rocks_colors = set()
            for code in can_push_rocks:
                if code not in MAP_COLOR_CODES:
                    raise ValueError(f"Invalid color code '{code}' in can_push_rocks. Must be one of: r, g, b, p, y, e")
                self._can_push_rocks_colors.add(MAP_COLOR_CODES[code])
        else:
            self._can_push_rocks_colors = None
        
        # If map is provided, parse it to get dimensions and agents
        if map is not None:
            map_width, map_height, cells, map_agents, has_interactive_objects = parse_map_string(map, objects_set)
            self._map_parsed = (map_width, map_height, cells, map_agents)
            self._has_interactive_objects = has_interactive_objects
            
            # Auto-select action class based on map contents if not explicitly set
            # SmallActions for simple movement, ObjectActions if pickup/toggle needed
            if actions_set is Actions:  # Default was used, auto-detect
                if has_interactive_objects:
                    actions_set = ObjectActions
                else:
                    actions_set = SmallActions
            
            # Override width/height with map dimensions
            width = map_width
            height = map_height
            
            # Auto-create agents from map if not provided
            if agents is None:
                agents = []
                for x, y, agent_params in map_agents:
                    color = agent_params.get('color', 'red')
                    color_idx = objects_set.COLOR_TO_IDX.get(color, 0)
                    # Determine if this agent can push rocks based on color
                    agent_can_push_rocks = (
                        hasattr(self, '_can_push_rocks_colors') and 
                        self._can_push_rocks_colors is not None and
                        color in self._can_push_rocks_colors
                    )
                    agents.append(Agent(objects_set, color_idx, can_push_rocks=agent_can_push_rocks))
            
            # Handle orientations: convert single value to list, then parse
            num_agents = len(map_agents)
            # Normalize orientations to a list
            if isinstance(orientations, (int, str)):
                # Single value: apply to all agents
                orientations = [orientations] * num_agents
            
            # Parse orientation codes/ints to direction indices
            self._agent_orientations = []
            for orient in orientations:
                if isinstance(orient, int):
                    if not (0 <= orient <= 3):
                        raise ValueError(f"Invalid orientation {orient}. Must be 0-3.")
                    self._agent_orientations.append(orient)
                elif orient in self.ORIENTATION_TO_DIR:
                    self._agent_orientations.append(self.ORIENTATION_TO_DIR[orient])
                else:
                    raise ValueError(f"Invalid orientation '{orient}'. Must be 0-3 or one of: w, n, e, s")
        else:
            self._agent_orientations = None
        
        self.agents = agents

        # Does the agents have partial or full observation?
        self.partial_obs = partial_obs

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions.available))

        self.objects=objects_set

        if partial_obs:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(agent_view_size, agent_view_size, self.objects.encode_dim),
                dtype='uint8'
            )

        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(width, height, self.objects.encode_dim),
                dtype='uint8'
            )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None
        
        # Video recording state
        self._recording = False
        self._video_frames = []

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # RNG already initialized earlier for random orientations, no need to call seed() again

        # Initialize the state
        self.reset()
        
        # Parse possible goals from config (needs to be after reset so env is initialized)
        # Goals are created on-the-fly for any human_agent_index passed to generate()/sample()
        if self._possible_goals_specs:
            self.possible_goal_sampler, self.possible_goal_generator = \
                create_config_goal_sampler_and_generator(self._possible_goals_specs, self)
        else:
            self.possible_goal_sampler = None
            self.possible_goal_generator = None
    
    def _get_construction_args(self) -> tuple:
        """Get positional arguments for reconstructing this environment."""
        return ()
    
    def _get_construction_kwargs(self) -> dict:
        """Get keyword arguments for reconstructing this environment."""
        # Store original __init__ parameters for reconstruction in workers
        return {
            'grid_size': getattr(self, '_init_grid_size', None),
            'width': self.width,
            'height': self.height,
            'max_steps': self.max_steps,
            'see_through_walls': self.see_through_walls,
            'seed': getattr(self, '_init_seed', 2),
            'agents': None,  # Will be reconstructed from map
            'partial_obs': self.partial_obs,
            'agent_view_size': getattr(self, 'observation_space', None).shape[0] if hasattr(self, 'observation_space') and self.partial_obs else 7,
            'actions_set': self.actions,
            'objects_set': self.objects,
            'map': self._map_spec,
            'orientations': getattr(self, '_init_orientations', None),
            'can_push_rocks': getattr(self, '_init_can_push_rocks', 'e'),
            'config_file': getattr(self, '_config_file', None),
            'stumble_probability': getattr(self, '_init_stumble_probability', 0.5),
            'solidify_probability': getattr(self, '_init_solidify_probability', 0.1)
        }
    
    @staticmethod
    def _load_config_file(config_path: str) -> dict:
        """
        Load environment configuration from a JSON or YAML file.
        
        Args:
            config_path: Path to the config file (JSON or YAML)
            
        Returns:
            dict: Configuration dictionary with keys:
                - map: ASCII map string or list of strings
                - max_steps: Maximum steps per episode
                - seed: Random seed
                - orientations: List of orientation codes
                - can_push_rocks: Color codes for rock-pushing agents
                - partial_obs: Whether to use partial observations
                - agent_view_size: Size of partial observation view
                - see_through_walls: Whether agents can see through walls
                - metadata: Optional metadata dict (name, description, author, etc.)
                
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid or missing required fields
            ImportError: If YAML file is requested but PyYAML is not installed
        """
        try:
            with open(config_path, 'r') as f:
                # Determine format based on file extension
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    if not YAML_AVAILABLE:
                        raise ImportError(
                            f"PyYAML is required to load YAML config files. "
                            f"Install it with: pip install pyyaml"
                        )
                    try:
                        config = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
                else:
                    # Default to JSON for .json or unknown extensions
                    try:
                        config = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in config file {config_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Validate required fields
        if 'map' not in config:
            raise ValueError(f"Config file {config_path} must contain a 'map' field")
        
        return config

    def reset(self):

        # Create terrain grid first, before _gen_grid
        # This stores overlappable terrain (like unsteady ground) that persists under agents
        self.terrain_grid = Grid(self.width, self.height)
        
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Reset all per-episode agent state
        for a in self.agents:
            a.carrying = None
            a.terminated = False
            a.started = True
            a.paused = False
            a.forced_next_action = None
            # Derive on_unsteady_ground from terrain grid at agent's position
            terrain_cell = self.terrain_grid.get(*a.pos)
            a.on_unsteady_ground = (
                terrain_cell is not None and terrain_cell.type == 'unsteadyground'
            )

        # Step count since episode start
        self.step_count = 0
        
        # Track cells where stumbling occurred in the current step (for visual feedback)
        self.stumbled_cells = set()
        
        # Build cache of mobile objects (blocks, rocks) and mutable objects (doors, boxes, magic walls)
        # This avoids full grid scans in get_state()
        self._build_object_cache()
        
        # Compute and cache the initial map hash (immutable grid structure)
        # This is computed once per reset and used to distinguish states from different maps
        self._initial_map_hash = self._compute_immutable_objects_hash()

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(self.agents))]
        obs=[self.objects.normalize_obs*ob for ob in obs]
        return obs
    
    def _build_object_cache(self):
        """
        Build cache of mobile and mutable objects to avoid full grid scans in get_state().
        
        Mobile objects: blocks and rocks (can be pushed)
        Mutable objects: doors, boxes, magic walls, killbuttons, pauseswitches, controlbuttons (have mutable state)
        
        This cache stores references to the objects themselves, not their positions.
        Positions are read from the grid when get_state() is called.
        """
        self._mobile_objects = []  # List of (initial_pos, obj) tuples for sorting
        self._mutable_objects = []  # List of (pos, obj) tuples - position is immutable for these
        
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell = self.grid.get(i, j)
                if cell is None:
                    continue
                
                obj_type = cell.type
                
                # Mobile objects: blocks and rocks
                if obj_type in ('block', 'rock'):
                    self._mobile_objects.append(((i, j), cell))
                
                # Mutable objects: doors, boxes, magic walls, killbuttons, pauseswitches, controlbuttons
                elif obj_type in ('door', 'box', 'magicwall', 'killbutton', 'pauseswitch', 'controlbutton'):
                    self._mutable_objects.append(((i, j), cell))
        
        # Sort mobile objects by initial position for deterministic ordering
        self._mobile_objects.sort(key=lambda x: x[0])
    
    def _compute_immutable_objects_hash(self) -> int:
        """
        Compute a hash of the initial/immutable grid structure.
        
        This captures the positions and types of immutable objects (walls, goals, lava, etc.)
        that don't change during an episode. This hash is used to distinguish states
        from different map configurations in ensemble mode, where different worlds
        may have the same mutable state but different wall layouts.
        
        The hash includes:
        - Grid dimensions (width, height)
        - Positions and types of immutable objects (walls, goals, lava, unsteady ground)
        
        Note: Mobile objects (blocks, rocks, agents) and mutable objects (doors, boxes, 
        magic walls, keys, balls) are NOT included since they can change during episodes
        and are tracked in get_state().
        
        Returns:
            int: A hash value representing the immutable grid structure.
        """
        # Start with grid dimensions
        immutable_data = [self.grid.width, self.grid.height]
        
        # Collect immutable objects from the grid
        # Types that don't change position and don't have mutable state
        # Note: keys and balls CAN be picked up/moved, so they're not truly immutable
        IMMUTABLE_TYPES = {'wall', 'goal', 'lava', 'unsteadyground'}
        
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type in IMMUTABLE_TYPES:
                    # Store (x, y, type, color) - color is important for distinguishing e.g. different colored goals
                    color = getattr(cell, 'color', None)
                    immutable_data.append((i, j, cell.type, color))
                
                # Also check terrain grid for unsteady ground
                terrain_cell = self.terrain_grid.get(i, j)
                if terrain_cell is not None and terrain_cell.type in IMMUTABLE_TYPES:
                    color = getattr(terrain_cell, 'color', None)
                    immutable_data.append((i, j, terrain_cell.type, color, 'terrain'))
        
        return hash(tuple(immutable_data))
    
    def get_initial_map_hash(self) -> int:
        """
        Get the hash of the initial/immutable grid structure.
        
        This hash uniquely identifies the map layout (walls, goals, etc.) and is used
        to distinguish states from different map configurations in ensemble mode.
        Without this, lookup tables would conflate states from different worlds
        that happen to have the same mutable state.
        
        The hash is computed once during reset() and cached for efficiency.
        
        Returns:
            int: A hash value representing the immutable grid structure.
            
        Raises:
            AttributeError: If called before reset() has been called.
        """
        if not hasattr(self, '_initial_map_hash'):
            raise AttributeError(
                "get_initial_map_hash() called before reset(). "
                "The initial map hash is computed during reset()."
            )
        return self._initial_map_hash

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Short string for opened door

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        """
        Generate the grid layout.
        
        If a map specification was provided to __init__, this will use it.
        Otherwise, subclasses must override this method.
        """
        if self._map_parsed is not None:
            self._gen_grid_from_map()
        else:
            assert False, "_gen_grid needs to be implemented by each environment"
    
    def _gen_grid_from_map(self):
        """
        Generate the grid from the parsed map specification.
        """
        map_width, map_height, cells, map_agents = self._map_parsed
        
        # Create the grid
        self.grid = Grid(map_width, map_height)
        
        # Place objects from the map (rocks are now simpler - just use create_object_from_spec)
        for y in range(map_height):
            for x in range(map_width):
                cell_spec = cells[y][x]
                if cell_spec is not None and cell_spec[0] != 'agent':
                    obj = create_object_from_spec(cell_spec, self.objects, self.actions, 
                                                  stumble_probability=self.stumble_probability,
                                                  solidify_probability=self.solidify_probability)
                    if obj is not None:
                        self.grid.set(x, y, obj)
        
        # Place agents with their orientations
        for agent_idx, (x, y, agent_params) in enumerate(map_agents):
            if agent_idx < len(self.agents):
                agent = self.agents[agent_idx]
                agent.pos = np.array([x, y])
                # Use stored orientation if available, otherwise default to 0 (east)
                if self._agent_orientations is not None and agent_idx < len(self._agent_orientations):
                    agent.dir = self._agent_orientations[agent_idx]
                else:
                    agent.dir = 0  # Default facing right/east
                self.grid.set(x, y, agent)

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        """
        Handle pickup action - agent picks up object in front of them.
        
        Default implementation that can be overridden by subclasses for
        game-specific pickup behavior (e.g., rewards, restrictions).
        
        Args:
            i: Agent index
            rewards: Rewards array to potentially modify
            fwd_pos: Position in front of the agent
            fwd_cell: Object at the forward position (or None)
        """
        if fwd_cell and fwd_cell.can_pickup():
            if self.agents[i].carrying is None:
                # Pick up the object
                self.agents[i].carrying = fwd_cell
                # Update object's position tracking if it has cur_pos
                if hasattr(fwd_cell, 'cur_pos'):
                    fwd_cell.cur_pos = np.array([-1, -1])
                # Remove from grid
                self.grid.set(*fwd_pos, None)

    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        """
        Handle drop action - agent drops carried object in front of them.
        
        Default implementation that can be overridden by subclasses for
        game-specific drop behavior (e.g., rewards, restrictions).
        
        Args:
            i: Agent index
            rewards: Rewards array to potentially modify
            fwd_pos: Position in front of the agent
            fwd_cell: Object at the forward position (or None)
        """
        carrying = self.agents[i].carrying
        if carrying is not None:
            # Can only drop on empty cells or cells that can be overlapped
            if fwd_cell is None or fwd_cell.can_overlap():
                # Place object on grid
                self.grid.set(*fwd_pos, carrying)
                # Update object's position tracking if it has cur_pos
                if hasattr(carrying, 'cur_pos'):
                    carrying.cur_pos = np.array(fwd_pos)
                # Clear agent's carrying state
                self.agents[i].carrying = None

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        """
        Handle special effects when an agent moves to a cell.
        
        This includes deactivating agents when they step on lava.
        """
        # Handle Lava effects - agent gets deactivated (terminated)
        if fwd_cell is not None and fwd_cell.type == 'lava':
            self.agents[i].terminated = True
    
    def can_forward(self, state, agent_index: int) -> bool:
        """
        Check if an agent can, in principle, move forward from the given state.
        
        This checks whether the forward cell is passable for the given agent,
        ignoring possible conflicts with other agents' actions. The check is:
        
        1. Forward cell is within grid bounds
        2. Forward cell is either:
           - Empty (None)
           - An object that can be overlapped (unsteady ground, control buttons, etc.)
           - A block or rock that can be pushed (considering agent's can_push_rocks)
           - A magic wall the agent can attempt to enter (if can_enter_magic_walls and active)
        
        This method is useful for exploration policies to bias toward passable
        directions without considering multi-agent conflicts.
        
        Args:
            state: The environment state tuple from get_state()
            agent_index: Index of the agent to check
        
        Returns:
            bool: True if forward movement is possible in principle
        """
        # Parse state
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Get agent state
        agent_state = agent_states[agent_index]
        agent_x, agent_y, agent_dir = agent_state[0], agent_state[1], agent_state[2]
        
        if agent_x is None or agent_y is None or agent_dir is None:
            return False  # Agent not on grid
        
        # Compute forward position
        dir_vec = [(1, 0), (0, 1), (-1, 0), (0, -1)][agent_dir]
        fwd_x = agent_x + dir_vec[0]
        fwd_y = agent_y + dir_vec[1]
        
        # Check bounds
        if fwd_x < 0 or fwd_x >= self.grid.width or fwd_y < 0 or fwd_y >= self.grid.height:
            return False
        
        # Get the agent object to check its capabilities
        agent = self.agents[agent_index]
        
        # Get the cell at forward position
        fwd_cell = self.grid.get(fwd_x, fwd_y)
        
        # Empty cell - can move
        if fwd_cell is None:
            return True
        
        # Cell with overlappable object - can move
        if fwd_cell.can_overlap():
            return True
        
        # Check for pushable objects (blocks/rocks)
        if fwd_cell.type == 'block':
            # All agents can push blocks - check if push is possible
            can_push, _, _ = self._can_push_objects(agent, np.array([fwd_x, fwd_y]))
            return can_push
        
        if fwd_cell.type == 'rock':
            # Only agents with can_push_rocks can push rocks
            if not getattr(agent, 'can_push_rocks', False):
                return False
            can_push, _, _ = self._can_push_objects(agent, np.array([fwd_x, fwd_y]))
            return can_push
        
        # Check for magic walls
        if fwd_cell.type == 'magicwall' and fwd_cell.active:
            # Only agents with can_enter_magic_walls can attempt entry
            if not getattr(agent, 'can_enter_magic_walls', False):
                return False
            # Check if approaching from the magic side
            approach_dir = (agent_dir + 2) % 4
            if fwd_cell.magic_side == 4 or approach_dir == fwd_cell.magic_side:
                return True
            return False
        
        # All other objects (walls, doors, etc.) - cannot pass
        return False
    
    def _can_push_objects(self, agent, start_pos):
        """
        Check if agent can push blocks/rocks starting at start_pos.
        Returns (can_push, num_objects, end_pos) where:
        - can_push: True if push is possible
        - num_objects: number of consecutive blocks/rocks
        - end_pos: position where the last object would move to
        """
        # Check if there are consecutive blocks/rocks in the direction the agent is facing
        direction = agent.dir_vec
        current_pos = np.array(start_pos)
        num_objects = 0
        
        # Count consecutive blocks/rocks
        while True:
            cell = self.grid.get(*current_pos)
            if cell is None:
                break
            if cell.type not in ['block', 'rock']:
                break
            # For rocks, check if agent can push this rock
            if cell.type == 'rock' and not cell.can_be_pushed_by(agent):
                return False, 0, None
            num_objects += 1
            current_pos = current_pos + direction
            
            # Bounds check
            if current_pos[0] < 0 or current_pos[0] >= self.grid.width or \
               current_pos[1] < 0 or current_pos[1] >= self.grid.height:
                return False, 0, None
        
        # current_pos is now the first empty cell after the objects
        # Check if this cell is empty or can be overlapped
        end_cell = self.grid.get(*current_pos)
        if end_cell is None or end_cell.can_overlap():
            # Also check if the end position was occupied by another agent at step start.
            # This prevents chain conflicts where pushing depends on execution order.
            end_pos_tuple = tuple(current_pos)
            if hasattr(self, '_initial_agent_positions') and end_pos_tuple in self._initial_agent_positions:
                # Cannot push onto a cell that was occupied by an agent at step start
                return False, 0, None
            return True, num_objects, current_pos
        else:
            return False, 0, None
    
    def _push_objects(self, agent, start_pos):
        """
        Push blocks/rocks starting at start_pos in the direction agent is facing.
        Returns True if push was successful.
        """
        can_push, num_objects, end_pos = self._can_push_objects(agent, start_pos)
        
        if not can_push or num_objects == 0:
            return False
        
        # Move objects from back to front to avoid overwriting
        direction = agent.dir_vec
        # Start from the end position and work backwards
        for j in range(num_objects):
            from_pos = end_pos - direction * (j + 1)
            to_pos = end_pos - direction * j
            obj = self.grid.get(*from_pos)
            self.grid.set(*to_pos, obj)
            if obj:
                obj.cur_pos = to_pos
        
        # Clear the original start position (now empty)
        self.grid.set(*start_pos, None)
        
        # Now agent can move into start_pos
        self.grid.set(*start_pos, agent)
        self.grid.set(*agent.pos, None)
        agent.pos = np.array(start_pos)
        
        return True

    def _move_agent_to_cell(self, agent_idx, target_pos, target_cell):
        """
        Move an agent to a target cell and update terrain tracking.
        
        Args:
            agent_idx: Index of the agent to move
            target_pos: Target position (numpy array or tuple)
            target_cell: The object/cell at the target position (can be None)
        """
        # Restore terrain at old position (if any)
        old_pos = self.agents[agent_idx].pos
        terrain_at_old_pos = self.terrain_grid.get(*old_pos)
        if terrain_at_old_pos is not None:
            self.grid.set(*old_pos, terrain_at_old_pos)
        else:
            self.grid.set(*old_pos, None)
        
        # Update agent position
        self.agents[agent_idx].pos = np.array(target_pos) if not isinstance(target_pos, np.ndarray) else target_pos
        
        # Save overlappable terrain at new position so it persists under the agent
        # and can be restored when the agent leaves.
        # Note: If terrain is already set at the target position (e.g., magic wall entry
        # pre-saves the magic wall), we preserve it rather than overwriting.
        existing_terrain = self.terrain_grid.get(*self.agents[agent_idx].pos)
        if existing_terrain is None:
            if target_cell is not None and target_cell.can_overlap():
                self.terrain_grid.set(*self.agents[agent_idx].pos, target_cell)
            else:
                self.terrain_grid.set(*self.agents[agent_idx].pos, None)
        # else: terrain already set (e.g., by magic wall entry), keep it
        
        # Set new position
        self.grid.set(*self.agents[agent_idx].pos, self.agents[agent_idx])
    
    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        # Handle both old and new numpy random API
        if hasattr(self.np_random, 'randint'):
            return self.np_random.randint(low, high)
        else:
            return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        # Handle both old and new numpy random API
        if hasattr(self.np_random, 'randint'):
            return (self.np_random.randint(0, 2) == 0)
        else:
            return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def agent_sees(self, a, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = a.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def _is_still_action(self, action) -> bool:
        """
        Check if an action is the 'still' (no-op) action.
        
        Handles action sets that don't have a 'still' action (like MinimalActions)
        by returning False when still is not defined or is None.
        
        Args:
            action: The action index to check
            
        Returns:
            bool: True if the action is the 'still' action, False otherwise
        """
        still_action = getattr(self.actions, 'still', None)
        return still_action is not None and action == still_action

    def _execute_single_agent_action(self, agent_idx, action, rewards):
        """
        Execute a single agent's action. This is the core action execution logic
        shared by step(), _compute_successor_state(), and _compute_successor_state_with_unsteady().
        
        Args:
            agent_idx: Index of the agent
            action: Action to execute
            rewards: Rewards array to update
            
        Returns:
            bool: True if the action resulted in reaching a goal (done condition)
        """
        done = False
        fwd_pos = self.agents[agent_idx].front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        
        # Check if the target cell was occupied by another agent at the start of this step.
        # This prevents chain conflicts where outcome depends on execution order.
        # An agent can only move into a cell the step AFTER it was vacated.
        fwd_pos_tuple = tuple(fwd_pos)
        own_pos_tuple = tuple(self.agents[agent_idx].pos)
        target_was_other_agent = (
            hasattr(self, '_initial_agent_positions') and 
            fwd_pos_tuple in self._initial_agent_positions and
            fwd_pos_tuple != own_pos_tuple  # Can always stay in own cell
        )
        
        if action == self.actions.left:
            self.agents[agent_idx].dir -= 1
            if self.agents[agent_idx].dir < 0:
                self.agents[agent_idx].dir += 4
        
        elif action == self.actions.right:
            self.agents[agent_idx].dir = (self.agents[agent_idx].dir + 1) % 4
        
        elif action == self.actions.forward:
            # Block movement if target was occupied by another agent at step start
            if target_was_other_agent:
                pass  # Movement blocked - agent stays in place
            elif fwd_cell is not None and fwd_cell.type in ['block', 'rock']:
                self._push_objects(self.agents[agent_idx], fwd_pos)
            elif fwd_cell is not None:
                if fwd_cell.type == 'goal':
                    done = True
                    self._reward(agent_idx, rewards, 1)
                    self._move_agent_to_cell(agent_idx, fwd_pos, fwd_cell)
                elif fwd_cell.type == 'switch':
                    self._handle_switch(agent_idx, rewards, fwd_pos, fwd_cell)
                    self._move_agent_to_cell(agent_idx, fwd_pos, fwd_cell)
                elif fwd_cell.can_overlap():
                    self._move_agent_to_cell(agent_idx, fwd_pos, fwd_cell)
            elif fwd_cell is None:
                self._move_agent_to_cell(agent_idx, fwd_pos, fwd_cell)
            self._handle_special_moves(agent_idx, rewards, fwd_pos, fwd_cell)
        
        elif 'build' in self.actions.available and action == self.actions.build:
            self._handle_build(agent_idx, rewards, fwd_pos, fwd_cell)
        
        elif action == self.actions.pickup:
            self._handle_pickup(agent_idx, rewards, fwd_pos, fwd_cell)
        
        elif action == self.actions.drop:
            self._handle_drop(agent_idx, rewards, fwd_pos, fwd_cell)
        
        elif action == self.actions.toggle:
            if fwd_cell:
                # Set env.carrying to agent's carrying for Door compatibility
                self.carrying = self.agents[agent_idx].carrying
                # Pass the agent_idx so toggle knows which agent is doing the action
                fwd_cell.toggle(self, fwd_pos, agent_idx=agent_idx)
                self.carrying = None  # Reset after toggle
        
        elif action == self.actions.done:
            pass
        
        else:
            assert False, "unknown action"
        
        # After executing, check if this agent was programming a control button.
        # Record the action for any control button awaiting this agent's action.
        for j in range(self.grid.height):
            for ii in range(self.grid.width):
                cell = self.grid.get(ii, j)
                if (cell is not None and cell.type == 'controlbutton' and 
                    cell._awaiting_action and cell.controlled_agent == agent_idx):
                    # Skip recording if this is the toggle that just activated programming mode
                    if cell._just_activated:
                        cell._just_activated = False  # Clear the flag for next step
                    else:
                        # Record any action (including toggle for controlling other switches)
                        cell.record_action(action)
        
        return done

    def _process_unsteady_forward_agents(self, unsteady_forward_agents, rewards, unsteady_outcomes=None):
        """
        Process agents on unsteady ground attempting forward movement.
        This handles stumbling stochasticity and conflict resolution for unsteady agents.
        
        Args:
            unsteady_forward_agents: List of agent indices on unsteady ground
            rewards: Rewards array to update
            unsteady_outcomes: Optional dict mapping agent_idx -> outcome_type 
                             ('forward', 'left-forward', 'right-forward')
                             If None, randomly determine outcomes
        
        Returns:
            bool: True if any agent reached a goal (done condition)
        """
        done = False
        unsteady_targets = {}  # agent_idx -> (target_pos, stumbled, turn_dir)
        contested_cells = {}  # agent_idx -> contested_cell
        occupied_targets = set()
        
        # Determine stumbling outcomes and target cells for all unsteady agents
        # Filter out agents that were terminated/paused by earlier actions this step
        unsteady_forward_agents = [
            i for i in unsteady_forward_agents
            if not self.agents[i].terminated and not self.agents[i].paused and self.agents[i].started
        ]
        for i in unsteady_forward_agents:
            # Determine outcome
            if unsteady_outcomes is not None and i in unsteady_outcomes:
                # Use provided outcome
                outcome_type = unsteady_outcomes[i]
                if outcome_type == 'left-forward':
                    turn_dir = 'left'
                    stumbles = True
                elif outcome_type == 'right-forward':
                    turn_dir = 'right'
                    stumbles = True
                else:  # 'forward'
                    turn_dir = None
                    stumbles = False
            else:
                # Randomly determine stumbling based on the cell's stumble_probability
                # Get the stumble probability from the unsteady ground cell
                current_cell = self.grid.get(*self.agents[i].pos)
                if current_cell and current_cell.type == 'unsteadyground':
                    stumble_prob = current_cell.stumble_probability
                else:
                    stumble_prob = 0.5  # Default fallback (shouldn't happen)
                stumbles = self.np_random.random() < stumble_prob
                turn_dir = self.np_random.choice(['left', 'right']) if stumbles else None
            
            # Apply turn if stumbling
            if stumbles:
                # Record the agent's position for visual feedback
                if hasattr(self, 'stumbled_cells'):
                    self.stumbled_cells.add(tuple(self.agents[i].pos))
                
                if turn_dir == 'left':
                    self.agents[i].dir -= 1
                    if self.agents[i].dir < 0:
                        self.agents[i].dir += 4
                else:  # right
                    self.agents[i].dir = (self.agents[i].dir + 1) % 4
            
            # Compute target position
            target_pos = tuple(self.agents[i].front_pos)
            unsteady_targets[i] = (target_pos, stumbles, turn_dir)
            
            # Determine contested cell (target or block end position)
            target_cell = self.grid.get(*target_pos)
            if target_cell is not None and target_cell.type in ['block', 'rock']:
                can_push, num_objects, end_pos = self._can_push_objects(self.agents[i], np.array(target_pos))
                contested_cells[i] = tuple(end_pos) if can_push else target_pos
            else:
                contested_cells[i] = target_pos
        
        # Check for conflicts
        contested_counts = {}
        for i, contested_cell in contested_cells.items():
            contested_counts[contested_cell] = contested_counts.get(contested_cell, 0) + 1
            cell_obj = self.grid.get(*contested_cell)
            if cell_obj is not None and cell_obj.type == 'agent':
                occupied_targets.add(contested_cell)
            # Also check if the contested cell was occupied by another agent at step start
            # This prevents chain conflicts for unsteady agents too
            if hasattr(self, '_initial_agent_positions') and contested_cell in self._initial_agent_positions:
                # Check it's not the agent's own position
                if contested_cell != tuple(self.agents[i].pos):
                    occupied_targets.add(contested_cell)
        
        for contested_cell, count in contested_counts.items():
            if count > 1:
                occupied_targets.add(contested_cell)
        
        # Execute movements for unsteady agents
        for i in unsteady_forward_agents:
            target_pos, stumbled, turn_dir = unsteady_targets[i]
            fwd_pos = np.array(target_pos)
            fwd_cell = self.grid.get(*fwd_pos)
            
            can_move = contested_cells[i] not in occupied_targets
            
            if can_move:
                if fwd_cell is not None and fwd_cell.type in ['block', 'rock']:
                    pushed = self._push_objects(self.agents[i], fwd_pos)
                    can_move = pushed
                elif fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        done = True
                        self._reward(i, rewards, 1)
                        can_move = True
                    elif fwd_cell.type == 'switch':
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                        can_move = True
                    elif fwd_cell.can_overlap():
                        can_move = True
                    else:
                        can_move = False
                elif fwd_cell is None:
                    can_move = True
                else:
                    can_move = False
            
            if can_move:
                self._move_agent_to_cell(i, fwd_pos, fwd_cell)
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
        
        return done
    
    def _process_magic_wall_agents(self, magic_wall_agents, rewards):
        """
        Process agents attempting to enter magic walls.
        These agents are processed last, and entry succeeds with the magic wall's probability.
        If entry fails, the magic wall may solidify into a normal wall based on solidify_probability.
        
        NOTE: This function uses np_random for stochastic outcomes and should ONLY be called 
        from step(), never from transition_probabilities() or its helpers!
        
        Args:
            magic_wall_agents: List of agent indices attempting to enter magic walls
            rewards: Rewards array to update
        
        Returns:
            bool: True if any agent reached a goal (done condition)
        """
        done = False
        
        for i in magic_wall_agents:
            # Skip agents terminated/paused by earlier actions this step
            if (self.agents[i].terminated or
                    self.agents[i].paused or
                    not self.agents[i].started):
                continue
            fwd_pos = self.agents[i].front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            
            # Verify it's still a magic wall (shouldn't change, but be safe)
            if fwd_cell is None or fwd_cell.type != 'magicwall':
                continue
            
            # Check if entry succeeds based on probability
            if self.np_random.random() < fwd_cell.entry_probability:
                # Entry succeeds - record the magic wall position for visual feedback
                if hasattr(self, 'magic_wall_entered_cells'):
                    self.magic_wall_entered_cells.add(tuple(fwd_pos))
                
                # Move agent into the magic wall cell
                # First, save the magic wall to terrain_grid so agent can step off it later
                self.terrain_grid.set(*fwd_pos, fwd_cell)
                self._move_agent_to_cell(i, fwd_pos, fwd_cell)
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            else:
                # Entry fails - check if magic wall should solidify (deactivate)
                if self.np_random.random() < fwd_cell.solidify_probability:
                    # Deactivate the magic wall (it now acts as a normal wall)
                    fwd_cell.active = False
        
        return done
    
    def _categorize_agents(self, actions, active_agents=None):
        """
        Helper function to categorize agents into normal, unsteady-forward, and magic-wall-entry groups.
        This logic is shared between step() and transition_probabilities().
        
        **IMPORTANT NOTE FOR DEVELOPERS:**
        When adding any new object type or stochastic behavior that affects agent actions:
        
        1. Add categorization logic HERE in _categorize_agents() to identify affected agents
        2. Update step() to process the new agent category appropriately
        3. Update transition_probabilities() to add corresponding uncertainty blocks
        4. Update _compute_successor_state_with_unsteady() (or create a new helper) to handle 
           deterministic execution based on resolved outcomes
        
        This ensures consistency between:
        - step(): which samples stochastic outcomes randomly
        - transition_probabilities(): which enumerates all possible outcomes with exact probabilities
        
        Both functions MUST produce the same distribution over successor states for any given 
        state-action pair. Failing to maintain this consistency will break the correctness of 
        probability computations and planning algorithms that depend on them.
        
        Examples of stochastic elements that follow this pattern:
        - Unsteady ground: agent stumbles with probability, creating 3 outcomes (forward, left+forward, right+forward)
        - Magic walls: agent enters with probability, creating 2 outcomes (succeed, fail)
        
        Args:
            actions: List of action indices, one per agent
            active_agents: List of active agent indices (if None, determines from agent states)
            
        Returns:
            tuple: (normal_agents, unsteady_forward_agents, magic_wall_agents)
        """
        normal_agents = []
        unsteady_forward_agents = []
        magic_wall_agents = []
        
        # Determine active agents if not provided
        if active_agents is None:
            active_agents = []
            for i in range(len(actions)):
                if (not self.agents[i].terminated and 
                    not self.agents[i].paused and 
                    self.agents[i].started and 
                    not self._is_still_action(actions[i])):
                    active_agents.append(i)
        
        for i in active_agents:
            # Check if agent is attempting to enter a magic wall
            if (actions[i] == self.actions.forward and 
                self.agents[i].can_enter_magic_walls):
                fwd_pos = self.agents[i].front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                # Check if it's an active magic wall (not solidified)
                if fwd_cell is not None and fwd_cell.type == 'magicwall' and fwd_cell.active:
                    # Check if agent is approaching from the magic side (or magic_side=4 means all sides)
                    # Agent's direction is where they're facing, magic_side is where wall can be entered from
                    # If agent faces right (dir=0), they approach from left (opposite of right=0 is left=2)
                    approach_dir = (self.agents[i].dir + 2) % 4
                    if fwd_cell.magic_side == 4 or approach_dir == fwd_cell.magic_side:
                        magic_wall_agents.append(i)
                        continue
                
            # Check if agent is on unsteady ground and attempting forward
            if (actions[i] == self.actions.forward and 
                self.agents[i].on_unsteady_ground):
                unsteady_forward_agents.append(i)
            else:
                normal_agents.append(i)
        
        return normal_agents, unsteady_forward_agents, magic_wall_agents

    def step(self, actions):
        # Clear visual feedback from previous step
        self.stumbled_cells = set()
        self.magic_wall_entered_cells = set()

        # Get current state and delegate transition to _transition_probabilities_impl
        # with sample_one=True to randomly sample a single outcome.
        # This guarantees step() and transition_probabilities() produce the same
        # distribution over successor states, avoiding any divergence.
        # Note: forced_next_action (from ControlButton triggers) is handled inside
        # _transition_probabilities_impl, which reads it from the state, overrides
        # the corresponding action, and clears it in the state.
        state = self.get_state()
        result = self._transition_probabilities_impl(state, actions, sample_one=True)
        
        if result is None:
            # Terminal state — no transition possible, just mark done
            done = True
        else:
            # result is [(1.0, successor_state)]
            # Note: _compute_successor_state* already left the env in the successor state
            done = self.step_count >= self.max_steps
        
        # Check if all agents are terminated
        if all(a.terminated for a in self.agents):
            done = True

        rewards = np.zeros(len(actions))

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(actions))]

        obs=[self.objects.normalize_obs*ob for ob in obs]

        return obs, rewards, done, {}

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:

            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(agent_pos=(a.view_size // 2, a.view_size - 1))
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grids, vis_masks = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        obs = [grid.encode_for_agents(self.objects, [grid.width // 2, grid.height - 1], vis_mask) for grid, vis_mask in zip(grids, vis_masks)]

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            self.objects,
            tile_size,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS, annotation_text=None,
               annotation_panel_width=200, annotation_font_size=11, goal_overlays=None):
        """
        Render the whole-grid human view.
        
        Args:
            mode: 'human' for window display, 'rgb_array' for numpy array
            close: If True, close the rendering window
            highlight: If True, highlight visible cells for each agent
            tile_size: Pixel size of each grid cell
            annotation_text: Optional text to display in a panel to the right of the grid.
                            Can be a string (rendered as-is) or a list of strings (one per line).
            annotation_panel_width: Width of the annotation panel in pixels (default 200).
            annotation_font_size: Font size for annotation text (default 11).
            goal_overlays: Optional dict mapping agent indices to their goals.
                          Each goal will be rendered as a dashed rectangle with a line to the agent.
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            self.window = Window('gym_multigrid')
            self.window.show(block=False)

        if highlight:

            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {(i, j): [] for i in range(self.width) for j in range(self.height)}

            for i, a in enumerate(self.agents):

                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            self.objects,
            tile_size,
            terrain_grid=self.terrain_grid if hasattr(self, 'terrain_grid') else None,
            highlight_masks=highlight_masks if highlight else None,
            stumbled_cells=self.stumbled_cells if hasattr(self, 'stumbled_cells') else None,
            magic_wall_entered_cells=self.magic_wall_entered_cells if hasattr(self, 'magic_wall_entered_cells') else None
        )
        
        # Draw dashed lines from control buttons to controlled agents
        self._draw_control_button_connections(img, tile_size)
        
        # Draw goal overlays if provided
        if goal_overlays is not None:
            for agent_idx, goal in goal_overlays.items():
                self.draw_goal_overlay(img, goal, agent_idx=agent_idx, tile_size=tile_size)
        
        # Add annotation panel if text provided
        if annotation_text is not None:
            img = self._add_annotation_panel(img, annotation_text, 
                                             panel_width=annotation_panel_width,
                                             font_size=annotation_font_size)

        if mode == 'human':
            self.window.show_img(img)
        
        # Auto-capture frame if recording
        if self._recording and mode == 'rgb_array':
            self._video_frames.append(img.copy())

        return img
    
    def _draw_control_button_connections(self, img, tile_size):
        """Draw dashed lines from control buttons to their controlled agents."""
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell = self.grid.get(i, j)
                if (cell is not None and cell.type == 'controlbutton' and 
                    cell.controlled_agent is not None and cell.triggered_action is not None):
                    # Get button center position in pixels
                    btn_x = int((i + 0.5) * tile_size)
                    btn_y = int((j + 0.5) * tile_size)
                    
                    # Get controlled agent position
                    agent = self.agents[cell.controlled_agent]
                    if agent.pos is not None:
                        agent_x = int((agent.pos[0] + 0.5) * tile_size)
                        agent_y = int((agent.pos[1] + 0.5) * tile_size)
                        
                        # Draw dashed line (muted color, thinner)
                        self._draw_dashed_line(img, btn_x, btn_y, agent_x, agent_y, 
                                              color=(80, 160, 80), dash_len=3, gap_len=4, thickness=1)
    
    def _draw_dashed_line(self, img, x1, y1, x2, y2, color=(255, 255, 255), dash_len=5, gap_len=3, thickness=1):
        """Draw a dashed line on the image."""
        import math
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0:
            return
        
        # Normalize direction
        dx /= dist
        dy /= dist
        
        # Draw dashes
        total_len = dash_len + gap_len
        num_segments = int(dist / total_len)
        
        for seg in range(num_segments + 1):
            start = seg * total_len
            end = min(start + dash_len, dist)
            
            sx = int(x1 + dx * start)
            sy = int(y1 + dy * start)
            ex = int(x1 + dx * end)
            ey = int(y1 + dy * end)
            
            # Draw the dash segment with simple line algorithm
            self._draw_line_segment(img, sx, sy, ex, ey, color, thickness)
    
    def _draw_line_segment(self, img, x1, y1, x2, y2, color, thickness=1):
        """Draw a simple line segment on the image using Bresenham-like approach."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy, 1)
        
        x_inc = (x2 - x1) / steps
        y_inc = (y2 - y1) / steps
        
        x, y = float(x1), float(y1)
        h, w = img.shape[:2]
        
        # Thickness offset (for thickness=1, only center pixel)
        half_t = thickness // 2
        
        for _ in range(int(steps) + 1):
            px, py = int(x), int(y)
            if 0 <= px < w and 0 <= py < h:
                # Draw pixels based on thickness
                for ox in range(-half_t, half_t + 1):
                    for oy in range(-half_t, half_t + 1):
                        if 0 <= px+ox < w and 0 <= py+oy < h:
                            img[py+oy, px+ox] = color
            x += x_inc
            y += y_inc

    def draw_goal_overlay(self, img, goal, agent_idx=None, tile_size=TILE_PIXELS,
                          color=(0, 102, 255), line_thickness=2, dash_len=6, gap_len=4,
                          inset_frac=0.08):
        """
        Draw a dashed rectangle around a goal region and optionally a line to the agent.
        
        Args:
            img: RGB numpy array (H, W, 3) to draw on (modified in place).
            goal: Goal object with target_rect, target_pos, or (x1, y1, x2, y2) tuple.
            agent_idx: If provided, draw a dashed line from this agent to the goal.
            tile_size: Size of each grid cell in pixels.
            color: RGB tuple for the goal visualization (default: blue).
            line_thickness: Thickness of the dashed lines.
            dash_len: Length of dashes in pixels.
            gap_len: Length of gaps between dashes in pixels.
            inset_frac: Fraction of cell to inset rectangle from cell edges (0.08 = 8%).
        """
        # Extract bounding box from goal
        x1, y1, x2, y2 = self._get_goal_bounding_box(goal)
        
        # Calculate pixel coordinates with inset
        left = int(x1 * tile_size + tile_size * inset_frac)
        top = int(y1 * tile_size + tile_size * inset_frac)
        right = int((x2 + 1) * tile_size - tile_size * inset_frac)
        bottom = int((y2 + 1) * tile_size - tile_size * inset_frac)
        
        # Draw dashed rectangle (4 sides)
        self._draw_dashed_line(img, left, top, right, top, color, dash_len, gap_len, line_thickness)  # Top
        self._draw_dashed_line(img, right, top, right, bottom, color, dash_len, gap_len, line_thickness)  # Right
        self._draw_dashed_line(img, right, bottom, left, bottom, color, dash_len, gap_len, line_thickness)  # Bottom
        self._draw_dashed_line(img, left, bottom, left, top, color, dash_len, gap_len, line_thickness)  # Left
        
        # Draw line from agent to closest point on goal if agent_idx provided
        if agent_idx is not None and agent_idx < len(self.agents):
            agent = self.agents[agent_idx]
            if agent.pos is not None:
                agent_px = int((agent.pos[0] + 0.5) * tile_size)
                agent_py = int((agent.pos[1] + 0.5) * tile_size)
                
                # Find closest point on rectangle boundary
                closest_x, closest_y = self._closest_point_on_rect(
                    left, top, right, bottom, agent_px, agent_py
                )
                
                self._draw_dashed_line(img, agent_px, agent_py, closest_x, closest_y,
                                       color, dash_len, gap_len, max(1, line_thickness - 1))
    
    def _get_goal_bounding_box(self, goal):
        """Extract bounding box (x1, y1, x2, y2) from various goal representations."""
        if hasattr(goal, 'target_rect'):
            x1, y1, x2, y2 = goal.target_rect
        elif hasattr(goal, 'target_pos'):
            x, y = goal.target_pos
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif hasattr(goal, 'position'):
            x, y = goal.position
            x1, y1, x2, y2 = int(x), int(y), int(x), int(y)
        elif isinstance(goal, (tuple, list)):
            if len(goal) == 4:
                x1, y1, x2, y2 = int(goal[0]), int(goal[1]), int(goal[2]), int(goal[3])
            elif len(goal) >= 2:
                x1, y1 = int(goal[0]), int(goal[1])
                x2, y2 = x1, y1
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0
        
        # Normalize order
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        return (x1, y1, x2, y2)
    
    def _closest_point_on_rect(self, left, top, right, bottom, px, py):
        """Find the closest point on rectangle boundary to a given point."""
        # If point is inside the rectangle, find closest edge
        if left <= px <= right and top <= py <= bottom:
            d_left = px - left
            d_right = right - px
            d_top = py - top
            d_bottom = bottom - py
            
            min_d = min(d_left, d_right, d_top, d_bottom)
            
            if min_d == d_left:
                return (left, py)
            elif min_d == d_right:
                return (right, py)
            elif min_d == d_top:
                return (px, top)
            else:
                return (px, bottom)
        
        # Clamp point to rectangle boundary
        cx = max(left, min(px, right))
        cy = max(top, min(py, bottom))
        
        return (cx, cy)

    def _add_annotation_panel(self, img, annotation_text, panel_width=200, font_size=11, bg_color=(255, 255, 255)):
        """
        Add a text annotation panel to the right side of the image.
        
        Args:
            img: RGB numpy array (H, W, 3)
            annotation_text: String or list of strings to display
            panel_width: Width of the annotation panel in pixels
            font_size: Font size for the text
            bg_color: Background color of the panel (R, G, B)
        
        Returns:
            New image with annotation panel appended on the right
        """
        import numpy as np
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # PIL not available, return original image
            return img
        
        h, w = img.shape[:2]
        
        # Create annotation panel
        panel = np.ones((h, panel_width, 3), dtype=np.uint8)
        panel[:, :] = bg_color
        
        # Concatenate grid and panel
        combined = np.concatenate([img, panel], axis=1)
        
        # Convert to PIL for text drawing
        pil_img = Image.fromarray(combined)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use a monospace font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Prepare text lines
        if isinstance(annotation_text, str):
            lines = annotation_text.split('\n')
        else:
            lines = list(annotation_text)
        
        # Draw text
        text_color = (0, 0, 0)  # Black
        x = w + 5
        y = 5
        line_height = font_size + 3
        
        for line in lines:
            # Handle special formatting: lines starting with '>' are highlighted
            if line.startswith('>'):
                draw.text((x, y), line[1:], fill=(0, 128, 0), font=font)  # Green
            elif line.startswith('!'):
                draw.text((x, y), line[1:], fill=(200, 0, 0), font=font)  # Red
            else:
                draw.text((x, y), line, fill=text_color, font=font)
            y += line_height
            
            # Stop if we run out of space
            if y > h - line_height:
                break
        
        return np.array(pil_img)

    # =========================================================================
    # Video Recording Methods
    # =========================================================================
    
    def start_video_recording(self):
        """
        Start recording frames for video.
        
        After calling this, each call to render() with mode='rgb_array' will
        automatically store the frame. Call save_video() to save the recording.
        
        Example:
            env.start_video_recording()
            for step in range(100):
                env.step(actions)
                env.render(mode='rgb_array')  # Frame automatically captured
            env.save_video('output.mp4', fps=10)
        """
        self._recording = True
        self._video_frames = []
    
    def stop_video_recording(self):
        """
        Stop recording frames without saving.
        
        Use this to cancel a recording. To save frames, use save_video() instead.
        """
        self._recording = False
        self._video_frames = []
    
    def capture_frame(self, tile_size=TILE_PIXELS):
        """
        Capture the current frame and add it to the video recording.
        
        This is called automatically by render() when recording is active,
        but can also be called manually to capture specific frames.
        
        Args:
            tile_size: Size of each grid cell in pixels (default: TILE_PIXELS)
            
        Returns:
            The captured frame as a numpy array, or None if not recording.
        """
        if not self._recording:
            return None
        
        img = self.render(mode='rgb_array', tile_size=tile_size)
        if img is not None:
            self._video_frames.append(img.copy())
        return img
    
    def save_video(self, filename='multigrid_video.mp4', fps=10):
        """
        Save recorded frames as video.
        
        Uses imageio for fast encoding. Supports MP4 (requires imageio-ffmpeg)
        and GIF formats. Falls back to GIF using PIL if imageio is not available.
        
        Args:
            filename: Output filename. Extension determines format:
                     - .mp4: H.264 video (requires imageio-ffmpeg)
                     - .gif: Animated GIF
            fps: Frames per second (default: 10)
            
        Example:
            env.start_video_recording()
            for _ in range(50):
                env.step(actions)
                env.render(mode='rgb_array')
            env.save_video('demo.mp4', fps=5)
        """
        if not self._video_frames:
            print("No frames recorded. Call start_video_recording() and render() first.")
            return
        
        n_frames = len(self._video_frames)
        print(f"Saving {n_frames} frames to {filename}...")
        
        # Try imageio first (fastest)
        try:
            import imageio.v3 as iio
            
            if filename.endswith('.gif'):
                # GIF format
                duration = 1000.0 / fps  # milliseconds per frame
                iio.imwrite(filename, self._video_frames, duration=duration, loop=0)
            else:
                # MP4 or other video format
                iio.imwrite(filename, self._video_frames, fps=fps)
            
            print(f"✓ Video saved to {filename} ({n_frames} frames)")
            self._recording = False
            self._video_frames = []
            return
            
        except ImportError:
            print("imageio not available, trying PIL for GIF...")
        except Exception as e:
            print(f"imageio failed ({e}), trying PIL for GIF...")
        
        # Fall back to PIL for GIF
        try:
            from PIL import Image
            
            gif_filename = filename if filename.endswith('.gif') else filename.rsplit('.', 1)[0] + '.gif'
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in self._video_frames]
            
            # Save as animated GIF
            duration_ms = int(1000 / fps)
            pil_frames[0].save(
                gif_filename,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"✓ Video saved as GIF to {gif_filename} ({n_frames} frames)")
            
        except Exception as e:
            print(f"Error saving video: {e}")
        
        # Reset state
        self._recording = False
        self._video_frames = []

    def get_state(self):
        """
        Get the current state of the environment as a compact representation.
        
        The compact state only stores mutable/mobile objects:
        1. Immutable objects (walls) are not stored
        2. Mobile objects (blocks/rocks) only store type and position (color is immutable)
        3. Mutable immobile objects only store their mutable state (e.g., active for magic walls)
        4. Uses fixed ordering instead of serializing entire grid
        
        Format:
        - step_count: int
        - agent_states: tuple of (pos_x, pos_y, dir, terminated, started, paused, carrying_type, carrying_color, forced_next_action)
        - mobile_objects: tuple of (obj_type, pos_x, pos_y) for blocks/rocks
        - mutable_objects: tuple of (obj_type, x, y, mutable_state...) for doors/boxes/magic walls
        
        Note: on_unsteady_ground is NOT stored in the state - it is derived from the
        agent's position and the terrain_grid when set_state() is called.
        
        Returns:
            tuple: A hashable compact state representation
        """
        # Rebuild cache if not present (e.g., objects were added after reset)
        if not hasattr(self, '_mobile_objects') or not hasattr(self, '_mutable_objects'):
            self._build_object_cache()
        
        # Agent states - fixed order by agent index
        # Note: on_unsteady_ground is NOT stored - it's derived from position + terrain_grid
        agent_states = []
        for agent in self.agents:
            carrying_type = agent.carrying.type if agent.carrying else None
            carrying_color = agent.carrying.color if agent.carrying else None
            agent_states.append((
                int(agent.pos[0]) if agent.pos is not None else None,
                int(agent.pos[1]) if agent.pos is not None else None,
                int(agent.dir) if agent.dir is not None else None,
                agent.terminated,
                agent.started,
                agent.paused,
                carrying_type,
                carrying_color,
                agent.forced_next_action,  # Include forced action in state
            ))
        
        # For environments with no mobile/mutable objects in cache, do a quick grid scan
        # This handles cases where objects are added after reset()
        if len(self._mobile_objects) == 0 and len(self._mutable_objects) == 0:
            # Quick check if there are any objects we should track
            has_trackable = False
            for j in range(self.grid.height):
                for i in range(self.grid.width):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type in ('block', 'rock', 'door', 'box', 'magicwall', 'killbutton', 'pauseswitch', 'controlbutton'):
                        has_trackable = True
                        break
                if has_trackable:
                    break
            if has_trackable:
                self._build_object_cache()
        
        # Use cached objects - no grid scanning needed
        # Mobile objects use cur_pos which is updated when they're pushed
        # Note: We only store (type, x, y) since color is immutable for blocks/rocks
        mobile_objects = []
        for initial_pos, obj in self._mobile_objects:
            if obj.cur_pos is not None:
                x, y = int(obj.cur_pos[0]), int(obj.cur_pos[1])
            else:
                # Fallback to initial position if cur_pos not set
                x, y = initial_pos
            mobile_objects.append((
                obj.type,
                x, y,
            ))
        
        # Mutable objects - their positions are fixed, just read their mutable state
        mutable_objects = []
        for (x, y), obj in self._mutable_objects:
            if obj.type == 'door':
                mutable_objects.append((
                    'door',
                    x, y,
                    obj.is_open,
                    obj.is_locked,
                ))
            elif obj.type == 'box':
                contains_type = obj.contains.type if obj.contains else None
                contains_color = obj.contains.color if obj.contains else None
                mutable_objects.append((
                    'box',
                    x, y,
                    contains_type,
                    contains_color,
                ))
            elif obj.type == 'magicwall':
                mutable_objects.append((
                    'magicwall',
                    x, y,
                    obj.active,
                ))
            elif obj.type == 'killbutton':
                mutable_objects.append((
                    'killbutton',
                    x, y,
                    obj.enabled,
                ))
            elif obj.type == 'pauseswitch':
                mutable_objects.append((
                    'pauseswitch',
                    x, y,
                    obj.is_on,
                    obj.enabled,
                ))
            elif obj.type == 'controlbutton':
                mutable_objects.append((
                    'controlbutton',
                    x, y,
                    obj.enabled,
                    obj.controlled_agent,
                    obj.triggered_action,
                    obj._awaiting_action,
                ))
        
        # Sort mobile objects for deterministic ordering (type, x, y)
        mobile_objects.sort(key=lambda obj: (obj[0], obj[1], obj[2]))
        
        return (
            self.step_count,
            tuple(agent_states),
            tuple(mobile_objects),
            tuple(mutable_objects),
        )
    
    @property
    def human_agent_indices(self):
        """
        Get the indices of human agents in the environment.
        
        If _human_agent_indices is set (e.g., from YAML config), use that.
        Otherwise, identify human agents by the color 'yellow'.
        
        Returns:
            List of indices for human agents.
        """
        if hasattr(self, '_human_agent_indices') and self._human_agent_indices is not None:
            return self._human_agent_indices
        return [i for i, agent in enumerate(self.agents) if agent.color == 'yellow']
    
    @human_agent_indices.setter
    def human_agent_indices(self, value):
        """Set the human agent indices explicitly."""
        self._human_agent_indices = value
    
    @property
    def robot_agent_indices(self):
        """
        Get the indices of robot agents in the environment.
        
        If _robot_agent_indices is set (e.g., from YAML config), use that.
        Otherwise, identify robot agents by the color 'grey'.
        
        Returns:
            List of indices for robot agents.
        """
        if hasattr(self, '_robot_agent_indices') and self._robot_agent_indices is not None:
            return self._robot_agent_indices
        return [i for i, agent in enumerate(self.agents) if agent.color == 'grey']
    
    @robot_agent_indices.setter
    def robot_agent_indices(self, value):
        """Set the robot agent indices explicitly."""
        self._robot_agent_indices = value
    
    def set_state(self, state):
        """
        Set the environment to a compact state.
        
        Note: This requires that the immutable grid structure is already correct.
        Only use this for states that came from the same environment instance
        or an identical environment setup.
        
        Args:
            state: A compact state tuple as returned by get_state()
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        self.step_count = step_count
        
        # Restore agent states in TWO PASSES to handle agents swapping positions correctly.
        # If we remove and place in the same pass, an agent moving to another agent's old 
        # position may see the other agent instead of the terrain.
        
        # PASS 1: Remove ALL agents from their current positions, restoring terrain
        for agent_idx, agent in enumerate(self.agents):
            if agent.pos is not None:
                old_cell = self.grid.get(*agent.pos)
                if old_cell is agent:
                    # Restore terrain at old position
                    old_terrain = self.terrain_grid.get(*agent.pos)
                    self.grid.set(*agent.pos, old_terrain)  # None is fine
                    self.terrain_grid.set(*agent.pos, None)
        
        # PASS 2: Update agent states and place them at new positions
        for agent_idx, agent_state in enumerate(agent_states):
            agent = self.agents[agent_idx]
            # Handle both old format (8 elements) and new format (9 elements with forced_next_action)
            if len(agent_state) == 9:
                pos_x, pos_y, dir_, terminated, started, paused, carrying_type, carrying_color, forced_next_action = agent_state
            else:
                pos_x, pos_y, dir_, terminated, started, paused, carrying_type, carrying_color = agent_state
                forced_next_action = None
            
            # Update agent state
            agent.pos = np.array([pos_x, pos_y]) if pos_x is not None else None
            agent.dir = dir_
            agent.terminated = terminated
            agent.started = started
            agent.paused = paused
            agent.forced_next_action = forced_next_action
            
            # Restore carrying state
            if carrying_type is not None:
                if carrying_type == 'ball':
                    agent.carrying = Ball(self.objects, color=carrying_color)
                elif carrying_type == 'key':
                    agent.carrying = Key(self.objects, color=carrying_color)
                elif carrying_type == 'box':
                    agent.carrying = Box(self.objects, color=carrying_color)
                else:
                    agent.carrying = None
            else:
                agent.carrying = None
            
            # Place agent in grid at new position, saving terrain if present
            if agent.pos is not None:
                current_cell = self.grid.get(*agent.pos)
                # Save terrain (overlappable objects like unsteady ground, magic walls)
                if current_cell is not None and current_cell.can_overlap():
                    self.terrain_grid.set(*agent.pos, current_cell)
                    # Derive on_unsteady_ground from the terrain
                    agent.on_unsteady_ground = (current_cell.type == 'unsteadyground')
                else:
                    self.terrain_grid.set(*agent.pos, None)
                    agent.on_unsteady_ground = False
                self.grid.set(*agent.pos, agent)
        
        # Restore mobile objects (blocks, rocks)
        # Use cached objects - clear from current positions and move to new positions
        # This avoids scanning the entire grid
        if hasattr(self, '_mobile_objects') and self._mobile_objects:
            # Clear cached mobile objects from their current positions
            for initial_pos, obj in self._mobile_objects:
                if obj.cur_pos is not None:
                    old_x, old_y = int(obj.cur_pos[0]), int(obj.cur_pos[1])
                    cell = self.grid.get(old_x, old_y)
                    if cell is obj:
                        self.grid.set(old_x, old_y, None)
            
            # Group cached objects by type for matching with state
            # We match by type since get_state() groups mobile objects by (type, x, y).
            # Within each type, objects are sorted by position, so we assign in the same order.
            # Since all blocks are identical and all rocks are identical (color is immutable),
            # it doesn't matter which physical object gets which position.
            cached_by_type = {}
            for initial_pos, obj in self._mobile_objects:
                if obj.type not in cached_by_type:
                    cached_by_type[obj.type] = []
                cached_by_type[obj.type].append(obj)
            
            # Group mobile objects from state by type
            state_by_type = {}
            for mobile_obj in mobile_objects:
                obj_type, x, y = mobile_obj
                if obj_type not in state_by_type:
                    state_by_type[obj_type] = []
                state_by_type[obj_type].append((x, y))
            
            # Place cached objects at new positions from state, matching by type
            for obj_type, positions in state_by_type.items():
                if obj_type in cached_by_type:
                    cached_objs = cached_by_type[obj_type]
                    for idx, (x, y) in enumerate(positions):
                        if idx < len(cached_objs):
                            obj = cached_objs[idx]
                            obj.cur_pos = np.array([x, y])
                            self.grid.set(x, y, obj)
        else:
            # Fallback: scan grid if no cache (shouldn't happen after reset)
            for j in range(self.grid.height):
                for i in range(self.grid.width):
                    cell = self.grid.get(i, j)
                    if cell is not None and cell.type in ('block', 'rock'):
                        self.grid.set(i, j, None)
            
            for mobile_obj in mobile_objects:
                obj_type, x, y = mobile_obj
                if obj_type == 'block':
                    obj = Block(self.objects)
                elif obj_type == 'rock':
                    obj = Rock(self.objects)
                else:
                    continue
                obj.cur_pos = np.array([x, y])
                self.grid.set(x, y, obj)
        
        # Restore mutable objects (doors, boxes, magic walls, killbuttons, pauseswitches)
        for mutable_obj in mutable_objects:
            obj_type = mutable_obj[0]
            x, y = mutable_obj[1], mutable_obj[2]
            cell = self.grid.get(x, y)
            
            if obj_type == 'door':
                if cell is not None and cell.type == 'door':
                    cell.is_open = mutable_obj[3]
                    cell.is_locked = mutable_obj[4]
            
            elif obj_type == 'box':
                if cell is not None and cell.type == 'box':
                    contains_type = mutable_obj[3]
                    contains_color = mutable_obj[4]
                    if contains_type is not None:
                        if contains_type == 'ball':
                            cell.contains = Ball(self.objects, color=contains_color)
                        elif contains_type == 'key':
                            cell.contains = Key(self.objects, color=contains_color)
                        else:
                            cell.contains = None
                    else:
                        cell.contains = None
            
            elif obj_type == 'magicwall':
                # Just update the 'active' attribute on the existing magic wall object
                active = mutable_obj[3]
                if cell is not None and cell.type == 'magicwall':
                    cell.active = active
            
            elif obj_type == 'killbutton':
                enabled = mutable_obj[3]
                if cell is not None and cell.type == 'killbutton':
                    cell.enabled = enabled
            
            elif obj_type == 'pauseswitch':
                is_on = mutable_obj[3]
                enabled = mutable_obj[4]
                if cell is not None and cell.type == 'pauseswitch':
                    cell.is_on = is_on
                    cell.enabled = enabled
            
            elif obj_type == 'controlbutton':
                enabled = mutable_obj[3]
                controlled_agent = mutable_obj[4]
                triggered_action = mutable_obj[5]
                awaiting_action = mutable_obj[6] if len(mutable_obj) > 6 else False
                if cell is not None and cell.type == 'controlbutton':
                    cell.enabled = enabled
                    cell.controlled_agent = controlled_agent
                    cell.triggered_action = triggered_action
                    cell._awaiting_action = awaiting_action
                    cell._just_activated = False  # Always False between steps
    
    def transition_probabilities(self, state, actions, sample_one=False):
        """
        Given a state and vector of actions, return possible transitions with exact probabilities.
        
        When sample_one=True, instead of enumerating all possible outcomes, randomly
        samples a single outcome (permutation winner, unsteady outcome, magic wall outcome)
        and returns [(1.0, successor_state)]. This is used by step() to guarantee
        consistency between step() and the full transition_probabilities().
        
        **When transitions are probabilistic vs deterministic:**
        
        Transitions in multigrid environments are **deterministic** EXCEPT in the following case:
        - When multiple agents are active (not terminated/paused/started=False) AND
        - At least 2+ agents choose non-"still" actions AND  
        - The order of action execution matters for the outcome
        
        The ONLY source of non-determinism is the random permutation of agent execution order
        at line 1257 of the step() function: `order = np.random.permutation(len(actions))`
        
        All individual actions (left, right, forward, pickup, drop, toggle, etc.) are 
        themselves deterministic - there is no randomness in their effects.
        
        **When order matters:**
        Order of execution can matter when:
        - Two agents try to move into the same cell
        - Two agents try to pick up the same object
        - Agents interact with each other (e.g., stealing carried objects)
        - An agent's action depends on the grid state that another agent modifies
        
        **Computing exact probabilities (optimized):**
        Since np.random.permutation creates a uniform distribution over all n! permutations,
        each permutation has probability 1/n! where n is the number of agents.
        
        **Optimization strategy:**
        Most of the time, most permutations lead to the same result because:
        1. Agents are often far apart and don't interact
        2. Only rotation/still actions are used (always commutative)
        3. Order only matters for conflicting actions
        
        We optimize by using **conflict block partitioning** (suggested by @mensch72):
        1. Early exit if ≤1 active agents (deterministic)
        2. Early exit if all actions are rotations (commutative)
        3. Partition agents into conflict blocks (agents competing for same resource)
        4. Compute Cartesian product of blocks instead of all k! permutations
        5. Each outcome has equal probability: 1 / product(block_sizes)
        
        This is MORE efficient than permutation enumeration:
        - If 2 blocks of 2 agents each: 2×2=4 outcomes instead of 4!=24 permutations
        - Most blocks are singletons (no conflicts), making this very fast
        - Only conflicting agents need to be considered in different orderings
        
        Args:
            state: A state tuple as returned by get_state()
            actions: List of action indices, one per agent
            sample_one: If True, randomly sample a single transition instead of
                       enumerating all possible outcomes. Returns [(1.0, successor_state)].
            
        Returns:
            list: List of (probability, successor_state) tuples describing all
                  possible transitions (or a single sampled transition if sample_one=True).
                  Returns None if the state is terminal
                  or if any action is not feasible in the given state.
        """
        # Check if we're in a terminal state (state[0] is step_count in compact format)
        step_count = state[0]
        if step_count >= self.max_steps:
            return None
        
        # Check if all actions are valid
        for action in actions:
            if action < 0 or action >= self.action_space.n:
                raise ValueError(f"Invalid action {action} in transition_probabilities")
        
        # Save original state to restore at the end
        original_state = self.get_state()
        
        # Set to query state
        self.set_state(state)
        
        try:
            return self._transition_probabilities_impl(state, actions, sample_one=sample_one)
        finally:
            # Always restore the original state
            self.set_state(original_state)
    
    def _transition_probabilities_impl(self, state, actions, sample_one=False):
        """
        Internal implementation of transition_probabilities.
        Called after state is set, original state will be restored by caller.
        
        When sample_one=True, randomly samples one outcome instead of enumerating all.
        """
        num_agents = len(self.agents)
        actions = list(actions)  # Copy to avoid modifying caller's list
        
        # Handle forced actions (from ControlButton triggers in a previous step):
        # If any agent has forced_next_action set in the state, override their action
        # and clear forced_next_action from the state so it doesn't persist into successor states.
        agent_states = state[1]
        modified_agent_states = None
        for i, agent_state in enumerate(agent_states):
            if len(agent_state) >= 9 and agent_state[8] is not None:
                actions[i] = agent_state[8]
                if modified_agent_states is None:
                    modified_agent_states = list(agent_states)
                modified_agent_states[i] = agent_state[:8] + (None,)
        if modified_agent_states is not None:
            state = (state[0], tuple(modified_agent_states), state[2], state[3])
            self.set_state(state)  # Re-set state with cleared forced actions
        
        # Identify which agents will actually act
        active_agents = []
        inactive_agents = []
        for i in range(num_agents):
            if (not self.agents[i].terminated and 
                not self.agents[i].paused and 
                self.agents[i].started and 
                not self._is_still_action(actions[i])):
                active_agents.append(i)
            else:
                inactive_agents.append(i)
        
        # OPTIMIZATION 1: If ≤1 agents active, check if transition is deterministic
        # (only deterministic if the agent is NOT on unsteady ground attempting forward
        # and NOT attempting to enter a magic wall)
        if len(active_agents) <= 1:
            # Check if the single agent is on unsteady ground or attempting magic wall entry
            is_stochastic = False
            if len(active_agents) == 1:
                agent_idx = active_agents[0]
                if actions[agent_idx] == self.actions.forward:
                    if self.agents[agent_idx].on_unsteady_ground:
                        is_stochastic = True
                    elif self.agents[agent_idx].can_enter_magic_walls:
                        fwd_pos = self.agents[agent_idx].front_pos
                        fwd_cell = self.grid.get(*fwd_pos)
                        # Check if it's an active magic wall (not solidified)
                        if fwd_cell is not None and fwd_cell.type == 'magicwall' and fwd_cell.active:
                            approach_dir = (self.agents[agent_idx].dir + 2) % 4
                            if approach_dir == fwd_cell.magic_side:
                                is_stochastic = True
            
            if not is_stochastic:
                # Only one or zero agents acting and not stochastic - order doesn't matter
                successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
                return [(1.0, successor_state)]
        
        # OPTIMIZATION 2: Check if all but at most one actions are rotations (left/right)
        # Rotations never interfere with each other, so order doesn't matter
        n_non_rotations = sum(
            actions[i] not in [self.actions.left, self.actions.right] 
            for i in active_agents
        )
        if n_non_rotations < 2:
            # Check if any agents are on unsteady ground or attempting magic wall entry
            has_stochastic = any(
                actions[i] == self.actions.forward and (
                    self.agents[i].on_unsteady_ground or
                    (self.agents[i].can_enter_magic_walls and
                     self.grid.get(*self.agents[i].front_pos) is not None and
                     self.grid.get(*self.agents[i].front_pos).type == 'magicwall' and
                     (self.agents[i].dir + 2) % 4 == self.grid.get(*self.agents[i].front_pos).magic_side)
                )
                for i in active_agents
            )
            if not has_stochastic:
                # Rotations are commutative and no stochastic agents - result is deterministic
                successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
                return [(1.0, successor_state)]
        
        # Use helper to categorize agents
        normal_active_agents, unsteady_forward_agents, magic_wall_agents = self._categorize_agents(actions, active_agents)
        
        # OPTIMIZATION 3: Partition ONLY normal (non-stochastic) agents into conflict blocks
        # Unsteady and magic wall agents are excluded because they're handled separately
        # This is MORE efficient than permuting all active agents
        # Instead of k! permutations, we compute the Cartesian product of conflict blocks
        conflict_blocks = self._identify_conflict_blocks(actions, normal_active_agents)
        
        # If no conflicts and no stochastic agents, result is deterministic
        if (all(len(block) == 1 for block in conflict_blocks) and 
            len(unsteady_forward_agents) == 0 and 
            len(magic_wall_agents) == 0):
            successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
            return [(1.0, successor_state)]
        
        # OPTIMIZATION 4: Compute outcomes via Cartesian product of conflict blocks, unsteady blocks, and magic wall blocks
        # Each outcome has probability = 1 / product(block_sizes)
        
        # Build list of all blocks (conflict blocks + unsteady agent blocks + magic wall blocks)
        all_blocks = []
        
        # Add conflict blocks
        for block in conflict_blocks:
            all_blocks.append(('conflict', block))
        
        # Add unsteady agent blocks (one per unsteady-forward agent)
        # Each block has 3 outcomes with probabilities
        for agent_idx in unsteady_forward_agents:
            # Get the stumble probability from the unsteady ground cell
            # Check terrain_grid first (where terrain is saved when agent stands on it)
            # Fall back to main grid if terrain_grid doesn't have it
            terrain_cell = self.terrain_grid.get(*self.agents[agent_idx].pos)
            if terrain_cell and terrain_cell.type == 'unsteadyground':
                stumble_prob = terrain_cell.stumble_probability
            else:
                # Fallback: check main grid (shouldn't normally happen if terrain_grid is maintained)
                current_cell = self.grid.get(*self.agents[agent_idx].pos)
                if current_cell and current_cell.type == 'unsteadyground':
                    stumble_prob = current_cell.stumble_probability
                else:
                    stumble_prob = 0.5  # Default fallback
            # Block elements are (probability, outcome) pairs
            # If stumbles, 50% chance of left-forward, 50% chance of right-forward
            # Optimization: if stumble_prob is 0, only forward is possible
            if stumble_prob == 0:
                outcomes = [(1.0, 'forward')]
            else:
                outcomes = [
                    (1.0 - stumble_prob, 'forward'),
                    (stumble_prob * 0.5, 'left-forward'),
                    (stumble_prob * 0.5, 'right-forward')
                ]
            all_blocks.append(('unsteady', agent_idx, outcomes))
        
        # Add magic wall agent blocks (one per magic-wall-entry agent)
        # Each block has 3 outcomes with probabilities: succeed, fail (stays magic), solidify (turns to wall)
        for agent_idx in magic_wall_agents:
            fwd_pos = self.agents[agent_idx].front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            entry_prob = fwd_cell.entry_probability if fwd_cell else 0.5
            solidify_prob = fwd_cell.solidify_probability if fwd_cell else 0.0
            # Block elements are (probability, outcome) pairs
            # On failure, there's a chance to solidify into a normal wall
            fail_prob = 1.0 - entry_prob
            outcomes = [
                (entry_prob, 'succeed'),
                (fail_prob * (1.0 - solidify_prob), 'fail'),
                (fail_prob * solidify_prob, 'solidify')
            ]
            all_blocks.append(('magicwall', agent_idx, outcomes))
        
        # Generate all possible outcome combinations
        # For conflict blocks: winner index (which agent wins) - uniform probability
        # For unsteady blocks: outcome index into (probability, outcome) pairs
        # For magic wall blocks: outcome index into (probability, outcome) pairs
        
        def get_block_size(block):
            if block[0] == 'conflict':
                return len(block[1])
            elif block[0] in ['unsteady', 'magicwall']:
                return len(block[2])  # Number of (probability, outcome) pairs
            else:
                return 1
        
        [get_block_size(block) for block in all_blocks]
        
        # --- sample_one mode: randomly sample one outcome per block ---
        if sample_one:
            outcome_indices = []
            for block in all_blocks:
                size = get_block_size(block)
                if block[0] == 'conflict':
                    # Uniform random winner
                    outcome_indices.append(np.random.randint(size))
                elif block[0] in ['unsteady', 'magicwall']:
                    # Sample according to outcome probabilities
                    probs = [block[2][j][0] for j in range(size)]
                    outcome_indices.append(np.random.choice(size, p=probs))
                else:
                    outcome_indices.append(0)
            outcome_indices = tuple(outcome_indices)
            
            # Process conflict blocks to determine winners
            conflict_winners = []
            conflict_block_idx = 0
            for i, block in enumerate(all_blocks):
                if block[0] == 'conflict':
                    winner_idx = outcome_indices[i]
                    conflict_winners.append((conflict_block_idx, block[1][winner_idx]))
                    conflict_block_idx += 1
            
            # Process unsteady blocks to determine modified actions
            modified_actions = list(actions)
            for i, block in enumerate(all_blocks):
                if block[0] == 'unsteady':
                    agent_idx = block[1]
                    _, outcome_type = block[2][outcome_indices[i]]
                    modified_actions[agent_idx] = (actions[agent_idx], outcome_type)
            
            # Process magic wall blocks to determine outcomes
            magic_wall_outcomes = {}
            for i, block in enumerate(all_blocks):
                if block[0] == 'magicwall':
                    agent_idx = block[1]
                    _, outcome_type = block[2][outcome_indices[i]]
                    magic_wall_outcomes[agent_idx] = outcome_type
            
            # Compute the single successor state
            succ_state = self._compute_successor_state_with_unsteady(
                state, modified_actions, num_agents, active_agents,
                conflict_blocks, conflict_winners, magic_wall_outcomes
            )
            return [(1.0, succ_state)]
        
        # --- Full enumeration mode: compute all outcomes ---
        
        # Compute successor state for each outcome
        successor_states = {}
        
        # Generate cartesian product of all block outcomes
        block_ranges = [range(get_block_size(block)) for block in all_blocks]
        
        for outcome_indices in product(*block_ranges):
            # outcome_indices[i] tells us which outcome for block i
            
            # Compute probability for this outcome combination
            outcome_probability = 1.0
            for i, block in enumerate(all_blocks):
                outcome_idx = outcome_indices[i]
                if block[0] == 'conflict':
                    # Uniform probability over conflict block members
                    outcome_probability *= 1.0 / len(block[1])
                elif block[0] in ['unsteady', 'magicwall']:
                    # Use the probability from the (probability, outcome) pair
                    prob, _ = block[2][outcome_idx]
                    outcome_probability *= prob
            
            # Process conflict blocks to determine winners
            conflict_winners = []
            conflict_block_idx = 0
            for i, block in enumerate(all_blocks):
                if block[0] == 'conflict':
                    winner_idx = outcome_indices[i]
                    conflict_winners.append((conflict_block_idx, block[1][winner_idx]))
                    conflict_block_idx += 1
            
            # Process unsteady blocks to determine modified actions
            modified_actions = list(actions)
            for i, block in enumerate(all_blocks):
                if block[0] == 'unsteady':
                    agent_idx = block[1]
                    outcome_idx = outcome_indices[i]
                    _, outcome_type = block[2][outcome_idx]  # Extract outcome from (probability, outcome) pair
                    
                    # Modify actions based on outcome
                    # We'll encode the outcome in the action temporarily
                    modified_actions[agent_idx] = (actions[agent_idx], outcome_type)
            
            # Process magic wall blocks to determine outcomes
            magic_wall_outcomes = {}
            for i, block in enumerate(all_blocks):
                if block[0] == 'magicwall':
                    agent_idx = block[1]
                    outcome_idx = outcome_indices[i]
                    _, outcome_type = block[2][outcome_idx]  # Extract outcome from (probability, outcome) pair
                    magic_wall_outcomes[agent_idx] = outcome_type
            
            # Compute the successor state for this outcome
            succ_state = self._compute_successor_state_with_unsteady(
                state, modified_actions, num_agents, active_agents, 
                conflict_blocks, conflict_winners, magic_wall_outcomes
            )
            
            # Aggregate probabilities for identical successor states
            if succ_state not in successor_states:
                successor_states[succ_state] = 0.0
            successor_states[succ_state] += outcome_probability
        
        # Convert to result list
        result = [(prob, state) for state, prob in successor_states.items()]
        
        # Sort by probability (descending) for consistency
        result.sort(key=lambda x: x[0], reverse=True)
        
        return result
    
    def _identify_conflict_blocks(self, actions, active_agents):
        """
        Partition active agents into conflict blocks where agents compete for resources.
        
        Agents are in the same block if they:
        - Try to move into the same cell (forward action to same position)
        - Try to pick up the same object
        - Interact with each other directly
        
        Note: This method assumes the environment is already set to the relevant state
        (i.e., the caller has already called set_state() before calling this method).
        
        Args:
            actions: List of action indices
            active_agents: List of active agent indices
            
        Returns:
            list of lists: Each inner list is a conflict block of agent indices
        """
        # Constants for resource types
        RESOURCE_INDEPENDENT = 'independent'
        RESOURCE_CELL = 'cell'
        RESOURCE_PICKUP = 'pickup'
        RESOURCE_DROP_AGENT = 'drop_agent'
        
        # Grid is already set up from set_state() call in transition_probabilities
        
        # Track which resource each agent targets
        agent_targets = {}  # agent_idx -> resource identifier
        
        for agent_idx in active_agents:
            action = actions[agent_idx]
            agent = self.agents[agent_idx]
            
            # Determine what resource this agent is targeting
            if action == self.actions.forward:
                # Check what the agent is trying to move into
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                
                # If pushing blocks/rocks, the resource is the final cell where objects land
                if fwd_cell and fwd_cell.type in ['block', 'rock']:
                    can_push, num_objects, end_pos = self._can_push_objects(agent, fwd_pos)
                    if can_push:
                        # Agent will push objects and move into fwd_pos
                        # The contested resource is the end_pos where objects land
                        agent_targets[agent_idx] = (RESOURCE_CELL, tuple(end_pos))
                    else:
                        # Can't push, agent won't move - independent action
                        agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                else:
                    # Normal movement - target is the cell they're moving into
                    target_pos = tuple(agent.front_pos)
                    agent_targets[agent_idx] = (RESOURCE_CELL, target_pos)
            
            elif hasattr(self.actions, 'pickup') and action == self.actions.pickup:
                # Target is the object at the forward position
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.can_pickup():
                    # Use object position as identifier
                    agent_targets[agent_idx] = (RESOURCE_PICKUP, tuple(fwd_pos))
                else:
                    # No valid target, agent acts independently
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            elif hasattr(self.actions, 'drop') and action == self.actions.drop:
                # Check if dropping on another agent or specific location
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.type == 'agent':
                    # Interacting with another agent
                    agent_targets[agent_idx] = (RESOURCE_DROP_AGENT, tuple(fwd_pos))
                else:
                    # Dropping on ground, independent action
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            else:
                # Other actions (toggle, build, done) - typically independent
                agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
        
        # Group agents by their target resource
        resource_to_agents = {}
        for agent_idx, resource in agent_targets.items():
            if resource not in resource_to_agents:
                resource_to_agents[resource] = []
            resource_to_agents[resource].append(agent_idx)
        
        # Convert to list of blocks
        # Only resources with multiple agents form conflict blocks
        conflict_blocks = []
        for resource, agent_list in resource_to_agents.items():
            if len(agent_list) > 1 and resource[0] != RESOURCE_INDEPENDENT:
                # Multiple agents competing for same resource
                conflict_blocks.append(agent_list)
            else:
                # Each agent acts independently (singleton blocks)
                for agent_idx in agent_list:
                    conflict_blocks.append([agent_idx])
        
        # Post-processing: detect KillButton/PauseSwitch toggles that create
        # ordering dependencies with target-color agents.
        # When an agent toggles a KillButton or PauseSwitch, the outcome depends
        # on whether target agents act before or after the toggle, so they must
        # be in the same conflict block.
        toggle_affected = []  # list of (toggling_agent_idx, set_of_affected_agent_indices)
        if hasattr(self.actions, 'toggle'):
            for agent_idx in active_agents:
                if actions[agent_idx] == self.actions.toggle:
                    agent = self.agents[agent_idx]
                    fwd_pos = agent.front_pos
                    fwd_cell = self.grid.get(*fwd_pos)
                    if fwd_cell is not None:
                        affected = set()
                        if (fwd_cell.type == 'killbutton' and fwd_cell.enabled
                                and agent.color == fwd_cell.trigger_color):
                            for other_idx in active_agents:
                                if other_idx != agent_idx and self.agents[other_idx].color == fwd_cell.target_color:
                                    affected.add(other_idx)
                        elif (fwd_cell.type == 'pauseswitch' and fwd_cell.enabled
                                and agent.color == fwd_cell.toggle_color):
                            for other_idx in active_agents:
                                if other_idx != agent_idx and self.agents[other_idx].color == fwd_cell.target_color:
                                    affected.add(other_idx)
                        if affected:
                            toggle_affected.append((agent_idx, affected))
        
        # Merge conflict blocks connected by toggle dependencies
        if toggle_affected:
            # Build agent -> block index mapping
            agent_to_block = {}
            for block_idx, block in enumerate(conflict_blocks):
                for a in block:
                    agent_to_block[a] = block_idx
            
            # For each toggle dependency, merge blocks using union-find
            block_parent = list(range(len(conflict_blocks)))
            
            def find(x):
                while block_parent[x] != x:
                    block_parent[x] = block_parent[block_parent[x]]
                    x = block_parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    block_parent[rx] = ry
            
            for toggler_idx, affected_set in toggle_affected:
                toggler_block = agent_to_block[toggler_idx]
                for affected_idx in affected_set:
                    affected_block = agent_to_block[affected_idx]
                    union(toggler_block, affected_block)
            
            # Rebuild conflict blocks from union-find
            merged = {}
            for block_idx, block in enumerate(conflict_blocks):
                root = find(block_idx)
                if root not in merged:
                    merged[root] = []
                merged[root].extend(block)
            conflict_blocks = list(merged.values())
        
        return conflict_blocks
    
    def _build_ordering_with_winners(self, num_agents, active_agents, conflict_blocks, winners):
        """
        Build an agent ordering that respects conflict block winners.
        
        Winners go first in their blocks, giving them priority for contested resources.
        
        Args:
            num_agents: Total number of agents
            active_agents: List of active agent indices
            conflict_blocks: List of conflict blocks
            winners: List of winning agent indices (one per block)
            
        Returns:
            tuple: Full ordering of all agent indices
        """
        # Build ordering: winners first, then other agents in original order
        ordering = []
        
        # Track which agents are winners
        winner_set = set(winners)
        
        # Add winners first (in the order they win)
        for winner in winners:
            ordering.append(winner)
        
        # Add all inactive agents in their original order
        for i in range(num_agents):
            if i not in active_agents:
                ordering.append(i)
        
        # Add non-winning active agents
        for agent_idx in active_agents:
            if agent_idx not in winner_set:
                ordering.append(agent_idx)
        
        return tuple(ordering)
    
    def _build_full_ordering(self, num_agents, active_perm, inactive_agents):
        """
        Build a full agent ordering from a permutation of active agents.
        
        Inactive agents are placed in their original positions, active agents
        are placed in the order specified by active_perm.
        
        Args:
            num_agents: Total number of agents
            active_perm: Tuple of active agent indices in some order
            inactive_agents: List of inactive agent indices
            
        Returns:
            tuple: Full ordering of all agent indices
        """
        # Build ordering that respects relative order of active agents
        # but keeps inactive agents in their original positions
        ordering = []
        active_iter = iter(active_perm)
        
        for i in range(num_agents):
            if i in inactive_agents:
                ordering.append(i)
            else:
                ordering.append(next(active_iter))
        
        return tuple(ordering)
    
    def _compute_successor_state(self, state, actions, ordering):
        """
        Compute the exact successor state for given actions and agent ordering.
        
        This is a pure computation that doesn't rely on RNG or executing step().
        It replicates the deterministic logic of step() with a fixed ordering.
        
        Args:
            state: Current state tuple
            actions: List of action indices
            ordering: Tuple specifying the order in which agents act
            
        Returns:
            tuple: The successor state
        """
        # Start from the given state
        self.set_state(state)
        
        # Increment step counter
        self.step_count += 1
        
        # Record initial agent positions to prevent chain conflicts.
        # This must be consistent with step() behavior.
        self._initial_agent_positions = set()
        for agent in self.agents:
            if agent.pos is not None and not agent.terminated:
                self._initial_agent_positions.add(tuple(agent.pos))
        
        # Execute each agent's action in the specified order
        rewards = np.zeros(len(actions))
        done = False
        
        for i in ordering:
            # Skip if agent shouldn't act
            if (self.agents[i].terminated or 
                self.agents[i].paused or 
                not self.agents[i].started or 
                self._is_still_action(actions[i])):
                continue
            
            # Execute the action using shared helper
            agent_done = self._execute_single_agent_action(i, actions[i], rewards)
            done = done or agent_done
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        # Return the resulting state
        return self.get_state()
    
    def _compute_successor_state_with_unsteady(self, state, modified_actions, num_agents, 
                                               active_agents, conflict_blocks, conflict_winners, 
                                               magic_wall_outcomes=None):
        """
        Compute successor state with unsteady ground and magic wall stochasticity.
        
        This handles the special processing order required for stochastic elements:
        1. Process normal agents first (with conflict resolution)
        2. Process unsteady-forward agents after (with stumbling outcomes)
        3. Process magic wall entry attempts last (with probabilistic outcomes)
        
        Args:
            state: Current state tuple
            modified_actions: List of actions, where unsteady agent actions are tuples (action, outcome_type)
            num_agents: Total number of agents
            active_agents: List of active agent indices
            conflict_blocks: List of conflict blocks
            conflict_winners: List of (block_idx, winner_agent_idx) tuples
            magic_wall_outcomes: Optional dict mapping agent_idx -> outcome_type ('succeed' or 'fail')
            
        Returns:
            tuple: The successor state
        """
        # Start from the given state
        self.set_state(state)
        
        # Increment step counter
        self.step_count += 1
        
        # Record initial agent positions to prevent chain conflicts.
        # This must be consistent with step() behavior.
        self._initial_agent_positions = set()
        for agent in self.agents:
            if agent.pos is not None and not agent.terminated:
                self._initial_agent_positions.add(tuple(agent.pos))
        
        # Separate agents into normal, unsteady-forward, and magic-wall
        normal_agents = []
        unsteady_forward_agents_list = []
        unsteady_outcomes_dict = {}  # agent_idx -> outcome_type
        magic_wall_agents_list = []
        
        for i in range(num_agents):
            if (self.agents[i].terminated or 
                self.agents[i].paused or 
                not self.agents[i].started):
                continue
            
            action = modified_actions[i]
            if isinstance(action, tuple):
                # This is an unsteady agent with (action, outcome_type)
                orig_action, outcome_type = action
                if not self._is_still_action(orig_action):
                    unsteady_forward_agents_list.append(i)
                    unsteady_outcomes_dict[i] = outcome_type
            elif not self._is_still_action(action):
                # Check if this is a magic wall agent
                if magic_wall_outcomes and i in magic_wall_outcomes:
                    magic_wall_agents_list.append(i)
                else:
                    normal_agents.append(i)
        
        # Build ordering for normal agents based on conflict winners
        winner_set = set(w[1] for w in conflict_winners)
        ordering = []
        
        # Add winners first
        for _, winner in conflict_winners:
            if winner in normal_agents:
                ordering.append(winner)
        
        # Add non-winning normal agents
        for i in normal_agents:
            if i not in winner_set:
                ordering.append(i)
        
        # Process normal agents using the shared helper
        rewards = np.zeros(num_agents)
        done = False
        
        for i in ordering:
            # Re-check: agent may have been terminated/paused by an earlier agent's action
            if (self.agents[i].terminated or
                    self.agents[i].paused or
                    not self.agents[i].started):
                continue
            action = modified_actions[i]
            if isinstance(action, tuple):
                action = action[0]  # Extract original action
            
            agent_done = self._execute_single_agent_action(i, action, rewards)
            done = done or agent_done
        
        # Process unsteady-forward agents using the shared helper
        if unsteady_forward_agents_list:
            unsteady_done = self._process_unsteady_forward_agents(
                unsteady_forward_agents_list, rewards, unsteady_outcomes_dict
            )
            done = done or unsteady_done
        
        # Process magic wall agents (deterministic based on outcome)
        if magic_wall_agents_list:
            for i in magic_wall_agents_list:
                outcome = magic_wall_outcomes[i]
                if outcome == 'succeed':
                    # Agent successfully enters the magic wall
                    fwd_pos = self.agents[i].front_pos
                    fwd_cell = self.grid.get(*fwd_pos)
                    # Save magic wall to terrain_grid so agent can step off it later
                    self.terrain_grid.set(*fwd_pos, fwd_cell)
                    self._move_agent_to_cell(i, fwd_pos, fwd_cell)
                    self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
                elif outcome == 'solidify':
                    # Entry failed and magic wall solidifies (deactivates)
                    fwd_pos = self.agents[i].front_pos
                    fwd_cell = self.grid.get(*fwd_pos)
                    if fwd_cell and fwd_cell.type == 'magicwall':
                        fwd_cell.active = False
                # If outcome is 'fail', agent stays in place and wall stays magic (no action)
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        # Return the resulting state
        return self.get_state()
