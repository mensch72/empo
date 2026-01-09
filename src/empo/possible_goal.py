"""
Possible Goal Abstractions for Human Behavior Modeling.

This module provides abstract base classes for defining and working with
possible goals that human agents might have. Goals are encoded as 0/1 reward
functions that indicate goal achievement.

Classes:
    PossibleGoal: Abstract base class for a single possible goal.
    PossibleGoalSampler: Abstract base class for stochastic goal sampling (for Monte Carlo methods).
    PossibleGoalGenerator: Abstract base class for deterministic goal enumeration.

Functions:
    approx_integral_over_possible_goals: Monte Carlo integration over goal space.

The goal abstraction is central to the human policy prior computation, where
we assume humans are goal-directed agents whose behavior can be modeled as
optimizing for possible goals with some uncertainty.

Example usage:
    >>> class ReachCell(PossibleGoal):
    ...     def __init__(self, env, target_pos):
    ...         super().__init__(env)
    ...         self.target_pos = target_pos
    ...         self._hash = hash(self.target_pos)  # Cache hash at init
    ...         super()._freeze()  # Must call last in __init__
    ...     
    ...     def is_achieved(self, state) -> int:
    ...         # Check if agent is at target position
    ...         agent_states = state[1]  # (step_count, agent_states, mobile_objects, mutable_objects)
    ...         agent_pos = (agent_states[0][0], agent_states[0][1])
    ...         return 1 if agent_pos == self.target_pos else 0
    ...     
    ...     def __hash__(self):
    ...         return self._hash
    ...     
    ...     def __eq__(self, other):
    ...         return isinstance(other, ReachCell) and self.target_pos == other.target_pos
"""

from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Callable, TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass


class PossibleGoal(ABC):
    """
    Abstract base class for a possible goal of a human agent.
    
    A goal is encoded as a 0/1 reward function that returns 1 when the goal
    is achieved in a given state and 0 otherwise. Goals must be hashable
    to be used as dictionary keys in the backward induction algorithm.
    
    Subclasses MUST implement:
        - is_achieved(state): Returns 1 if goal achieved, 0 otherwise
        - __hash__(): Returns hash for use as dict key
        - __eq__(other): Equality comparison
    
    Attributes:
        env: Reference to the gymnasium environment this goal applies to.
    
    Immutability:
        Goals are immutable after creation. The base class enforces this by
        raising AttributeError if you try to modify or delete attributes after
        __init__ completes. Subclasses must set all attributes (including
        self._hash for cached hash) during __init__, calling super().__init__(env)
        first and super()._freeze() last.
    
    Performance:
        For performance, subclasses should cache the hash value at initialization
        (store in self._hash) and return it from __hash__(), since goals are
        frequently used as dictionary keys in backward induction.
    
    Example:
        >>> class MyGoal(PossibleGoal):
        ...     def __init__(self, env, target):
        ...         super().__init__(env)
        ...         self.target = target
        ...         self._hash = hash(self.target)
        ...         super()._freeze()  # Must call last in __init__
    """

    env: Any  # gymnasium.Env or compatible
    _frozen: bool  # Whether the object is frozen (immutable)
    index: Optional[int]  # Optional index of this goal in a list (set by YAML loader)
    
    def __init__(self, env: Any, index: Optional[int] = None):
        """
        Initialize the possible goal.
        
        Args:
            env: The gymnasium environment (or compatible) this goal applies to.
            index: Optional index of this goal in a list (set by YAML loader).
        
        Note:
            Subclasses must call super()._freeze() at the END of their __init__
            to enable immutability enforcement.
        """
        object.__setattr__(self, '_frozen', False)
        self.env = env
        self.index = index
    
    def _freeze(self) -> None:
        """Freeze the object to prevent further modifications.
        
        Subclasses MUST call this at the end of their __init__ method.
        """
        object.__setattr__(self, '_frozen', True)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification after initialization."""
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"Cannot modify attribute '{name}' of immutable {self.__class__.__name__}. "
                f"PossibleGoal objects are immutable after creation."
            )
        object.__setattr__(self, name, value)
    
    def __delattr__(self, name: str) -> None:
        """Prevent attribute deletion."""
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"Cannot delete attribute '{name}' of immutable {self.__class__.__name__}. "
                f"PossibleGoal objects are immutable after creation."
            )
        object.__delattr__(self, name)

    def __getstate__(self):
        """Exclude env from pickling (it may contain unpicklable objects like thread locks)."""
        state = self.__dict__.copy()
        state['env'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling.
        
        Temporarily unfreeze to allow setting attributes, then re-freeze.
        The env/world_model will need to be set separately after unpickling
        if needed.
        """
        # Must set _frozen to False first to allow attribute setting
        object.__setattr__(self, '_frozen', False)
        self.__dict__.update(state)
        # Re-freeze the object
        object.__setattr__(self, '_frozen', True)

    @abstractmethod
    def is_achieved(self, state) -> int:
        """
        Check if this goal is achieved in the given state.
        
        Args:
            state: A hashable state tuple as returned by env.get_state() if available.
                   Format is typically: (step_count, agent_states, mobile_objects, mutable_objects)
        
        Returns:
            int: 1 if the goal is achieved in this state, 0 otherwise.
        """
    
    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use as dictionary key."""
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality with another goal."""


class PossibleGoalSampler(ABC):
    """
    Abstract base class for stochastic sampling of possible goals.
    
    Used for Monte Carlo approximation of integrals over goal space.
    Each sample returns a goal along with a weight for computing X_h.
    
    This is useful when the goal space is too large for exact enumeration.
    
    Attributes:
        env: Reference to the gymnasium environment this sampler applies to.
        indexed: Boolean indicating whether goals have indices (set by YAML loader).
    """

    env: Any  # gymnasium.Env or compatible
    indexed: bool  # Whether goals have indices
    
    def __init__(self, env: Any, indexed: bool = False):
        """
        Initialize the goal sampler.
        
        Args:
            env: The gymnasium environment (or compatible) this sampler applies to.
            indexed: Boolean indicating whether goals have indices (default False).
        """
        self.env = self.world_model = env
        self.indexed = indexed
    
    def set_world_model(self, world_model: Any) -> None:
        """
        Set or update the world model reference.
        
        This is used for async training where the world_model cannot be pickled
        and must be recreated in child processes.
        
        Args:
            world_model: The world model (environment) to use.
        """
        self.env = self.world_model = world_model
    
    def __getstate__(self):
        """Exclude world_model/env from pickling (it contains thread locks)."""
        state = self.__dict__.copy()
        state['env'] = None
        state['world_model'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling (world_model will be set later)."""
        self.__dict__.update(state)

    @abstractmethod
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        """
        Sample a possible goal for a human agent in the given state.
        
        The returned weight is used for computing X_h = E[weight * V_h^e].
        
        Args:
            state: Current world state (hashable tuple from get_state()).
            human_agent_index: Index of the human agent whose goal to sample.
        
        Returns:
            Tuple of (goal, weight) where:
                - goal: A PossibleGoal instance
                - weight: Weight for this goal in X_h computation (float > 0)
        """


def approx_integral_over_possible_goals(
    state, 
    human_agent_index: int, 
    sampler: PossibleGoalSampler, 
    func: Callable[['PossibleGoal'], float], 
    sample_size: int
) -> float:
    """
    Approximate an integral over possible goals using Monte Carlo sampling.
    
    Computes the weighted average of func(goal) over goals sampled from the
    sampler, using the weights returned by the sampler.
    
    The approximation converges to the true integral as sample_size increases,
    assuming the sampler has support over all goals.
    
    Args:
        state: Current world state.
        human_agent_index: Index of the human agent.
        sampler: A PossibleGoalSampler for drawing goal samples.
        func: A function that takes a PossibleGoal and returns a float.
        sample_size: Number of Monte Carlo samples to draw.
    
    Returns:
        float: Monte Carlo estimate of the integral.
    
    Example:
        >>> def goal_value(goal):
        ...     return policy_prior(state, agent_idx, goal).max()
        >>> expected_value = approx_integral_over_possible_goals(
        ...     state, 0, uniform_sampler, goal_value, 1000
        ... )
    """
    total = 0.0
    for _ in range(sample_size):
        possible_goal, weight = sampler.sample(state, human_agent_index)
        total += func(possible_goal) * weight
    return total / sample_size


class PossibleGoalGenerator(ABC):
    """
    Abstract base class for deterministic enumeration of possible goals.
    
    Used for exact computation of integrals over goal space when the
    number of possible goals is finite and small enough to enumerate.
    
    This is a Python generator that yields (goal, weight) pairs.

    If there is a corresponding PossibleGoalSampler, the weights produced by
    this generator must be consistent with the sampler's weights and sampling
    probabilities.

    Important:
        - The generator is used for **exact integration** over goals, where you
          compute a sum of the form:
            
            sum_over_goals[ generator_weight(goal) * value(goal) ].

        - The sampler is used for **Monte Carlo approximation**, where you draw
          goals with some sampling probability `p(goal)` and weight `sampler_weight(goal)`
          and compute an expected value:
            
            E_{goal ~ p}[ sampler_weight(goal) * value(goal) ].

        - To make these two views consistent, each generator weight should equal
          the sampler weight multiplied by the corresponding sampling probability:
            
            generator_weight(goal) = sampler_weight(goal) * p(goal).
    
    Attributes:
        env: Reference to the gymnasium environment this generator applies to.
        indexed: Boolean indicating whether goals have indices (set by YAML loader).
    
    Example implementation:
        >>> class AllCellsGenerator(PossibleGoalGenerator):
        ...     def generate(self, state, human_agent_index: int):
        ...         for x in range(self.env.width):
        ...             for y in range(self.env.height):
        ...                 goal = ReachCell(self.env, (x, y))
        ...                 weight = 1.0 / (self.env.width * self.env.height)
        ...                 yield goal, weight
    """

    env: Any  # gymnasium.Env or compatible
    world_model: Any  # Alias for env for compatibility
    indexed: bool  # Whether goals have indices
    
    def __init__(self, env: Any, indexed: bool = False):
        """
        Initialize the goal generator.
        
        Args:
            env: The gymnasium environment (or compatible) this generator applies to.
            indexed: Boolean indicating whether goals have indices (default False).
        """
        self.env = env
        self.world_model = env  # Alias for compatibility
        self.indexed = indexed

    @abstractmethod
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple['PossibleGoal', float]]:
        """
        Generate all possible goals for a human agent in the given state.
        
        This is a generator that yields (goal, weight) pairs. The weights
        can represent prior probabilities over goals, or all be 1.0 for
        uniform weighting.
        
        Args:
            state: Current world state (hashable tuple from get_state()).
            human_agent_index: Index of the human agent whose goals to generate.
        
        Yields:
            Tuple[PossibleGoal, float]: Pairs of (goal, aggregation_weight).
        """


class DeterministicGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that always returns a single fixed goal.
    
    Useful when the goal is known/fixed and you want to use it with
    interfaces that expect a PossibleGoalSampler.
    
    Args:
        goal: The fixed PossibleGoal instance to return.
        weight: The weight for X_h computation (default 1.0).
    """
    
    def __init__(self, goal: 'PossibleGoal', weight: float = 1.0, indexed: bool = False):
        """
        Initialize with a fixed goal.
        
        Args:
            goal: The PossibleGoal instance to always return.
            weight: The weight for X_h computation (default 1.0).
            indexed: Boolean indicating whether goals have indices (default False).
        """
        super().__init__(goal.env, indexed=indexed)
        self.goal = goal
        self.weight = weight
    
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        """
        Return the fixed goal.
        
        Args:
            state: Current world state (unused).
            human_agent_index: Index of the human agent (unused).
        
        Returns:
            Tuple of (goal, weight).
        """
        return self.goal, self.weight


class DeterministicGoalGenerator(PossibleGoalGenerator):
    """
    A goal generator that yields a single fixed goal.
    
    Useful when the goal is known/fixed and you want to use it with
    interfaces that expect a PossibleGoalGenerator.
    
    Args:
        goal: The fixed PossibleGoal instance to yield.
        weight: The aggregation weight to yield (default 1.0).
    """
    
    def __init__(self, goal: 'PossibleGoal', weight: float = 1.0, indexed: bool = False):
        """
        Initialize with a fixed goal.
        
        Args:
            goal: The fixed PossibleGoal instance to yield.
            weight: The aggregation weight to yield (default 1.0).
            indexed: Boolean indicating whether goals have indices (default False).
        """
        super().__init__(goal.env, indexed=indexed)
        self.goal = goal
        self.weight = weight
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple['PossibleGoal', float]]:
        """
        Yield the single fixed goal.
        
        Args:
            state: Current world state (unused).
            human_agent_index: Index of the human agent (unused).
        
        Yields:
            Single tuple of (goal, weight).
        """
        yield self.goal, self.weight


class TabularGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that samples from a fixed list of goals.
    
    Samples goals according to the provided probabilities (or uniformly if not given),
    and returns the specified weights for X_h computation (or 1.0 if not given).
    
    Args:
        goals: Iterable of PossibleGoal instances.
        probabilities: Optional iterable of sampling probabilities. If None, uniform (1/n) is used.
        weights: Optional iterable of weights for X_h computation. If None, 1.0 is used for all.
    """
    
    def __init__(self, goals, probabilities=None, weights=None, indexed: bool = False):
        """
        Initialize with a list of goals and optional probabilities/weights.
        
        Args:
            goals: Iterable of PossibleGoal instances.
            probabilities: Optional iterable of sampling probabilities (will be normalized).
                          If None, uses uniform 1/n probabilities.
            weights: Optional iterable of weights for X_h computation.
                    If None, uses 1.0 for all goals.
            indexed: Boolean indicating whether goals have indices (default False).
        """
        self.goals = list(goals)
        if len(self.goals) == 0:
            raise ValueError("TabularGoalSampler requires at least one goal")
        
        super().__init__(self.goals[0].env, indexed=indexed)
        
        n = len(self.goals)
        
        # Set up sampling probabilities
        if probabilities is None:
            self.probs = [1.0 / n] * n
        else:
            probs = list(probabilities)
            if len(probs) != n:
                raise ValueError("Number of probabilities must match number of goals")
            # Normalize to sum to 1
            total = sum(probs)
            self.probs = [p / total for p in probs]
        
        # Set up weights for X_h computation
        if weights is None:
            self.weights = [1.0] * n
        else:
            self.weights = list(weights)
            if len(self.weights) != n:
                raise ValueError("Number of weights must match number of goals")
    
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        """
        Sample a goal according to the probabilities.
        
        Args:
            state: Current world state (unused).
            human_agent_index: Index of the human agent (unused).
        
        Returns:
            Tuple of (goal, weight).
        """
        import random
        idx = random.choices(range(len(self.goals)), weights=self.probs, k=1)[0]
        return self.goals[idx], self.weights[idx]
    
    def get_generator(self) -> 'TabularGoalGenerator':
        """
        Create a TabularGoalGenerator from this sampler.
        
        The generator's weights are computed as probs * weights from the sampler,
        which represents the expected contribution of each goal to X_h computation.
        
        Returns:
            TabularGoalGenerator with weights = probs * weights.
        """
        combined_weights = [p * w for p, w in zip(self.probs, self.weights)]
        return TabularGoalGenerator(self.goals, weights=combined_weights, indexed=self.indexed)
    
    def validate_coverage(self, raise_on_error: bool = True) -> list:
        """
        Validate that goals cover all walkable cells in the grid.
        
        A cell is considered walkable if it doesn't contain an immutable object
        (wall, lava, magicwall). The method checks that every such cell is covered
        by at least one goal (cell or rectangle).
        
        This only works for goals that have a target_rect attribute (ReachCellGoal,
        ReachRectangleGoal) which define the cells they cover.
        
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
            raise ValueError("Cannot validate coverage: env/world_model is None")
        
        # Build set of all cells covered by goals
        covered_cells = set()
        for goal in self.goals:
            if hasattr(goal, 'target_rect'):
                x1, y1, x2, y2 = goal.target_rect
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        covered_cells.add((x, y))
            elif hasattr(goal, 'target_pos'):
                covered_cells.add(goal.target_pos)
        
        # Find all walkable cells (not containing immutable objects)
        immutable_types = {'wall', 'lava', 'magicwall'}
        uncovered = []
        
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = self.env.grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell else None
                
                # Skip cells with immutable objects
                if cell_type in immutable_types:
                    continue
                
                # Check if this walkable cell is covered
                if (x, y) not in covered_cells:
                    uncovered.append((x, y))
        
        if uncovered and raise_on_error:
            raise ValueError(
                f"Goals do not cover all walkable cells. Uncovered cells: {uncovered}\n"
                f"Goals: {self.goals}\n"
                f"Add goals covering these cells or use a rectangle goal covering the entire grid."
            )
        
        return uncovered


class TabularGoalGenerator(PossibleGoalGenerator):
    """
    A goal generator that yields from a fixed list of goals.
    
    Yields all goals with their associated weights.
    
    Args:
        goals: Iterable of PossibleGoal instances.
        weights: Optional iterable of weights. If None, uniform 1/n weights are used.
    """
    
    def __init__(self, goals, weights=None, indexed: bool = False):
        """
        Initialize with a list of goals and optional weights.
        
        Args:
            goals: Iterable of PossibleGoal instances.
            weights: Optional iterable of weights. If None, uses uniform 1/n weights.
            indexed: Boolean indicating whether goals have indices (default False).
        """
        self.goals = list(goals)
        if len(self.goals) == 0:
            raise ValueError("TabularGoalGenerator requires at least one goal")
        
        super().__init__(self.goals[0].env, indexed=indexed)
        
        if weights is None:
            n = len(self.goals)
            self.weights = [1.0 / n] * n
        else:
            self.weights = list(weights)
            if len(self.weights) != len(self.goals):
                raise ValueError("Number of weights must match number of goals")
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple['PossibleGoal', float]]:
        """
        Yield all goals with their weights.
        
        Args:
            state: Current world state (unused).
            human_agent_index: Index of the human agent (unused).
        
        Yields:
            Tuples of (goal, weight) for each goal.
        """
        for goal, weight in zip(self.goals, self.weights):
            yield goal, weight
    
    def get_sampler(self) -> 'TabularGoalSampler':
        """
        Create a TabularGoalSampler from this generator.
        
        The sampler's probabilities are set to the generator's weights (normalized),
        and the sampler's weights are set to 1.0 for all goals.
        
        Returns:
            TabularGoalSampler with probs = weights (normalized), weights = 1.0.
        """
        return TabularGoalSampler(self.goals, probabilities=self.weights, weights=None, indexed=self.indexed)
    
    def validate_coverage(self, raise_on_error: bool = True) -> list:
        """
        Validate that goals cover all walkable cells in the grid.
        
        A cell is considered walkable if it doesn't contain an immutable object
        (wall, lava, magicwall). The method checks that every such cell is covered
        by at least one goal (cell or rectangle).
        
        This only works for goals that have a target_rect attribute (ReachCellGoal,
        ReachRectangleGoal) which define the cells they cover.
        
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
            raise ValueError("Cannot validate coverage: env/world_model is None")
        
        # Build set of all cells covered by goals
        covered_cells = set()
        for goal in self.goals:
            if hasattr(goal, 'target_rect'):
                x1, y1, x2, y2 = goal.target_rect
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                for x in range(x1, x2 + 1):
                    for y in range(y1, y2 + 1):
                        covered_cells.add((x, y))
            elif hasattr(goal, 'target_pos'):
                covered_cells.add(goal.target_pos)
        
        # Find all walkable cells (not containing immutable objects)
        immutable_types = {'wall', 'lava', 'magicwall'}
        uncovered = []
        
        for x in range(self.env.width):
            for y in range(self.env.height):
                cell = self.env.grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell else None
                
                # Skip cells with immutable objects
                if cell_type in immutable_types:
                    continue
                
                # Check if this walkable cell is covered
                if (x, y) not in covered_cells:
                    uncovered.append((x, y))
        
        if uncovered and raise_on_error:
            raise ValueError(
                f"Goals do not cover all walkable cells. Uncovered cells: {uncovered}\n"
                f"Goals: {self.goals}\n"
                f"Add goals covering these cells or use a rectangle goal covering the entire grid."
            )
        
        return uncovered