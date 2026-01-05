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
    ...     
    ...     def is_achieved(self, state) -> int:
    ...         # Check if agent is at target position
    ...         agent_states = state[1]  # (step_count, agent_states, mobile_objects, mutable_objects)
    ...         agent_pos = (agent_states[0][0], agent_states[0][1])
    ...         return 1 if agent_pos == self.target_pos else 0
    ...     
    ...     def __hash__(self):
    ...         return hash(self.target_pos)
    ...     
    ...     def __eq__(self, other):
    ...         return isinstance(other, ReachCell) and self.target_pos == other.target_pos
"""

from abc import ABC, abstractmethod
from typing import Tuple, Iterator, Callable, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import gymnasium as gym


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
    
    Note:
        Goals should be immutable after creation to ensure consistent hashing.
    """

    env: Any  # gymnasium.Env or compatible
    
    def __init__(self, env: Any):
        """
        Initialize the possible goal.
        
        Args:
            env: The gymnasium environment (or compatible) this goal applies to.
        """
        self.env = env

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
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        """Return hash for use as dictionary key."""
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        """Check equality with another goal."""
        pass


class PossibleGoalSampler(ABC):
    """
    Abstract base class for stochastic sampling of possible goals.
    
    Used for Monte Carlo approximation of integrals over goal space.
    Each sample returns a goal along with a weight for computing X_h.
    
    This is useful when the goal space is too large for exact enumeration.
    
    Attributes:
        env: Reference to the gymnasium environment this sampler applies to.
    """

    env: Any  # gymnasium.Env or compatible
    
    def __init__(self, env: Any):
        """
        Initialize the goal sampler.
        
        Args:
            env: The gymnasium environment (or compatible) this sampler applies to.
        """
        self.env = self.world_model = env
    
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
        pass


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
    The weights should sum to 1.0 for proper probability weighting,
    or all be 1.0 for uniform weighting.
    
    Attributes:
        env: Reference to the gymnasium environment this generator applies to.
    
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
    
    def __init__(self, env: Any):
        """
        Initialize the goal generator.
        
        Args:
            env: The gymnasium environment (or compatible) this generator applies to.
        """
        self.env = env
        self.world_model = env  # Alias for compatibility

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
        pass


class DeterministicGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that always returns a single fixed goal.
    
    Useful when the goal is known/fixed and you want to use it with
    interfaces that expect a PossibleGoalSampler.
    
    Args:
        goal: The fixed PossibleGoal instance to return.
        weight: The weight for X_h computation (default 1.0).
    """
    
    def __init__(self, goal: 'PossibleGoal', weight: float = 1.0):
        """
        Initialize with a fixed goal.
        
        Args:
            goal: The PossibleGoal instance to always return.
            weight: The weight for X_h computation (default 1.0).
        """
        super().__init__(goal.env)
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
    
    def __init__(self, goal: 'PossibleGoal', weight: float = 1.0):
        """
        Initialize with a fixed goal.
        
        Args:
            goal: The PossibleGoal instance to always yield.
            weight: The aggregation weight (default 1.0).
        """
        super().__init__(goal.env)
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
    
    def __init__(self, goals, probabilities=None, weights=None):
        """
        Initialize with a list of goals and optional probabilities/weights.
        
        Args:
            goals: Iterable of PossibleGoal instances.
            probabilities: Optional iterable of sampling probabilities (will be normalized).
                          If None, uses uniform 1/n probabilities.
            weights: Optional iterable of weights for X_h computation.
                    If None, uses 1.0 for all goals.
        """
        self.goals = list(goals)
        if len(self.goals) == 0:
            raise ValueError("TabularGoalSampler requires at least one goal")
        
        super().__init__(self.goals[0].env)
        
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


class TabularGoalGenerator(PossibleGoalGenerator):
    """
    A goal generator that yields from a fixed list of goals.
    
    Yields all goals with their associated weights.
    
    Args:
        goals: Iterable of PossibleGoal instances.
        weights: Optional iterable of weights. If None, uniform 1/n weights are used.
    """
    
    def __init__(self, goals, weights=None):
        """
        Initialize with a list of goals and optional weights.
        
        Args:
            goals: Iterable of PossibleGoal instances.
            weights: Optional iterable of weights. If None, uses uniform 1/n weights.
        """
        self.goals = list(goals)
        if len(self.goals) == 0:
            raise ValueError("TabularGoalGenerator requires at least one goal")
        
        super().__init__(self.goals[0].env)
        
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