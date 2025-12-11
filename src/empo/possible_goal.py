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
    Each sample returns a goal along with an importance weight for
    weighted averaging.
    
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

    @abstractmethod
    def sample(self, state, human_agent_index: int) -> Tuple['PossibleGoal', float]:
        """
        Sample a possible goal for a human agent in the given state.
        
        The returned weight is used for importance sampling. If sampling
        uniformly from all goals, the weight should be 1.0 for all samples.
        
        Args:
            state: Current world state (hashable tuple from get_state()).
            human_agent_index: Index of the human agent whose goal to sample.
        
        Returns:
            Tuple of (goal, weight) where:
                - goal: A PossibleGoal instance
                - weight: Importance weight for this sample (float > 0)
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
    sampler. Uses importance sampling with the weights returned by the sampler.
    
    The approximation converges to the true integral as sample_size increases,
    assuming the sampler has support over all goals and the weights are
    valid importance weights.
    
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