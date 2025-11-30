"""
Human Policy Prior for Modeling Human Behavior.

This module provides abstract and concrete implementations for modeling
human behavior as goal-directed policies. The policy prior represents
our belief about what action a human agent would take in a given state,
optionally conditioned on a specific goal.

Classes:
    HumanPolicyPrior: Abstract base class for human policy priors.
    TabularHumanPolicyPrior: Concrete implementation using precomputed lookup tables.

The policy prior is central to the EMPO framework where robots reason about
human behavior to compute empowerment and select helpful actions.

Key concepts:
    - A policy prior maps (state, agent, goal) -> action distribution
    - When called without a goal, returns marginal over all possible goals
    - The sample() method enables Monte Carlo simulation of human behavior

Example usage:
    >>> # Using a precomputed policy prior
    >>> from empo.backward_induction import compute_human_policy_prior
    >>> policy_prior = compute_human_policy_prior(env, [0], goal_generator)
    >>> 
    >>> # Get action distribution for agent 0 with specific goal
    >>> action_dist = policy_prior(state, 0, my_goal)  # numpy array
    >>> 
    >>> # Sample an action
    >>> action = policy_prior.sample(state, 0, my_goal)  # int
    >>> 
    >>> # Get marginal action distribution (averaging over goals)
    >>> marginal_dist = policy_prior(state, 0)  # numpy array
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from empo.world_model import WorldModel
    from empo.possible_goal import PossibleGoal, PossibleGoalGenerator


class HumanPolicyPrior(ABC):
    """
    Abstract base class for human policy priors.
    
    A human policy prior is a callable that returns a probability distribution
    over actions for a human agent, optionally conditioned on a possible goal.
    
    When called with a goal, returns P(action | state, agent, goal).
    When called without a goal, returns the marginal P(action | state, agent)
    obtained by averaging over possible goals weighted by their prior probabilities.
    
    Attributes:
        world_model: The world model (environment) this prior applies to.
        human_agent_indices: List of agent indices considered as "human" agents.
    
    Note:
        This class assumes independence between human agents when sampling
        joint actions without conditioning on specific goals.
    """

    world_model: 'WorldModel'
    human_agent_indices: List[int]
    
    def __init__(self, world_model: 'WorldModel', human_agent_indices: List[int]):
        """
        Initialize the human policy prior.
        
        Args:
            world_model: The world model (environment) this prior applies to.
            human_agent_indices: List of indices of agents to model as humans.
        """
        self.world_model = world_model
        self.human_agent_indices = human_agent_indices

    @abstractmethod
    def __call__(
        self, 
        state, 
        human_agent_index: int, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> np.ndarray:
        """
        Get the action distribution for a human agent.
        
        Args:
            state: Current world state (hashable tuple from get_state()).
            human_agent_index: Index of the human agent.
            possible_goal: If provided, condition the distribution on this goal.
                          If None, return marginal distribution over goals.
        
        Returns:
            np.ndarray: Probability distribution over actions (sums to 1.0).
                       Shape is (num_actions,) where num_actions = action_space.n.
        """
        pass

    def sample(
        self, 
        state, 
        human_agent_index: Optional[int] = None, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> Union[int, List[int]]:
        """
        Sample action(s) from the policy prior.
        
        If human_agent_index is provided, samples a single action for that agent.
        If human_agent_index is None, samples actions for ALL human agents (assuming
        independence and marginalizing over goals for each).
        
        Args:
            state: Current world state.
            human_agent_index: If provided, sample for this specific agent.
                              If None, sample for all human agents.
            possible_goal: If provided, condition on this goal.
                          Only valid when human_agent_index is also provided.
        
        Returns:
            int: If human_agent_index is provided, the sampled action index.
            List[int]: If human_agent_index is None, list of sampled actions
                      for all human agents (in order of human_agent_indices).
        
        Raises:
            AssertionError: If possible_goal is provided without human_agent_index.
        """
        if human_agent_index is not None:
            action_distribution = self(state, human_agent_index, possible_goal)
            return int(np.random.choice(len(action_distribution), p=action_distribution))
        else:
            assert possible_goal is None, \
                "When sampling actions for all human agents, no possible goal can be given."
            actions = []
            for agent_index in self.human_agent_indices:
                action_distribution = self(state, agent_index)
                action = int(np.random.choice(len(action_distribution), p=action_distribution))
                actions.append(action)
            return actions


class TabularHumanPolicyPrior(HumanPolicyPrior):
    """
    Tabular (lookup-table) implementation of human policy prior.
    
    This implementation stores precomputed policy distributions in a nested
    dictionary structure, indexed by (state, agent_index, goal).
    
    Typically created by the `compute_human_policy_prior()` function which
    performs backward induction to compute optimal Boltzmann policies.
    
    Attributes:
        values: Nested dict mapping state -> agent_index -> goal -> action_distribution.
        possible_goal_generator: Generator for enumerating possible goals.
    
    Structure of `values`:
        {
            state1: {
                agent_idx1: {
                    goal1: np.array([p_action0, p_action1, ...]),
                    goal2: np.array([...]),
                    ...
                },
                agent_idx2: {...},
            },
            state2: {...},
        }
    """

    values: dict
    possible_goal_generator: 'PossibleGoalGenerator'

    def __init__(
        self, 
        world_model: 'WorldModel', 
        human_agent_indices: List[int], 
        possible_goal_generator: 'PossibleGoalGenerator', 
        values: dict
    ):
        """
        Initialize the tabular policy prior.
        
        Args:
            world_model: The world model (environment) this prior applies to.
            human_agent_indices: List of indices of human agents.
            possible_goal_generator: Generator for enumerating possible goals.
            values: Precomputed policy lookup table (state -> agent -> goal -> distribution).
        """
        super().__init__(world_model, human_agent_indices)
        self.values = values
        self.possible_goal_generator = possible_goal_generator

    def __call__(
        self, 
        state, 
        human_agent_index: int, 
        possible_goal: Optional['PossibleGoal'] = None
    ) -> np.ndarray:
        """
        Look up or compute the action distribution.
        
        Args:
            state: Current world state (must be a key in self.values).
            human_agent_index: Index of the human agent.
            possible_goal: If provided, return distribution conditioned on this goal.
                          If None, compute marginal by averaging over goals.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        
        Raises:
            KeyError: If state or agent_index not found in lookup table.
        """
        if possible_goal is not None:
            return self.values[state][human_agent_index][possible_goal]
        else:
            # Compute marginal by averaging over goals weighted by their prior
            vs = self.values[state][human_agent_index]
            num_actions: int = self.world_model.action_space.n  # type: ignore[attr-defined]
            total = np.zeros(num_actions)
            for goal, weight in self.possible_goal_generator.generate(state, human_agent_index):
                total += vs[goal] * weight
            return total