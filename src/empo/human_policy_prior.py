# ABC for the human policy prior used in by the robot to model human behavior.

# It's a callable that has a world model attribute and accepts a world state, the index of a human agent, and a possible goal, and returns a distribution over the possible actions of that human agent encoded as a numpy array of probabilities by action index.

# If called without a possible goal, it returns a distribution over actions marginalizing over possible goals.

# It has a sample() method that samples an action according to the distribution returned by __call__. If that sample methods is called without an agent and goal, it samples an action combination for all human agents according to the joint distribution obtained by assuming independence between the human agents, marginalizing over their possible goals. This method has a default implementation based on __call__.

# This will later be implemented e.g. by a lookup table or a neural network.

from abc import ABC, abstractmethod
from empo.world_model import WorldModel
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
import numpy as np

# output type:

class HumanPolicyPrior(ABC):

    world_model: WorldModel
    human_agent_indices: 'list[int]'
    
    def __init__(self, world_model: WorldModel, human_agent_indices: 'list[int]'):
        self.world_model = world_model
        self.human_agent_indices = human_agent_indices

    @abstractmethod
    def __call__(self, state, human_agent_index: int, possible_goal: PossibleGoal = None) -> 'np.ndarray':
        """Returns a distribution over the possible actions of the human agent with the given index in the given world state, conditioned on the possible goal."""
        pass

    def sample(self, state, human_agent_index: int = None, possible_goal: PossibleGoal = None) -> int or 'list[int]':
        """Samples an action for the human agent with the given index in the given world state according to the distribution returned by __call__. If no agent index is given, samples an action combination for all human agents according to the joint distribution obtained by assuming independence between the human agents, marginalizing over their possible goals."""
        if human_agent_index is not None:
            action_distribution = self(state, human_agent_index, possible_goal)
            return np.random.choice(len(action_distribution), p=action_distribution)
        else:
            assert possible_goal is None, "When sampling actions for all human agents, no possible goal can be given."
            actions = []
            for agent_index in self.human_agent_indices:
                action_distribution = self(state, agent_index)
                action = np.random.choice(len(action_distribution), p=action_distribution)
                actions.append(action)
            return actions
        
class TabularHumanPolicyPrior(HumanPolicyPrior):

    values: dict = None
    possible_goal_generator: PossibleGoalGenerator = None

    def __init__(self, world_model: WorldModel, human_agent_indices: list, possible_goal_generator: PossibleGoalGenerator, values: dict):
        super().__init__(world_model, human_agent_indices)
        self.values = values
        self.possible_goal_generator = possible_goal_generator

    def __call__(self, state, human_agent_index: int, possible_goal: PossibleGoal = None):
        if possible_goal is not None:
            return self.values[state][human_agent_index][possible_goal]
        else:
            vs = self.values[state][human_agent_index]
            for possible_goal, weight in self.possible_goal_generator.generate(state, human_agent_index):
                total += vs[possible_goal] * weight
            return total