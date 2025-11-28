# ABC for potential goals of humans. A possible goal is encoded by a 1-0 reward function taking a world state and returning 1 if the goal is achieved in that state and 0 otherwise.

from abc import ABC, abstractmethod
from empo.world_model import WorldModel

class PossibleGoal(ABC):

    world_model: 'WorldModel'
    def __init__(self, world_model: 'WorldModel'):
        self.world_model = world_model

    @abstractmethod
    def is_achieved(self, state) -> int:
        """Returns whether the goal is achieved in the given world state."""
        pass

# Another ABC for a goal sampler over possible goals for a human agent in a given world state, to be used in stochastic approximation of integrals over goals, e.g. by means of importance sampling. It samples a goal and returns an aggregation weight for that goal. The aggregation weight is used to weight the contribution of that goal in the approximation of the integral.

class PossibleGoalSampler(ABC):

    world_model: 'WorldModel'
    def __init__(self, world_model: 'WorldModel'):
        self.world_model = world_model

    @abstractmethod
    def sample(self, state, human_agent_index: int) -> ('PossibleGoal', float):
        """Samples a possible goal for the human agent with the given index in the given world state, and returns it along with an aggregation weight."""
        pass

# A function for stochastic approximation of integrals over possible goals for a human agent in a given world state. It takes a world state, the index of a human agent, a goal sampler, a function accepting a possible goal and returning a float, and a sample size, and returns the stochastic approximation of the integral of that function over possible goals for that human agent in that world state.

def approx_integral_over_possible_goals(state, human_agent_index: int, sampler: PossibleGoalSampler, func, sample_size: int) -> float:
    total = 0.0
    for _ in range(sample_size):
        possible_goal, weight = sampler.sample(state, human_agent_index)
        total += func(possible_goal) * weight
    return total / sample_size


# Another class for looping over all possible goals for a human agent in a given world state, to be used in exact computation of integrals over goals, acting as a proper python generator rather than returning a list.

class PossibleGoalGenerator(ABC):

    world_model: 'WorldModel'
    def __init__(self, world_model: 'WorldModel'):
        self.world_model = world_model

    @abstractmethod
    def generate(self, state, human_agent_index: int):
        """Yields all pairs of (possible goal, aggregation weight) for the human agent with the given index in the given world state."""
        pass