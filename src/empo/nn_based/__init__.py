"""
Neural network-based policy priors.

This package provides neural network function approximators for computing
policy priors when the state space is too large for tabular methods.

For multigrid environments, use the `multigrid` subpackage:

    from empo.nn_based.multigrid import (
        MultiGridNeuralHumanPolicyPrior,
        train_multigrid_neural_policy_prior,
        MultiGridQNetwork,
        MultiGridStateEncoder,
        MultiGridAgentEncoder,
    )

Base classes for custom implementations:

    from empo.nn_based import (
        BaseStateEncoder,
        BaseGoalEncoder,
        BaseAgentEncoder,
        BaseQNetwork,
        BasePolicyPriorNetwork,
        BaseNeuralHumanPolicyPrior,
    )
"""

from .state_encoder import BaseStateEncoder
from .goal_encoder import BaseGoalEncoder
from .agent_encoder import BaseAgentEncoder
from .q_network import BaseQNetwork
from .policy_prior_network import BasePolicyPriorNetwork
from .neural_policy_prior import BaseNeuralHumanPolicyPrior
from .replay_buffer import ReplayBuffer
from .trainer import Trainer

from . import multigrid

__all__ = [
    # Base classes
    'BaseStateEncoder',
    'BaseGoalEncoder',
    'BaseAgentEncoder',
    'BaseQNetwork',
    'BasePolicyPriorNetwork',
    'BaseNeuralHumanPolicyPrior',
    # Utilities
    'ReplayBuffer',
    'Trainer',
    # Subpackage
    'multigrid',
]
