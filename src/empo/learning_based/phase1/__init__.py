"""
Phase 1: Human Policy Prior Learning for Empowerment-Based AI

This module implements the neural network-based learning approach for Phase 1
of the EMPO framework - computing human policy priors.

Phase 1 computes:
- Q(s, a, g): Goal-conditioned action values
- π(a|s, g): Goal-conditioned Boltzmann policies
- π(a|s): Marginal action probabilities (averaged over goals)

These priors are used in Phase 2 to model human goal-achievement abilities.
"""

from .state_encoder import BaseStateEncoder
from .goal_encoder import BaseGoalEncoder
from .q_network import BaseQNetwork
from .policy_prior_network import BasePolicyPriorNetwork
from .neural_policy_prior import BaseNeuralHumanPolicyPrior
from .replay_buffer import ReplayBuffer
from .trainer import Trainer

__all__ = [
    'BaseStateEncoder',
    'BaseGoalEncoder',
    'BaseQNetwork',
    'BasePolicyPriorNetwork',
    'BaseNeuralHumanPolicyPrior',
    'ReplayBuffer',
    'Trainer',
]
