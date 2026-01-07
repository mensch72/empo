"""
Phase 1: Transport-specific Human Policy Prior Learning

This module implements the transport environment-specific learning approach for
Phase 1 of the EMPO framework - computing human policy priors for multi-agent
transport/logistics scenarios.

Transport-specific features:
- GNN-based network encoding for road network topology
- Support for node-based and cluster-based routing goals
- Step-type-aware action masking (boarding, destination, departure)
- Distance-based reward shaping for sparse reward environments

Phase 1 computes:
- Q(s, a, g): Goal-conditioned action values using GNN state encoding
- π(a|s, g): Goal-conditioned Boltzmann policies
- π(a|s): Marginal action probabilities (averaged over goals)

These priors are used in Phase 2 to model human goal-achievement abilities
in transport coordination scenarios.
"""

from .state_encoder import TransportStateEncoder
from .goal_encoder import TransportGoalEncoder
from .q_network import TransportQNetwork
from .policy_prior_network import TransportPolicyPriorNetwork
from .neural_policy_prior import (
    TransportNeuralHumanPolicyPrior,
    train_transport_neural_policy_prior,
)
from .constants import NUM_TRANSPORT_ACTIONS

__all__ = [
    'TransportStateEncoder',
    'TransportGoalEncoder',
    'TransportQNetwork',
    'TransportPolicyPriorNetwork',
    'TransportNeuralHumanPolicyPrior',
    'train_transport_neural_policy_prior',
    'NUM_TRANSPORT_ACTIONS',
]
