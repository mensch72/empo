"""
Phase 1: MultiGrid-specific Human Policy Prior Learning

This module implements the MultiGrid environment-specific learning approach for
Phase 1 of the EMPO framework - computing human policy priors for multi-agent
grid-based environments.

MultiGrid-specific features:
- CNN-based grid encoding for spatial reasoning
- Agent feature encoding (position, direction, carrying)
- Interactive object encoding (doors, keys, buttons, switches)
- Goal position encoding with orientation
- Support for colored agents and objects

Phase 1 computes:
- Q(s, a, g): Goal-conditioned action values using grid/agent encoders
- π(a|s, g): Goal-conditioned Boltzmann policies
- π(a|s): Marginal action probabilities (averaged over goals)

These priors are used in Phase 2 to model human goal-achievement abilities
in cooperative multi-agent scenarios.
"""

from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .direct_phi_network import DirectPhiNetwork
from .neural_policy_prior import (
    MultiGridNeuralHumanPolicyPrior,
    train_multigrid_neural_policy_prior,
)

__all__ = [
    'MultiGridQNetwork',
    'MultiGridPolicyPriorNetwork',
    'DirectPhiNetwork',
    'MultiGridNeuralHumanPolicyPrior',
    'train_multigrid_neural_policy_prior',
]
