"""
Phase 1: BushWorld-specific Human Policy Prior Learning (DQN path).

This subpackage is the BushWorld analogue of
:mod:`empo.learning_based.multigrid.phase1`. It learns a goal-conditioned human
policy prior using the shared, generic Q-learning
:class:`~empo.learning_based.phase1.trainer.Trainer` — no new training algorithm
is introduced.

Phase 1 computes:
- ``Q(s, a, g)``: Goal-conditioned action values (:class:`BushWorldQNetwork`).
- ``π(a|s, g)``: Goal-conditioned Boltzmann policies.
- ``π(a|s)``: Marginal action probabilities averaged over goals
  (:class:`BushWorldPolicyPriorNetwork`, or the fast
  :class:`BushWorldDirectPhiNetwork`).

BushWorld's human prior is normally the heuristic
:class:`~empo.bushworld.human_policy.ShortestPathHumanPolicyPrior`; this package
provides the *learned* alternative for full structural parity with multigrid.
"""

from .direct_phi_network import BushWorldDirectPhiNetwork
from .neural_policy_prior import (
    BushWorldNeuralHumanPolicyPrior,
    train_bushworld_neural_policy_prior,
)
from .policy_prior_network import BushWorldPolicyPriorNetwork
from .q_network import BushWorldQNetwork

__all__ = [
    "BushWorldQNetwork",
    "BushWorldPolicyPriorNetwork",
    "BushWorldDirectPhiNetwork",
    "BushWorldNeuralHumanPolicyPrior",
    "train_bushworld_neural_policy_prior",
]
