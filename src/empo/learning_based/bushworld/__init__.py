"""
BushWorld-specific neural network encoders and Phase 1 / Phase 2 implementations.

This subpackage is the BushWorld analogue of
:mod:`empo.learning_based.multigrid`. It contains the encoders that convert a
BushWorld state into neural-network inputs, plus subpackages that reuse the
shared training infrastructure (no new training algorithms) for both phases.

Main components:
    - constants: Channel indices and feature sizes.
    - feature_extraction: Convert raw state/goal/agent into input vectors.
    - state_encoder: ``BushWorldStateEncoder``.
    - goal_encoder: ``BushWorldGoalEncoder`` (cell and rectangle goals).
    - agent_encoder: ``BushWorldAgentEncoder``.
    - phase1: Phase 1 (human policy prior) DQN networks, neural human policy
      prior, and trainer (reuses the generic Phase 1 ``Trainer``).
    - phase1_ppo: PPO-based Phase 1 env wrapper and network factory (reuses the
      shared ``Phase1PPOEnv`` / ``GoalConditionedActorCritic``).
    - phase2: Phase 2 (robot policy) networks, trainer, and deployable robot
      policy.
    - phase2_ppo: PPO-based Phase 2 env wrapper and network factory (reuses the
      shared ``EMPOWorldModelEnv`` / ``EMPOActorCritic`` infrastructure).
"""

from .agent_encoder import BushWorldAgentEncoder
from .constants import (
    AGENT_FEATURE_SIZE,
    GOAL_COORD_DIM,
    NUM_GLOBAL_WORLD_FEATURES,
    NUM_GRID_CHANNELS,
)
from .goal_encoder import BushWorldGoalEncoder
from .state_encoder import BushWorldStateEncoder

__all__ = [
    "NUM_GRID_CHANNELS",
    "NUM_GLOBAL_WORLD_FEATURES",
    "GOAL_COORD_DIM",
    "AGENT_FEATURE_SIZE",
    "BushWorldStateEncoder",
    "BushWorldGoalEncoder",
    "BushWorldAgentEncoder",
]

