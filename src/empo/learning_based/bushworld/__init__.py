"""
BushWorld-specific neural network encoders and Phase 2 implementations.

This subpackage is the BushWorld analogue of
:mod:`empo.learning_based.multigrid`. It contains the encoders that convert a
BushWorld state into neural-network inputs, plus (in the ``phase2`` subpackage)
the Phase 2 networks and trainer that reuse the shared Phase 2 infrastructure.

Main components:
    - constants: Channel indices and feature sizes.
    - feature_extraction: Convert raw state/goal/agent into input vectors.
    - state_encoder: ``BushWorldStateEncoder``.
    - goal_encoder: ``BushWorldGoalEncoder`` (cell and rectangle goals).
    - agent_encoder: ``BushWorldAgentEncoder``.
    - phase2: Phase 2 networks, trainer, and deployable robot policy.
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
