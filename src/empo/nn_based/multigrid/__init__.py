"""
Multigrid-specific neural network encoders and policy priors.

This subpackage contains all multigrid-specific implementations for encoding
environment state into neural network inputs and computing policy priors.

MAINTAINER NOTE:
When adding new object types or features to the multigrid environment,
the encoders in this package may need to be updated. See docs/ENCODER_ARCHITECTURE.md.

Main components:
    - constants: Object type mappings, color indices, feature sizes
    - feature_extraction: Extract features from state/world_model
    - state_encoder_unified: Unified state encoder (grid + agents + interactive)
    - goal_encoder: Goal position encoder (separate - not part of world state)
    - q_network: Q-network combining state and goal encoders
    - policy_prior_network: Marginal policy computation via goal enumeration
    - direct_phi_network: Trainable network for stochastic marginal approximation
    - neural_policy_prior: Main class with save/load and training
    - path_distance: BFS-based path distance calculator
"""

from .constants import (
    STANDARD_COLORS,
    NUM_STANDARD_COLORS,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_BASE_OBJECT_CHANNELS,
    NUM_OBJECT_TYPE_CHANNELS,
    DOOR_CHANNEL_START,
    KEY_CHANNEL_START,
    MAGICWALL_CHANNEL,
    AGENT_FEATURE_SIZE,
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
    NUM_GLOBAL_WORLD_FEATURES,
    DEFAULT_ACTION_ENCODING,
    SMALL_ACTION_ENCODING,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
)

from .feature_extraction import (
    extract_agent_features,
    extract_all_agent_features,
    extract_interactive_objects,
    extract_global_world_features,
    extract_door_states,
    extract_magic_wall_states,
    extract_agent_colors,
    get_num_agents_per_color,
)

from .state_encoder import MultiGridStateEncoder
from .goal_encoder import MultiGridGoalEncoder
from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .direct_phi_network import DirectPhiNetwork
from .neural_policy_prior import (
    MultiGridNeuralHumanPolicyPrior,
    train_multigrid_neural_policy_prior,
)
from .path_distance import PathDistanceCalculator

__all__ = [
    # Constants
    'STANDARD_COLORS',
    'NUM_STANDARD_COLORS',
    'COLOR_TO_IDX',
    'IDX_TO_COLOR',
    'OBJECT_TYPE_TO_CHANNEL',
    'NUM_BASE_OBJECT_CHANNELS',
    'NUM_OBJECT_TYPE_CHANNELS',
    'DOOR_CHANNEL_START',
    'KEY_CHANNEL_START',
    'MAGICWALL_CHANNEL',
    'AGENT_FEATURE_SIZE',
    'KILLBUTTON_FEATURE_SIZE',
    'PAUSESWITCH_FEATURE_SIZE',
    'DISABLINGSWITCH_FEATURE_SIZE',
    'CONTROLBUTTON_FEATURE_SIZE',
    'NUM_GLOBAL_WORLD_FEATURES',
    'DEFAULT_ACTION_ENCODING',
    'SMALL_ACTION_ENCODING',
    'OVERLAPPABLE_OBJECTS',
    'NON_OVERLAPPABLE_IMMOBILE_OBJECTS',
    'NON_OVERLAPPABLE_MOBILE_OBJECTS',
    # Feature extraction
    'extract_agent_features',
    'extract_all_agent_features',
    'extract_interactive_objects',
    'extract_global_world_features',
    'extract_door_states',
    'extract_magic_wall_states',
    'extract_agent_colors',
    'get_num_agents_per_color',
    # Encoders
    'MultiGridStateEncoder',
    'MultiGridGoalEncoder',
    # Networks
    'MultiGridQNetwork',
    'MultiGridPolicyPriorNetwork',
    'DirectPhiNetwork',
    'MultiGridNeuralHumanPolicyPrior',
    'train_multigrid_neural_policy_prior',
    # Utilities
    'PathDistanceCalculator',
]
