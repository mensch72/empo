"""
Multigrid-specific neural network encoders and networks.

This subpackage contains all multigrid-specific implementations for encoding
environment state into neural network inputs. These encoders are designed
specifically for the multigrid environment and handle all its object types,
agent features, and interactive objects.

Main components:
    - constants: Object type mappings, color indices, feature sizes
    - state_encoder: Grid-based CNN encoder for spatial state
    - agent_encoder: List-based encoder for agent features
    - goal_encoder: Goal position encoder
    - interactive_encoder: List-based encoder for buttons/switches
    - feature_extraction: Helper functions to extract features from state/world_model
    - q_network: Multigrid-specific Q-network combining all encoders
    - policy_prior: Complete neural policy prior for multigrid

MAINTAINER NOTE:
When adding new object types or features to the multigrid environment,
the encoders in this package may need to be updated. See docs/ENCODER_ARCHITECTURE.md
for the full encoding specification.
"""

from .constants import (
    STANDARD_COLORS,
    NUM_STANDARD_COLORS,
    COLOR_TO_IDX,
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
)

from .feature_extraction import (
    extract_agent_features,
    extract_interactive_objects,
    extract_global_world_features,
    extract_remaining_time,
    extract_door_states,
    extract_magic_wall_states,
    extract_agent_colors,
)

from .state_encoder import MultiGridStateEncoder
from .agent_encoder import MultiGridAgentEncoder
from .goal_encoder import MultiGridGoalEncoder
from .interactive_encoder import MultiGridInteractiveObjectEncoder
from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .neural_policy_prior import MultiGridNeuralHumanPolicyPrior, train_multigrid_neural_policy_prior

__all__ = [
    # Constants
    'STANDARD_COLORS',
    'NUM_STANDARD_COLORS', 
    'COLOR_TO_IDX',
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
    # Feature extraction
    'extract_agent_features',
    'extract_interactive_objects',
    'extract_global_world_features',
    'extract_remaining_time',
    'extract_door_states',
    'extract_magic_wall_states',
    'extract_agent_colors',
    # Encoders
    'MultiGridStateEncoder',
    'MultiGridAgentEncoder',
    'MultiGridGoalEncoder',
    'MultiGridInteractiveObjectEncoder',
    # Networks
    'MultiGridQNetwork',
    'MultiGridPolicyPriorNetwork',
    'MultiGridNeuralHumanPolicyPrior',
    'train_multigrid_neural_policy_prior',
]
