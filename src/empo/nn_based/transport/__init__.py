"""
Transport-specific neural network encoders and policy components.

This subpackage contains all transport-specific implementations for encoding
environment state into neural network inputs using Graph Neural Networks (GNNs)
instead of CNNs (as used in multigrid).

The transport environment is graph-based (road network), so we use GNNs to
process node and edge features. The encoders support both node-based and
cluster-based routing modes.

Main components:
    - constants: Feature dimensions, step type encoding, action encoding
    - feature_extraction: Extract features from transport env state
    - state_encoder: GNN-based state encoder for network topology
    - goal_encoder: Encode node/cluster goals
    - q_network: Q-network combining state and goal encoders

Example usage:
    >>> from empo.nn_based.transport import (
    ...     TransportStateEncoder,
    ...     TransportGoalEncoder,
    ...     TransportQNetwork,
    ...     observation_to_graph_data,
    ... )
    >>> 
    >>> # Create encoders
    >>> state_encoder = TransportStateEncoder(
    ...     num_clusters=10, max_nodes=100, feature_dim=128
    ... )
    >>> goal_encoder = TransportGoalEncoder(max_nodes=100, num_clusters=10)
    >>> 
    >>> # Create Q-network
    >>> q_network = TransportQNetwork(
    ...     state_encoder=state_encoder,
    ...     goal_encoder=goal_encoder,
    ...     num_actions=42
    ... )
"""

from .constants import (
    # Step type encoding
    STEP_TYPE_TO_IDX,
    IDX_TO_STEP_TYPE,
    NUM_STEP_TYPES,
    # Feature dimensions
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    AGENT_FEATURE_DIM,
    # Action encoding
    NUM_TRANSPORT_ACTIONS,
    MAX_VEHICLES_AT_NODE,
    MAX_OUTGOING_EDGES,
    MAX_CLUSTERS,
)

from .feature_extraction import (
    extract_node_features,
    extract_edge_features,
    extract_global_features,
    extract_agent_features,
    observation_to_graph_data,
)

from .state_encoder import TransportStateEncoder
from .goal_encoder import TransportGoalEncoder
from .q_network import TransportQNetwork

__all__ = [
    # Constants
    'STEP_TYPE_TO_IDX',
    'IDX_TO_STEP_TYPE',
    'NUM_STEP_TYPES',
    'NODE_FEATURE_DIM',
    'EDGE_FEATURE_DIM',
    'GLOBAL_FEATURE_DIM',
    'AGENT_FEATURE_DIM',
    'NUM_TRANSPORT_ACTIONS',
    'MAX_VEHICLES_AT_NODE',
    'MAX_OUTGOING_EDGES',
    'MAX_CLUSTERS',
    # Feature extraction
    'extract_node_features',
    'extract_edge_features',
    'extract_global_features',
    'extract_agent_features',
    'observation_to_graph_data',
    # Encoders
    'TransportStateEncoder',
    'TransportGoalEncoder',
    # Networks
    'TransportQNetwork',
]
