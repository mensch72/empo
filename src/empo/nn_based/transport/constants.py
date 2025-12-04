"""
Constants for transport neural network encoders.

This module defines:
- Step type encoding (routing, unboarding, boarding, departing)
- Feature dimensions for nodes, edges, agents, and global state
- Action space constants
"""

from typing import Dict

# =============================================================================
# Step Type Encoding
# =============================================================================

STEP_TYPE_TO_IDX: Dict[str, int] = {
    'routing': 0,
    'unboarding': 1,
    'boarding': 2,
    'departing': 3,
}

IDX_TO_STEP_TYPE: Dict[int, str] = {v: k for k, v in STEP_TYPE_TO_IDX.items()}

NUM_STEP_TYPES = len(STEP_TYPE_TO_IDX)


# =============================================================================
# Feature Dimensions
# =============================================================================

# Node features (per network node):
# - cluster_id_onehot (MAX_CLUSTERS)
# - is_agent_here (1) - whether query agent is at this node
# - num_humans (1) - number of humans at this node
# - num_vehicles (1) - number of vehicles at this node
# - is_destination (1) - whether this is the query agent's destination
# - num_outgoing_edges (1) - number of outgoing edges
# - num_incoming_edges (1) - number of incoming edges
# Total base: 6 + MAX_CLUSTERS

MAX_CLUSTERS = 20
NODE_BASE_FEATURES = 6
NODE_FEATURE_DIM = NODE_BASE_FEATURES + MAX_CLUSTERS  # 26


# Edge features (per network edge):
# - length (1) - edge length/travel time
# - speed_limit (1) - max speed on this edge
# - capacity (1) - vehicle capacity
# - num_agents_on_edge (1) - number of agents currently on this edge
# - is_agent_on_edge (1) - whether query agent is on this edge
# Total: 5

EDGE_FEATURE_DIM = 5


# Global features:
# - step_type_onehot (4)
# - real_time (1) - simulation time
# - num_humans (1) - total humans in env
# - num_vehicles (1) - total vehicles in env
# - use_clusters (1) - whether cluster-based routing is enabled
# - num_clusters (1) - number of clusters (0 if not using clusters)
# Total: 9

GLOBAL_FEATURE_DIM = 9


# Agent features (per agent, for query agent embedding):
# - is_human (1)
# - is_vehicle (1)
# - position_type (2) - [at_node, on_edge]
# - current_node_cluster_onehot (MAX_CLUSTERS) - if at node
# - destination_cluster_onehot (MAX_CLUSTERS) - if vehicle with destination
# - aboard_vehicle (1) - if human is aboard a vehicle
# - has_destination (1) - if vehicle has a destination set
# Total: 6 + 2 * MAX_CLUSTERS = 46

AGENT_BASE_FEATURES = 6
AGENT_FEATURE_DIM = AGENT_BASE_FEATURES + 2 * MAX_CLUSTERS  # 46


# =============================================================================
# Action Space Constants
# =============================================================================

# Number of actions in fixed action space
NUM_TRANSPORT_ACTIONS = 42

# Maximum supported values
MAX_VEHICLES_AT_NODE = 10  # Actions 2-11 for boarding
MAX_DESTINATION_SLOTS = 20  # Actions 12-31 for destinations (clusters or nodes)
MAX_OUTGOING_EDGES = 10  # Actions 32-41 for departing


# =============================================================================
# GNN Constants
# =============================================================================

# Default hidden dimensions
DEFAULT_GNN_HIDDEN_DIM = 128
DEFAULT_GNN_NUM_LAYERS = 3
DEFAULT_OUTPUT_FEATURE_DIM = 128
