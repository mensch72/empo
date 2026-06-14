"""
Constants for BushWorld neural network encoders.

BushWorld is a deliberately minimal grid world, so the encoder needs far fewer
constants than multigrid. The state is encoded as a small multi-channel grid
(bush density + robot occupancy + human occupancy) plus a single global feature
(normalised step count).
"""

# Grid channels produced by :class:`BushWorldStateEncoder`.
DENSITY_CHANNEL = 0  # bush density, normalised to [0, 1]
ROBOT_CHANNEL = 1  # 1.0 where a robot stands, else 0.0
HUMAN_CHANNEL = 2  # 1.0 where a human stands, else 0.0

NUM_GRID_CHANNELS = 3

# Number of global (non-spatial) world features: just the normalised step count.
NUM_GLOBAL_WORLD_FEATURES = 1

# Goal bounding box is encoded as (x1, y1, x2, y2), normalised by grid size.
GOAL_COORD_DIM = 4

# Agent identity features: (normalised index, normalised x, normalised y).
AGENT_FEATURE_SIZE = 3

__all__ = [
    "DENSITY_CHANNEL",
    "ROBOT_CHANNEL",
    "HUMAN_CHANNEL",
    "NUM_GRID_CHANNELS",
    "NUM_GLOBAL_WORLD_FEATURES",
    "GOAL_COORD_DIM",
    "AGENT_FEATURE_SIZE",
]
