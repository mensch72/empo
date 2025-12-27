"""
Multigrid-specific constants for neural network encoding.

This module defines all the constants needed for encoding multigrid environment
states into neural network inputs. This includes object type mappings, color
indices, feature sizes, and channel layouts.

MAINTAINER NOTE:
When adding new object types or features to the multigrid environment,
update the constants in this file and the corresponding encoder implementations.
See docs/ENCODER_ARCHITECTURE.md for the full encoding specification.
"""

# =============================================================================
# COLORS
# =============================================================================

STANDARD_COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey', 'brown']
NUM_STANDARD_COLORS = len(STANDARD_COLORS)
COLOR_TO_IDX = {color: i for i, color in enumerate(STANDARD_COLORS)}
IDX_TO_COLOR = {i: color for i, color in enumerate(STANDARD_COLORS)}

# =============================================================================
# OBJECT TYPE CHANNELS
# =============================================================================

# Base object type channels (simple presence, value = 1.0 if present)
OBJECT_TYPE_TO_CHANNEL = {
    'wall': 0,
    'ball': 1,
    'box': 2,
    'goal': 3,
    'lava': 4,
    'block': 5,
    'rock': 6,
    'unsteadyground': 7,
    'switch': 8,
    'floor': 9,
    'killbutton': 10,
    'pauseswitch': 11,
    'disablingswitch': 12,
    'controlbutton': 13,
}
NUM_BASE_OBJECT_CHANNELS = 14

# Per-color door channels: value encodes state (1=open, 2=closed, 3=locked)
DOOR_CHANNEL_START = NUM_BASE_OBJECT_CHANNELS  # 14
DOOR_STATE_NONE = 0
DOOR_STATE_OPEN = 1
DOOR_STATE_CLOSED = 2
DOOR_STATE_LOCKED = 3

# Per-color key channels: value = 1 if present
KEY_CHANNEL_START = DOOR_CHANNEL_START + NUM_STANDARD_COLORS  # 21

# Magic wall channel: value encodes magic_side (1-5) or inactive (6)
MAGICWALL_CHANNEL = KEY_CHANNEL_START + NUM_STANDARD_COLORS  # 28
MAGICWALL_STATE_NONE = 0
MAGICWALL_STATE_ACTIVE_BASE = 1  # Add magic_side (0-4) to get 1-5
MAGICWALL_STATE_INACTIVE = 6

NUM_OBJECT_TYPE_CHANNELS = MAGICWALL_CHANNEL + 1  # 29

# =============================================================================
# OBJECT CATEGORIES
# =============================================================================

OVERLAPPABLE_OBJECTS = {
    'goal', 'floor', 'switch', 'killbutton',
    'controlbutton', 'unsteadyground', 'objectgoal'
}
NON_OVERLAPPABLE_IMMOBILE_OBJECTS = {
    'wall', 'magicwall', 'lava', 'door', 'pauseswitch', 'disablingswitch'
}
NON_OVERLAPPABLE_MOBILE_OBJECTS = {'block', 'rock'}

# =============================================================================
# AGENT FEATURES
# =============================================================================

# Features per agent (13 total):
# - position (2): x, y (raw integers)
# - direction (4): one-hot encoding of direction 0-3
# - can_enter_magic_walls (1): 0 or 1
# - can_push_rocks (1): 0 or 1
# - carrying_type (1): object type index or -1
# - carrying_color (1): color index or -1
# - paused (1): 0 or 1
# - terminated (1): 0 or 1
# - forced_next_action (1): action index or -1
AGENT_FEATURE_SIZE = 13

# =============================================================================
# INTERACTIVE OBJECT FEATURES
# =============================================================================

# KillButton (5 features): position(2), enabled(1), trigger_color(1), target_color(1)
KILLBUTTON_FEATURE_SIZE = 5

# PauseSwitch (6 features): position(2), enabled(1), is_on(1), toggle_color(1), target_color(1)
PAUSESWITCH_FEATURE_SIZE = 6

# DisablingSwitch (6 features): position(2), enabled(1), is_on(1), toggle_color(1), target_type(1)
DISABLINGSWITCH_FEATURE_SIZE = 6

# ControlButton (7 features): position(2), enabled(1), trigger_color(1), target_color(1), 
#                             forced_action(1), awaiting_action(1)
CONTROLBUTTON_FEATURE_SIZE = 7

# =============================================================================
# GLOBAL WORLD FEATURES
# =============================================================================

# Global features (4): remaining_time, stumble_prob, magic_entry_prob, magic_solidify_prob
NUM_GLOBAL_WORLD_FEATURES = 4

# =============================================================================
# ACTION ENCODINGS
# =============================================================================

DEFAULT_ACTION_ENCODING = {
    0: 'still',
    1: 'left',
    2: 'right',
    3: 'forward',
    4: 'pickup',
    5: 'drop',
    6: 'toggle',
    7: 'done',
}

SMALL_ACTION_ENCODING = {
    0: 'still',
    1: 'left',
    2: 'right',
    3: 'forward',
}

# =============================================================================
# COMPRESSED GRID FORMAT (for replay buffer storage)
# =============================================================================
# 
# Each cell is encoded as a single int32 with the following bit layout:
# Bits 0-4 (5 bits): object_type (0-31, maps to OBJECT_TYPE_TO_CHANNEL or special values)
# Bits 5-7 (3 bits): object_color (0-7, maps to COLOR_TO_IDX)
# Bits 8-9 (2 bits): object_state (0-3, for doors: 0=none, 1=open, 2=closed, 3=locked)
# Bits 10-12 (3 bits): agent_color (0-7, 7 means no agent)
# Bits 13-15 (3 bits): magic_wall_state (0-6, 0=none, 1-5=active with magic_side, 6=inactive)
# Bits 16-18 (3 bits): other_category (0=none, 1=overlappable, 2=immobile, 3=mobile)
#
# Special object_type values:
#   30 = door (color in bits 5-7, state in bits 8-9)
#   31 = key (color in bits 5-7)
#
# This encoding allows reconstructing the full grid tensor without access to world_model.

COMPRESSED_GRID_OBJECT_TYPE_BITS = 5
COMPRESSED_GRID_OBJECT_TYPE_MASK = (1 << COMPRESSED_GRID_OBJECT_TYPE_BITS) - 1  # 0x1F

COMPRESSED_GRID_COLOR_SHIFT = 5
COMPRESSED_GRID_COLOR_BITS = 3
COMPRESSED_GRID_COLOR_MASK = ((1 << COMPRESSED_GRID_COLOR_BITS) - 1) << COMPRESSED_GRID_COLOR_SHIFT  # 0xE0

COMPRESSED_GRID_STATE_SHIFT = 8
COMPRESSED_GRID_STATE_BITS = 2
COMPRESSED_GRID_STATE_MASK = ((1 << COMPRESSED_GRID_STATE_BITS) - 1) << COMPRESSED_GRID_STATE_SHIFT  # 0x300

COMPRESSED_GRID_AGENT_COLOR_SHIFT = 10
COMPRESSED_GRID_AGENT_COLOR_BITS = 3
COMPRESSED_GRID_AGENT_COLOR_MASK = ((1 << COMPRESSED_GRID_AGENT_COLOR_BITS) - 1) << COMPRESSED_GRID_AGENT_COLOR_SHIFT  # 0x1C00
COMPRESSED_GRID_NO_AGENT = 7  # Special value meaning no agent at this cell

COMPRESSED_GRID_MAGIC_SHIFT = 13
COMPRESSED_GRID_MAGIC_BITS = 3
COMPRESSED_GRID_MAGIC_MASK = ((1 << COMPRESSED_GRID_MAGIC_BITS) - 1) << COMPRESSED_GRID_MAGIC_SHIFT  # 0xE000

COMPRESSED_GRID_OTHER_SHIFT = 16
COMPRESSED_GRID_OTHER_BITS = 2
COMPRESSED_GRID_OTHER_MASK = ((1 << COMPRESSED_GRID_OTHER_BITS) - 1) << COMPRESSED_GRID_OTHER_SHIFT  # 0x30000
COMPRESSED_GRID_OTHER_NONE = 0
COMPRESSED_GRID_OTHER_OVERLAPPABLE = 1
COMPRESSED_GRID_OTHER_IMMOBILE = 2
COMPRESSED_GRID_OTHER_MOBILE = 3

# Special object type codes for door and key (which need color)
COMPRESSED_GRID_DOOR_TYPE = 30
COMPRESSED_GRID_KEY_TYPE = 31
