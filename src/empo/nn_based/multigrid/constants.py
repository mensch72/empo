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
