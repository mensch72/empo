"""
Constants for multigrid neural network encoding.

This module defines all the mappings and sizes used for encoding multigrid
environment state into neural network inputs.

Object Type Channels:
    The grid uses separate channels for different object types. Some objects
    have special encoding (doors encode state, keys are per-color, etc.).

Agent Features:
    Each agent is encoded with 13 features capturing position, direction,
    abilities, carried object, and status.

Interactive Object Features:
    Complex interactive objects (buttons, switches) use list-based encoding
    with configurable maximum counts per type.
"""

# =============================================================================
# COLOR CONSTANTS
# =============================================================================

# Standard colors used in multigrid environments (same order as in multigrid.py)
STANDARD_COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey', 'brown']
NUM_STANDARD_COLORS = len(STANDARD_COLORS)
COLOR_TO_IDX = {color: i for i, color in enumerate(STANDARD_COLORS)}
IDX_TO_COLOR = {i: color for i, color in enumerate(STANDARD_COLORS)}

# =============================================================================
# OBJECT TYPE CHANNEL MAPPING
# =============================================================================

# Base object type channels (value = 1.0 if present, 0.0 otherwise)
OBJECT_TYPE_TO_CHANNEL = {
    'wall': 0,
    'ball': 1,
    'box': 2,
    'goal': 3,
    'lava': 4,
    'block': 5,
    'rock': 6,
    'unsteadyground': 7,
    'switch': 8,  # Simple switch (just presence)
    'floor': 9,
    'killbutton': 10,  # Grid presence only, details in list encoder
    'pauseswitch': 11,  # Grid presence only, details in list encoder
    'disablingswitch': 12,  # Grid presence only, details in list encoder
    'controlbutton': 13,  # Grid presence only, details in list encoder
}
NUM_BASE_OBJECT_CHANNELS = 14

# Per-color door channels (one per color): value encodes state
# 0=none, 1=open, 2=closed, 3=locked (raw integers, not normalized)
DOOR_CHANNEL_START = NUM_BASE_OBJECT_CHANNELS  # 14
DOOR_STATE_NONE = 0
DOOR_STATE_OPEN = 1
DOOR_STATE_CLOSED = 2
DOOR_STATE_LOCKED = 3

# Per-color key channels (one per color): value = 1.0 if present
KEY_CHANNEL_START = DOOR_CHANNEL_START + NUM_STANDARD_COLORS  # 21

# Magic wall channel: value encodes state (raw integers)
# 0=none, 1=active magic_side 0 (right), 2=active magic_side 1 (down),
# 3=active magic_side 2 (left), 4=active magic_side 3 (up), 5=inactive/solidified
MAGICWALL_CHANNEL = KEY_CHANNEL_START + NUM_STANDARD_COLORS  # 28
MAGICWALL_STATE_NONE = 0
MAGICWALL_STATE_RIGHT = 1   # magic_side = 0, active
MAGICWALL_STATE_DOWN = 2    # magic_side = 1, active
MAGICWALL_STATE_LEFT = 3    # magic_side = 2, active
MAGICWALL_STATE_UP = 4      # magic_side = 3, active
MAGICWALL_STATE_INACTIVE = 5  # solidified (no longer magic)

NUM_OBJECT_TYPE_CHANNELS = MAGICWALL_CHANNEL + 1  # 29

# =============================================================================
# OBJECT PROPERTY CATEGORIES (for "other objects" fallback channels)
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
# AGENT FEATURE SIZES
# =============================================================================

# Agent features (per agent, all raw values):
# - position (2): raw x, y coordinates
# - direction (4): one-hot encoding (right=0, down=1, left=2, up=3)
# - abilities (2): can_enter_magic_walls (0/1), can_push_rocks (0/1)
# - carried object (2): type_idx (-1 if none), color_idx (-1 if none)
# - status (3): paused (0/1), terminated (0/1), forced_next_action (-1 if none)
AGENT_FEATURE_SIZE = 13

# Direction encoding indices
DIR_RIGHT = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_UP = 3

# =============================================================================
# INTERACTIVE OBJECT FEATURE SIZES
# =============================================================================

# KillButton features (per button):
# - position (2): raw x, y coordinates
# - enabled (1): 0 or 1
# - trigger_color (1): color index (integer)
# - target_color (1): color index (integer)
KILLBUTTON_FEATURE_SIZE = 5

# PauseSwitch features (per switch):
# - position (2): raw x, y coordinates
# - enabled (1): 0 or 1
# - is_on (1): 0 or 1
# - toggle_color (1): color index (integer)
# - target_color (1): color index (integer)
PAUSESWITCH_FEATURE_SIZE = 6

# DisablingSwitch features (per switch):
# - position (2): raw x, y coordinates
# - enabled (1): 0 or 1
# - is_on (1): 0 or 1
# - toggle_color (1): color index (integer)
# - target_type (1): object type index (integer)
DISABLINGSWITCH_FEATURE_SIZE = 6

# ControlButton features (per button):
# - position (2): raw x, y coordinates
# - enabled (1): 0 or 1
# - trigger_color (1): color index (integer)
# - controlled_color (1): color index of controlled agent (integer)
# - triggered_action (1): action index (-1 if not programmed)
# - awaiting_action (1): 0 or 1 (whether in programming mode)
CONTROLBUTTON_FEATURE_SIZE = 7

# =============================================================================
# GLOBAL WORLD FEATURES
# =============================================================================

# Global world parameters that can vary across environments
# These are encoded as a separate global feature vector
# - stumble_prob: probability of stumbling on unsteady ground
# - magic_entry_prob: probability of successfully entering magic wall
# - magic_solidify_prob: probability of magic wall solidifying after entry
# - remaining_time: raw integer (max_steps - step_count)
NUM_GLOBAL_WORLD_FEATURES = 4

# =============================================================================
# DEFAULT ACTION ENCODING
# =============================================================================

# Standard multigrid actions
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

# Small action set (for training on simpler problems)
SMALL_ACTION_ENCODING = {
    0: 'still',
    1: 'left',
    2: 'right',
    3: 'forward',
}

# =============================================================================
# STATE TUPLE FORMAT
# =============================================================================
# State format from multigrid.get_state():
#   (step_count, agent_states, mobile_objects, mutable_objects)
#
# Agent state format:
#   (pos_x, pos_y, dir, terminated, started, paused, 
#    carrying_type, carrying_color, forced_next_action)
#
# Mobile object format:
#   (obj_type, pos_x, pos_y)
#
# Mutable object formats vary by type:
#   Door: ('door', x, y, is_open, is_locked)
#   Box: ('box', x, y, contains_type, contains_color)
#   MagicWall: ('magicwall', x, y, active)
#   KillButton: ('killbutton', x, y, enabled)
#   PauseSwitch: ('pauseswitch', x, y, is_on, enabled)
#   ControlButton: ('controlbutton', x, y, enabled, controlled_agent, triggered_action)

# Agent state tuple indices
AGENT_STATE_POS_X = 0
AGENT_STATE_POS_Y = 1
AGENT_STATE_DIR = 2
AGENT_STATE_TERMINATED = 3
AGENT_STATE_STARTED = 4
AGENT_STATE_PAUSED = 5
AGENT_STATE_CARRYING_TYPE = 6
AGENT_STATE_CARRYING_COLOR = 7
AGENT_STATE_FORCED_ACTION = 8
