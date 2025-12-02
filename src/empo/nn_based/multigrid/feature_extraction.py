"""
Feature extraction functions for multigrid environment.

This module provides functions to extract ALL transition-relevant features
from the multigrid state tuple and world_model. These functions are used
by the encoders to convert environment state into tensor representations.

IMPORTANT: These functions extract ACTUAL values, not placeholders.
All features that can influence or change during a transition must be extracted.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    STANDARD_COLORS,
    COLOR_TO_IDX,
    OBJECT_TYPE_TO_CHANNEL,
    AGENT_FEATURE_SIZE,
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
    NUM_GLOBAL_WORLD_FEATURES,
    DOOR_STATE_NONE,
    DOOR_STATE_OPEN,
    DOOR_STATE_CLOSED,
    DOOR_STATE_LOCKED,
    MAGICWALL_STATE_NONE,
    MAGICWALL_STATE_RIGHT,
    MAGICWALL_STATE_DOWN,
    MAGICWALL_STATE_LEFT,
    MAGICWALL_STATE_UP,
    MAGICWALL_STATE_INACTIVE,
    AGENT_STATE_POS_X,
    AGENT_STATE_POS_Y,
    AGENT_STATE_DIR,
    AGENT_STATE_TERMINATED,
    AGENT_STATE_STARTED,
    AGENT_STATE_PAUSED,
    AGENT_STATE_CARRYING_TYPE,
    AGENT_STATE_CARRYING_COLOR,
    AGENT_STATE_FORCED_ACTION,
)


def extract_agent_features(
    state: Tuple,
    world_model: Any,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract ALL agent features from state tuple and world_model.
    
    This extracts the complete set of 13 features per agent:
    - Position (2): raw x, y coordinates
    - Direction (4): one-hot encoding
    - Abilities (2): can_enter_magic_walls, can_push_rocks
    - Carried object (2): type_idx, color_idx
    - Status (3): paused, terminated, forced_next_action
    
    Args:
        state: State tuple (step_count, agent_states, mobile_objects, mutable_objects)
        world_model: Environment with agents list for abilities and colors
        device: Torch device
    
    Returns:
        Tuple of tensors:
            - positions: (num_agents, 2) raw x, y
            - directions: (num_agents, 4) one-hot
            - abilities: (num_agents, 2) can_enter_magic_walls, can_push_rocks
            - carried: (num_agents, 2) type_idx, color_idx
            - status: (num_agents, 3) paused, terminated, forced_next_action
            - colors: (num_agents,) color indices
    """
    _, agent_states, _, _ = state
    num_agents = len(agent_states)
    
    positions = torch.zeros(num_agents, 2, device=device)
    directions = torch.zeros(num_agents, 4, device=device)
    abilities = torch.zeros(num_agents, 2, device=device)
    carried = torch.zeros(num_agents, 2, device=device)
    carried[:, :] = -1.0  # Default: nothing carried
    status = torch.zeros(num_agents, 3, device=device)
    status[:, 2] = -1.0  # Default: no forced action
    colors = torch.zeros(num_agents, dtype=torch.long, device=device)
    
    for i, agent_state in enumerate(agent_states):
        # Position (raw)
        if agent_state[AGENT_STATE_POS_X] is not None:
            positions[i, 0] = float(agent_state[AGENT_STATE_POS_X])
            positions[i, 1] = float(agent_state[AGENT_STATE_POS_Y])
        
        # Direction (one-hot)
        if agent_state[AGENT_STATE_DIR] is not None:
            dir_idx = int(agent_state[AGENT_STATE_DIR]) % 4
            directions[i, dir_idx] = 1.0
        
        # Status from state tuple
        status[i, 0] = 1.0 if agent_state[AGENT_STATE_PAUSED] else 0.0
        status[i, 1] = 1.0 if agent_state[AGENT_STATE_TERMINATED] else 0.0
        if agent_state[AGENT_STATE_FORCED_ACTION] is not None:
            status[i, 2] = float(agent_state[AGENT_STATE_FORCED_ACTION])
        
        # Carried object from state tuple
        carrying_type = agent_state[AGENT_STATE_CARRYING_TYPE]
        carrying_color = agent_state[AGENT_STATE_CARRYING_COLOR]
        
        if carrying_type is not None:
            if carrying_type in OBJECT_TYPE_TO_CHANNEL:
                carried[i, 0] = float(OBJECT_TYPE_TO_CHANNEL[carrying_type])
        
        if carrying_color is not None:
            if carrying_color in COLOR_TO_IDX:
                carried[i, 1] = float(COLOR_TO_IDX[carrying_color])
        
        # Abilities and color from world_model.agents
        if world_model is not None and hasattr(world_model, 'agents') and i < len(world_model.agents):
            agent = world_model.agents[i]
            abilities[i, 0] = 1.0 if getattr(agent, 'can_enter_magic_walls', False) else 0.0
            abilities[i, 1] = 1.0 if getattr(agent, 'can_push_rocks', False) else 0.0
            
            agent_color = getattr(agent, 'color', None)
            if agent_color in COLOR_TO_IDX:
                colors[i] = COLOR_TO_IDX[agent_color]
    
    return positions, directions, abilities, carried, status, colors


def extract_agent_colors(world_model: Any) -> List[str]:
    """
    Extract agent color strings from world_model.
    
    Args:
        world_model: Environment with agents list
    
    Returns:
        List of color strings, one per agent
    """
    if world_model is None or not hasattr(world_model, 'agents'):
        return []
    
    colors = []
    for agent in world_model.agents:
        color = getattr(agent, 'color', 'grey')
        colors.append(color)
    return colors


def extract_interactive_objects(
    world_model: Any,
    state: Optional[Tuple] = None,
    max_kill_buttons: int = 4,
    max_pause_switches: int = 4,
    max_disabling_switches: int = 4,
    max_control_buttons: int = 4,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract ALL interactive object features from world_model and state.
    
    This combines information from:
    1. world_model.grid - for object positions and static attributes
    2. state mutable_objects - for dynamic state (enabled, is_on, etc.)
    
    Args:
        world_model: Environment with grid attribute
        state: Optional state tuple for mutable object states
        max_kill_buttons: Maximum number of KillButtons to track
        max_pause_switches: Maximum number of PauseSwitches to track
        max_disabling_switches: Maximum number of DisablingSwitches to track
        max_control_buttons: Maximum number of ControlButtons to track
        device: Torch device
    
    Returns:
        Tuple of tensors:
            - kill_buttons: (max_kill_buttons, KILLBUTTON_FEATURE_SIZE)
            - pause_switches: (max_pause_switches, PAUSESWITCH_FEATURE_SIZE)
            - disabling_switches: (max_disabling_switches, DISABLINGSWITCH_FEATURE_SIZE)
            - control_buttons: (max_control_buttons, CONTROLBUTTON_FEATURE_SIZE)
    """
    kill_buttons = torch.zeros(max_kill_buttons, KILLBUTTON_FEATURE_SIZE, device=device)
    pause_switches = torch.zeros(max_pause_switches, PAUSESWITCH_FEATURE_SIZE, device=device)
    disabling_switches = torch.zeros(max_disabling_switches, DISABLINGSWITCH_FEATURE_SIZE, device=device)
    control_buttons = torch.zeros(max_control_buttons, CONTROLBUTTON_FEATURE_SIZE, device=device)
    
    # Set default forced_action to -1 (not programmed)
    control_buttons[:, 5] = -1.0
    
    if world_model is None or not hasattr(world_model, 'grid') or world_model.grid is None:
        return kill_buttons, pause_switches, disabling_switches, control_buttons
    
    # Build lookup from state mutable_objects for dynamic state
    mutable_state = {}
    if state is not None:
        _, _, _, mutable_objects = state
        for obj_data in mutable_objects:
            obj_type = obj_data[0]
            x, y = obj_data[1], obj_data[2]
            mutable_state[(x, y, obj_type)] = obj_data
    
    kb_count = 0
    ps_count = 0
    ds_count = 0
    cb_count = 0
    
    for y in range(world_model.height):
        for x in range(world_model.width):
            cell = world_model.grid.get(x, y)
            if cell is None:
                continue
            
            cell_type = getattr(cell, 'type', None)
            
            if cell_type == 'killbutton' and kb_count < max_kill_buttons:
                _encode_kill_button(cell, x, y, mutable_state, kill_buttons, kb_count)
                kb_count += 1
            
            elif cell_type == 'pauseswitch' and ps_count < max_pause_switches:
                _encode_pause_switch(cell, x, y, mutable_state, pause_switches, ps_count)
                ps_count += 1
            
            elif cell_type == 'disablingswitch' and ds_count < max_disabling_switches:
                _encode_disabling_switch(cell, x, y, mutable_state, disabling_switches, ds_count)
                ds_count += 1
            
            elif cell_type == 'controlbutton' and cb_count < max_control_buttons:
                _encode_control_button(cell, x, y, mutable_state, control_buttons, cb_count)
                cb_count += 1
    
    return kill_buttons, pause_switches, disabling_switches, control_buttons


def _encode_kill_button(cell, x: int, y: int, mutable_state: dict, 
                        tensor: torch.Tensor, idx: int):
    """Encode a KillButton into the tensor."""
    tensor[idx, 0] = float(x)
    tensor[idx, 1] = float(y)
    
    # Get enabled state from mutable_state if available, else from cell
    key = (x, y, 'killbutton')
    if key in mutable_state:
        # mutable format: ('killbutton', x, y, enabled)
        tensor[idx, 2] = 1.0 if mutable_state[key][3] else 0.0
    else:
        tensor[idx, 2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    
    # Trigger and target colors
    trigger_color = getattr(cell, 'trigger_color', None)
    if trigger_color in COLOR_TO_IDX:
        tensor[idx, 3] = float(COLOR_TO_IDX[trigger_color])
    
    target_color = getattr(cell, 'target_color', None)
    if target_color in COLOR_TO_IDX:
        tensor[idx, 4] = float(COLOR_TO_IDX[target_color])


def _encode_pause_switch(cell, x: int, y: int, mutable_state: dict,
                         tensor: torch.Tensor, idx: int):
    """Encode a PauseSwitch into the tensor."""
    tensor[idx, 0] = float(x)
    tensor[idx, 1] = float(y)
    
    # Get dynamic state from mutable_state if available
    key = (x, y, 'pauseswitch')
    if key in mutable_state:
        # mutable format: ('pauseswitch', x, y, is_on, enabled)
        tensor[idx, 2] = 1.0 if mutable_state[key][4] else 0.0  # enabled
        tensor[idx, 3] = 1.0 if mutable_state[key][3] else 0.0  # is_on
    else:
        tensor[idx, 2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
        tensor[idx, 3] = 1.0 if getattr(cell, 'is_on', False) else 0.0
    
    # Toggle and target colors
    toggle_color = getattr(cell, 'toggle_color', None)
    if toggle_color in COLOR_TO_IDX:
        tensor[idx, 4] = float(COLOR_TO_IDX[toggle_color])
    
    target_color = getattr(cell, 'target_color', None)
    if target_color in COLOR_TO_IDX:
        tensor[idx, 5] = float(COLOR_TO_IDX[target_color])


def _encode_disabling_switch(cell, x: int, y: int, mutable_state: dict,
                             tensor: torch.Tensor, idx: int):
    """Encode a DisablingSwitch into the tensor."""
    tensor[idx, 0] = float(x)
    tensor[idx, 1] = float(y)
    
    # DisablingSwitch doesn't appear in mutable_objects in current implementation
    # so we read directly from cell
    tensor[idx, 2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    tensor[idx, 3] = 1.0 if getattr(cell, 'is_on', False) else 0.0
    
    toggle_color = getattr(cell, 'toggle_color', None)
    if toggle_color in COLOR_TO_IDX:
        tensor[idx, 4] = float(COLOR_TO_IDX[toggle_color])
    
    target_type = getattr(cell, 'target_type', None)
    if target_type in OBJECT_TYPE_TO_CHANNEL:
        tensor[idx, 5] = float(OBJECT_TYPE_TO_CHANNEL[target_type])


def _encode_control_button(cell, x: int, y: int, mutable_state: dict,
                           tensor: torch.Tensor, idx: int):
    """Encode a ControlButton into the tensor."""
    tensor[idx, 0] = float(x)
    tensor[idx, 1] = float(y)
    
    # Get dynamic state from mutable_state if available
    key = (x, y, 'controlbutton')
    if key in mutable_state:
        # mutable format: ('controlbutton', x, y, enabled, controlled_agent, triggered_action)
        tensor[idx, 2] = 1.0 if mutable_state[key][3] else 0.0  # enabled
        # controlled_agent is stored but we encode controlled_color from cell
        triggered_action = mutable_state[key][5]
        tensor[idx, 5] = float(triggered_action) if triggered_action is not None else -1.0
    else:
        tensor[idx, 2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
        triggered_action = getattr(cell, 'triggered_action', None)
        tensor[idx, 5] = float(triggered_action) if triggered_action is not None else -1.0
    
    # Trigger and controlled colors (from cell attributes)
    trigger_color = getattr(cell, 'trigger_color', None)
    if trigger_color in COLOR_TO_IDX:
        tensor[idx, 3] = float(COLOR_TO_IDX[trigger_color])
    
    controlled_color = getattr(cell, 'controlled_color', None)
    if controlled_color in COLOR_TO_IDX:
        tensor[idx, 4] = float(COLOR_TO_IDX[controlled_color])
    
    # _awaiting_action - critical transient state that persists across steps
    tensor[idx, 6] = 1.0 if getattr(cell, '_awaiting_action', False) else 0.0


def extract_global_world_features(
    state: Tuple,
    world_model: Any,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract global world parameters that influence transitions.
    
    Returns a tensor with:
    - stumble_prob: probability of stumbling on unsteady ground
    - magic_entry_prob: probability of successfully entering magic wall
    - magic_solidify_prob: probability of magic wall solidifying
    - remaining_time: raw integer (max_steps - step_count)
    
    Args:
        state: State tuple (step_count, ...)
        world_model: Environment with global parameters
        device: Torch device
    
    Returns:
        global_features: (NUM_GLOBAL_WORLD_FEATURES,) tensor
    """
    global_features = torch.zeros(NUM_GLOBAL_WORLD_FEATURES, device=device)
    
    # Extract probabilities from world_model
    if world_model is not None:
        global_features[0] = float(getattr(world_model, 'stumble_prob', 0.0))
        global_features[1] = float(getattr(world_model, 'magic_entry_prob', 0.0))
        global_features[2] = float(getattr(world_model, 'magic_solidify_prob', 0.0))
    
    # Remaining time (raw integer)
    step_count = state[0]
    max_steps = getattr(world_model, 'max_steps', 100) if world_model else 100
    global_features[3] = float(max_steps - step_count)
    
    return global_features


def extract_remaining_time(state: Tuple, world_model: Any) -> int:
    """
    Extract remaining time steps from state.
    
    Args:
        state: State tuple (step_count, ...)
        world_model: Environment with max_steps attribute
    
    Returns:
        remaining_time: max_steps - step_count (raw integer)
    """
    step_count = state[0]
    max_steps = getattr(world_model, 'max_steps', 100) if world_model else 100
    return max_steps - step_count


def extract_door_states(
    world_model: Any,
    state: Optional[Tuple] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract per-color door states from world_model and state.
    
    Args:
        world_model: Environment with grid
        state: Optional state tuple for mutable door states
        device: Torch device
    
    Returns:
        door_states: (num_colors, height, width) tensor with door state values
    """
    if world_model is None or not hasattr(world_model, 'grid'):
        return torch.zeros(len(STANDARD_COLORS), 1, 1, device=device)
    
    H, W = world_model.height, world_model.width
    door_states = torch.zeros(len(STANDARD_COLORS), H, W, device=device)
    
    # Build lookup from state mutable_objects
    door_state_lookup = {}
    if state is not None:
        _, _, _, mutable_objects = state
        for obj_data in mutable_objects:
            if obj_data[0] == 'door':
                # ('door', x, y, is_open, is_locked)
                x, y = obj_data[1], obj_data[2]
                is_open, is_locked = obj_data[3], obj_data[4]
                door_state_lookup[(x, y)] = (is_open, is_locked)
    
    # Scan grid for doors
    for y in range(H):
        for x in range(W):
            cell = world_model.grid.get(x, y)
            if cell is None or getattr(cell, 'type', None) != 'door':
                continue
            
            door_color = getattr(cell, 'color', None)
            if door_color not in COLOR_TO_IDX:
                continue
            
            color_idx = COLOR_TO_IDX[door_color]
            
            # Get state from lookup or cell
            if (x, y) in door_state_lookup:
                is_open, is_locked = door_state_lookup[(x, y)]
            else:
                is_open = getattr(cell, 'is_open', False)
                is_locked = getattr(cell, 'is_locked', False)
            
            # Encode state
            if is_open:
                door_states[color_idx, y, x] = DOOR_STATE_OPEN
            elif is_locked:
                door_states[color_idx, y, x] = DOOR_STATE_LOCKED
            else:
                door_states[color_idx, y, x] = DOOR_STATE_CLOSED
    
    return door_states


def extract_magic_wall_states(
    world_model: Any,
    state: Optional[Tuple] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract magic wall states from world_model and state.
    
    Args:
        world_model: Environment with grid
        state: Optional state tuple for mutable magic wall states
        device: Torch device
    
    Returns:
        magic_walls: (height, width) tensor with magic wall state values
    """
    if world_model is None or not hasattr(world_model, 'grid'):
        return torch.zeros(1, 1, device=device)
    
    H, W = world_model.height, world_model.width
    magic_walls = torch.zeros(H, W, device=device)
    
    # Build lookup from state mutable_objects
    magic_wall_lookup = {}
    if state is not None:
        _, _, _, mutable_objects = state
        for obj_data in mutable_objects:
            if obj_data[0] == 'magicwall':
                # ('magicwall', x, y, active)
                x, y = obj_data[1], obj_data[2]
                active = obj_data[3]
                magic_wall_lookup[(x, y)] = active
    
    # Scan grid for magic walls
    for y in range(H):
        for x in range(W):
            cell = world_model.grid.get(x, y)
            if cell is None or getattr(cell, 'type', None) != 'magicwall':
                continue
            
            # Get active state from lookup or cell
            if (x, y) in magic_wall_lookup:
                active = magic_wall_lookup[(x, y)]
            else:
                active = getattr(cell, 'active', True)
            
            if not active:
                magic_walls[y, x] = MAGICWALL_STATE_INACTIVE
            else:
                # Encode magic_side
                magic_side = getattr(cell, 'magic_side', 0)
                if magic_side == 0:
                    magic_walls[y, x] = MAGICWALL_STATE_RIGHT
                elif magic_side == 1:
                    magic_walls[y, x] = MAGICWALL_STATE_DOWN
                elif magic_side == 2:
                    magic_walls[y, x] = MAGICWALL_STATE_LEFT
                elif magic_side == 3:
                    magic_walls[y, x] = MAGICWALL_STATE_UP
                else:
                    magic_walls[y, x] = MAGICWALL_STATE_RIGHT  # default
    
    return magic_walls


def extract_key_positions(
    world_model: Any,
    state: Optional[Tuple] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract per-color key positions from world_model.
    
    Args:
        world_model: Environment with grid
        state: Optional state tuple (keys are in mobile_objects if mobile)
        device: Torch device
    
    Returns:
        keys: (num_colors, height, width) tensor with key presence (1.0 if present)
    """
    if world_model is None or not hasattr(world_model, 'grid'):
        return torch.zeros(len(STANDARD_COLORS), 1, 1, device=device)
    
    H, W = world_model.height, world_model.width
    keys = torch.zeros(len(STANDARD_COLORS), H, W, device=device)
    
    # Scan grid for keys
    for y in range(H):
        for x in range(W):
            cell = world_model.grid.get(x, y)
            if cell is None or getattr(cell, 'type', None) != 'key':
                continue
            
            key_color = getattr(cell, 'color', None)
            if key_color in COLOR_TO_IDX:
                color_idx = COLOR_TO_IDX[key_color]
                keys[color_idx, y, x] = 1.0
    
    # Also check mobile_objects in state for keys that have been moved
    if state is not None:
        _, _, mobile_objects, _ = state
        for obj_data in mobile_objects:
            if obj_data[0] == 'key':
                x, y = obj_data[1], obj_data[2]
                if 0 <= x < W and 0 <= y < H:
                    # Color would need to be in mobile_objects format
                    # Currently mobile_objects is (type, x, y) - no color
                    # Keys are typically not mobile, so this may not apply
                    pass
    
    return keys
