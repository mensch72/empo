"""
Feature extraction functions for multigrid environments.

These functions extract features from multigrid environment states and world models
for use in neural network encoding. All values are raw (not normalized).
"""

import torch
from typing import Any, Dict, List, Tuple

from .constants import (
    COLOR_TO_IDX,
    OBJECT_TYPE_TO_CHANNEL,
    AGENT_FEATURE_SIZE,
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
)


def extract_agent_features(
    agent_states: List[Tuple],
    world_model: Any,
    agent_idx: int
) -> torch.Tensor:
    """
    Extract features for a single agent.
    
    Args:
        agent_states: List of agent state tuples from environment state.
        world_model: Environment with agents attribute.
        agent_idx: Index of the agent to extract features for.
    
    Returns:
        Tensor of shape (AGENT_FEATURE_SIZE,) with agent features.
    """
    features = torch.zeros(AGENT_FEATURE_SIZE)
    
    if agent_idx >= len(agent_states):
        return features
    
    agent_state = agent_states[agent_idx]
    
    # Position (2): x, y
    if agent_state[0] is not None:
        features[0] = float(agent_state[0])  # x
        features[1] = float(agent_state[1])  # y
    
    # Direction (4): one-hot
    if len(agent_state) > 2 and agent_state[2] is not None:
        direction = int(agent_state[2])
        if 0 <= direction < 4:
            features[2 + direction] = 1.0
    
    # Get agent object for additional features
    agent = None
    if world_model is not None and hasattr(world_model, 'agents'):
        if agent_idx < len(world_model.agents):
            agent = world_model.agents[agent_idx]
    
    if agent is not None:
        # can_enter_magic_walls (1)
        features[6] = 1.0 if getattr(agent, 'can_enter_magic_walls', False) else 0.0
        
        # can_push_rocks (1)
        features[7] = 1.0 if getattr(agent, 'can_push_rocks', False) else 0.0
        
        # Carrying object (2): type and color
        carrying = getattr(agent, 'carrying', None)
        if carrying is not None:
            obj_type = getattr(carrying, 'type', None)
            if obj_type in OBJECT_TYPE_TO_CHANNEL:
                features[8] = float(OBJECT_TYPE_TO_CHANNEL[obj_type])
            else:
                features[8] = -1.0
            
            obj_color = getattr(carrying, 'color', None)
            if obj_color in COLOR_TO_IDX:
                features[9] = float(COLOR_TO_IDX[obj_color])
            else:
                features[9] = -1.0
        else:
            features[8] = -1.0
            features[9] = -1.0
        
        # paused (1)
        features[10] = 1.0 if getattr(agent, 'paused', False) else 0.0
        
        # terminated (1)
        features[11] = 1.0 if getattr(agent, 'terminated', False) else 0.0
        
        # forced_next_action (1)
        forced = getattr(agent, 'forced_next_action', None)
        features[12] = float(forced) if forced is not None else -1.0
    else:
        features[8] = -1.0  # carrying type
        features[9] = -1.0  # carrying color
        features[12] = -1.0  # forced action
    
    return features


def extract_all_agent_features(
    agent_states: List[Tuple],
    world_model: Any,
    num_agents_per_color: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """
    Extract features for all agents organized by color.
    
    This is agent-agnostic - it extracts features for all agents without
    any query-agent-specific processing. Agent identity is handled separately
    by the AgentIdentityEncoder.
    
    Args:
        agent_states: List of agent state tuples.
        world_model: Environment with agents.
        num_agents_per_color: Dict mapping color to max number of agents of that color.
    
    Returns:
        Dict mapping color to Tensor (num_agents, AGENT_FEATURE_SIZE)
    """
    # Group agents by color
    color_to_agents: Dict[str, List[int]] = {color: [] for color in num_agents_per_color}
    
    if world_model is not None and hasattr(world_model, 'agents'):
        for i, agent in enumerate(world_model.agents):
            color = getattr(agent, 'color', 'grey')
            if color in color_to_agents:
                color_to_agents[color].append(i)
    
    # Extract features for each color group
    color_features = {}
    for color, max_count in num_agents_per_color.items():
        agent_indices = color_to_agents.get(color, [])[:max_count]
        features = torch.zeros(max_count, AGENT_FEATURE_SIZE)
        for i, agent_idx in enumerate(agent_indices):
            features[i] = extract_agent_features(agent_states, world_model, agent_idx)
        color_features[color] = features
    
    return color_features


def extract_interactive_objects(
    state: Tuple,
    world_model: Any,
    max_kill_buttons: int = 4,
    max_pause_switches: int = 4,
    max_disabling_switches: int = 4,
    max_control_buttons: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Extract features for all interactive objects.
    
    Args:
        state: Environment state tuple.
        world_model: Environment with grid.
        max_*: Maximum number of each object type to encode.
    
    Returns:
        Dict with tensors for each object type.
    """
    result = {
        'kill_buttons': torch.zeros(max_kill_buttons, KILLBUTTON_FEATURE_SIZE),
        'pause_switches': torch.zeros(max_pause_switches, PAUSESWITCH_FEATURE_SIZE),
        'disabling_switches': torch.zeros(max_disabling_switches, DISABLINGSWITCH_FEATURE_SIZE),
        'control_buttons': torch.zeros(max_control_buttons, CONTROLBUTTON_FEATURE_SIZE),
    }
    
    if world_model is None or not hasattr(world_model, 'grid') or world_model.grid is None:
        return result
    
    kill_idx = 0
    pause_idx = 0
    disable_idx = 0
    control_idx = 0
    
    H = getattr(world_model.grid, 'height', getattr(world_model, 'height', 10))
    W = getattr(world_model.grid, 'width', getattr(world_model, 'width', 10))
    
    for y in range(H):
        for x in range(W):
            cell = world_model.grid.get(x, y)
            if cell is None:
                continue
            
            cell_type = getattr(cell, 'type', None)
            
            if cell_type == 'killbutton' and kill_idx < max_kill_buttons:
                result['kill_buttons'][kill_idx] = _extract_kill_button(cell, x, y)
                kill_idx += 1
            
            elif cell_type == 'pauseswitch' and pause_idx < max_pause_switches:
                result['pause_switches'][pause_idx] = _extract_pause_switch(cell, x, y)
                pause_idx += 1
            
            elif cell_type == 'disablingswitch' and disable_idx < max_disabling_switches:
                result['disabling_switches'][disable_idx] = _extract_disabling_switch(cell, x, y)
                disable_idx += 1
            
            elif cell_type == 'controlbutton' and control_idx < max_control_buttons:
                result['control_buttons'][control_idx] = _extract_control_button(cell, x, y)
                control_idx += 1
    
    return result


def _extract_kill_button(cell: Any, x: int, y: int) -> torch.Tensor:
    """Extract KillButton features."""
    features = torch.zeros(KILLBUTTON_FEATURE_SIZE)
    features[0] = float(x)
    features[1] = float(y)
    features[2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    
    trigger_color = getattr(cell, 'trigger_color', None)
    features[3] = float(COLOR_TO_IDX.get(trigger_color, -1))
    
    target_color = getattr(cell, 'target_color', None)
    features[4] = float(COLOR_TO_IDX.get(target_color, -1))
    
    return features


def _extract_pause_switch(cell: Any, x: int, y: int) -> torch.Tensor:
    """Extract PauseSwitch features."""
    features = torch.zeros(PAUSESWITCH_FEATURE_SIZE)
    features[0] = float(x)
    features[1] = float(y)
    features[2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    features[3] = 1.0 if getattr(cell, 'is_on', False) else 0.0
    
    toggle_color = getattr(cell, 'toggle_color', None)
    features[4] = float(COLOR_TO_IDX.get(toggle_color, -1))
    
    target_color = getattr(cell, 'target_color', None)
    features[5] = float(COLOR_TO_IDX.get(target_color, -1))
    
    return features


def _extract_disabling_switch(cell: Any, x: int, y: int) -> torch.Tensor:
    """Extract DisablingSwitch features."""
    features = torch.zeros(DISABLINGSWITCH_FEATURE_SIZE)
    features[0] = float(x)
    features[1] = float(y)
    features[2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    features[3] = 1.0 if getattr(cell, 'is_on', False) else 0.0
    
    toggle_color = getattr(cell, 'toggle_color', None)
    features[4] = float(COLOR_TO_IDX.get(toggle_color, -1))
    
    target_type = getattr(cell, 'target_type', None)
    features[5] = float(OBJECT_TYPE_TO_CHANNEL.get(target_type, -1))
    
    return features


def _extract_control_button(cell: Any, x: int, y: int) -> torch.Tensor:
    """Extract ControlButton features."""
    features = torch.zeros(CONTROLBUTTON_FEATURE_SIZE)
    features[0] = float(x)
    features[1] = float(y)
    features[2] = 1.0 if getattr(cell, 'enabled', True) else 0.0
    
    trigger_color = getattr(cell, 'trigger_color', None)
    features[3] = float(COLOR_TO_IDX.get(trigger_color, -1))
    
    target_color = getattr(cell, 'controlled_color', None)
    features[4] = float(COLOR_TO_IDX.get(target_color, -1))
    
    triggered_action = getattr(cell, 'triggered_action', None)
    features[5] = float(triggered_action) if triggered_action is not None else -1.0
    
    # _awaiting_action persists across time steps
    features[6] = 1.0 if getattr(cell, '_awaiting_action', False) else 0.0
    
    return features


def extract_global_world_features(
    state: Tuple,
    world_model: Any,
    device: str = 'cpu',
    include_step_count: bool = True
) -> torch.Tensor:
    """
    Extract global world features.
    
    Returns tensor with: remaining_time, stumble_prob, magic_entry_prob, magic_solidify_prob
    
    Args:
        state: Environment state tuple.
        world_model: Environment with max_steps, stumble_prob, etc.
        device: Torch device.
        include_step_count: If False, set remaining_time to 0 (for debugging time influence).
    """
    features = torch.zeros(4, device=device)
    
    if include_step_count:
        step_count = state[0] if state else 0
        max_steps = getattr(world_model, 'max_steps', 100) if world_model else 100
        features[0] = float(max_steps - step_count)  # remaining_time (raw integer)
    # else: features[0] remains 0
    
    if world_model is not None:
        features[1] = float(getattr(world_model, 'stumble_prob', 0.0))
        features[2] = float(getattr(world_model, 'magic_entry_prob', 1.0))
        features[3] = float(getattr(world_model, 'magic_solidify_prob', 0.0))
    
    return features


def extract_door_states(state: Tuple, world_model: Any) -> Dict[str, Dict[Tuple[int, int], Dict]]:
    """
    Extract door states organized by color.
    
    Returns dict mapping color -> (x, y) -> {is_open, is_locked}
    """
    result: Dict[str, Dict[Tuple[int, int], Dict]] = {}
    
    _, _, _, mutable_objects = state
    
    for obj_data in mutable_objects:
        if obj_data[0] == 'door':
            x, y = obj_data[1], obj_data[2]
            is_open = obj_data[3]
            is_locked = obj_data[4]
            
            # Get color from grid
            color = 'grey'
            if world_model and hasattr(world_model, 'grid'):
                cell = world_model.grid.get(x, y)
                if cell:
                    color = getattr(cell, 'color', 'grey')
            
            if color not in result:
                result[color] = {}
            result[color][(x, y)] = {'is_open': is_open, 'is_locked': is_locked}
    
    return result


def extract_magic_wall_states(state: Tuple, world_model: Any) -> Dict[Tuple[int, int], Dict]:
    """
    Extract magic wall states.
    
    Returns dict mapping (x, y) -> {magic_side, active}
    """
    result: Dict[Tuple[int, int], Dict] = {}
    
    _, _, _, mutable_objects = state
    
    for obj_data in mutable_objects:
        if obj_data[0] == 'magicwall':
            x, y = obj_data[1], obj_data[2]
            active = obj_data[3] if len(obj_data) > 3 else True
            
            magic_side = 0
            if world_model and hasattr(world_model, 'grid'):
                cell = world_model.grid.get(x, y)
                if cell:
                    magic_side = getattr(cell, 'magic_side', 0)
            
            result[(x, y)] = {'magic_side': magic_side, 'active': active}
    
    return result


def extract_agent_colors(world_model: Any) -> List[str]:
    """Extract colors for all agents in order."""
    if world_model is None or not hasattr(world_model, 'agents'):
        return []
    return [getattr(agent, 'color', 'grey') for agent in world_model.agents]


def get_num_agents_per_color(world_model: Any) -> Dict[str, int]:
    """Count agents per color."""
    result: Dict[str, int] = {}
    if world_model is None or not hasattr(world_model, 'agents'):
        return result
    
    for agent in world_model.agents:
        color = getattr(agent, 'color', 'grey')
        result[color] = result.get(color, 0) + 1
    
    return result
