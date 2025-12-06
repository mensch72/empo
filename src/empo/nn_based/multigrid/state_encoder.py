"""
Unified state encoder for multigrid environments.

Encodes the complete world state as seen by a query agent:
- Grid-based spatial information (objects, doors, magic walls)
- Agent features (query agent + per-color agent lists)
- Interactive object features (buttons/switches)
- Global world features

This unified approach keeps all state encoding in one class.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from ..state_encoder import BaseStateEncoder
from .constants import (
    NUM_STANDARD_COLORS,
    COLOR_TO_IDX,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
    DOOR_CHANNEL_START,
    KEY_CHANNEL_START,
    MAGICWALL_CHANNEL,
    DOOR_STATE_OPEN,
    DOOR_STATE_CLOSED,
    DOOR_STATE_LOCKED,
    MAGICWALL_STATE_ACTIVE_BASE,
    MAGICWALL_STATE_INACTIVE,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
    NUM_GLOBAL_WORLD_FEATURES,
    AGENT_FEATURE_SIZE,
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
)
from .feature_extraction import (
    extract_global_world_features,
    extract_all_agent_features,
    extract_interactive_objects,
)


class MultiGridStateEncoder(BaseStateEncoder):
    """
    Unified encoder for multigrid environment states.
    
    Combines:
    1. Grid-based CNN for spatial information (objects, doors, agents by color)
    2. MLP for agent features (query agent + per-color lists)
    3. MLP for interactive objects (buttons/switches)
    4. Global world features
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents of that color.
        num_agent_colors: Number of distinct agent colors for grid channels.
        feature_dim: Total output feature dimension.
        max_kill_buttons: Max KillButtons to encode.
        max_pause_switches: Max PauseSwitches to encode.
        max_disabling_switches: Max DisablingSwitches to encode.
        max_control_buttons: Max ControlButtons to encode.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = NUM_STANDARD_COLORS,
        feature_dim: int = 256,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_agents_per_color = num_agents_per_color
        self.num_agent_colors = num_agent_colors
        self.max_kill_buttons = max_kill_buttons
        self.max_pause_switches = max_pause_switches
        self.max_disabling_switches = max_disabling_switches
        self.max_control_buttons = max_control_buttons
        
        # Grid channel structure
        # Use all standard colors for agent channels to support any color combination
        self.num_object_channels = NUM_OBJECT_TYPE_CHANNELS
        self.num_other_channels = 3  # overlappable, immobile, mobile
        self.agent_channels_start = self.num_object_channels + self.num_other_channels
        self.query_agent_channel = self.agent_channels_start + NUM_STANDARD_COLORS  # Use all colors
        self.num_grid_channels = self.query_agent_channel + 1
        
        # Grid encoder (CNN)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(self.num_grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        grid_conv_out_size = 64 * grid_height * grid_width
        grid_feature_dim = 128
        self.grid_fc = nn.Sequential(
            nn.Linear(grid_conv_out_size + NUM_GLOBAL_WORLD_FEATURES, grid_feature_dim),
            nn.ReLU(),
        )
        
        # Agent encoder (MLP)
        self.color_order = sorted(num_agents_per_color.keys())
        total_agents = sum(num_agents_per_color.values())
        agent_input_size = AGENT_FEATURE_SIZE * (1 + total_agents)  # query + per-color lists
        agent_feature_dim = 64
        self.agent_fc = nn.Sequential(
            nn.Linear(agent_input_size, agent_feature_dim * 2),
            nn.ReLU(),
            nn.Linear(agent_feature_dim * 2, agent_feature_dim),
            nn.ReLU(),
        )
        
        # Interactive object encoder (MLP)
        interactive_input_size = (
            max_kill_buttons * KILLBUTTON_FEATURE_SIZE +
            max_pause_switches * PAUSESWITCH_FEATURE_SIZE +
            max_disabling_switches * DISABLINGSWITCH_FEATURE_SIZE +
            max_control_buttons * CONTROLBUTTON_FEATURE_SIZE
        )
        interactive_feature_dim = 32
        self.interactive_fc = nn.Sequential(
            nn.Linear(interactive_input_size, interactive_feature_dim),
            nn.ReLU(),
        )
        
        # Combined feature dimension
        combined_dim = grid_feature_dim + agent_feature_dim + interactive_feature_dim
        
        # Final projection to feature_dim
        self.output_fc = nn.Sequential(
            nn.Linear(combined_dim, feature_dim),
            nn.ReLU(),
        )
        
        # Store sub-feature dimensions for get_config
        self._grid_feature_dim = grid_feature_dim
        self._agent_feature_dim = agent_feature_dim
        self._interactive_feature_dim = interactive_feature_dim
        self._agent_input_size = agent_input_size
        self._interactive_input_size = interactive_input_size
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode complete state.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        batch_size = grid_tensor.shape[0]
        
        # Grid encoding
        conv_out = self.grid_conv(grid_tensor)
        conv_flat = conv_out.view(batch_size, -1)
        grid_combined = torch.cat([conv_flat, global_features], dim=1)
        grid_emb = self.grid_fc(grid_combined)
        
        # Agent encoding
        agent_emb = self.agent_fc(agent_features)
        
        # Interactive object encoding
        interactive_emb = self.interactive_fc(interactive_features)
        
        # Combine all features
        combined = torch.cat([grid_emb, agent_emb, interactive_emb], dim=1)
        return self.output_fc(combined)
    
    def encode_state(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: int,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single state.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents.
            query_agent_idx: Index of agent being queried.
            device: Torch device.
        
        Returns:
            Tuple of (grid_tensor, global_features, agent_features, interactive_features)
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        # Encode grid
        grid_tensor = torch.zeros(1, self.num_grid_channels, H, W, device=device)
        self._encode_grid(grid_tensor, world_model, agent_states, 
                          mobile_objects, mutable_objects, query_agent_idx)
        
        # Global features
        global_features = extract_global_world_features(state, world_model, device)
        global_features = global_features.unsqueeze(0)
        
        # Agent features
        agent_features = self._encode_agents(agent_states, world_model, query_agent_idx, device)
        
        # Interactive object features
        interactive_features = self._encode_interactive(state, world_model, device)
        
        return grid_tensor, global_features, agent_features, interactive_features
    
    def _encode_grid(
        self,
        grid_tensor: torch.Tensor,
        world_model: Any,
        agent_states: list,
        mobile_objects: list,
        mutable_objects: list,
        query_agent_idx: int
    ):
        """Encode grid objects into tensor.
        
        If world_model has smaller dimensions than the encoder was configured for,
        only the actual world area is encoded and the rest is padded with walls.
        This allows policies trained on larger grids to work on smaller grids.
        """
        H, W = self.grid_height, self.grid_width
        
        # Get actual world dimensions (may be smaller than encoder dimensions)
        actual_height = getattr(world_model, 'height', H) if world_model is not None else H
        actual_width = getattr(world_model, 'width', W) if world_model is not None else W
        
        # Pad area outside actual world with walls (grey walls, channel 0)
        # This allows policies trained on larger grids to work on smaller grids
        if actual_height < H or actual_width < W:
            wall_channel = OBJECT_TYPE_TO_CHANNEL['wall']
            # Fill entire grid with walls first
            grid_tensor[0, wall_channel, :, :] = 1.0
            # Clear the actual world area (will be filled below)
            grid_tensor[0, wall_channel, :actual_height, :actual_width] = 0.0
        
        # Encode grid objects from actual world only
        if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
            for y in range(min(actual_height, H)):
                for x in range(min(actual_width, W)):
                    cell = world_model.grid.get(x, y)
                    if cell is None:
                        continue
                    
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is None:
                        continue
                    
                    if cell_type == 'door':
                        self._encode_door(cell, x, y, mutable_objects, grid_tensor)
                    elif cell_type == 'key':
                        self._encode_key(cell, x, y, grid_tensor)
                    elif cell_type == 'magicwall':
                        self._encode_magic_wall(cell, x, y, mutable_objects, grid_tensor)
                    elif cell_type in OBJECT_TYPE_TO_CHANNEL:
                        channel = OBJECT_TYPE_TO_CHANNEL[cell_type]
                        grid_tensor[0, channel, y, x] = 1.0
                    else:
                        self._encode_other_object(cell, cell_type, x, y, grid_tensor)
        
        # Encode mobile objects
        if mobile_objects:
            for obj_data in mobile_objects:
                obj_type, obj_x, obj_y = obj_data[0], obj_data[1], obj_data[2]
                if 0 <= obj_x < W and 0 <= obj_y < H:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        grid_tensor[0, channel, obj_y, obj_x] = 1.0
                    else:
                        other_mobile_idx = self.num_object_channels + 2
                        grid_tensor[0, other_mobile_idx, obj_y, obj_x] = 1.0
        
        # Encode agents by color
        if world_model is not None and hasattr(world_model, 'agents'):
            for i, agent_state in enumerate(agent_states):
                if agent_state[0] is None:
                    continue
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H and i < len(world_model.agents):
                    color = getattr(world_model.agents[i], 'color', 'grey')
                    if color in COLOR_TO_IDX:
                        channel = self.agent_channels_start + COLOR_TO_IDX[color]
                        grid_tensor[0, channel, y, x] = 1.0
        
        # Encode query agent
        if query_agent_idx is not None and query_agent_idx < len(agent_states):
            agent_state = agent_states[query_agent_idx]
            if agent_state[0] is not None:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    grid_tensor[0, self.query_agent_channel, y, x] = 1.0
    
    def _encode_door(self, cell, x: int, y: int, mutable_objects, grid_tensor: torch.Tensor):
        """Encode door into per-color channel."""
        color = getattr(cell, 'color', None)
        if color not in COLOR_TO_IDX:
            return
        
        channel = DOOR_CHANNEL_START + COLOR_TO_IDX[color]
        
        is_open, is_locked = False, False
        for obj_data in mutable_objects:
            if obj_data[0] == 'door' and obj_data[1] == x and obj_data[2] == y:
                is_open, is_locked = obj_data[3], obj_data[4]
                break
        else:
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
        
        if is_open:
            grid_tensor[0, channel, y, x] = DOOR_STATE_OPEN
        elif is_locked:
            grid_tensor[0, channel, y, x] = DOOR_STATE_LOCKED
        else:
            grid_tensor[0, channel, y, x] = DOOR_STATE_CLOSED
    
    def _encode_key(self, cell, x: int, y: int, grid_tensor: torch.Tensor):
        """Encode key into per-color channel."""
        color = getattr(cell, 'color', None)
        if color not in COLOR_TO_IDX:
            return
        channel = KEY_CHANNEL_START + COLOR_TO_IDX[color]
        grid_tensor[0, channel, y, x] = 1.0
    
    def _encode_magic_wall(self, cell, x: int, y: int, mutable_objects, grid_tensor: torch.Tensor):
        """Encode magic wall with state."""
        active = True
        for obj_data in mutable_objects:
            if obj_data[0] == 'magicwall' and obj_data[1] == x and obj_data[2] == y:
                active = obj_data[3] if len(obj_data) > 3 else True
                break
        else:
            active = getattr(cell, 'active', True)
        
        magic_side = getattr(cell, 'magic_side', 0)
        
        if active:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = MAGICWALL_STATE_ACTIVE_BASE + magic_side
        else:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = MAGICWALL_STATE_INACTIVE
    
    def _encode_other_object(self, cell, cell_type: str, x: int, y: int, grid_tensor: torch.Tensor):
        """Encode object not in main channels."""
        if cell_type in OVERLAPPABLE_OBJECTS:
            channel = self.num_object_channels
        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
            channel = self.num_object_channels + 2
        elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
            channel = self.num_object_channels + 1
        elif hasattr(cell, 'can_overlap') and cell.can_overlap():
            channel = self.num_object_channels
        else:
            channel = self.num_object_channels + 1
        
        grid_tensor[0, channel, y, x] = 1.0
    
    def _encode_agents(
        self,
        agent_states: list,
        world_model: Any,
        query_agent_idx: int,
        device: str
    ) -> torch.Tensor:
        """Encode agent features."""
        query_features, color_features = extract_all_agent_features(
            agent_states, world_model, query_agent_idx, self.num_agents_per_color
        )
        
        all_features = [query_features]
        for color in self.color_order:
            if color in color_features:
                all_features.append(color_features[color].flatten())
        
        return torch.cat(all_features).unsqueeze(0).to(device)
    
    def _encode_interactive(
        self,
        state: Tuple,
        world_model: Any,
        device: str
    ) -> torch.Tensor:
        """Encode interactive object features."""
        objects = extract_interactive_objects(
            state, world_model,
            self.max_kill_buttons,
            self.max_pause_switches,
            self.max_disabling_switches,
            self.max_control_buttons
        )
        
        features = torch.cat([
            objects['kill_buttons'].flatten(),
            objects['pause_switches'].flatten(),
            objects['disabling_switches'].flatten(),
            objects['control_buttons'].flatten(),
        ]).unsqueeze(0).to(device)
        
        return features
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'num_agents_per_color': self.num_agents_per_color,
            'num_agent_colors': self.num_agent_colors,
            'feature_dim': self.feature_dim,
            'max_kill_buttons': self.max_kill_buttons,
            'max_pause_switches': self.max_pause_switches,
            'max_disabling_switches': self.max_disabling_switches,
            'max_control_buttons': self.max_control_buttons,
        }
