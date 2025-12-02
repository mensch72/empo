"""
Grid-based state encoder for multigrid environments.

This encoder uses a CNN to process a multi-channel grid representation of the
environment state. Each channel encodes a different aspect of the state.

Channel Structure:
    - Object type channels (29): Walls, doors, keys, magic walls, etc.
    - Other object channels (3): Fallback for unknown object types
    - Per-color agent channels: One channel per agent color
    - Query agent channel: Marks the specific agent being queried
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    STANDARD_COLORS,
    NUM_STANDARD_COLORS,
    COLOR_TO_IDX,
    OBJECT_TYPE_TO_CHANNEL,
    NUM_OBJECT_TYPE_CHANNELS,
    NUM_BASE_OBJECT_CHANNELS,
    DOOR_CHANNEL_START,
    KEY_CHANNEL_START,
    MAGICWALL_CHANNEL,
    OVERLAPPABLE_OBJECTS,
    NON_OVERLAPPABLE_IMMOBILE_OBJECTS,
    NON_OVERLAPPABLE_MOBILE_OBJECTS,
    NUM_GLOBAL_WORLD_FEATURES,
    DOOR_STATE_OPEN,
    DOOR_STATE_CLOSED,
    DOOR_STATE_LOCKED,
)
from .feature_extraction import (
    extract_door_states,
    extract_magic_wall_states,
    extract_key_positions,
    extract_global_world_features,
)


class MultiGridStateEncoder(nn.Module):
    """
    CNN-based encoder for multigrid environment states.
    
    Encodes the grid state into a feature vector using convolutional layers.
    Handles all object types, per-color doors/keys, magic walls, and agents.
    
    Channel Layout:
        0-13: Base object types (wall, ball, box, goal, lava, block, rock, etc.)
        14-20: Per-color doors (value encodes open/closed/locked state)
        21-27: Per-color keys (1.0 if present)
        28: Magic walls (value encodes magic_side and active state)
        29-31: "Other objects" channels (overlappable, immobile, mobile)
        32+: Per-color agent channels
        Last: Query agent channel
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_object_types: Number of object type channels.
        num_agent_colors: Number of distinct agent colors.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
        num_agent_colors: int = NUM_STANDARD_COLORS,
        feature_dim: int = 128
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_object_types = num_object_types
        self.num_agent_colors = num_agent_colors
        self.feature_dim = feature_dim
        
        # Channel structure:
        # - num_object_types: object type channels
        # - 3: "other" object channels (overlappable, immobile, mobile)
        # - num_agent_colors: per-color agent channels
        # - 1: query agent channel
        self.num_other_channels = 3
        self.num_channels = (
            num_object_types + 
            self.num_other_channels + 
            num_agent_colors + 
            1  # query agent
        )
        
        # CNN layers with padding to preserve spatial dimensions
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Compute flattened size after conv
        conv_out_size = 64 * grid_height * grid_width
        
        # MLP to combine spatial features with global features
        # Global features: remaining_time + stumble_prob + magic_entry_prob + magic_solidify_prob
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + NUM_GLOBAL_WORLD_FEATURES, feature_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode grid state into feature vector.
        
        Args:
            grid_tensor: (batch, num_channels, H, W) grid representation
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES) global world features
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        batch_size = grid_tensor.shape[0]
        
        # Apply CNN
        conv_out = self.conv(grid_tensor)
        conv_flat = conv_out.view(batch_size, -1)
        
        # Concatenate with global features
        combined = torch.cat([conv_flat, global_features], dim=1)
        
        # Apply MLP
        features = self.fc(combined)
        
        return features
    
    def encode_state(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_index: Optional[int] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single state from the environment.
        
        Args:
            state: State tuple (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents
            query_agent_index: Index of the agent being queried (for query agent channel)
            device: Torch device
        
        Returns:
            Tuple of (grid_tensor, global_features) ready for forward()
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        grid_tensor = torch.zeros(1, self.num_channels, H, W, device=device)
        
        # Channel indices
        other_overlappable_idx = self.num_object_types
        other_immobile_idx = self.num_object_types + 1
        other_mobile_idx = self.num_object_types + 2
        agent_channels_start = self.num_object_types + self.num_other_channels
        query_agent_channel = agent_channels_start + self.num_agent_colors
        
        # 1. Encode base objects from grid
        if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
            for y in range(H):
                for x in range(W):
                    cell = world_model.grid.get(x, y)
                    if cell is None:
                        continue
                    
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is None:
                        continue
                    
                    # Handle doors specially - they go in per-color channels
                    if cell_type == 'door':
                        self._encode_door(cell, x, y, state, grid_tensor)
                    # Handle keys specially - they go in per-color channels
                    elif cell_type == 'key':
                        self._encode_key(cell, x, y, grid_tensor)
                    # Handle magic walls specially
                    elif cell_type == 'magicwall':
                        self._encode_magic_wall(cell, x, y, state, grid_tensor)
                    # Standard object types
                    elif cell_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                        if channel_idx < self.num_object_types:
                            grid_tensor[0, channel_idx, y, x] = 1.0
                    # Fallback to "other" channels
                    else:
                        if cell_type in OVERLAPPABLE_OBJECTS:
                            grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                            grid_tensor[0, other_mobile_idx, y, x] = 1.0
                        elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                            grid_tensor[0, other_immobile_idx, y, x] = 1.0
                        elif hasattr(cell, 'can_overlap') and cell.can_overlap():
                            grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                        else:
                            grid_tensor[0, other_immobile_idx, y, x] = 1.0
        
        # 2. Encode mobile objects (blocks, rocks)
        if mobile_objects:
            for obj_data in mobile_objects:
                obj_type = obj_data[0]
                obj_x, obj_y = obj_data[1], obj_data[2]
                if 0 <= obj_x < W and 0 <= obj_y < H:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        if channel_idx < self.num_object_types:
                            grid_tensor[0, channel_idx, obj_y, obj_x] = 1.0
                    else:
                        grid_tensor[0, other_mobile_idx, obj_y, obj_x] = 1.0
        
        # 3. Encode agents in per-color channels
        if world_model is not None and hasattr(world_model, 'agents'):
            for i, agent_state in enumerate(agent_states):
                if agent_state[0] is None:
                    continue
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    # Get agent color
                    if i < len(world_model.agents):
                        agent_color = getattr(world_model.agents[i], 'color', 'grey')
                        if agent_color in COLOR_TO_IDX:
                            color_idx = COLOR_TO_IDX[agent_color]
                            channel_idx = agent_channels_start + color_idx
                            grid_tensor[0, channel_idx, y, x] = 1.0
        
        # 4. Encode query agent channel
        if query_agent_index is not None and query_agent_index < len(agent_states):
            agent_state = agent_states[query_agent_index]
            if agent_state[0] is not None:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    grid_tensor[0, query_agent_channel, y, x] = 1.0
        
        # 5. Extract global features
        global_features = extract_global_world_features(state, world_model, device)
        global_features = global_features.unsqueeze(0)  # Add batch dim
        
        return grid_tensor, global_features
    
    def _encode_door(self, cell, x: int, y: int, state: Tuple, grid_tensor: torch.Tensor):
        """Encode a door into the appropriate per-color channel."""
        door_color = getattr(cell, 'color', None)
        if door_color not in COLOR_TO_IDX:
            return
        
        color_idx = COLOR_TO_IDX[door_color]
        channel_idx = DOOR_CHANNEL_START + color_idx
        
        # Get door state from mutable_objects
        _, _, _, mutable_objects = state
        is_open, is_locked = False, False
        
        for obj_data in mutable_objects:
            if obj_data[0] == 'door' and obj_data[1] == x and obj_data[2] == y:
                is_open = obj_data[3]
                is_locked = obj_data[4]
                break
        else:
            # Fallback to cell attributes
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
        
        # Encode state as raw integer
        if is_open:
            grid_tensor[0, channel_idx, y, x] = DOOR_STATE_OPEN
        elif is_locked:
            grid_tensor[0, channel_idx, y, x] = DOOR_STATE_LOCKED
        else:
            grid_tensor[0, channel_idx, y, x] = DOOR_STATE_CLOSED
    
    def _encode_key(self, cell, x: int, y: int, grid_tensor: torch.Tensor):
        """Encode a key into the appropriate per-color channel."""
        key_color = getattr(cell, 'color', None)
        if key_color not in COLOR_TO_IDX:
            return
        
        color_idx = COLOR_TO_IDX[key_color]
        channel_idx = KEY_CHANNEL_START + color_idx
        grid_tensor[0, channel_idx, y, x] = 1.0
    
    def _encode_magic_wall(self, cell, x: int, y: int, state: Tuple, grid_tensor: torch.Tensor):
        """Encode a magic wall with its state."""
        _, _, _, mutable_objects = state
        active = True
        
        for obj_data in mutable_objects:
            if obj_data[0] == 'magicwall' and obj_data[1] == x and obj_data[2] == y:
                active = obj_data[3]
                break
        else:
            active = getattr(cell, 'active', True)
        
        magic_side = getattr(cell, 'magic_side', 0)
        
        # Encode as raw integer value
        if not active:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = 5.0  # inactive
        else:
            grid_tensor[0, MAGICWALL_CHANNEL, y, x] = float(magic_side + 1)  # 1-4 for sides
