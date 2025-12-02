"""
State encoder for multigrid environments.

Encodes grid-based states into feature vectors using a CNN.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

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
)
from .feature_extraction import extract_global_world_features


class MultiGridStateEncoder(BaseStateEncoder):
    """
    CNN-based encoder for multigrid environment states.
    
    Channel layout:
        0-13: Base object types
        14-20: Per-color doors (value encodes state)
        21-27: Per-color keys
        28: Magic walls (value encodes state)
        29-31: "Other objects" (overlappable, immobile, mobile)
        32+: Per-color agent channels
        Last: Query agent channel
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agent_colors: Number of distinct agent colors.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_agent_colors: int = NUM_STANDARD_COLORS,
        feature_dim: int = 128
    ):
        super().__init__(feature_dim)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_agent_colors = num_agent_colors
        
        # Channel structure
        self.num_object_channels = NUM_OBJECT_TYPE_CHANNELS
        self.num_other_channels = 3  # overlappable, immobile, mobile
        self.agent_channels_start = self.num_object_channels + self.num_other_channels
        self.query_agent_channel = self.agent_channels_start + num_agent_colors
        self.num_channels = self.query_agent_channel + 1
        
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        conv_out_size = 64 * grid_height * grid_width
        
        # MLP combining spatial and global features
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
        Encode grid state.
        
        Args:
            grid_tensor: (batch, num_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        batch_size = grid_tensor.shape[0]
        conv_out = self.conv(grid_tensor)
        conv_flat = conv_out.view(batch_size, -1)
        combined = torch.cat([conv_flat, global_features], dim=1)
        return self.fc(combined)
    
    def encode_state(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: Optional[int] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single state.
        
        Args:
            state: (step_count, agent_states, mobile_objects, mutable_objects)
            world_model: Environment with grid and agents.
            query_agent_idx: Index of agent being queried.
            device: Torch device.
        
        Returns:
            (grid_tensor, global_features)
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        grid_tensor = torch.zeros(1, self.num_channels, H, W, device=device)
        
        # Encode grid objects
        if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
            for y in range(H):
                for x in range(W):
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
        
        # Global features
        global_features = extract_global_world_features(state, world_model, device)
        global_features = global_features.unsqueeze(0)
        
        return grid_tensor, global_features
    
    def _encode_door(self, cell, x: int, y: int, mutable_objects, grid_tensor: torch.Tensor):
        """Encode door into per-color channel."""
        color = getattr(cell, 'color', None)
        if color not in COLOR_TO_IDX:
            return
        
        channel = DOOR_CHANNEL_START + COLOR_TO_IDX[color]
        
        # Get state from mutable_objects
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
            channel = self.num_object_channels  # other_overlappable
        elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
            channel = self.num_object_channels + 2  # other_mobile
        elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
            channel = self.num_object_channels + 1  # other_immobile
        elif hasattr(cell, 'can_overlap') and cell.can_overlap():
            channel = self.num_object_channels
        else:
            channel = self.num_object_channels + 1
        
        grid_tensor[0, channel, y, x] = 1.0
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'num_agent_colors': self.num_agent_colors,
            'feature_dim': self.feature_dim,
        }
