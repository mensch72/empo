"""
Interactive object encoder for multigrid environments.

Encodes buttons and switches using list-based approach.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from .constants import (
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
)
from .feature_extraction import extract_interactive_objects


class MultiGridInteractiveObjectEncoder(nn.Module):
    """
    Encoder for interactive objects (buttons/switches).
    
    Uses list-based encoding with configurable max counts per type.
    
    Args:
        max_kill_buttons: Max KillButtons to encode.
        max_pause_switches: Max PauseSwitches to encode.
        max_disabling_switches: Max DisablingSwitches to encode.
        max_control_buttons: Max ControlButtons to encode.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        feature_dim: int = 32
    ):
        super().__init__()
        self.max_kill_buttons = max_kill_buttons
        self.max_pause_switches = max_pause_switches
        self.max_disabling_switches = max_disabling_switches
        self.max_control_buttons = max_control_buttons
        self.feature_dim = feature_dim
        
        # Calculate input size
        input_size = (
            max_kill_buttons * KILLBUTTON_FEATURE_SIZE +
            max_pause_switches * PAUSESWITCH_FEATURE_SIZE +
            max_disabling_switches * DISABLINGSWITCH_FEATURE_SIZE +
            max_control_buttons * CONTROLBUTTON_FEATURE_SIZE
        )
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, object_features: torch.Tensor) -> torch.Tensor:
        """
        Encode interactive object features.
        
        Args:
            object_features: (batch, input_size) flattened features
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        return self.fc(object_features)
    
    def encode_objects(
        self,
        state: Tuple,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode interactive objects from state.
        
        Args:
            state: Environment state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Tensor (1, input_size) ready for forward().
        """
        objects = extract_interactive_objects(
            state, world_model,
            self.max_kill_buttons,
            self.max_pause_switches,
            self.max_disabling_switches,
            self.max_control_buttons
        )
        
        # Flatten all object features
        features = torch.cat([
            objects['kill_buttons'].flatten(),
            objects['pause_switches'].flatten(),
            objects['disabling_switches'].flatten(),
            objects['control_buttons'].flatten(),
        ]).unsqueeze(0).to(device)
        
        return features
    
    def get_input_size(self) -> int:
        """Return the input feature size."""
        return (
            self.max_kill_buttons * KILLBUTTON_FEATURE_SIZE +
            self.max_pause_switches * PAUSESWITCH_FEATURE_SIZE +
            self.max_disabling_switches * DISABLINGSWITCH_FEATURE_SIZE +
            self.max_control_buttons * CONTROLBUTTON_FEATURE_SIZE
        )
