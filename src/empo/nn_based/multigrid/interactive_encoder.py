"""
Interactive object encoder for multigrid environments.

This encoder handles complex interactive objects (buttons, switches) using
list-based encoding. Each object type has a configurable maximum count, and
objects are encoded with all their transition-relevant features.

Object Types:
    - KillButton: Terminates agents of target_color when triggered
    - PauseSwitch: Pauses/unpauses agents of target_color
    - DisablingSwitch: Enables/disables objects of target_type
    - ControlButton: Programs and triggers agent actions
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from .constants import (
    KILLBUTTON_FEATURE_SIZE,
    PAUSESWITCH_FEATURE_SIZE,
    DISABLINGSWITCH_FEATURE_SIZE,
    CONTROLBUTTON_FEATURE_SIZE,
)
from .feature_extraction import extract_interactive_objects


class MultiGridInteractiveObjectEncoder(nn.Module):
    """
    List-based encoder for interactive objects in multigrid environments.
    
    Encodes buttons and switches with all their transition-relevant features.
    Each object type has a configurable maximum count.
    
    Args:
        max_kill_buttons: Maximum number of KillButtons to track.
        max_pause_switches: Maximum number of PauseSwitches to track.
        max_disabling_switches: Maximum number of DisablingSwitches to track.
        max_control_buttons: Maximum number of ControlButtons to track.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        feature_dim: int = 64
    ):
        super().__init__()
        self.max_kill_buttons = max_kill_buttons
        self.max_pause_switches = max_pause_switches
        self.max_disabling_switches = max_disabling_switches
        self.max_control_buttons = max_control_buttons
        self.feature_dim = feature_dim
        
        # Total input size
        self.input_dim = (
            max_kill_buttons * KILLBUTTON_FEATURE_SIZE +
            max_pause_switches * PAUSESWITCH_FEATURE_SIZE +
            max_disabling_switches * DISABLINGSWITCH_FEATURE_SIZE +
            max_control_buttons * CONTROLBUTTON_FEATURE_SIZE
        )
        
        # MLP to produce feature vector
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        kill_buttons: torch.Tensor,
        pause_switches: torch.Tensor,
        disabling_switches: torch.Tensor,
        control_buttons: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode interactive objects into feature vector.
        
        Args:
            kill_buttons: (batch, max_kill_buttons, KILLBUTTON_FEATURE_SIZE)
                Features: [x, y, enabled, trigger_color, target_color]
            pause_switches: (batch, max_pause_switches, PAUSESWITCH_FEATURE_SIZE)
                Features: [x, y, enabled, is_on, toggle_color, target_color]
            disabling_switches: (batch, max_disabling_switches, DISABLINGSWITCH_FEATURE_SIZE)
                Features: [x, y, enabled, is_on, toggle_color, target_type]
            control_buttons: (batch, max_control_buttons, CONTROLBUTTON_FEATURE_SIZE)
                Features: [x, y, enabled, trigger_color, controlled_color, 
                          triggered_action, awaiting_action]
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        batch_size = kill_buttons.shape[0]
        
        # Flatten each object type
        kb_flat = kill_buttons.view(batch_size, -1)
        ps_flat = pause_switches.view(batch_size, -1)
        ds_flat = disabling_switches.view(batch_size, -1)
        cb_flat = control_buttons.view(batch_size, -1)
        
        # Concatenate all features
        all_features = torch.cat([kb_flat, ps_flat, ds_flat, cb_flat], dim=1)
        
        # Apply MLP
        return self.fc(all_features)
    
    def encode_interactive_objects(
        self,
        world_model: Any,
        state: Optional[Tuple] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract and encode interactive objects from environment.
        
        Args:
            world_model: Environment with grid
            state: Optional state tuple for dynamic object states
            device: Torch device
        
        Returns:
            Tuple of tensors ready for forward():
                kill_buttons, pause_switches, disabling_switches, control_buttons
                Each with shape (1, max_count, feature_size)
        """
        kb, ps, ds, cb = extract_interactive_objects(
            world_model=world_model,
            state=state,
            max_kill_buttons=self.max_kill_buttons,
            max_pause_switches=self.max_pause_switches,
            max_disabling_switches=self.max_disabling_switches,
            max_control_buttons=self.max_control_buttons,
            device=device
        )
        
        # Add batch dimension
        return kb.unsqueeze(0), ps.unsqueeze(0), ds.unsqueeze(0), cb.unsqueeze(0)
