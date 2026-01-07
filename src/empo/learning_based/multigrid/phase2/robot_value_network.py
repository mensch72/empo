"""
Multigrid-specific Robot Value Network for Phase 2.

Implements V_r(s) from equation (9) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ...phase2.robot_value_network import BaseRobotValueNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridRobotValueNetwork(BaseRobotValueNetwork):
    """
    Robot Value Network for multigrid environments.
    
    Estimates V_r(s) = U_r(s) + E_{a_r ~ π_r}[Q_r(s, a_r)] - the robot's
    value function representing expected long-term aggregate human power.
    
    .. warning:: ASYNC TRAINING / PICKLE COMPATIBILITY
    
        This class (via its encoders) is pickled and sent to spawned actor
        processes during async training. See warnings in MultiGridStateEncoder
        for details on maintaining pickle compatibility.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dimension.
        hidden_dim: Hidden layer dimension.
        gamma_r: Robot discount factor.
        dropout: Dropout rate for hidden layers (0 = no dropout).
        max_kill_buttons: Max KillButtons.
        max_pause_switches: Max PauseSwitches.
        max_disabling_switches: Max DisablingSwitches.
        max_control_buttons: Max ControlButtons.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        gamma_r: float = 0.99,
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        state_encoder: Optional[MultiGridStateEncoder] = None,
        own_state_encoder: Optional[MultiGridStateEncoder] = None
    ):
        super().__init__(gamma_r=gamma_r)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        # Use shared state encoder or create own
        if state_encoder is not None:
            self.state_encoder = state_encoder
        else:
            self.state_encoder = MultiGridStateEncoder(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                num_agent_colors=num_agent_colors,
                feature_dim=state_feature_dim,
                max_kill_buttons=max_kill_buttons,
                max_pause_switches=max_pause_switches,
                max_disabling_switches=max_disabling_switches,
                max_control_buttons=max_control_buttons
            )
        
        # Own state encoder for V_r-specific features (trained with V_r loss)
        # This allows V_r to learn additional state features beyond those learned by V_h^e
        # Note: own_state_encoder shares cache with state_encoder to avoid redundant tensorization
        if own_state_encoder is not None:
            self.own_state_encoder = own_state_encoder
        else:
            self.own_state_encoder = MultiGridStateEncoder(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                num_agent_colors=num_agent_colors,
                feature_dim=state_feature_dim,
                max_kill_buttons=max_kill_buttons,
                max_pause_switches=max_pause_switches,
                max_disabling_switches=max_disabling_switches,
                max_control_buttons=max_control_buttons,
                share_cache_with=self.state_encoder
            )
        
        # V_r value head with optional dropout
        # Uses BOTH shared encoder (frozen/detached) AND own encoder (trained with V_r loss)
        actual_state_dim = self.state_encoder.feature_dim + self.own_state_encoder.feature_dim
        if dropout > 0.0:
            self.value_head = nn.Sequential(
                nn.Linear(actual_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(actual_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
    
    def _network_forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Internal: Compute V_r(s) from pre-encoded tensors.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            V_r values tensor (batch,) with V_r < 0.
        """
        # Encode state with SHARED encoder (DETACHED - no gradients flow to shared encoder)
        # The shared state encoder is trained ONLY by V_h^e loss.
        shared_state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        ).detach()
        
        # Encode state with OWN encoder (trained with V_r loss)
        own_state_features = self.own_state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Combine BOTH shared (detached) and own (trainable) features
        state_features = torch.cat([shared_state_features, own_state_features], dim=-1)
        
        # Compute raw value
        raw_value = self.value_head(state_features).squeeze(-1)
        
        # Ensure V_r < 0
        return self.ensure_negative(raw_value)
    
    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute V_r.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            V_r tensor of shape (1,).
        """
        # Encode state (agent-agnostic)
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.tensorize_state(state, world_model, device)
        
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def forward_from_encoded(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with pre-encoded state features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            V_r values tensor (batch,) with V_r < 0.
        """
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict V_r (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            V_r tensor of shape (1,).
        """
        with torch.no_grad():
            return self.forward(state, world_model, device)
    
    def forward_batch(
        self,
        states: List[Any],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states.
        
        Batch-tensorizes all states and computes V_r in a single forward pass.
        This is the primary interface for batched training.
        
        Args:
            states: List of raw environment states.
            world_model: Environment with grid (for tensorization).
            device: Torch device.
        
        Returns:
            V_r values tensor (batch,) with V_r < 0.
        """
        # Batch tensorize states
        grid_list, glob_list, agent_list, inter_list = [], [], [], []
        for state in states:
            grid, glob, agent, inter = self.state_encoder.tensorize_state(state, world_model, device)
            grid_list.append(grid)
            glob_list.append(glob)
            agent_list.append(agent)
            inter_list.append(inter)
        
        grid_tensor = torch.cat(grid_list, dim=0)
        global_features = torch.cat(glob_list, dim=0)
        agent_features = torch.cat(agent_list, dim=0)
        interactive_features = torch.cat(inter_list, dim=0)
        
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'hidden_dim': self.hidden_dim,
            'gamma_r': self.gamma_r,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
        }
