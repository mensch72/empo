"""
Multigrid-specific Intrinsic Reward Network for Phase 2.

Implements U_r(s) from equation (8) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ...phase2.intrinsic_reward_network import BaseIntrinsicRewardNetwork
from ..state_encoder import MultiGridStateEncoder


class MultiGridIntrinsicRewardNetwork(BaseIntrinsicRewardNetwork):
    """
    Intrinsic Reward Network for multigrid environments.
    
    Estimates U_r(s) = -(E_h[X_h(s)^{-ξ}])^η - the robot's intrinsic reward
    based on aggregate human power.
    
    Network predicts log(y-1) where y = E_h[X_h^{-ξ}], then computes U_r = -y^η.
    
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
        xi: Inter-human inequality aversion parameter.
        eta: Intertemporal inequality aversion parameter.
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
        xi: float = 1.0,
        eta: float = 1.1,
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        state_encoder: Optional[MultiGridStateEncoder] = None,
        own_state_encoder: Optional[MultiGridStateEncoder] = None
    ):
        super().__init__(xi=xi, eta=eta)
        
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
        
        # Own state encoder for U_r-specific features (trained with U_r loss)
        # This allows U_r to learn additional state features beyond those learned by V_h^e
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
        
        # Network predicts log(y-1) for numerical stability with optional dropout
        # y = 1 + exp(log(y-1)) ensures y > 1
        # Uses BOTH shared encoder (frozen/detached) AND own encoder (trained with U_r loss)
        actual_state_dim = self.state_encoder.feature_dim + self.own_state_encoder.feature_dim
        if dropout > 0.0:
            self.y_head = nn.Sequential(
                nn.Linear(actual_state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.y_head = nn.Sequential(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal: Compute y and U_r(s) from pre-encoded tensors.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Tuple (y, U_r) where:
            - y: intermediate value (batch,), y > 1
            - U_r: intrinsic reward (batch,), U_r < 0
        """
        # Encode state with SHARED encoder (DETACHED - no gradients flow to shared encoder)
        # The shared state encoder is trained ONLY by V_h^e loss.
        shared_state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        ).detach()
        
        # Encode state with OWN encoder (trained with U_r loss)
        own_state_features = self.own_state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Combine BOTH shared (detached) and own (trainable) features
        state_features = torch.cat([shared_state_features, own_state_features], dim=-1)
        
        # Predict log(y-1)
        log_y_minus_1 = self.y_head(state_features).squeeze(-1)
        
        # Convert to y
        y = self.log_y_minus_1_to_y(log_y_minus_1)
        
        # Compute U_r = -y^η
        u_r = self.y_to_u_r(y)
        
        return y, u_r
    
    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode state and compute y and U_r.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Tuple (y, U_r), each of shape (1,).
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with pre-encoded state features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Tuple (y, U_r) where:
            - y: intermediate value (batch,), y > 1
            - U_r: intrinsic reward (batch,), U_r < 0
        """
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict y and U_r (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            device: Torch device.
        
        Returns:
            Tuple (y, U_r), each of shape (1,).
        """
        with torch.no_grad():
            return self.forward(state, world_model, device)
    
    def forward_batch(
        self,
        states: List[Any],
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch forward pass from raw states.
        
        Batch-tensorizes all states and computes y and U_r in a single forward pass.
        This is the primary interface for batched training.
        
        Args:
            states: List of raw environment states.
            world_model: Environment with grid (for tensorization).
            device: Torch device.
        
        Returns:
            Tuple (y, U_r) where:
            - y: intermediate value (batch,), y > 1
            - U_r: intrinsic reward (batch,), U_r < 0
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
            'xi': self.xi,
            'eta': self.eta,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
        }
