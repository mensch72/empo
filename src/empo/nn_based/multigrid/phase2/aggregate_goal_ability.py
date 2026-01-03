"""
Multigrid-specific Aggregate Goal Ability Network for Phase 2.

Implements X_h(s) from equation (7) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ...phase2.aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from ..state_encoder import MultiGridStateEncoder
from ..agent_encoder import AgentIdentityEncoder


class MultiGridAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """
    Aggregate Goal Ability Network for multigrid environments.
    
    Estimates X_h(s) = E_{g_h}[V_h^e(s, g_h)^ζ] - the aggregate ability
    of human h to achieve various goals.
    
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
        zeta: Risk/reliability preference parameter.
        feasible_range: Output bounds for X_h.
        dropout: Dropout rate (0 = no dropout).
        max_kill_buttons: Max KillButtons.
        max_pause_switches: Max PauseSwitches.
        max_disabling_switches: Max DisablingSwitches.
        max_control_buttons: Max ControlButtons.
        max_agents: Max number of agents for identity encoding.
        agent_embedding_dim: Dimension of agent identity embedding.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        zeta: float = 2.0,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        max_agents: int = 10,
        agent_embedding_dim: int = 16,
        state_encoder: Optional[MultiGridStateEncoder] = None,
        agent_encoder: Optional[AgentIdentityEncoder] = None,
        own_state_encoder: Optional[MultiGridStateEncoder] = None,
        own_agent_encoder: Optional[AgentIdentityEncoder] = None
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.max_agents = max_agents
        self.agent_embedding_dim = agent_embedding_dim
        
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
        
        # Use shared agent identity encoder or create own
        if agent_encoder is not None:
            self.agent_encoder = agent_encoder
        else:
            self.agent_encoder = AgentIdentityEncoder(
                num_agents=max_agents,
                embedding_dim=agent_embedding_dim,
                grid_height=grid_height,
                grid_width=grid_width
            )
        
        # Own state encoder for X_h-specific features (trained with X_h loss)
        # This allows X_h to learn additional state features beyond those learned by V_h^e
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
        
        # Own agent encoder for X_h-specific features (trained with X_h loss)
        # This allows X_h to learn additional agent features beyond those learned by V_h^e
        # Note: own_agent_encoder shares cache with agent_encoder to avoid redundant tensorization
        if own_agent_encoder is not None:
            self.own_agent_encoder = own_agent_encoder
        else:
            self.own_agent_encoder = AgentIdentityEncoder(
                num_agents=max_agents,
                embedding_dim=agent_embedding_dim,
                grid_height=grid_height,
                grid_width=grid_width,
                share_cache_with=self.agent_encoder
            )
        
        # X_h value head with optional dropout
        # Note: X_h depends on both state and human identity
        # Uses BOTH shared encoders (frozen/detached) AND own encoders (trained with X_h loss)
        # Use actual encoder feature_dim (may differ from passed parameter when use_encoders=False)
        combined_dim = (self.state_encoder.feature_dim + self.own_state_encoder.feature_dim +
                       self.agent_encoder.output_dim + self.own_agent_encoder.output_dim)
        if dropout > 0.0:
            self.value_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.value_head = nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
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
        interactive_features: torch.Tensor,
        query_agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor,
        own_query_agent_indices: Optional[torch.Tensor] = None,
        own_query_agent_grid: Optional[torch.Tensor] = None,
        own_query_agent_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Internal: Compute X_h(s) from pre-encoded tensors.
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            query_agent_indices: (batch,) agent indices for shared encoder
            query_agent_grid: (batch, 1, H, W) grid marking query agent position for shared encoder
            query_agent_features: (batch, AGENT_FEATURE_SIZE) query agent features for shared encoder
            own_query_agent_indices: (batch,) agent indices for own encoder (optional, uses same as shared if None)
            own_query_agent_grid: (batch, 1, H, W) for own encoder
            own_query_agent_features: (batch, AGENT_FEATURE_SIZE) for own encoder
        
        Returns:
            X_h values tensor (batch,) in (0, 1].
        """
        # Encode state with SHARED encoder (DETACHED - no gradients flow to shared encoder)
        # The shared state encoder is trained ONLY by V_h^e loss.
        shared_state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        ).detach()
        
        # Encode state with OWN encoder (trained with X_h loss)
        own_state_features = self.own_state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Encode agent identity with shared encoder (DETACHED - frozen during X_h training)
        # The shared agent encoder is trained ONLY by V_h^e loss.
        shared_agent_embedding = self.agent_encoder(
            query_agent_indices, query_agent_grid, query_agent_features
        ).detach()
        
        # Encode agent identity with own encoder (trained with X_h loss)
        # Use same inputs if own_ versions not provided
        own_idx = own_query_agent_indices if own_query_agent_indices is not None else query_agent_indices
        own_grid = own_query_agent_grid if own_query_agent_grid is not None else query_agent_grid
        own_feat = own_query_agent_features if own_query_agent_features is not None else query_agent_features
        own_agent_embedding = self.own_agent_encoder(own_idx, own_grid, own_feat)
        
        # Combine BOTH shared (detached) and own (trainable) features
        combined = torch.cat([shared_state_features, own_state_features, shared_agent_embedding, own_agent_embedding], dim=-1)
        
        # Compute raw value
        raw_value = self.value_head(combined).squeeze(-1)
        
        # Apply soft clamp to keep in (0, 1]
        return self.apply_clamp(raw_value)
    
    def forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute X_h for a specific human.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h tensor of shape (1,).
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Encode state (agent-agnostic)
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.tensorize_state(state, world_model, device)
        
        # Tensorize agent identity with shared encoder
        query_idx, query_grid, query_features = self.agent_encoder.tensorize_single(
            human_agent_idx, state, world_model, device
        )
        
        # Tensorize agent identity with own encoder (same input, separate encoding)
        own_idx, own_grid, own_features = self.own_agent_encoder.tensorize_single(
            human_agent_idx, state, world_model, device
        )
        
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features,
            query_idx, query_grid, query_features,
            own_idx, own_grid, own_features
        )
    
    def forward_from_encoded(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor,
        query_agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with pre-encoded state features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            query_agent_indices: (batch,) agent indices
            query_agent_grid: (batch, 1, H, W) grid marking query agent position
            query_agent_features: (batch, AGENT_FEATURE_SIZE) query agent features
        
        Returns:
            X_h values tensor (batch,).
        """
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features,
            query_agent_indices, query_agent_grid, query_agent_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict X_h with hard clamping (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h tensor with strict (0, 1] bounds.
        """
        with torch.no_grad():
            x_h = self.forward(state, world_model, human_agent_idx, device)
            return self.apply_hard_clamp(x_h)
    
    def forward_batch(
        self,
        states: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states and human indices.
        
        Batch-tensorizes all inputs and computes X_h in a single forward pass.
        This is the primary interface for batched training.
        
        Args:
            states: List of raw environment states.
            human_indices: List of human agent indices (one per state).
            world_model: Environment with grid (for tensorization).
            device: Torch device.
        
        Returns:
            X_h values tensor (batch,) in (0, 1].
        """
        if len(states) != len(human_indices):
            raise ValueError("states and human_indices must have same length")
        
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
        
        # Batch tensorize agent identities (shared encoder)
        idx_list, grid_list, feat_list = [], [], []
        for h_idx, state in zip(human_indices, states):
            idx, grid, feat = self.agent_encoder.tensorize_single(h_idx, state, world_model, device)
            idx_list.append(idx)
            grid_list.append(grid)
            feat_list.append(feat)
        
        query_agent_indices = torch.cat(idx_list, dim=0)
        query_agent_grid = torch.cat(grid_list, dim=0)
        query_agent_features = torch.cat(feat_list, dim=0)
        
        # Batch tensorize agent identities (own encoder)
        own_idx_list, own_grid_list, own_feat_list = [], [], []
        for h_idx, state in zip(human_indices, states):
            idx, grid, feat = self.own_agent_encoder.tensorize_single(h_idx, state, world_model, device)
            own_idx_list.append(idx)
            own_grid_list.append(grid)
            own_feat_list.append(feat)
        
        own_idx = torch.cat(own_idx_list, dim=0)
        own_grid = torch.cat(own_grid_list, dim=0)
        own_features = torch.cat(own_feat_list, dim=0)
        
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features,
            query_agent_indices, query_agent_grid, query_agent_features,
            own_idx, own_grid, own_features
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'hidden_dim': self.hidden_dim,
            'zeta': self.zeta,
            'max_agents': self.max_agents,
            'agent_embedding_dim': self.agent_embedding_dim,
            'feasible_range': self.feasible_range,
            'dropout': self.dropout_rate,
            'state_encoder_config': self.state_encoder.get_config(),
            'agent_encoder_config': self.agent_encoder.get_config(),
            'own_agent_encoder_config': self.own_agent_encoder.get_config(),
        }
