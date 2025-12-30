"""
Multigrid-specific Human Goal Achievement Network for Phase 2.

Implements V_h^e(s, g_h) from equation (6) for multigrid environments.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ...phase2.human_goal_ability import BaseHumanGoalAchievementNetwork
from ..state_encoder import MultiGridStateEncoder
from ..goal_encoder import MultiGridGoalEncoder
from ..agent_encoder import AgentIdentityEncoder


class MultiGridHumanGoalAchievementNetwork(BaseHumanGoalAchievementNetwork):
    """
    Human Goal Achievement Network for multigrid environments.
    
    Estimates V_h^e(s, g_h) - the probability that human h achieves goal g_h
    under the current robot policy π_r.
    
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
        goal_feature_dim: Goal encoder output dimension.
        hidden_dim: Hidden layer dimension.
        gamma_h: Human discount factor.
        feasible_range: Output bounds for V_h^e.
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
        goal_feature_dim: int = 64,
        hidden_dim: int = 256,
        gamma_h: float = 0.99,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        dropout: float = 0.0,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4,
        max_agents: int = 10,
        agent_embedding_dim: int = 16,
        state_encoder: Optional[MultiGridStateEncoder] = None,
        goal_encoder: Optional[MultiGridGoalEncoder] = None,
        agent_encoder: Optional[AgentIdentityEncoder] = None
    ):
        super().__init__(gamma_h=gamma_h, feasible_range=feasible_range)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.goal_feature_dim = goal_feature_dim
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
        
        # Use shared goal encoder or create own
        if goal_encoder is not None:
            self.goal_encoder = goal_encoder
        else:
            self.goal_encoder = MultiGridGoalEncoder(
                grid_height=grid_height,
                grid_width=grid_width,
                feature_dim=goal_feature_dim
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
        
        # Value head: combines state + goal + agent identity features
        # Use actual encoder feature_dim (may differ from passed parameter when use_encoders=False)
        combined_dim = self.state_encoder.feature_dim + self.goal_encoder.feature_dim + self.agent_encoder.output_dim
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
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor,
        goal_features: torch.Tensor,
        query_agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute V_h^e(s, g_h).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            goal_features: (batch, goal_feature_dim)
            query_agent_indices: (batch,) agent indices for identity encoding
            query_agent_grid: (batch, 1, H, W) grid marking query agent position
            query_agent_features: (batch, AGENT_FEATURE_SIZE) query agent features
        
        Returns:
            V_h^e values tensor (batch,) in [0, 1].
        """
        # Encode state
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        
        # Encode agent identity (index + position + features)
        agent_embedding = self.agent_encoder(
            query_agent_indices, query_agent_grid, query_agent_features
        )
        
        # Combine state, goal, and agent identity features
        combined = torch.cat([state_features, goal_features, agent_embedding], dim=-1)
        
        # Compute raw value
        raw_value = self.value_head(combined).squeeze(-1)
        
        # Apply soft clamp to keep in [0, 1]
        return self.apply_clamp(raw_value)
    
    def encode_and_forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and goal, then compute V_h^e.
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            goal: The goal g_h for this human.
            device: Torch device.
        
        Returns:
            V_h^e tensor of shape (1,).
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Encode state (agent-agnostic)
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.tensorize_state(state, world_model, device)
        
        # Encode goal: first extract coordinates, then pass through encoder network
        goal_coords = self.goal_encoder.tensorize_goal(goal, device)
        goal_features = self.goal_encoder(goal_coords)
        
        # Encode agent identity (index + position grid + features)
        query_idx, query_grid, query_features = self.agent_encoder.encode_single(
            human_agent_idx, state, world_model, device
        )
        
        return self.forward(
            grid_tensor, global_features, agent_features,
            interactive_features, goal_features,
            query_idx, query_grid, query_features
        )
    
    def forward_with_goal_features(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor,
        goal_features: torch.Tensor,
        query_agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with pre-encoded state and goal features (for batched training).
        
        Args:
            grid_tensor: (batch, num_grid_channels, H, W)
            global_features: (batch, NUM_GLOBAL_WORLD_FEATURES)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
            goal_features: (batch, goal_feature_dim)
            query_agent_indices: (batch,) agent indices
            query_agent_grid: (batch, 1, H, W) grid marking query agent position
            query_agent_features: (batch, AGENT_FEATURE_SIZE) query agent features
        
        Returns:
            V_h^e values tensor (batch,).
        """
        return self.forward(
            grid_tensor, global_features, agent_features,
            interactive_features, goal_features,
            query_agent_indices, query_agent_grid, query_agent_features
        )
    
    def predict(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Predict V_h^e with hard clamping (for inference).
        
        Args:
            state: Multigrid state tuple.
            world_model: Environment with grid.
            human_agent_idx: Index of the human agent.
            goal: The goal g_h for this human.
            device: Torch device.
        
        Returns:
            V_h^e tensor with strict [0, 1] bounds.
        """
        with torch.no_grad():
            v_h_e = self.encode_and_forward(
                state, world_model, human_agent_idx, goal, device
            )
            return self.apply_hard_clamp(v_h_e)
    
    def forward_batch(
        self,
        states: List[Any],
        goals: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states, goals, and human indices.
        
        Batch-tensorizes all inputs and computes V_h^e in a single forward pass.
        This is the primary interface for batched training.
        
        Args:
            states: List of raw environment states.
            goals: List of goals (one per state).
            human_indices: List of human agent indices (one per state).
            world_model: Environment with grid (for tensorization).
            device: Torch device.
        
        Returns:
            V_h^e values tensor (batch,) in [0, 1].
        """
        if len(states) != len(goals) or len(states) != len(human_indices):
            raise ValueError("states, goals, and human_indices must have same length")
        
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
        
        # Batch tensorize goals
        goal_coords_list = [self.goal_encoder.tensorize_goal(g, device) for g in goals]
        goal_coords_batch = torch.cat(goal_coords_list, dim=0)
        goal_features = self.goal_encoder(goal_coords_batch)
        
        # Batch tensorize agent identities
        idx_list, grid_list, feat_list = [], [], []
        for h_idx, state in zip(human_indices, states):
            idx, grid, feat = self.agent_encoder.encode_single(h_idx, state, world_model, device)
            idx_list.append(idx)
            grid_list.append(grid)
            feat_list.append(feat)
        
        query_agent_indices = torch.cat(idx_list, dim=0)
        query_agent_grid = torch.cat(grid_list, dim=0)
        query_agent_features = torch.cat(feat_list, dim=0)
        
        return self.forward(
            grid_tensor, global_features, agent_features, interactive_features,
            goal_features,
            query_agent_indices, query_agent_grid, query_agent_features
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'state_feature_dim': self.state_feature_dim,
            'goal_feature_dim': self.goal_feature_dim,
            'hidden_dim': self.hidden_dim,
            'gamma_h': self.gamma_h,
            'feasible_range': self.feasible_range,
            'dropout': self.dropout_rate,
            'max_agents': self.max_agents,
            'agent_embedding_dim': self.agent_embedding_dim,
            'state_encoder_config': self.state_encoder.get_config(),
            'goal_encoder_config': self.goal_encoder.get_config(),
            'agent_encoder_config': self.agent_encoder.get_config(),
        }
