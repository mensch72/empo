import torch
import torch.nn as nn
from typing import Any, List, Optional, Dict, Tuple

from empo.learning_based.phase2.inverse_dynamics import BaseInverseDynamicsNetwork
from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.agent_encoder import AgentIdentityEncoder
from empo.learning_based.multigrid.constants import NUM_GLOBAL_WORLD_FEATURES

class MultiGridInverseDynamicsNetwork(BaseInverseDynamicsNetwork):
    """
    Transition Dynamics (Inverse Dynamics) Network for MultiGrid.
    Uses BOTH shared and own encoders.
    """
    def __init__(
        self,
        config: Any,
        num_actions: int,
        grid_height: int,
        grid_width: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        max_agents: int = 10,
        agent_embedding_dim: int = 16,
        state_encoder: Optional[MultiGridStateEncoder] = None,
        agent_encoder: Optional[AgentIdentityEncoder] = None,
        own_state_encoder: Optional[MultiGridStateEncoder] = None,
        own_agent_encoder: Optional[AgentIdentityEncoder] = None
    ):
        super().__init__(config, num_actions)
        
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.max_agents = max_agents
        self.agent_embedding_dim = agent_embedding_dim
        
        # Use shared state encoder or create own (NOTE: Not detached)
        if state_encoder is not None:
            self.state_encoder = state_encoder
        else:
            self.state_encoder = MultiGridStateEncoder(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                num_agent_colors=num_agent_colors,
                feature_dim=state_feature_dim,
                include_step_count=config.include_step_count,
                use_encoders=config.use_encoders
            )
            
        if agent_encoder is not None:
            self.agent_encoder = agent_encoder
        else:
            self.agent_encoder = AgentIdentityEncoder(
                num_agents=max_agents,
                embedding_dim=agent_embedding_dim,
                grid_height=grid_height,
                grid_width=grid_width,
                use_encoders=config.use_encoders
            )
            
        if own_state_encoder is not None:
            self.own_state_encoder = own_state_encoder
        else:
            self.own_state_encoder = MultiGridStateEncoder(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                num_agent_colors=num_agent_colors,
                feature_dim=state_feature_dim,
                include_step_count=config.include_step_count,
                use_encoders=config.use_encoders,
                share_cache_with=self.state_encoder,
            )
            
        if own_agent_encoder is not None:
            self.own_agent_encoder = own_agent_encoder
        else:
            self.own_agent_encoder = AgentIdentityEncoder(
                num_agents=max_agents,
                embedding_dim=agent_embedding_dim,
                grid_height=grid_height,
                grid_width=grid_width,
                use_encoders=config.use_encoders,
                share_cache_with=self.agent_encoder,
            )

        self.actual_state_feature_dim = self.state_encoder.feature_dim
        self.actual_own_state_feature_dim = self.own_state_encoder.feature_dim
        self.actual_agent_feature_dim = self.agent_encoder.output_dim
        self.actual_own_agent_feature_dim = self.own_agent_encoder.output_dim

        total_input_dim = (self.actual_state_feature_dim + self.actual_own_state_feature_dim) * 2 + \
                          self.actual_agent_feature_dim + self.actual_own_agent_feature_dim
                          
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward_from_encoded(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor,
        next_grid_tensor: torch.Tensor,
        next_global_features: torch.Tensor,
        next_agent_features: torch.Tensor,
        next_interactive_features: torch.Tensor,
        query_agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor,
        own_query_agent_indices: Optional[torch.Tensor] = None,
        own_query_agent_grid: Optional[torch.Tensor] = None,
        own_query_agent_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass from pre-encoded raw tensors.
        """
        if own_query_agent_indices is None:
            own_query_agent_indices = query_agent_indices
        if own_query_agent_grid is None:
            own_query_agent_grid = query_agent_grid
        if own_query_agent_features is None:
            own_query_agent_features = query_agent_features
            
        # Encode State
        shared_state_features = self.state_encoder(
            grid_tensor=grid_tensor,
            global_features=global_features,
            agent_features=agent_features,
            interactive_features=interactive_features
        )
        own_state_features = self.own_state_encoder(
            grid_tensor=grid_tensor,
            global_features=global_features,
            agent_features=agent_features,
            interactive_features=interactive_features
        )
        
        # Encode Next State
        shared_next_state_features = self.state_encoder(
            grid_tensor=next_grid_tensor,
            global_features=next_global_features,
            agent_features=next_agent_features,
            interactive_features=next_interactive_features
        )
        own_next_state_features = self.own_state_encoder(
            grid_tensor=next_grid_tensor,
            global_features=next_global_features,
            agent_features=next_agent_features,
            interactive_features=next_interactive_features
        )
        
        # Encode Agent
        shared_agent_embedding = self.agent_encoder(
            agent_indices=query_agent_indices,
            query_agent_grid=query_agent_grid,
            query_agent_features=query_agent_features
        )
        own_agent_embedding = self.own_agent_encoder(
            agent_indices=own_query_agent_indices,
            query_agent_grid=own_query_agent_grid,
            query_agent_features=own_query_agent_features
        )
        
        # Concatenate everything
        combined = torch.cat([
            shared_state_features, own_state_features,
            shared_agent_embedding, own_agent_embedding,
            shared_next_state_features, own_next_state_features
        ], dim=-1)
        
        return self.fc(combined)

    def forward(self, state: Any, next_state: Any, world_model: Any, human_agent_idx: int, device: str = 'cpu') -> torch.Tensor:
        return self.forward_batch([state], [next_state], [human_agent_idx], world_model, device)

    def forward_batch(self, states: List[Any], next_states: List[Any], human_indices: List[int], world_model: Any, device: str = 'cpu') -> torch.Tensor:
        if not states:
            return torch.zeros((0, self.num_actions), device=device)
            
        # Tensorize states
        grid_list, global_list, agent_list, inter_list = [], [], [], []
        for s in states:
            g, glob, a, i = self.state_encoder.tensorize_state(s, world_model)
            grid_list.append(g)
            global_list.append(glob)
            agent_list.append(a)
            inter_list.append(i)
            
        grid_tensor = torch.cat(grid_list, dim=0).to(device)
        global_tensor = torch.cat(global_list, dim=0).to(device)
        agent_tensor = torch.cat(agent_list, dim=0).to(device)
        inter_tensor = torch.cat(inter_list, dim=0).to(device)
        
        # Tensorize next states
        n_grid_list, n_global_list, n_agent_list, n_inter_list = [], [], [], []
        for ns in next_states:
            g, glob, a, i = self.state_encoder.tensorize_state(ns, world_model)
            n_grid_list.append(g)
            n_global_list.append(glob)
            n_agent_list.append(a)
            n_inter_list.append(i)
            
        n_grid_tensor = torch.cat(n_grid_list, dim=0).to(device)
        n_global_tensor = torch.cat(n_global_list, dim=0).to(device)
        n_agent_tensor = torch.cat(n_agent_list, dim=0).to(device)
        n_inter_tensor = torch.cat(n_inter_list, dim=0).to(device)
        
        # Tensorize agent
        a_idx_list, a_grid_list, a_feat_list = [], [], []
        for s, h_idx in zip(states, human_indices):
            a_i, a_g, a_f = self.agent_encoder.tensorize_single(h_idx, s, world_model)
            a_idx_list.append(a_i)
            a_grid_list.append(a_g)
            a_feat_list.append(a_f)
            
        a_idx_tensor = torch.cat(a_idx_list, dim=0).to(device)
        a_grid_tensor = torch.cat(a_grid_list, dim=0).to(device)
        a_feat_tensor = torch.cat(a_feat_list, dim=0).to(device)
        
        return self.forward_from_encoded(
            grid_tensor, global_tensor, agent_tensor, inter_tensor,
            n_grid_tensor, n_global_tensor, n_agent_tensor, n_inter_tensor,
            a_idx_tensor, a_grid_tensor, a_feat_tensor
        )
