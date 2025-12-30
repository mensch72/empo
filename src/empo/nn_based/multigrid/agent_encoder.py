"""
Agent identity encoder for multigrid environments.

Encodes agent identity (index) into features that can be combined
with state features for agent-specific networks like V_h^e and X_h.

Provides:
1. Agent index embedding (learnable)
2. Grid-shaped channel marking the query agent's position
3. Query agent's features (position, direction, carrying, etc.)

The encoder supports internal caching of raw tensor extraction (before NN forward)
to avoid redundant computation when the same agent identity is encoded multiple times.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .feature_extraction import extract_agent_features
from .constants import AGENT_FEATURE_SIZE


def _make_hashable(obj):
    """
    Recursively convert lists to tuples to make state hashable for caching.
    
    The state tuple contains lists (agent_states, mobile_objects, mutable_objects)
    which are not hashable. This function converts them to tuples.
    """
    if isinstance(obj, list):
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(_make_hashable(item) for item in obj)
    else:
        return obj


class AgentIdentityEncoder(nn.Module):
    """
    Encodes agent identity for agent-specific networks.
    
    This provides networks with:
    1. A learnable embedding for the agent index
    2. A grid channel marking the query agent's position (important when
       multiple agents have the same color)
    3. The query agent's features (even though redundant with color-grouped
       features, this makes it easier for the network to learn)
    
    Args:
        num_agents: Maximum number of agents.
        embedding_dim: Dimension of the agent index embedding.
        position_feature_dim: Output dimension for position encoding.
        agent_feature_dim: Output dimension for agent feature encoding.
        grid_height: Height of the grid (for position channel).
        grid_width: Width of the grid (for position channel).
        use_encoders: If False, forward() returns identity (flattened+padded input).
    """
    
    def __init__(
        self,
        num_agents: int,
        embedding_dim: int = 16,
        position_feature_dim: int = 32,
        agent_feature_dim: int = 32,
        grid_height: int = 10,
        grid_width: int = 10,
        use_encoders: bool = True,
        identity_output_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_agents = num_agents
        self.embedding_dim = embedding_dim
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.use_encoders = use_encoders
        self.position_feature_dim = position_feature_dim
        self.agent_feature_dim = agent_feature_dim
        
        # Learnable embedding for each agent index
        self.index_embedding = nn.Embedding(num_agents, embedding_dim)
        
        # Small CNN to process the query agent position channel
        # This converts the grid-shaped position marker into a feature vector
        self.position_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        position_conv_out = 8 * grid_height * grid_width
        self.position_fc = nn.Sequential(
            nn.Linear(position_conv_out, position_feature_dim),
            nn.ReLU(),
        )
        
        # MLP to process query agent features
        self.agent_feature_fc = nn.Sequential(
            nn.Linear(AGENT_FEATURE_SIZE, agent_feature_dim),
            nn.ReLU(),
        )
        
        # Total output dimension
        # If use_encoders=False, use identity_output_dim if provided, else compute it
        if not use_encoders:
            if identity_output_dim is not None:
                self.output_dim = identity_output_dim
            else:
                # Compute: agent_idx (1) + grid (H*W) + agent_features (AGENT_FEATURE_SIZE)
                self.output_dim = 1 + grid_height * grid_width + AGENT_FEATURE_SIZE
        else:
            self.output_dim = embedding_dim + position_feature_dim + agent_feature_dim
        
        # Internal cache for raw tensor extraction (before NN forward)
        # Keys are (state_tuple, agent_idx), values are (idx_tensor, grid, features)
        self._raw_cache: Dict[Tuple[Tuple, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_cache(self):
        """Clear the internal raw tensor cache."""
        self._raw_cache.clear()
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (hits, misses) cache statistics."""
        return self._cache_hits, self._cache_misses
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        self._cache_hits = 0
        self._cache_misses = 0
    
    def forward(
        self,
        agent_indices: torch.Tensor,
        query_agent_grid: torch.Tensor,
        query_agent_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode agent identity into a feature vector.
        
        Args:
            agent_indices: (batch,) agent indices
            query_agent_grid: (batch, 1, H, W) grid with 1 at query agent position
            query_agent_features: (batch, AGENT_FEATURE_SIZE) query agent features
        
        Returns:
            Agent identity features of shape (batch, output_dim).
        
        If use_encoders=False, bypasses neural network and returns flattened+padded
        input tensors directly (identity mode for debugging).
        """
        batch_size = agent_indices.shape[0]
        
        if not self.use_encoders:
            # Identity mode: flatten all inputs and concatenate unchanged
            # Convert agent_indices to float for concatenation
            idx_float = agent_indices.float().unsqueeze(1)  # (batch, 1)
            grid_flat = query_agent_grid.view(batch_size, -1)  # (batch, H*W)
            return torch.cat([idx_float, grid_flat, query_agent_features], dim=1)
        
        # Agent index embedding
        idx_emb = self.index_embedding(agent_indices)  # (batch, embedding_dim)
        
        # Position channel encoding
        pos_conv = self.position_conv(query_agent_grid)  # (batch, 8, H, W)
        pos_flat = pos_conv.view(pos_conv.size(0), -1)  # (batch, 8*H*W)
        pos_emb = self.position_fc(pos_flat)  # (batch, 32)
        
        # Agent features encoding
        feat_emb = self.agent_feature_fc(query_agent_features)  # (batch, 32)
        
        # Combine all
        return torch.cat([idx_emb, pos_emb, feat_emb], dim=-1)
    
    def encode_single(
        self,
        agent_idx: int,
        state: Tuple,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract raw tensors for a single agent identity (before NN forward).
        
        This method extracts index, position grid, and agent features from the
        state. Results are cached by (state_id, agent_idx) to avoid redundant
        extraction. Call forward() on these tensors to get the final encoding.
        
        Args:
            agent_idx: Agent index.
            state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
            world_model: Environment with agents.
            device: Torch device.
        
        Returns:
            Tuple of (agent_index_tensor, query_agent_grid, query_agent_features)
            ready for batching or direct forward pass.
        """
        # Check cache using content-based key (state tuple + agent index)
        # State contains lists, so convert to hashable form (tuples)
        cache_key = (_make_hashable(state), agent_idx)
        if cache_key in self._raw_cache:
            self._cache_hits += 1
            # Clone to avoid in-place operation conflicts during gradient computation
            cached = self._raw_cache[cache_key]
            return tuple(t.clone() for t in cached)
        
        self._cache_misses += 1
        
        step_count, agent_states, mobile_objects, mutable_objects = state
        H, W = self.grid_height, self.grid_width
        
        # Agent index tensor
        idx_tensor = torch.tensor([agent_idx], device=device)
        
        # Query agent position grid
        query_grid = torch.zeros(1, 1, H, W, device=device)
        if agent_idx < len(agent_states):
            agent_state = agent_states[agent_idx]
            if agent_state[0] is not None:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    query_grid[0, 0, y, x] = 1.0
        
        # Query agent features
        query_features = extract_agent_features(agent_states, world_model, agent_idx)
        query_features = query_features.unsqueeze(0).to(device)
        
        result = (idx_tensor, query_grid, query_features)
        self._raw_cache[cache_key] = result
        return result
    
    def encode_batch(
        self,
        agent_indices: list,
        states: list,
        world_models: list,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract raw tensors for a batch of agent identities (before NN forward).
        
        Uses caching to avoid redundant extraction.
        
        Args:
            agent_indices: List of agent indices.
            states: List of state tuples for each sample.
            world_models: List of world_model for each sample (can be single model).
            device: Torch device.
        
        Returns:
            Tuple of batched (agent_indices, query_agent_grids, query_agent_features).
        """
        batch_size = len(agent_indices)
        H, W = self.grid_height, self.grid_width
        
        idx_list = []
        grid_list = []
        features_list = []
        
        # Handle single world_model for all
        if not isinstance(world_models, (list, tuple)):
            world_models = [world_models] * batch_size
        
        for agent_idx, state, world_model in zip(agent_indices, states, world_models):
            idx, grid, features = self.encode_single(agent_idx, state, world_model, device)
            idx_list.append(idx)
            grid_list.append(grid)
            features_list.append(features)
        
        return (
            torch.cat(idx_list, dim=0),
            torch.cat(grid_list, dim=0),
            torch.cat(features_list, dim=0)
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'num_agents': self.num_agents,
            'embedding_dim': self.embedding_dim,
            'position_feature_dim': self.position_feature_dim,
            'agent_feature_dim': self.agent_feature_dim,
            'grid_height': self.grid_height,
            'grid_width': self.grid_width,
            'use_encoders': self.use_encoders,
        }
