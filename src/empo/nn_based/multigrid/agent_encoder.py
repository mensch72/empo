"""
Agent encoder for multigrid environments.

Encodes agent features into vectors using a list-based approach:
- Query agent features first
- Then per-color agent lists
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ..agent_encoder import BaseAgentEncoder
from .constants import AGENT_FEATURE_SIZE, STANDARD_COLORS
from .feature_extraction import extract_agent_features, extract_all_agent_features


class MultiGridAgentEncoder(BaseAgentEncoder):
    """
    List-based encoder for multigrid agents.
    
    Encodes agents as:
    1. Query agent features (AGENT_FEATURE_SIZE)
    2. For each color: features of first N agents of that color
    
    Args:
        num_agents_per_color: Dict mapping color to max agents of that color.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        num_agents_per_color: Dict[str, int],
        feature_dim: int = 64
    ):
        super().__init__(feature_dim)
        self.num_agents_per_color = num_agents_per_color
        self.color_order = sorted(num_agents_per_color.keys())
        
        # Input size: query agent + per-color agent lists
        total_agents = sum(num_agents_per_color.values())
        input_size = AGENT_FEATURE_SIZE * (1 + total_agents)
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        Encode agent features.
        
        Args:
            agent_features: (batch, input_size) flattened agent features
        
        Returns:
            Feature tensor (batch, feature_dim)
        """
        return self.fc(agent_features)
    
    def encode_agents(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode agents from a state.
        
        Args:
            state: Environment state tuple.
            world_model: Environment with agents.
            query_agent_idx: Index of query agent.
            device: Torch device.
        
        Returns:
            Tensor (1, input_size) ready for forward().
        """
        _, agent_states, _, _ = state
        
        query_features, color_features = extract_all_agent_features(
            agent_states, world_model, query_agent_idx, self.num_agents_per_color
        )
        
        # Concatenate: query agent + per-color lists
        all_features = [query_features]
        for color in self.color_order:
            if color in color_features:
                all_features.append(color_features[color].flatten())
        
        combined = torch.cat(all_features).unsqueeze(0).to(device)
        return combined
    
    def get_input_size(self) -> int:
        """Return the input feature size."""
        total_agents = sum(self.num_agents_per_color.values())
        return AGENT_FEATURE_SIZE * (1 + total_agents)
