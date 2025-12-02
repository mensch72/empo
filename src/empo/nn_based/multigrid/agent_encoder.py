"""
Agent encoder for multigrid environments.

This encoder uses list-based encoding for agents, capturing detailed features
for each agent. The encoding structure supports policy transfer across different
numbers of agents by using per-color agent lists.

Encoding Structure:
    [query_agent_features] + [per_color_agent_lists]
    
    Query agent features (13): position, direction, abilities, carried, status
    Per-color list: For each color k, features of first num_agents_per_color[k] agents
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    STANDARD_COLORS,
    COLOR_TO_IDX,
    AGENT_FEATURE_SIZE,
)
from .feature_extraction import extract_agent_features


class MultiGridAgentEncoder(nn.Module):
    """
    List-based encoder for agent features in multigrid environments.
    
    Encodes agents using a structure that supports policy transfer:
    1. Query agent features come first (always present)
    2. Per-color agent lists follow (for each color, up to max agents)
    
    Each agent is encoded with 13 features:
    - Position (2): raw x, y coordinates
    - Direction (4): one-hot encoding
    - Abilities (2): can_enter_magic_walls, can_push_rocks
    - Carried object (2): type_idx, color_idx
    - Status (3): paused, terminated, forced_next_action
    
    Args:
        num_agents_per_color: Dict mapping color name to max agents of that color.
            Example: {'yellow': 2, 'grey': 1} for 2 yellow humans + 1 grey robot.
        feature_dim: Output feature dimension.
    """
    
    def __init__(
        self,
        num_agents_per_color: Dict[str, int],
        feature_dim: int = 64
    ):
        super().__init__()
        self.num_agents_per_color = num_agents_per_color
        self.feature_dim = feature_dim
        self.agent_feature_size = AGENT_FEATURE_SIZE
        
        # Sort colors for deterministic ordering
        self.color_order = sorted(num_agents_per_color.keys())
        self.color_to_idx = {c: i for i, c in enumerate(self.color_order)}
        
        # Total input size: query agent + all per-color agents
        total_agents = sum(num_agents_per_color.values())
        self.input_dim = (1 + total_agents) * AGENT_FEATURE_SIZE
        
        # MLP to produce feature vector
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        query_position: torch.Tensor,
        query_direction: torch.Tensor,
        query_abilities: torch.Tensor,
        query_carried: torch.Tensor,
        query_status: torch.Tensor,
        all_agent_positions: torch.Tensor,
        all_agent_directions: torch.Tensor,
        all_agent_abilities: torch.Tensor,
        all_agent_carried: torch.Tensor,
        all_agent_status: torch.Tensor,
        agent_color_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode agents into feature vector.
        
        Args:
            query_position: (batch, 2) query agent position
            query_direction: (batch, 4) query agent direction one-hot
            query_abilities: (batch, 2) query agent abilities
            query_carried: (batch, 2) query agent carried object
            query_status: (batch, 3) query agent status
            all_agent_positions: (batch, num_agents, 2) all agent positions
            all_agent_directions: (batch, num_agents, 4) all agent directions
            all_agent_abilities: (batch, num_agents, 2) all agent abilities
            all_agent_carried: (batch, num_agents, 2) all agent carried objects
            all_agent_status: (batch, num_agents, 3) all agent statuses
            agent_color_indices: (batch, num_agents) color index for each agent
        
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        batch_size = query_position.shape[0]
        device = query_position.device
        
        # Concatenate query agent features
        query_features = torch.cat([
            query_position, query_direction, query_abilities, 
            query_carried, query_status
        ], dim=1)  # (batch, 13)
        
        # Build per-color agent lists
        color_features_list = [query_features]
        
        for color in self.color_order:
            max_agents_this_color = self.num_agents_per_color[color]
            color_idx = self.color_to_idx[color]
            
            # Initialize features for this color (zeros for missing agents)
            color_features = torch.zeros(
                batch_size, max_agents_this_color * self.agent_feature_size, 
                device=device
            )
            
            # Fill in features for agents of this color
            for slot in range(max_agents_this_color):
                for b in range(batch_size):
                    agent_count_this_color = 0
                    for agent_i in range(all_agent_positions.shape[1]):
                        if agent_color_indices[b, agent_i] == COLOR_TO_IDX.get(color, -1):
                            if agent_count_this_color == slot:
                                # This agent fills this slot
                                start_idx = slot * self.agent_feature_size
                                pos = all_agent_positions[b, agent_i]
                                dir_ = all_agent_directions[b, agent_i]
                                abilities = all_agent_abilities[b, agent_i]
                                carried = all_agent_carried[b, agent_i]
                                status = all_agent_status[b, agent_i]
                                
                                color_features[b, start_idx:start_idx+2] = pos
                                color_features[b, start_idx+2:start_idx+6] = dir_
                                color_features[b, start_idx+6:start_idx+8] = abilities
                                color_features[b, start_idx+8:start_idx+10] = carried
                                color_features[b, start_idx+10:start_idx+13] = status
                                break
                            agent_count_this_color += 1
            
            color_features_list.append(color_features)
        
        # Concatenate all features
        all_features = torch.cat(color_features_list, dim=1)
        
        # Apply MLP
        return self.fc(all_features)
    
    def encode_agents(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_index: int,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, ...]:
        """
        Extract and encode agents from a single state.
        
        Args:
            state: State tuple
            world_model: Environment with agents
            query_agent_index: Index of the query agent
            device: Torch device
        
        Returns:
            Tuple of tensors ready for forward():
                query_position, query_direction, query_abilities, query_carried, query_status,
                all_agent_positions, all_agent_directions, all_agent_abilities, 
                all_agent_carried, all_agent_status, agent_color_indices
        """
        positions, directions, abilities, carried, status, colors = extract_agent_features(
            state, world_model, device
        )
        
        # Add batch dimension
        positions = positions.unsqueeze(0)
        directions = directions.unsqueeze(0)
        abilities = abilities.unsqueeze(0)
        carried = carried.unsqueeze(0)
        status = status.unsqueeze(0)
        colors = colors.unsqueeze(0)
        
        # Extract query agent features
        query_position = positions[:, query_agent_index]
        query_direction = directions[:, query_agent_index]
        query_abilities = abilities[:, query_agent_index]
        query_carried = carried[:, query_agent_index]
        query_status = status[:, query_agent_index]
        
        return (
            query_position, query_direction, query_abilities, query_carried, query_status,
            positions, directions, abilities, carried, status, colors
        )
