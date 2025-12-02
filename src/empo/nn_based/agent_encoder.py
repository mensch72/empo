"""
Base agent encoder class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


class BaseAgentEncoder(nn.Module, ABC):
    """
    Abstract base class for agent encoders.
    
    Agent encoders convert agent state information into fixed-size feature vectors.
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode agent state(s) into feature vectors."""
        pass
    
    @abstractmethod
    def encode_agents(self, state: Any, world_model: Any, query_agent_idx: int,
                      device: str = 'cpu') -> torch.Tensor:
        """Encode agents from a state."""
        pass
