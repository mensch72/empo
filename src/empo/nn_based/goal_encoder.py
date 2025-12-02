"""
Base goal encoder class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


class BaseGoalEncoder(nn.Module, ABC):
    """
    Abstract base class for goal encoders.
    
    Goal encoders convert goal specifications into fixed-size feature vectors.
    """
    
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode goal(s) into feature vectors."""
        pass
    
    @abstractmethod
    def encode_goal(self, goal: Any, device: str = 'cpu') -> torch.Tensor:
        """Encode a single goal."""
        pass
