"""
Base state encoder class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStateEncoder(nn.Module, ABC):
    """
    Abstract base class for state encoders.
    
    State encoders convert environment states into fixed-size feature vectors.
    Subclasses implement domain-specific encoding logic.
    """
    
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode state(s) into feature vectors."""
        pass
    
    @abstractmethod
    def encode_state(self, state: Any, world_model: Any, **kwargs) -> Any:
        """Encode a single state from the environment."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
