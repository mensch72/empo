"""
Base state encoder class.

A StateEncoder encodes the complete world state as seen by a query agent,
including grid/spatial information, agent features, and interactive object features.
This unified approach allows domain-specific implementations to choose their
own internal structure (grid vs list, separate vs combined encoders).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStateEncoder(nn.Module, ABC):
    """
    Abstract base class for state encoders.
    
    State encoders convert environment states into fixed-size feature vectors.
    
    The internal structure (CNN, MLP, separate encoders) is domain-specific.
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
    def tensorize_state(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Convert a raw state to input tensors (preprocessing, NOT neural network encoding).
        
        This is tensorization/featurization, not encoding. Call forward() on the
        returned tensors to get the actual neural network encoding.
        
        Args:
            state: Environment state.
            world_model: Environment/world model (or None).
            device: Torch device.
        
        Returns:
            Input tensors ready for forward().
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for save/load."""
        pass
