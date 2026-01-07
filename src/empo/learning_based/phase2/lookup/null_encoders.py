"""
Null (dummy) encoders for lookup table networks.

When V_h^e is a lookup table, other neural networks still expect shared encoders
from it. These null encoders output zeros, allowing the neural networks to rely
entirely on their own encoders while maintaining API compatibility.

These encoders:
- Have no learnable parameters
- Output zero tensors of the expected dimension
- Implement the same interface as real encoders (cache methods, tensorize, etc.)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple


class NullStateEncoder(nn.Module):
    """
    Null state encoder that outputs zeros.
    
    Used when V_h^e is a lookup table and other networks need a "shared" encoder
    to concatenate features with. Since the output is always zeros, networks
    must rely on their own encoders for actual learning.
    
    Args:
        feature_dim: Output dimension (must match what networks expect).
    """
    
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self._feature_dim = feature_dim
        # No learnable parameters - just a placeholder
        
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """Return zeros of shape (batch, feature_dim)."""
        batch_size = grid_tensor.shape[0]
        device = grid_tensor.device
        return torch.zeros(batch_size, self._feature_dim, device=device)
    
    def tensorize_state(
        self,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return dummy tensors for state tensorization.
        
        The actual values don't matter since forward() ignores them and outputs zeros.
        """
        # Return minimal tensors - the forward() will output zeros anyway
        dummy = torch.zeros(1, 1, device=device)
        return dummy, dummy, dummy, dummy
    
    def clear_cache(self):
        """No-op (no cache to clear)."""
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (0, 0) - no cache stats."""
        return (0, 0)
    
    def reset_cache_stats(self):
        """No-op (no cache stats to reset)."""
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {'feature_dim': self._feature_dim, 'type': 'NullStateEncoder'}


class NullGoalEncoder(nn.Module):
    """
    Null goal encoder that outputs zeros.
    
    Args:
        feature_dim: Output dimension (must match what networks expect).
    """
    
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        self._feature_dim = feature_dim
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def forward(self, goal_tensor: torch.Tensor) -> torch.Tensor:
        """Return zeros of shape (batch, feature_dim)."""
        batch_size = goal_tensor.shape[0]
        device = goal_tensor.device
        return torch.zeros(batch_size, self._feature_dim, device=device)
    
    def tensorize_goal(
        self,
        goal: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Return dummy tensor for goal tensorization."""
        return torch.zeros(1, 1, device=device)
    
    def clear_cache(self):
        """No-op (no cache to clear)."""
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (0, 0) - no cache stats."""
        return (0, 0)
    
    def reset_cache_stats(self):
        """No-op (no cache stats to reset)."""
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {'feature_dim': self._feature_dim, 'type': 'NullGoalEncoder'}


class NullAgentEncoder(nn.Module):
    """
    Null agent identity encoder that outputs zeros.
    
    Args:
        output_dim: Output dimension (must match what networks expect).
    """
    
    def __init__(self, output_dim: int = 32):
        super().__init__()
        self._output_dim = output_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        agent_idx: torch.Tensor,
        agent_position: torch.Tensor,
        agent_features: torch.Tensor
    ) -> torch.Tensor:
        """Return zeros of shape (batch, output_dim)."""
        batch_size = agent_idx.shape[0]
        device = agent_idx.device
        return torch.zeros(batch_size, self._output_dim, device=device)
    
    def tensorize_agent(
        self,
        agent_idx: int,
        state: Any,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return dummy tensors for agent tensorization."""
        dummy = torch.zeros(1, 1, device=device)
        return dummy, dummy, dummy
    
    def clear_cache(self):
        """No-op (no cache to clear)."""
    
    def get_cache_stats(self) -> Tuple[int, int]:
        """Return (0, 0) - no cache stats."""
        return (0, 0)
    
    def reset_cache_stats(self):
        """No-op (no cache stats to reset)."""
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict."""
        return {'output_dim': self._output_dim, 'type': 'NullAgentEncoder'}
