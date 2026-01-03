"""
Lookup Table Robot Value Network for Phase 2.

Implements V_r(s) as a dictionary-based lookup table with one entry per unique
(state, map_hash) pair. The map_hash distinguishes states from different map
configurations in ensemble mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..robot_value_network import BaseRobotValueNetwork
from .robot_q_network import _get_map_hash


class LookupTableRobotValueNetwork(BaseRobotValueNetwork):
    """
    Lookup table implementation of robot value network (V_r) for Phase 2.
    
    Stores V_r values in a dictionary keyed by (state, map_hash) hash. Each entry is a
    torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    The map_hash distinguishes states from different map configurations in ensemble mode,
    where different worlds may have the same mutable state but different wall layouts.
    
    V_r(s) < 0 always (since U_r < 0 and Q_r < 0), ensured via ensure_negative().
    
    Note: In practice, V_r is often computed directly from U_r and Q_r rather
    than learned separately (see v_r_use_network config option).
    
    Args:
        gamma_r: Robot discount factor.
        default_v_r: Initial V-value for unseen states (should be negative).
        include_step_count: If False, strip step_count from state before hashing.
        state_encoder: Optional state encoder (for API compatibility with neural networks).
    """
    
    def __init__(
        self,
        gamma_r: float = 0.99,
        default_v_r: float = -1.0,
        include_step_count: bool = True,
        state_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(gamma_r=gamma_r)
        self.default_v_r = default_v_r
        self.include_step_count = include_step_count
        
        # Store encoder for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        
        # Main lookup table: hash(state) -> Parameter(V_r value)
        self.table: Dict[int, nn.Parameter] = {}
        
        # Track new parameters for incremental optimizer updates
        self._new_params: List[nn.Parameter] = []
    
    def _normalize_state(self, state: Hashable) -> Hashable:
        """
        Normalize state for lookup key generation.
        
        If include_step_count is False and state is a tuple with step_count as first
        element (as in MultiGrid states), strip the step_count.
        """
        if not self.include_step_count and isinstance(state, tuple) and len(state) >= 2:
            return state[1:]
        return state

    def get_new_params(self) -> List[nn.Parameter]:
        """
        Get newly created parameters and clear the tracking list.
        
        Returns:
            List of parameters created since last call.
        """
        params = self._new_params
        self._new_params = []
        return params

    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the (state, map_hash) pair.
            device: Target device for the parameter.
        
        Returns:
            Parameter containing V_r value for this state.
        """
        if key not in self.table:
            # Create new entry with default value
            # Use raw value that will produce default_v_r after ensure_negative()
            raw_default = -self.default_v_r
            param = nn.Parameter(
                torch.tensor([raw_default], dtype=torch.float32, device=device)
            )
            self.table[key] = param
            self._new_params.append(param)
        return self.table[key]
    
    def _batch_forward(
        self,
        states: List[Hashable],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Internal: Batch forward pass from raw states.
        
        Args:
            states: List of hashable states.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            V_r values of shape (batch_size,), all negative.
        """
        
        # Collect parameters for all states
        params = []
        for state in states:
            # Use (state, map_hash) as the cache key to distinguish states from different maps
            normalized_state = self._normalize_state(state)
            key = hash((normalized_state, map_hash))
            param = self._get_or_create_entry(key, device)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Ensure V_r < 0
        return self.ensure_negative(raw_output)
    
    def forward(
        self,
        state: Hashable,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Single-state forward pass.
        
        Args:
            state: Environment state (must be hashable).
            world_model: Environment/world model (used to get initial map hash).
            device: Torch device.
        
        Returns:
            V_r value of shape (1,), negative.
        """
        map_hash = _get_map_hash(world_model)
        normalized_state = self._normalize_state(state)
        key = hash((normalized_state, map_hash))
        param = self._get_or_create_entry(key, device)
        raw_output = param.view(1)
        return self.ensure_negative(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states (explicit interface).
        
        Args:
            states: List of hashable states.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            V_r values of shape (batch_size,).
        """
        return self._batch_forward(states, map_hash, device)
    
    def forward_batch(
        self,
        states: List[Hashable],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states (unified interface).
        
        This matches the neural network's forward_batch signature.
        Uses world_model to get the initial map hash for ensemble mode.
        
        Args:
            states: List of hashable states.
            world_model: Environment (used to get initial map hash).
            device: Target device.
        
        Returns:
            V_r values of shape (batch_size,), negative.
        """
        map_hash = _get_map_hash(world_model)
        return self._batch_forward(states, map_hash, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'gamma_r': self.gamma_r,
            'default_v_r': self.default_v_r,
            'include_step_count': self.include_step_count,
            'table_size': len(self.table)
        }
    
    def parameters(self, recurse: bool = True):
        """Return iterator over all lookup table entries."""
        return iter(self.table.values())
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters (table entries with their keys as names)."""
        for key, param in self.table.items():
            name = f"{prefix}table.{key}" if prefix else f"table.{key}"
            yield name, param
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return state dict containing all table entries."""
        state = {}
        for key, param in self.table.items():
            state[f"{prefix}table.{key}"] = param if keep_vars else param.data.clone()
        state[f"{prefix}_table_keys"] = list(self.table.keys())
        state[f"{prefix}_config"] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict containing table entries."""
        # Restore config if present
        config_key = '_config'
        if config_key in state_dict:
            config = state_dict[config_key]
            if 'include_step_count' in config:
                self.include_step_count = config['include_step_count']
        
        prefix = ''
        keys_key = f"{prefix}_table_keys"
        if keys_key not in state_dict:
            keys = [
                int(k.replace(f"{prefix}table.", ''))
                for k in state_dict.keys()
                if k.startswith(f"{prefix}table.")
            ]
        else:
            keys = state_dict[keys_key]
        
        self.table.clear()
        for key in keys:
            param_key = f"{prefix}table.{key}"
            if param_key in state_dict:
                self.table[key] = nn.Parameter(state_dict[param_key].clone())
    
    def to(self, device):
        """Move all table entries to device."""
        for key in list(self.table.keys()):
            self.table[key] = nn.Parameter(self.table[key].to(device))
        return self
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def __repr__(self):
        return (
            f"LookupTableRobotValueNetwork("
            f"gamma_r={self.gamma_r}, "
            f"table_size={len(self.table)})"
        )
