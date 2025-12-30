"""
Lookup Table Robot Value Network for Phase 2.

Implements V_r(s) as a dictionary-based lookup table with one entry per unique state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..robot_value_network import BaseRobotValueNetwork


class LookupTableRobotValueNetwork(BaseRobotValueNetwork):
    """
    Lookup table implementation of robot value network (V_r) for Phase 2.
    
    Stores V_r values in a dictionary keyed by state hash. Each entry is a
    torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    V_r(s) < 0 always (since U_r < 0 and Q_r < 0), ensured via ensure_negative().
    
    Note: In practice, V_r is often computed directly from U_r and Q_r rather
    than learned separately (see v_r_use_network config option).
    
    Args:
        gamma_r: Robot discount factor.
        default_v_r: Initial V-value for unseen states (should be negative).
    """
    
    def __init__(
        self,
        gamma_r: float = 0.99,
        default_v_r: float = -1.0
    ):
        super().__init__(gamma_r=gamma_r)
        self.default_v_r = default_v_r
        
        # Main lookup table: hash(state) -> Parameter(V_r value)
        self.table: Dict[int, nn.Parameter] = {}
    
    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the state.
            device: Target device for the parameter.
        
        Returns:
            Parameter containing V_r value for this state.
        """
        if key not in self.table:
            # Create new entry with default value
            # Use raw value that will produce default_v_r after ensure_negative()
            raw_default = -self.default_v_r
            self.table[key] = nn.Parameter(
                torch.tensor([raw_default], dtype=torch.float32, device=device)
            )
        return self.table[key]
    
    def forward(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states.
        
        Args:
            states: List of hashable states.
            device: Target device.
        
        Returns:
            V_r values of shape (batch_size,), all negative.
        """
        batch_size = len(states)
        
        # Collect parameters for all states
        params = []
        for state in states:
            key = hash(state)
            param = self._get_or_create_entry(key, device)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Ensure V_r < 0
        return self.ensure_negative(raw_output)
    
    def encode_and_forward(
        self,
        state: Hashable,
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Single-state forward pass.
        
        Args:
            state: Environment state (must be hashable).
            world_model: Environment/world model (not used for lookup tables).
            device: Torch device.
        
        Returns:
            V_r value of shape (1,), negative.
        """
        key = hash(state)
        param = self._get_or_create_entry(key, device)
        raw_output = param.view(1)
        return self.ensure_negative(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states (explicit interface).
        
        Args:
            states: List of hashable states.
            device: Target device.
        
        Returns:
            V_r values of shape (batch_size,).
        """
        return self.forward(states, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'gamma_r': self.gamma_r,
            'default_v_r': self.default_v_r,
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
