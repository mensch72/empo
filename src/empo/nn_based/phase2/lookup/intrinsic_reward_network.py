"""
Lookup Table Intrinsic Reward Network for Phase 2.

Implements U_r(s) as a dictionary-based lookup table with one entry per unique
(state, map_hash) pair. The map_hash distinguishes states from different map
configurations in ensemble mode.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..intrinsic_reward_network import BaseIntrinsicRewardNetwork
from .robot_q_network import _get_map_hash


class LookupTableIntrinsicRewardNetwork(BaseIntrinsicRewardNetwork):
    """
    Lookup table implementation of U_r (intrinsic reward) for Phase 2.
    
    Stores the intermediate value y in a dictionary keyed by (state, map_hash) hash,
    where y = E_h[X_h(s)^{-ξ}] and U_r = -y^η.
    
    Each entry is a torch.nn.Parameter to enable gradient tracking and
    optimizer compatibility.
    
    The map_hash distinguishes states from different map configurations in ensemble mode,
    where different worlds may have the same mutable state but different wall layouts.
    
    Note: In practice, U_r is often computed directly from X_h values rather
    than learned separately (see u_r_use_network config option).
    
    Properties:
    - y ∈ [1, ∞) since X_h ∈ (0, 1] and X_h^{-ξ} ≥ 1 for ξ > 0
    - U_r = -y^η < 0 always
    
    Args:
        xi: Inter-human inequality aversion parameter (ξ >= 1).
        eta: Intertemporal inequality aversion parameter (η >= 1).
        default_y: Initial y value for unseen states (must be >= 1).
        include_step_count: If False, strip step_count from state before hashing.
        state_encoder: Optional state encoder (for API compatibility with neural networks).
    """
    
    def __init__(
        self,
        xi: float = 1.0,
        eta: float = 1.1,
        default_y: float = 2.0,
        include_step_count: bool = True,
        state_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(xi=xi, eta=eta)
        
        if default_y < 1.0:
            raise ValueError(f"default_y must be >= 1.0, got {default_y}")
        
        self.default_y = default_y
        self.include_step_count = include_step_count
        
        # Store encoder for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        
        # Main lookup table: hash(state) -> Parameter(log(y-1))
        # We store log(y-1) so that y = 1 + exp(log(y-1)) > 1 always
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
            Parameter containing log(y-1) for this state.
        """
        if key not in self.table:
            # Convert default_y to log(y-1) representation
            log_y_minus_1 = torch.log(torch.tensor(self.default_y - 1.0))
            param = nn.Parameter(
                torch.tensor([log_y_minus_1.item()], dtype=torch.float32, device=device)
            )
            self.table[key] = param
            self._new_params.append(param)
        return self.table[key]
    
    def _batch_forward(
        self,
        states: List[Hashable],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal: Batch forward pass from raw states.
        
        Args:
            states: List of hashable states.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            Tuple of (y, U_r) where:
            - y: shape (batch_size,), y > 1
            - U_r: shape (batch_size,), U_r < 0
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
        log_y_minus_1 = torch.stack(params, dim=0)
        
        # Convert to y and U_r
        y = self.log_y_minus_1_to_y(log_y_minus_1)
        u_r = self.y_to_u_r(y)
        
        return y, u_r
    
    def forward(
        self,
        state: Hashable,
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-state forward pass.
        
        Args:
            state: Environment state (must be hashable).
            world_model: Environment/world model (used to get initial map hash).
            device: Torch device.
        
        Returns:
            Tuple of (y, U_r) tensors, each shape (1,).
        """
        map_hash = _get_map_hash(world_model)
        normalized_state = self._normalize_state(state)
        key = hash((normalized_state, map_hash))
        param = self._get_or_create_entry(key, device)
        log_y_minus_1 = param.view(1)
        
        y = self.log_y_minus_1_to_y(log_y_minus_1)
        u_r = self.y_to_u_r(y)
        
        return y, u_r
    
    def forward_from_states(
        self,
        states: List[Hashable],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass from raw states (explicit interface).
        
        Args:
            states: List of hashable states.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            Tuple of (y, U_r) tensors.
        """
        return self._batch_forward(states, map_hash, device)
    
    def forward_batch(
        self,
        states: List[Hashable],
        world_model: Any,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch forward pass from raw states (unified interface).
        
        This matches the neural network's forward_batch signature.
        Uses world_model to get the initial map hash for ensemble mode.
        
        Args:
            states: List of hashable states.
            world_model: Environment (used to get initial map hash).
            device: Target device.
        
        Returns:
            Tuple of (y, U_r) tensors.
        """
        map_hash = _get_map_hash(world_model)
        return self._batch_forward(states, map_hash, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'xi': self.xi,
            'eta': self.eta,
            'default_y': self.default_y,
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
            f"LookupTableIntrinsicRewardNetwork("
            f"xi={self.xi}, eta={self.eta}, "
            f"table_size={len(self.table)}, "
            f"default_y={self.default_y})"
        )
