"""
Lookup Table Robot Q-Network for Phase 2.

Implements Q_r(s, a_r) as a dictionary-based lookup table with one entry per unique
(state, map_hash) pair. The map_hash distinguishes states from different map
configurations in ensemble mode.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..robot_q_network import BaseRobotQNetwork


def _get_map_hash(world_model: Any) -> int:
    """
    Safely get the initial map hash from a world model.
    
    Returns 0 if the world model doesn't support get_initial_map_hash(),
    which maintains backward compatibility with non-MultiGrid environments.
    
    Args:
        world_model: The world model (environment).
        
    Returns:
        The map hash, or 0 if not available.
    """
    if hasattr(world_model, 'get_initial_map_hash'):
        return world_model.get_initial_map_hash()
    return 0


class LookupTableRobotQNetwork(BaseRobotQNetwork):
    """
    Lookup table implementation of robot Q-network for Phase 2.
    
    Stores Q_r values in a dictionary keyed by (state, map_hash) hash. Each entry is a
    torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    The map_hash distinguishes states from different map configurations in ensemble mode,
    where different worlds may have the same mutable state but different wall layouts.
    
    Q_r(s, a_r) < 0 always (since V_r < 0), ensured via -softplus output transformation.
    
    Args:
        num_actions: Number of actions per robot.
        num_robots: Number of robots in the fleet.
        beta_r: Power-law policy exponent (nominal value).
        default_q_r: Initial Q-value for unseen states (should be negative).
        feasible_range: Optional (min, max) bounds for Q-values.
        include_step_count: If False, strip step_count from state before hashing.
        state_encoder: Optional state encoder (for API compatibility with neural networks).
            Typically an identity encoder with use_encoders=False.
        own_state_encoder: Optional own state encoder (for API compatibility).
    """
    
    def __init__(
        self,
        num_actions: int,
        num_robots: int,
        beta_r: float = 10.0,
        default_q_r: float = -1.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        include_step_count: bool = True,
        state_encoder: Optional[nn.Module] = None,
        own_state_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=beta_r,
            feasible_range=feasible_range
        )
        self.default_q_r = default_q_r
        self.include_step_count = include_step_count
        
        # Store encoders for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        self.own_state_encoder = own_state_encoder
        
        # Main lookup table: hash(state) -> Parameter(Q-values)
        # Using a regular dict, entries are created lazily on first access
        self.table: Dict[int, nn.Parameter] = {}
        
        # Track new parameters for incremental optimizer updates
        self._new_params: List[nn.Parameter] = []
        
        # Track which keys were accessed in current forward pass (for gradient flow)
        self._accessed_keys: List[int] = []
    
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
            Parameter containing Q-values for this state.
        """
        if key not in self.table:
            # Create new entry with default value
            # Use raw values that will produce default_q_r after ensure_negative()
            # Since ensure_negative uses -softplus, we need to solve:
            # -softplus(x) = default_q_r => softplus(x) = -default_q_r
            # For small x: softplus(x) ≈ x, so x ≈ -default_q_r
            raw_default = -self.default_q_r  # This gives us ≈ default_q_r after -softplus
            param = nn.Parameter(
                torch.full(
                    (self.num_action_combinations,),
                    raw_default,
                    dtype=torch.float32,
                    device=device
                )
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
            Q_r values of shape (batch_size, num_action_combinations), all negative.
        """
        
        # Collect parameters for all states
        params = []
        for state in states:
            # Use (state, map_hash) as the cache key to distinguish states from different maps
            normalized_state = self._normalize_state(state)
            key = hash((normalized_state, map_hash))
            param = self._get_or_create_entry(key, device)
            params.append(param)
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Ensure Q_r < 0
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
            Q_r values of shape (1, num_action_combinations), all negative.
        """
        map_hash = _get_map_hash(world_model)
        normalized_state = self._normalize_state(state)
        key = hash((normalized_state, map_hash))
        param = self._get_or_create_entry(key, device)
        raw_output = param.unsqueeze(0)
        return self.ensure_negative(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states (explicit interface).
        
        This is an alias for _batch_forward() that matches the naming convention
        expected by the trainer for lookup table compatibility.
        
        Args:
            states: List of hashable states.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            Q_r values of shape (batch_size, num_action_combinations).
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
            Q_r values of shape (batch_size, num_action_combinations).
        """
        map_hash = _get_map_hash(world_model)
        return self._batch_forward(states, map_hash, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'num_actions': self.num_actions,
            'num_robots': self.num_robots,
            'beta_r': self.beta_r,
            'default_q_r': self.default_q_r,
            'feasible_range': self.feasible_range,
            'table_size': len(self.table)
        }
    
    def parameters(self, recurse: bool = True):
        """
        Return iterator over all lookup table entries.
        
        This allows standard PyTorch optimizers to update the table entries.
        Note: New entries added after optimizer creation won't be tracked
        unless the optimizer is recreated.
        """
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
        # Also save metadata
        state[f"{prefix}_table_keys"] = list(self.table.keys())
        state[f"{prefix}_config"] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict containing table entries."""
        # Find table keys
        prefix = ''
        keys_key = f"{prefix}_table_keys"
        if keys_key not in state_dict:
            # Try to infer keys from state_dict
            keys = [
                int(k.replace(f"{prefix}table.", ''))
                for k in state_dict.keys()
                if k.startswith(f"{prefix}table.")
            ]
        else:
            keys = state_dict[keys_key]
        
        # Load each entry
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
            f"LookupTableRobotQNetwork("
            f"num_actions={self.num_actions}, "
            f"num_robots={self.num_robots}, "
            f"table_size={len(self.table)}, "
            f"beta_r={self.beta_r})"
        )
