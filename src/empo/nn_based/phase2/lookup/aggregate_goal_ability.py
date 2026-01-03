"""
Lookup Table Aggregate Goal Ability Network for Phase 2.

Implements X_h(s) as a dictionary-based lookup table with one entry
per unique (state, human_idx, map_hash) triple. The map_hash distinguishes
states from different map configurations in ensemble mode.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from .robot_q_network import _get_map_hash


class LookupTableAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """
    Lookup table implementation of X_h (aggregate goal ability) for Phase 2.
    
    Stores X_h values in a dictionary keyed by (state, human_idx, map_hash) hash. Each
    entry is a torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    The map_hash distinguishes states from different map configurations in ensemble mode,
    where different worlds may have the same mutable state but different wall layouts.
    
    X_h(s) = E_{g_h}[V_h^e(s, g_h)^ζ] ∈ (0, 1]
    
    This represents the aggregate ability of human h to achieve various goals,
    weighted by the goal sampler and raised to power ζ (risk preference).
    
    Args:
        zeta: Risk/reliability preference parameter (ζ >= 1, 1 = neutral).
        default_x_h: Initial X_h value for unseen (state, human) pairs.
            Should be in (0, 1]. 0.5 is neutral.
        feasible_range: Output bounds for X_h (default (0, 1]).
        include_step_count: If False, strip step_count from state before hashing.
        state_encoder: Optional state encoder (for API compatibility with neural networks).
        agent_encoder: Optional agent encoder (for API compatibility with neural networks).
        own_agent_encoder: Optional own agent encoder (for API compatibility).
    """
    
    def __init__(
        self,
        zeta: float = 2.0,
        default_x_h: float = 0.5,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        include_step_count: bool = True,
        state_encoder: Optional[nn.Module] = None,
        agent_encoder: Optional[nn.Module] = None,
        own_agent_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        self.default_x_h = default_x_h
        self.include_step_count = include_step_count
        
        # Store encoders for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        self.agent_encoder = agent_encoder
        self.own_agent_encoder = own_agent_encoder
        
        # Main lookup table: hash((state, human_idx)) -> Parameter(X_h value)
        self.table: Dict[int, nn.Parameter] = {}
        
        # Track new parameters for incremental optimizer updates
        self._new_params: List[nn.Parameter] = []
        
        # Track update counts per entry for adaptive learning rate
        self._update_counts: Dict[int, int] = {}
    
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

    def get_update_count(self, key: int) -> int:
        """Get the update count for a specific entry."""
        return self._update_counts.get(key, 0)
    
    def increment_update_counts(self, keys: List[int]) -> None:
        """Increment update counts for the given keys."""
        for key in keys:
            self._update_counts[key] = self._update_counts.get(key, 0) + 1
    
    def scale_gradients_by_update_count(self, min_lr: float = 1e-6) -> List[int]:
        """
        Scale gradients by 1/update_count for adaptive learning rate.
        
        Returns list of keys that had gradients (for incrementing update counts).
        """
        keys_with_grads = []
        for key, param in self.table.items():
            if param.grad is not None and param.grad.abs().sum() > 0:
                update_count = self._update_counts.get(key, 0) + 1
                effective_lr = max(min_lr, 1.0 / update_count)
                param.grad.mul_(effective_lr)
                keys_with_grads.append(key)
        return keys_with_grads

    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the (state, human_idx, map_hash) triple.
            device: Target device for the parameter.
        
        Returns:
            Parameter containing X_h value for this (state, human, map_hash).
        """
        if key not in self.table:
            param = nn.Parameter(
                torch.tensor([self.default_x_h], dtype=torch.float32, device=device)
            )
            self.table[key] = param
            self._new_params.append(param)
        return self.table[key]
    
    def _batch_forward(
        self,
        states: List[Hashable],
        human_indices: List[int],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Internal: Batch forward pass from raw states and human indices.
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices (one per state).
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,), in (0, 1].
        """
        if len(states) != len(human_indices):
            raise ValueError(f"states and human_indices must have same length")
        
        # Collect parameters for all (state, human_idx, map_hash) triples
        params = []
        for state, human_idx in zip(states, human_indices):
            # Use (state, human_idx, map_hash) as the cache key to distinguish states from different maps
            normalized_state = self._normalize_state(state)
            key = hash((normalized_state, human_idx, map_hash))
            param = self._get_or_create_entry(key, device)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Apply clamping to (0, 1]
        return self.apply_clamp(raw_output)
    
    def forward(
        self,
        state: Hashable,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Single-state forward pass.
        
        Args:
            state: Environment state (must be hashable).
            world_model: Environment/world model (used to get initial map hash).
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h value of shape (1,), in (0, 1].
        """
        map_hash = _get_map_hash(world_model)
        normalized_state = self._normalize_state(state)
        key = hash((normalized_state, human_agent_idx, map_hash))
        param = self._get_or_create_entry(key, device)
        raw_output = param.view(1)
        return self.apply_clamp(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        human_indices: List[int],
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states and human indices (explicit interface).
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,).
        """
        return self._batch_forward(states, human_indices, map_hash, device)
    
    def forward_batch(
        self,
        states: List[Hashable],
        human_indices: List[int],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states and human indices (unified interface).
        
        This matches the neural network's forward_batch signature.
        Uses world_model to get the initial map hash for ensemble mode.
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices.
            world_model: Environment (used to get initial map hash).
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,), in (0, 1].
        """
        map_hash = _get_map_hash(world_model)
        return self._batch_forward(states, human_indices, map_hash, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'zeta': self.zeta,
            'default_x_h': self.default_x_h,
            'feasible_range': self.feasible_range,
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
        """Return state dict containing all table entries and update counts."""
        state = {}
        for key, param in self.table.items():
            state[f"{prefix}table.{key}"] = param if keep_vars else param.data.clone()
        state[f"{prefix}_table_keys"] = list(self.table.keys())
        state[f"{prefix}_config"] = self.get_config()
        state[f"{prefix}_update_counts"] = dict(self._update_counts)
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
        
        # Load update counts if present
        update_counts_key = f"{prefix}_update_counts"
        if update_counts_key in state_dict:
            self._update_counts = dict(state_dict[update_counts_key])
        else:
            self._update_counts = {}
    
    def to(self, device):
        """Move all table entries to device."""
        for key in list(self.table.keys()):
            self.table[key] = nn.Parameter(self.table[key].to(device))
        return self
    
    def zero_grad(self):
        """Zero gradients for all table entries."""
        for param in self.table.values():
            if param.grad is not None:
                param.grad.zero_()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def __repr__(self):
        return (
            f"LookupTableAggregateGoalAbilityNetwork("
            f"zeta={self.zeta}, "
            f"table_size={len(self.table)}, "
            f"default_x_h={self.default_x_h})"
        )
