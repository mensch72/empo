"""
Lookup Table Aggregate Goal Ability Network for Phase 2.

Implements X_h(s) as a dictionary-based lookup table with one entry
per unique (state, human_idx) pair.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..aggregate_goal_ability import BaseAggregateGoalAbilityNetwork


class LookupTableAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """
    Lookup table implementation of X_h (aggregate goal ability) for Phase 2.
    
    Stores X_h values in a dictionary keyed by (state, human_idx) hash. Each entry
    is a torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    X_h(s) = E_{g_h}[V_h^e(s, g_h)^ζ] ∈ (0, 1]
    
    This represents the aggregate ability of human h to achieve various goals,
    weighted by the goal sampler and raised to power ζ (risk preference).
    
    Args:
        zeta: Risk/reliability preference parameter (ζ >= 1, 1 = neutral).
        default_x_h: Initial X_h value for unseen (state, human) pairs.
            Should be in (0, 1]. 0.5 is neutral.
        feasible_range: Output bounds for X_h (default (0, 1]).
        state_encoder: Optional state encoder (for API compatibility with neural networks).
        agent_encoder: Optional agent encoder (for API compatibility with neural networks).
        own_agent_encoder: Optional own agent encoder (for API compatibility).
    """
    
    def __init__(
        self,
        zeta: float = 2.0,
        default_x_h: float = 0.5,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        state_encoder: Optional[nn.Module] = None,
        agent_encoder: Optional[nn.Module] = None,
        own_agent_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        self.default_x_h = default_x_h
        
        # Store encoders for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        self.agent_encoder = agent_encoder
        self.own_agent_encoder = own_agent_encoder
        
        # Main lookup table: hash((state, human_idx)) -> Parameter(X_h value)
        self.table: Dict[int, nn.Parameter] = {}
    
    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the (state, human_idx) pair.
            device: Target device for the parameter.
        
        Returns:
            Parameter containing X_h value for this (state, human).
        """
        if key not in self.table:
            self.table[key] = nn.Parameter(
                torch.tensor([self.default_x_h], dtype=torch.float32, device=device)
            )
        return self.table[key]
    
    def forward(
        self,
        states: List[Hashable],
        human_indices: List[int],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states and human indices.
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices (one per state).
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,), in (0, 1].
        """
        if len(states) != len(human_indices):
            raise ValueError(f"states and human_indices must have same length")
        
        batch_size = len(states)
        
        # Collect parameters for all (state, human_idx) pairs
        params = []
        for state, human_idx in zip(states, human_indices):
            key = hash((state, human_idx))
            param = self._get_or_create_entry(key, device)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Apply clamping to (0, 1]
        return self.apply_clamp(raw_output)
    
    def encode_and_forward(
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
            world_model: Environment/world model (not used for lookup tables).
            human_agent_idx: Index of the human agent.
            device: Torch device.
        
        Returns:
            X_h value of shape (1,), in (0, 1].
        """
        key = hash((state, human_agent_idx))
        param = self._get_or_create_entry(key, device)
        raw_output = param.view(1)
        return self.apply_clamp(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        human_indices: List[int],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states and human indices (explicit interface).
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices.
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,).
        """
        return self.forward(states, human_indices, device)
    
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
        The world_model parameter is ignored (lookup tables don't need tensorization).
        
        Args:
            states: List of hashable states.
            human_indices: List of human agent indices.
            world_model: Environment (ignored for lookup tables).
            device: Target device.
        
        Returns:
            X_h values of shape (batch_size,), in (0, 1].
        """
        return self.forward(states, human_indices, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'zeta': self.zeta,
            'default_x_h': self.default_x_h,
            'feasible_range': self.feasible_range,
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
            f"LookupTableAggregateGoalAbilityNetwork("
            f"zeta={self.zeta}, "
            f"table_size={len(self.table)}, "
            f"default_x_h={self.default_x_h})"
        )
