"""
Lookup Table Human Goal Achievement Network for Phase 2.

Implements V_h^e(s, g_h) as a dictionary-based lookup table with one entry
per unique (state, goal) pair.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..human_goal_ability import BaseHumanGoalAchievementNetwork


class LookupTableHumanGoalAbilityNetwork(BaseHumanGoalAchievementNetwork):
    """
    Lookup table implementation of V_h^e (human goal achievement ability) for Phase 2.
    
    Stores V_h^e values in a dictionary keyed by (state, goal) hash. Each entry is a
    torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    V_h^e(s, g_h) ∈ [0, 1] represents the probability that human h achieves goal g_h
    under the current robot policy.
    
    Args:
        gamma_h: Human discount factor.
        default_v_h_e: Initial V_h^e value for unseen (state, goal) pairs.
            Should be in [0, 1]. 0.5 is neutral, higher is optimistic.
        feasible_range: Output bounds for V_h^e (default [0, 1]).
        state_encoder: Optional state encoder (for API compatibility with neural networks).
        goal_encoder: Optional goal encoder (for API compatibility with neural networks).
        agent_encoder: Optional agent encoder (for API compatibility with neural networks).
    """
    
    def __init__(
        self,
        gamma_h: float = 0.99,
        default_v_h_e: float = 0.5,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        state_encoder: Optional[nn.Module] = None,
        goal_encoder: Optional[nn.Module] = None,
        agent_encoder: Optional[nn.Module] = None,
    ):
        super().__init__(gamma_h=gamma_h, feasible_range=feasible_range)
        self.default_v_h_e = default_v_h_e
        
        # Store encoders for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.agent_encoder = agent_encoder
        
        # Main lookup table: hash((state, goal)) -> Parameter(V_h^e value)
        self.table: Dict[int, nn.Parameter] = {}
    
    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the (state, goal) pair.
            device: Target device for the parameter.
        
        Returns:
            Parameter containing V_h^e value for this (state, goal).
        """
        if key not in self.table:
            # Store raw value that will become default_v_h_e after clamping
            self.table[key] = nn.Parameter(
                torch.tensor([self.default_v_h_e], dtype=torch.float32, device=device)
            )
        return self.table[key]
    
    def forward(
        self,
        states: List[Hashable],
        goals: List[Hashable],
        human_indices: Optional[List[int]] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states and goals.
        
        Args:
            states: List of hashable states.
            goals: List of hashable goals (one per state).
            human_indices: Optional list of human indices (not used for lookup).
            device: Target device.
        
        Returns:
            V_h^e values of shape (batch_size,), in [0, 1].
        """
        if len(states) != len(goals):
            raise ValueError(f"states and goals must have same length, got {len(states)} and {len(goals)}")
        
        batch_size = len(states)
        
        # Collect parameters for all (state, goal) pairs
        params = []
        for state, goal in zip(states, goals):
            key = hash((state, goal))
            param = self._get_or_create_entry(key, device)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Apply clamping to [0, 1]
        return self.apply_clamp(raw_output)
    
    def encode_and_forward(
        self,
        state: Hashable,
        world_model: Any,
        human_agent_idx: int,
        goal: Hashable,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Single-state forward pass.
        
        Args:
            state: Environment state (must be hashable).
            world_model: Environment/world model (not used for lookup tables).
            human_agent_idx: Index of the human agent (not used for lookup).
            goal: The goal g_h (must be hashable).
            device: Torch device.
        
        Returns:
            V_h^e value of shape (1,), in [0, 1].
        """
        key = hash((state, goal))
        param = self._get_or_create_entry(key, device)
        raw_output = param.view(1)
        return self.apply_clamp(raw_output)
    
    def forward_from_states_and_goals(
        self,
        states: List[Hashable],
        goals: List[Hashable],
        human_indices: Optional[List[int]] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states and goals (explicit interface).
        
        Args:
            states: List of hashable states.
            goals: List of hashable goals.
            human_indices: Optional list of human indices.
            device: Target device.
        
        Returns:
            V_h^e values of shape (batch_size,).
        """
        return self.forward(states, goals, human_indices, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'gamma_h': self.gamma_h,
            'default_v_h_e': self.default_v_h_e,
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
            f"LookupTableHumanGoalAbilityNetwork("
            f"gamma_h={self.gamma_h}, "
            f"table_size={len(self.table)}, "
            f"default_v_h_e={self.default_v_h_e})"
        )
