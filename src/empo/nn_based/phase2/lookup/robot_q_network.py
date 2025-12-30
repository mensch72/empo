"""
Lookup Table Robot Q-Network for Phase 2.

Implements Q_r(s, a_r) as a dictionary-based lookup table with one entry per unique state.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..robot_q_network import BaseRobotQNetwork


class LookupTableRobotQNetwork(BaseRobotQNetwork):
    """
    Lookup table implementation of robot Q-network for Phase 2.
    
    Stores Q_r values in a dictionary keyed by state hash. Each entry is a
    torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    Q_r(s, a_r) < 0 always (since V_r < 0), ensured via -softplus output transformation.
    
    Args:
        num_actions: Number of actions per robot.
        num_robots: Number of robots in the fleet.
        beta_r: Power-law policy exponent (nominal value).
        default_q_r: Initial Q-value for unseen states (should be negative).
        feasible_range: Optional (min, max) bounds for Q-values.
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
        
        # Store encoders for API compatibility (not used for computation)
        self.state_encoder = state_encoder
        self.own_state_encoder = own_state_encoder
        
        # Main lookup table: hash(state) -> Parameter(Q-values)
        # Using a regular dict, entries are created lazily on first access
        self.table: Dict[int, nn.Parameter] = {}
        
        # Track which keys were accessed in current forward pass (for gradient flow)
        self._accessed_keys: List[int] = []
    
    def _get_or_create_entry(self, key: int, device: str = 'cpu') -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the state.
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
            self.table[key] = nn.Parameter(
                torch.full(
                    (self.num_action_combinations,),
                    raw_default,
                    dtype=torch.float32,
                    device=device
                )
            )
        return self.table[key]
    
    def forward(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states.
        
        This is the primary forward method for lookup tables.
        Neural networks use tensorized inputs; lookup tables use raw states.
        
        Args:
            states: List of hashable states.
            device: Target device.
        
        Returns:
            Q_r values of shape (batch_size, num_action_combinations), all negative.
        """
        batch_size = len(states)
        
        # Collect parameters for all states
        params = []
        for state in states:
            key = hash(state)
            param = self._get_or_create_entry(key, device)
            params.append(param)
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Ensure Q_r < 0
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
            Q_r values of shape (1, num_action_combinations), all negative.
        """
        key = hash(state)
        param = self._get_or_create_entry(key, device)
        raw_output = param.unsqueeze(0)
        return self.ensure_negative(raw_output)
    
    def forward_from_states(
        self,
        states: List[Hashable],
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states (explicit interface).
        
        This is an alias for forward() that matches the naming convention
        expected by the trainer for lookup table compatibility.
        
        Args:
            states: List of hashable states.
            device: Target device.
        
        Returns:
            Q_r values of shape (batch_size, num_action_combinations).
        """
        return self.forward(states, device)
    
    def forward_batch(
        self,
        states: List[Hashable],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states (unified interface).
        
        This matches the neural network's forward_batch signature.
        The world_model parameter is ignored (lookup tables don't need tensorization).
        
        Args:
            states: List of hashable states.
            world_model: Environment (ignored for lookup tables).
            device: Target device.
        
        Returns:
            Q_r values of shape (batch_size, num_action_combinations).
        """
        return self.forward(states, device)
    
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
