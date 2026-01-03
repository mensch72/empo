"""
Lookup Table Human Goal Achievement Network for Phase 2.

Implements V_h^e(s, g_h) as a dictionary-based lookup table with one entry
per unique (state, goal, map_hash) triple. The map_hash distinguishes states
from different map configurations in ensemble mode.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Hashable, List, Optional, Tuple

from ..human_goal_ability import BaseHumanGoalAchievementNetwork
from .robot_q_network import _get_map_hash


class LookupTableHumanGoalAbilityNetwork(BaseHumanGoalAchievementNetwork):
    """
    Lookup table implementation of V_h^e (human goal achievement ability) for Phase 2.
    
    Stores V_h^e values in a dictionary keyed by (state, goal, map_hash) hash. Each entry
    is a torch.nn.Parameter to enable gradient tracking and optimizer compatibility.
    
    The map_hash distinguishes states from different map configurations in ensemble mode,
    where different worlds may have the same mutable state but different wall layouts.
    
    V_h^e(s, g_h) ∈ [0, 1] represents the probability that human h achieves goal g_h
    under the current robot policy.
    
    Args:
        gamma_h: Human discount factor.
        default_v_h_e: Initial V_h^e value for unseen (state, goal) pairs.
            Should be in [0, 1]. 0.5 is neutral, higher is optimistic.
        feasible_range: Output bounds for V_h^e (default [0, 1]).
        include_step_count: If False, strip step_count from state before hashing.
            This allows states that differ only in step_count to share the same
            lookup entry, which is important for environments where step_count
            varies but doesn't affect goal achievability.
        state_encoder: Optional state encoder (for API compatibility with neural networks).
        goal_encoder: Optional goal encoder (for API compatibility with neural networks).
            If None, a NullGoalEncoder is created.
        agent_encoder: Optional agent encoder (for API compatibility with neural networks).
            If None, a NullAgentEncoder is created.
        state_feature_dim: Output dimension for null state encoder (default: 64).
        goal_feature_dim: Output dimension for null goal encoder (default: 32).
        agent_feature_dim: Output dimension for null agent encoder (default: 32).
    """
    
    def __init__(
        self,
        gamma_h: float = 0.99,
        default_v_h_e: float = 0.0,  # Start pessimistic - assume goal won't be achieved
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        include_step_count: bool = True,
        state_encoder: Optional[nn.Module] = None,
        goal_encoder: Optional[nn.Module] = None,
        agent_encoder: Optional[nn.Module] = None,
        state_feature_dim: int = 64,
        goal_feature_dim: int = 32,
        agent_feature_dim: int = 32,
    ):
        super().__init__(gamma_h=gamma_h, feasible_range=feasible_range)
        self.default_v_h_e = default_v_h_e
        self.include_step_count = include_step_count
        
        # Create null encoders for API compatibility with neural networks.
        # These output zeros, so networks using them must rely on their own encoders.
        from .null_encoders import NullStateEncoder, NullGoalEncoder, NullAgentEncoder
        
        self.state_encoder = state_encoder if state_encoder is not None else NullStateEncoder(state_feature_dim)
        self.goal_encoder = goal_encoder if goal_encoder is not None else NullGoalEncoder(goal_feature_dim)
        self.agent_encoder = agent_encoder if agent_encoder is not None else NullAgentEncoder(agent_feature_dim)
        
        # Main lookup table: hash((state, goal)) -> Parameter(V_h^e value)
        self.table: Dict[int, nn.Parameter] = {}
        
        # Track newly created parameters for incremental optimizer updates
        self._new_params: List[nn.Parameter] = []
        
        # Track update counts per entry for adaptive learning rate
        self._update_counts: Dict[int, int] = {}
    
    def get_new_params(self) -> List[nn.Parameter]:
        """
        Get and clear the list of newly created parameters.
        
        This allows the optimizer to incrementally add new parameters
        without full recreation.
        
        Returns:
            List of parameters created since last call.
        """
        new_params = self._new_params
        self._new_params = []
        return new_params

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
    
    def _normalize_state(self, state: Hashable) -> Hashable:
        """
        Normalize state for lookup key generation.
        
        If include_step_count is False and state is a tuple with step_count as first
        element (as in MultiGrid states), strip the step_count to allow states that
        differ only in step_count to share the same lookup entry.
        
        Args:
            state: Raw state from environment.
        
        Returns:
            Normalized state for use in lookup keys.
        """
        if not self.include_step_count and isinstance(state, tuple) and len(state) >= 2:
            # MultiGrid state format: (step_count, agent_states, mobile_objects, mutable_objects)
            # Strip step_count (first element)
            return state[1:]
        return state
    
    def _get_or_create_entry(self, key: int, device: str = 'cpu', state_for_debug: Hashable = None, goal_for_debug: Any = None, map_hash_for_debug: int = None) -> nn.Parameter:
        """
        Get entry for key, creating with default value if not present.
        
        Args:
            key: Hash key for the (state, goal, map_hash) triple.
            device: Target device for the parameter.
            state_for_debug: Original state (for debug warnings).
            goal_for_debug: Original goal (for debug warnings).
            map_hash_for_debug: Map hash (for debug warnings).
        
        Returns:
            Parameter containing V_h^e value for this (state, goal, map_hash).
        """
        is_new = key not in self.table
        if is_new:
            # DEBUG: Print when new key is created during training
            if getattr(self, 'debug_new_keys', False):
                goal_hash = hash(goal_for_debug) if goal_for_debug else 'unknown'
                goal_target = getattr(goal_for_debug, 'target_pos', str(goal_for_debug)[:30]) if goal_for_debug else 'unknown'
                print(f"[DEBUG V_h^e NEW #{len(self.table)+1}] key={key} = hash((state={state_for_debug}, goal={goal_hash} ({goal_target}), map_hash={map_hash_for_debug})), init_value={self.default_v_h_e:.4f}")
            # Store raw value that will become default_v_h_e after clamping
            param = nn.Parameter(
                torch.tensor([self.default_v_h_e], dtype=torch.float32, device=device)
            )
            self.table[key] = param
            self._new_params.append(param)
        
        # DEBUG: Print every lookup when debug_all_lookups is enabled (for test map generation)
        if getattr(self, 'debug_all_lookups', False):
            goal_hash = hash(goal_for_debug) if goal_for_debug else 'unknown'
            goal_target = getattr(goal_for_debug, 'target_pos', str(goal_for_debug)[:30]) if goal_for_debug else 'unknown'
            status = "NEW (not found!)" if is_new else "exists"
            value = self.table[key].item()
            print(f"[DEBUG V_h^e LOOKUP] key={key} = hash((state={state_for_debug}, goal={goal_hash} ({goal_target}), map_hash={map_hash_for_debug})) -> {status}, value={value:.4f}")
        
        return self.table[key]
    
    def _batch_forward(
        self,
        states: List[Hashable],
        goals: List[Hashable],
        human_indices: Optional[List[int]] = None,
        map_hash: int = 0,
        device: str = 'cpu',
        debug: bool = False
    ) -> torch.Tensor:
        """
        Internal: Batch forward pass from raw states and goals.
        
        Args:
            states: List of hashable states.
            goals: List of hashable goals (one per state).
            human_indices: Optional list of human indices (not used for lookup).
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
            debug: If True, print debug info about key collisions.
        
        Returns:
            V_h^e values of shape (batch_size,), in [0, 1].
        """
        if len(states) != len(goals):
            raise ValueError(f"states and goals must have same length, got {len(states)} and {len(goals)}")
                
        # Collect parameters for all (state, goal, map_hash) triples
        params = []
        if debug:
            key_to_inputs = {}  # Track which (state, goal) pairs map to each key
        for state, goal in zip(states, goals):
            # Use (state, goal, map_hash) as the cache key to distinguish states from different maps
            normalized_state = self._normalize_state(state)
            key = hash((normalized_state, goal, map_hash))
            if debug:
                goal_target = getattr(goal, 'target_pos', str(goal)[:30])
                if key in key_to_inputs:
                    prev_goal = key_to_inputs[key]
                    if prev_goal != goal_target:
                        print(f"[DEBUG] HASH COLLISION! key={key} maps to both {prev_goal} and {goal_target}")
                else:
                    key_to_inputs[key] = goal_target
            param = self._get_or_create_entry(key, device, state_for_debug=normalized_state, goal_for_debug=goal, map_hash_for_debug=map_hash)
            params.append(param.squeeze())
        
        # Stack into batch tensor
        raw_output = torch.stack(params, dim=0)
        
        # Apply clamping to [0, 1]
        return self.apply_clamp(raw_output)
    
    def forward(
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
            world_model: Environment/world model (used to get initial map hash).
            human_agent_idx: Index of the human agent (not used for lookup).
            goal: The goal g_h (must be hashable).
            device: Torch device.
        
        Returns:
            V_h^e value of shape (1,), in [0, 1].
        """
        map_hash = _get_map_hash(world_model)
        normalized_state = self._normalize_state(state)
        key = hash((normalized_state, goal, map_hash))
        param = self._get_or_create_entry(key, device, state_for_debug=normalized_state, goal_for_debug=goal, map_hash_for_debug=map_hash)
        raw_output = param.view(1)
        return self.apply_clamp(raw_output)
    
    def forward_from_states_and_goals(
        self,
        states: List[Hashable],
        goals: List[Hashable],
        human_indices: Optional[List[int]] = None,
        map_hash: int = 0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Forward pass from raw states and goals (explicit interface).
        
        Args:
            states: List of hashable states.
            goals: List of hashable goals.
            human_indices: Optional list of human indices.
            map_hash: Hash of the initial map configuration (for ensemble mode).
            device: Target device.
        
        Returns:
            V_h^e values of shape (batch_size,).
        """
        return self._batch_forward(states, goals, human_indices, map_hash, device)
    
    def forward_batch(
        self,
        states: List[Hashable],
        goals: List[Hashable],
        human_indices: List[int],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Batch forward pass from raw states, goals, and human indices (unified interface).
        
        This matches the neural network's forward_batch signature.
        Uses world_model to get the initial map hash for ensemble mode.
        
        Args:
            states: List of hashable states.
            goals: List of hashable goals (one per state).
            human_indices: List of human agent indices.
            world_model: Environment (used to get initial map hash).
            device: Target device.
        
        Returns:
            V_h^e values of shape (batch_size,), in [0, 1].
        """
        map_hash = _get_map_hash(world_model)
        return self._batch_forward(states, goals, human_indices, map_hash, device)
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            'type': 'lookup_table',
            'gamma_h': self.gamma_h,
            'default_v_h_e': self.default_v_h_e,
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
            f"LookupTableHumanGoalAbilityNetwork("
            f"gamma_h={self.gamma_h}, "
            f"table_size={len(self.table)}, "
            f"default_v_h_e={self.default_v_h_e})"
        )
