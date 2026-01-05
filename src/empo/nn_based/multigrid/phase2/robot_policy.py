"""
Multigrid Robot Policy for Phase 2.

This module provides deployable robot policy classes for multigrid environments.

Includes:
- MultiGridRobotPolicy: Learned policy from trained Q network
- MultiGridRobotExplorationPolicy: Simple exploration policy for epsilon-greedy training
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from ...phase2.robot_policy import RobotPolicy
from ...phase2.lookup.robot_q_network import LookupTableRobotQNetwork
from ..state_encoder import MultiGridStateEncoder
from .robot_q_network import MultiGridRobotQNetwork


class MultiGridRobotPolicy(RobotPolicy):
    """
    Robot policy for multigrid environments.
    
    This provides a minimal interface for running a trained robot policy
    in rollouts without the full training infrastructure.
    
    The policy loads from a checkpoint containing the Q network, its config,
    and the beta_r parameter. At deployment time:
    1. Call reset(world_model) at the start of each episode
    2. Call sample(state) to get actions during the episode
    
    Example usage:
        # Load policy from checkpoint
        policy = MultiGridRobotPolicy(path="policy.pt")
        
        # At start of episode
        policy.reset(env)
        
        # During episode
        state = env.get_state()
        action = policy.sample(state)
    """
    
    def __init__(
        self,
        q_network: Optional[Union[MultiGridRobotQNetwork, LookupTableRobotQNetwork]] = None,
        beta_r: float = 10.0,
        device: str = 'cpu',
        path: Optional[str] = None
    ):
        """
        Initialize the multigrid robot policy.
        
        Args:
            q_network: Optional Q_r network (neural or lookup table). If None, must provide path to load from.
            beta_r: Power-law policy exponent (default if not in checkpoint).
            device: Torch device for inference.
            path: Path to policy checkpoint saved by trainer.save_policy().
                  If provided without q_network, reconstructs network from config.
        
        Examples:
            # From checkpoint (recommended):
            policy = MultiGridRobotPolicy(path="policy.pt")
            
            # With custom Q network:
            policy = MultiGridRobotPolicy(q_network=my_q_network)
        """
        self.device = device
        self._world_model = None  # Set by reset()
        self._is_lookup_table = False  # Track policy type for sample()
        
        if q_network is None:
            if path is None:
                raise ValueError(
                    "Either q_network or path must be provided. "
                    "Use: MultiGridRobotPolicy(path='policy.pt')"
                )
            
            # Load checkpoint and reconstruct Q network
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
            if 'q_r_config' not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'q_r_config'. Was it saved with trainer.save_policy()?"
                )
            
            config = checkpoint['q_r_config']
            self.beta_r = checkpoint.get('beta_r', beta_r)
            
            # Check if this is a lookup table policy
            if config.get('type') == 'lookup_table':
                self._is_lookup_table = True
                self.q_network = LookupTableRobotQNetwork(
                    num_actions=config['num_actions'],
                    num_robots=config['num_robots'],
                    beta_r=self.beta_r,
                    default_q_r=config.get('default_q_r', -1.0),
                    feasible_range=config.get('feasible_range'),
                )
                self.q_network.load_state_dict(checkpoint['q_r'])
            else:
                # Neural network policy - get state encoder config
                state_enc_config = config['state_encoder_config']
                own_state_enc_config = config.get('own_state_encoder_config', state_enc_config)
                
                # Check if networks were trained with use_encoders=False (identity mode)
                use_encoders = state_enc_config.get('use_encoders', True)
                
                # Create state encoders with correct use_encoders flag
                # This ensures identity mode produces correct feature_dim
                state_encoder = MultiGridStateEncoder(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_agents_per_color=state_enc_config['num_agents_per_color'],
                    num_agent_colors=state_enc_config.get('num_agent_colors', 7),
                    feature_dim=state_enc_config.get('feature_dim', config['hidden_dim']),
                    max_kill_buttons=state_enc_config.get('max_kill_buttons', 4),
                    max_pause_switches=state_enc_config.get('max_pause_switches', 4),
                    max_disabling_switches=state_enc_config.get('max_disabling_switches', 4),
                    max_control_buttons=state_enc_config.get('max_control_buttons', 4),
                    use_encoders=use_encoders,
                )
                
                own_use_encoders = own_state_enc_config.get('use_encoders', True)
                own_state_encoder = MultiGridStateEncoder(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_agents_per_color=own_state_enc_config['num_agents_per_color'],
                    num_agent_colors=own_state_enc_config.get('num_agent_colors', 7),
                    feature_dim=own_state_enc_config.get('feature_dim', config['hidden_dim']),
                    max_kill_buttons=own_state_enc_config.get('max_kill_buttons', 4),
                    max_pause_switches=own_state_enc_config.get('max_pause_switches', 4),
                    max_disabling_switches=own_state_enc_config.get('max_disabling_switches', 4),
                    max_control_buttons=own_state_enc_config.get('max_control_buttons', 4),
                    use_encoders=own_use_encoders,
                    share_cache_with=state_encoder,
                )
                
                # Reconstruct Q network from config with pre-built state encoders
                self.q_network = MultiGridRobotQNetwork(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_robot_actions=config['num_robot_actions'],
                    num_robots=config['num_robots'],
                    hidden_dim=config['hidden_dim'],
                    beta_r=self.beta_r,
                    feasible_range=config.get('feasible_range'),
                    dropout=config.get('dropout', 0.0),
                    # Pass pre-built state encoders instead of individual params
                    num_agents_per_color=state_enc_config['num_agents_per_color'],
                    num_agent_colors=state_enc_config.get('num_agent_colors', 7),
                    state_feature_dim=state_enc_config.get('feature_dim', config['hidden_dim']),
                    state_encoder=state_encoder,
                    own_state_encoder=own_state_encoder,
                )
                
                # Load weights with validation
                checkpoint_keys = set(checkpoint['q_r'].keys())
                model_keys = set(self.q_network.state_dict().keys())
                
                if checkpoint_keys != model_keys:
                    missing_in_checkpoint = model_keys - checkpoint_keys
                    extra_in_checkpoint = checkpoint_keys - model_keys
                    
                    # Check if this is an encoder mismatch
                    encoder_keys_missing = any('encoder' in k for k in missing_in_checkpoint)
                    
                    if encoder_keys_missing:
                        raise ValueError(
                            f"State dict mismatch - likely encoder configuration issue.\n"
                            f"Config says use_encoders={use_encoders}, own_use_encoders={own_use_encoders}\n"
                            f"Missing in checkpoint: {sorted(missing_in_checkpoint)[:5]}{'...' if len(missing_in_checkpoint) > 5 else ''}\n"
                            f"Extra in checkpoint: {sorted(extra_in_checkpoint)[:5]}{'...' if len(extra_in_checkpoint) > 5 else ''}\n"
                            f"This may happen if the checkpoint was saved with different use_encoders settings.\n"
                            f"Try deleting old checkpoint files and re-training."
                        )
                
                self.q_network.load_state_dict(checkpoint['q_r'])
        else:
            self.q_network = q_network
            self.beta_r = beta_r
            self._is_lookup_table = isinstance(q_network, LookupTableRobotQNetwork)
            
            # Load weights if path provided
            if path is not None:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                self.q_network.load_state_dict(checkpoint['q_r'])
                self.beta_r = checkpoint.get('beta_r', beta_r)
        
        self.q_network.to(device)
        self.q_network.eval()
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at the start of an episode.
        
        Caches the world model reference needed for state tensorization.
        The world model provides static grid information (walls, doors, etc.)
        that is needed to convert states to tensor form.
        
        Args:
            world_model: MultiGridEnv or similar with grid information.
        """
        self._world_model = world_model
    
    def sample(self, state: Any) -> Any:
        """
        Sample an action for the given state.
        
        Args:
            state: Multigrid state tuple (step_count, agent_states, mobile_objects, mutable_objects).
        
        Returns:
            Tuple of actions, one per robot. Each action is an integer.
        
        Raises:
            RuntimeError: If reset() was not called before sample().
        """
        if self._world_model is None:
            raise RuntimeError(
                "Must call reset(world_model) before sample(). "
                "The world model is needed to tensorize the state."
            )
        
        with torch.no_grad():
            if self._is_lookup_table:
                # Lookup table uses forward() with raw state directly
                q_values = self.q_network.forward(state, self._world_model, self.device)
                # Sample action from power-law policy
                actions = self.q_network.sample_action(q_values)
            else:
                # Neural network needs tensorized state
                shared_tensors = self.q_network.state_encoder.tensorize_state(
                    state, self._world_model, self.device
                )
                own_tensors = self.q_network.own_state_encoder.tensorize_state(
                    state, self._world_model, self.device
                )
                
                # Combine into input format expected by Q network
                # shared: (grid, global, agent, interactive)
                # own: (grid, global, agent, interactive)
                all_tensors = (*shared_tensors, *own_tensors)
                
                # Sample action using the power-law policy
                actions = self.q_network.sample_action(*all_tensors)
            
            return actions


# Direction vectors for forward movement (matching DIR_TO_VEC in multigrid.py)
DIR_TO_VEC = [
    (1, 0),   # right (positive X)
    (0, 1),   # down (positive Y)
    (-1, 0),  # left (negative X)
    (0, -1),  # up (negative Y)
]


# Exploration policies moved to exploration_policies.py
# Import them here for backwards compatibility
from .exploration_policies import (
    MultiGridRobotExplorationPolicy,
    MultiGridHumanExplorationPolicy,
    MultiGridMultiStepExplorationPolicy,
)
