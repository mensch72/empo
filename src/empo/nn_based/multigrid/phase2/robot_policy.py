"""
Multigrid Robot Policy for Phase 2.

This module provides a deployable robot policy class for multigrid environments.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ...phase2.robot_policy import BaseRobotPolicy
from .robot_q_network import MultiGridRobotQNetwork


class MultiGridRobotPolicy(BaseRobotPolicy):
    """
    Robot policy for multigrid environments.
    
    This provides a minimal interface for running a trained robot policy
    in rollouts without the full training infrastructure.
    
    Example usage:
        # Load policy from checkpoint (recommended)
        policy = MultiGridRobotPolicy(path="policy.pt")
        
        # Get actions during rollout
        action = policy.get_action(state, env)
        
        # Alternative: provide your own Q network
        q_network = MultiGridRobotQNetwork(...)
        policy = MultiGridRobotPolicy(q_network=q_network, policy_path="policy.pt")
    """
    
    def __init__(
        self,
        q_network: Optional[MultiGridRobotQNetwork] = None,
        beta_r: float = 10.0,
        device: str = 'cpu',
        policy_path: Optional[str] = None,
        path: Optional[str] = None
    ):
        """
        Initialize the multigrid robot policy.
        
        Can be initialized either from a checkpoint path (recommended) or with
        a pre-created Q network.
        
        Args:
            q_network: Optional Q_r network. If None, must provide path to load from.
            beta_r: Power-law policy exponent (used if q_network provided without path).
            device: Torch device for inference.
            policy_path: Deprecated, use 'path' instead.
            path: Path to policy checkpoint saved by trainer.save_policy().
                  If provided, reconstructs Q network from checkpoint config.
        
        Examples:
            # From checkpoint (recommended):
            policy = MultiGridRobotPolicy(path="policy.pt")
            
            # With custom Q network:
            policy = MultiGridRobotPolicy(q_network=my_q_network, path="policy.pt")
        """
        # Handle path parameter (prefer 'path' over 'policy_path')
        checkpoint_path = path or policy_path
        
        if q_network is None:
            # Must have path to reconstruct network
            if checkpoint_path is None:
                raise ValueError(
                    "Either q_network or path must be provided. "
                    "Use: MultiGridRobotPolicy(path='policy.pt')"
                )
            
            # Load checkpoint and reconstruct Q network
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if 'q_r_config' not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'q_r_config'. Was it saved with trainer.save_policy()?"
                )
            
            config = checkpoint['q_r_config']
            beta_r = checkpoint.get('beta_r', beta_r)
            
            # Reconstruct Q network from config
            q_network = MultiGridRobotQNetwork(
                grid_height=config['grid_height'],
                grid_width=config['grid_width'],
                num_robot_actions=config['num_robot_actions'],
                num_robots=config['num_robots'],
                hidden_dim=config['hidden_dim'],
                beta_r=config.get('beta_r', beta_r),
                feasible_range=config.get('feasible_range'),
                dropout=config.get('dropout', 0.0),
                # State encoder config
                num_agents_per_color=config['state_encoder_config']['num_agents_per_color'],
                num_agent_colors=config['state_encoder_config'].get('num_agent_colors', 7),
                state_feature_dim=config['state_encoder_config'].get('state_feature_dim', 256),
                max_kill_buttons=config['state_encoder_config'].get('max_kill_buttons', 4),
                max_pause_switches=config['state_encoder_config'].get('max_pause_switches', 4),
                max_disabling_switches=config['state_encoder_config'].get('max_disabling_switches', 4),
                max_control_buttons=config['state_encoder_config'].get('max_control_buttons', 4),
            )
            
            # Initialize base class without loading (we'll load below)
            super().__init__(q_network, beta_r, device, policy_path=None)
            
            # Load weights
            self.q_network.load_state_dict(checkpoint['q_r'])
            self.q_network.eval()
        else:
            # Q network provided, use normal initialization
            super().__init__(q_network, beta_r, device, checkpoint_path)
    
    def encode_state(self, state: Any, world_model: Any) -> Tuple[torch.Tensor, ...]:
        """
        Encode multigrid state for the Q network.
        
        Args:
            state: Multigrid state tuple (step_count, agent_states, mobile_objects, mutable_objects).
            world_model: MultiGridEnv or similar with grid information.
        
        Returns:
            Tuple of encoded state tensors (shared_encoded..., own_encoded...).
        """
        # Encode with shared state encoder
        shared = self.q_network.state_encoder.tensorize_state(
            state, world_model, self.device
        )
        
        # Encode with own state encoder
        own = self.q_network.own_state_encoder.tensorize_state(
            state, world_model, self.device
        )
        
        return (*shared, *own)
