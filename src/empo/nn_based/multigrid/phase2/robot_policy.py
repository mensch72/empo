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
        # Load policy directly from checkpoint (recommended)
        policy = MultiGridRobotPolicy.from_checkpoint("policy.pt", device="cpu")
        
        # Get actions during rollout
        action = policy.get_action(state, env)
        
        # Alternative: provide your own Q network
        q_network = MultiGridRobotQNetwork(...)
        policy = MultiGridRobotPolicy(q_network, policy_path="policy.pt")
    """
    
    def __init__(
        self,
        q_network: MultiGridRobotQNetwork,
        beta_r: float = 10.0,
        device: str = 'cpu',
        policy_path: Optional[str] = None
    ):
        """
        Initialize the multigrid robot policy.
        
        Args:
            q_network: The Q_r network with MultiGridStateEncoders.
            beta_r: Power-law policy exponent.
            device: Torch device for inference.
            policy_path: Optional path to load pre-trained policy weights.
        """
        super().__init__(q_network, beta_r, device, policy_path)
    
    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = 'cpu'
    ) -> 'MultiGridRobotPolicy':
        """
        Create a policy directly from a saved checkpoint.
        
        This is the recommended way to load a trained policy for rollouts.
        The checkpoint must have been created by trainer.save_policy().
        
        Args:
            path: Path to the policy checkpoint.
            device: Torch device for inference.
        
        Returns:
            MultiGridRobotPolicy ready for inference.
        
        Example:
            policy = MultiGridRobotPolicy.from_checkpoint("policy.pt")
            action = policy.get_action(state, env)
        """
        # Load checkpoint to get config
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        if 'q_r_config' not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'q_r_config'. Was it saved with trainer.save_policy()?"
            )
        
        config = checkpoint['q_r_config']
        beta_r = checkpoint.get('beta_r', 10.0)
        
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
        
        # Create policy (will load weights)
        policy = cls(q_network, beta_r=beta_r, device=device)
        policy.q_network.load_state_dict(checkpoint['q_r'])
        policy.q_network.eval()
        
        return policy
    
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
