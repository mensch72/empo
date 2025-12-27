"""
Multigrid Robot Policy for Phase 2.

This module provides a deployable robot policy class for multigrid environments.
"""

from typing import Any, Optional, Tuple

import torch

from ...phase2.robot_policy import BaseRobotPolicy
from .robot_q_network import MultiGridRobotQNetwork


class MultiGridRobotPolicy(BaseRobotPolicy):
    """
    Robot policy for multigrid environments.
    
    This provides a minimal interface for running a trained robot policy
    in rollouts without the full training infrastructure.
    
    Example usage:
        # Create Q network (same architecture as training)
        q_network = MultiGridRobotQNetwork(...)
        
        # Create policy and load from checkpoint
        policy = MultiGridRobotPolicy(q_network, policy_path="policy.pt")
        
        # Get actions during rollout
        action = policy.get_action(state, env)
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
