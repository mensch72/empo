"""
Robot Policy for Phase 2.

This module provides a deployable robot policy class that can be loaded
from a saved checkpoint and used for rollouts/inference.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseRobotPolicy(ABC):
    """
    Base class for deployable robot policies.
    
    This provides a minimal interface for running a trained robot policy
    in rollouts without the full training infrastructure.
    
    Subclasses must implement environment-specific state encoding.
    """
    
    def __init__(
        self,
        q_network: nn.Module,
        beta_r: float = 10.0,
        device: str = 'cpu',
        policy_path: Optional[str] = None
    ):
        """
        Initialize the robot policy.
        
        Args:
            q_network: The Q_r network (must have state encoders).
            beta_r: Power-law policy exponent.
            device: Torch device for inference.
            policy_path: Optional path to load pre-trained policy weights.
        """
        self.q_network = q_network
        self.beta_r = beta_r
        self.device = device
        
        self.q_network.to(device)
        self.q_network.eval()
        
        if policy_path is not None:
            self.load_policy(policy_path)
    
    def load_policy(self, path: str, strict: bool = True) -> None:
        """
        Load policy weights from a saved checkpoint.
        
        Args:
            path: Path to the policy checkpoint saved by trainer.save_policy().
            strict: If True, requires all keys to match exactly.
        """
        # Note: weights_only=False is required for loading checkpoints with
        # nested state dicts. The checkpoint is trusted since it was created
        # by trainer.save_policy().
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_r'], strict=strict)
        
        if 'beta_r' in checkpoint:
            self.beta_r = checkpoint['beta_r']
        
        self.q_network.eval()
    
    @abstractmethod
    def encode_state(self, state: Any, world_model: Any) -> Tuple[torch.Tensor, ...]:
        """
        Encode state for the Q network.
        
        Args:
            state: Raw environment state.
            world_model: Environment/world model for encoding.
        
        Returns:
            Tuple of encoded state tensors ready for Q network forward pass.
        """
        pass
    
    def sample(
        self,
        state: Any,
        world_model: Any
    ) -> Tuple[int, ...]:
        """
        Sample robot action for a state.
        
        Uses the power-law policy with beta_r to sample from Q-values.
        
        Args:
            state: Current environment state.
            world_model: Environment/world model for state encoding.
        
        Returns:
            Tuple of robot actions.
        """
        with torch.no_grad():
            encoded = self.encode_state(state, world_model)
            q_values = self.q_network.forward(*encoded)
            return self.q_network.sample_action(q_values, epsilon=0.0, beta_r=self.beta_r)
    
    def get_q_values(self, state: Any, world_model: Any) -> torch.Tensor:
        """
        Get Q-values for all actions.
        
        Args:
            state: Current environment state.
            world_model: Environment/world model for state encoding.
        
        Returns:
            Q-values tensor (1, num_action_combinations).
        """
        with torch.no_grad():
            encoded = self.encode_state(state, world_model)
            return self.q_network.forward(*encoded)
