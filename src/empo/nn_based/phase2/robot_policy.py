"""
Robot Policy for Phase 2.

This module provides the abstract base class for robot policies that can be used
for rollouts/inference after training.

The policy is trained on an ENSEMBLE of environments, so it does not store
any specific environment reference. The policy's sample(state) method takes
only the state - any environment-specific context needed should be provided
via reset(world_model) at the start of each episode.
"""

from abc import ABC, abstractmethod
from typing import Any


class RobotPolicy(ABC):
    """
    Abstract base class for robot policies.
    
    Defines the minimal interface for sampling actions from a policy.
    All robot policies (learned, random, etc.) should inherit from this.
    
    The interface has two methods:
    - sample(state): Sample action(s) for the given state (required)
    - reset(world_model): Called at episode start to provide environment context (optional)
    
    Example usage:
        policy = SomeRobotPolicy(path="policy.pt")
        
        # At start of each episode
        policy.reset(env)
        
        # During episode
        state = env.get_state()
        action = policy.sample(state)
    """
    
    @abstractmethod
    def sample(self, state: Any) -> Any:
        """
        Sample an action for the given state.
        
        Args:
            state: Environment state. Format depends on implementation.
        
        Returns:
            Action to take. Format depends on implementation.
        """
        pass
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at the start of an episode.
        
        Called at the start of each episode to provide the policy with
        any world-model-specific context it needs (e.g., static grid layout).
        
        Some policies may not need this (e.g., if state contains all info).
        
        Args:
            world_model: The environment/world model for this episode.
        """
        pass  # Default: no-op
