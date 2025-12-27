"""
Robot Policy for Phase 2.

This module provides a deployable robot policy class that can be loaded
from a saved checkpoint and used for rollouts/inference on any compatible
environment.

The policy is trained on an ENSEMBLE of environments, so it does not store
any specific environment reference. The policy's sample(state) method takes
a state that contains all information needed for action selection.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class RobotPolicy(ABC):
    """
    Abstract base class for robot policies.
    
    Defines the minimal interface for sampling actions from a policy.
    All robot policies (learned, random, etc.) should inherit from this.
    """
    
    @abstractmethod
    def sample(self, state: Any) -> Any:
        """
        Sample an action for the given state.
        
        Args:
            state: Environment state. The format depends on the implementation.
                   For neural policies on multigrid, this should be a tuple of
                   (state_tuple, env) where env provides tensorization context.
        
        Returns:
            Action to take (format depends on implementation).
        """
        pass
