import torch
import torch.nn as nn
from typing import Any, List
from abc import ABC, abstractmethod

class BaseInverseDynamicsNetwork(nn.Module, ABC):
    """
    Learns the inverse-dynamics posterior P_theta(a_h | s, s').

    Takes (s, s') and the focal human index as input and outputs a vector of
    length num_actions containing logits over the focal human action.
    This acts as an inverse-dynamics model trained via cross-entropy loss.
    """
    def __init__(self, config: Any, num_actions: int):
        super().__init__()
        self.config = config
        self.num_actions = num_actions

    @abstractmethod
    def forward(
        self,
        state: Any,
        next_state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Compute P_theta(a_h | s, s') logits for a single transition.
        
        Returns:
            Tensor of shape (1, num_actions) containing unnormalized logits.
        """
        pass

    @abstractmethod
    def forward_batch(
        self,
        states: List[Any],
        next_states: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Compute P_theta(a_h | s, s') logits for a batch.
        
        Returns:
            Tensor of shape (batch_size, num_actions) containing unnormalized logits.
        """
        pass
