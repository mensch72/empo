"""
Trainer class for neural policy priors.
"""

import torch
import torch.optim as optim
from typing import Any, Optional
import random

from .q_network import BaseQNetwork
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    Generic trainer for neural policy priors.
    
    Handles the training loop, experience replay, and target network updates.
    Domain-specific state encoding is delegated to the Q-network.
    """
    
    def __init__(
        self,
        q_network: BaseQNetwork,
        target_network: BaseQNetwork,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.total_steps = 0
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def should_update_target(self) -> bool:
        """Check if target network should be updated."""
        return self.total_steps % self.target_update_freq == 0
    
    def store_transition(
        self,
        state: Any,
        action: int,
        next_state: Any,
        agent_idx: int,
        goal: Any
    ):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, next_state, agent_idx, goal)
        self.total_steps += 1
    
    def sample_action(
        self,
        state: Any,
        world_model: Any,
        agent_idx: int,
        goal: Any,
        epsilon: float = 0.0
    ) -> int:
        """
        Sample an action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            world_model: Environment.
            agent_idx: Agent index.
            goal: Current goal.
            epsilon: Exploration probability.
        
        Returns:
            Selected action index.
        """
        if random.random() < epsilon:
            return random.randint(0, self.q_network.num_actions - 1)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network.encode_and_forward(
                state, world_model, agent_idx, goal, self.device
            )
            probs = self.q_network.get_policy(q_values).squeeze(0)
            return torch.multinomial(probs, 1).item()
