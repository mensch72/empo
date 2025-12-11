"""
Trainer class for neural policy priors.
"""

import torch
import torch.optim as optim
from typing import Any, Callable, List, Optional, Dict
import random

from .q_network import BaseQNetwork
from .replay_buffer import ReplayBuffer


class Trainer:
    """
    Generic trainer for neural policy priors.
    
    Handles the training loop, experience replay, and target network updates.
    Domain-specific state encoding is delegated to the Q-network.
    
    The reward function computes rewards for transitions. It should take:
        (state, action, next_state, agent_idx, goal) -> reward (float)
    
    If no reward_fn is provided, a default goal-achievement reward is used:
        reward = 1.0 if goal.is_achieved(next_state) else 0.0
    """
    
    def __init__(
        self,
        q_network: BaseQNetwork,
        target_network: BaseQNetwork,
        optimizer: optim.Optimizer,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        target_update_freq: int = 100,
        device: str = 'cpu',
        exploration_policy: Optional[List[float]] = None,
        reward_fn: Optional[Callable[[Any, int, Any, int, Any], float]] = None
    ):
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device
        self.exploration_policy = exploration_policy
        self.reward_fn = reward_fn
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
            if self.exploration_policy is not None:
                # Support both callable (state-dependent) and list (static) policies
                if callable(self.exploration_policy):
                    weights = self.exploration_policy(state, world_model, agent_idx)
                else:
                    weights = self.exploration_policy
                return random.choices(range(self.q_network.num_actions), weights=weights)[0]
            else:
                return random.randint(0, self.q_network.num_actions - 1)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network.encode_and_forward(
                state, world_model, agent_idx, goal, self.device
            )
            probs = self.q_network.get_policy(q_values).squeeze(0)
            return torch.multinomial(probs, 1).item()
    
    def train_step(self, batch_size: int) -> Optional[float]:
        """
        Perform one training step with experience replay.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            Training loss if training occurred, None if not enough samples.
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        self.q_network.train()
        batch = self.replay_buffer.sample(batch_size)
        
        loss = self._compute_td_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        if self.should_update_target():
            self.update_target_network()
        
        return loss.item()
    
    def _compute_td_loss(self, batch: List[Dict]) -> torch.Tensor:
        """
        Compute TD loss for a batch of transitions.
        
        This is the generic TD loss computation that works with any Q-network
        that implements encode_and_forward().
        
        Uses the reward function if provided, otherwise uses goal achievement:
            reward = 1.0 if goal.is_achieved(next_state) else 0.0
        
        Args:
            batch: List of transition dicts with 'state', 'action', 
                   'next_state', 'agent_idx', 'goal'.
        
        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        for transition in batch:
            state = transition['state']
            action = transition['action']
            next_state = transition['next_state']
            agent_idx = transition['agent_idx']
            goal = transition['goal']
            
            # Compute reward
            if self.reward_fn is not None:
                reward = self.reward_fn(state, action, next_state, agent_idx, goal)
            else:
                # Default: goal achievement reward
                if hasattr(goal, 'is_achieved'):
                    reward = float(goal.is_achieved(next_state))
                else:
                    reward = 0.0
            
            # Current Q-value
            q_values = self.q_network.encode_and_forward(
                state, None, agent_idx, goal, self.device
            )
            current_q = q_values[0, action]
            
            # Target Q-value (soft)
            with torch.no_grad():
                next_q = self.target_network.encode_and_forward(
                    next_state, None, agent_idx, goal, self.device
                )
                next_v = self.q_network.get_value(next_q)
            
            # TD target: r + γ * V(s')
            target = reward + self.gamma * next_v
            loss = (current_q - target) ** 2
            total_loss = total_loss + loss
        
        return total_loss / len(batch)
