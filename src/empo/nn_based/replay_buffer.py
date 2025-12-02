"""
Replay buffer for experience replay during training.
"""

import random
from typing import Any, Dict, List, Optional


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions.
    
    Stores (state, action, next_state, agent_idx, goal) tuples for training.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
    
    def push(
        self,
        state: Any,
        action: int,
        next_state: Any,
        agent_idx: int,
        goal: Any
    ):
        """Add a transition to the buffer."""
        transition = {
            'state': state,
            'action': action,
            'next_state': next_state,
            'agent_idx': agent_idx,
            'goal': goal
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.position = 0
