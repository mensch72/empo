"""
Policy prior network for transport environment.

Computes marginal action probabilities by averaging over goals.
Uses GNN-based encoding for transport network state.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from ..policy_prior_network import BasePolicyPriorNetwork
from .q_network import TransportQNetwork
from .feature_extraction import observation_to_graph_data


class TransportPolicyPriorNetwork(BasePolicyPriorNetwork):
    """
    Policy prior network for transport environment.
    
    Computes π(a|s) = E_g[π(a|s,g)] by averaging Boltzmann policies
    over sampled goals.
    
    Uses GNN-based state encoding via TransportQNetwork.
    
    Args:
        q_network: The TransportQNetwork to use.
    
    Example:
        >>> q_network = TransportQNetwork(max_nodes=100, num_clusters=10)
        >>> policy_network = TransportPolicyPriorNetwork(q_network)
        >>> 
        >>> # Compute marginal policy
        >>> marginal = policy_network.compute_marginal(
        ...     state=None, world_model=env, query_agent_idx=0, goals=goals
        ... )
    """
    
    def __init__(self, q_network: TransportQNetwork):
        super().__init__(q_network)
        # q_network is already set by parent class
    
    def forward(
        self,
        graph_data: Dict[str, torch.Tensor],
        goal_tensors: torch.Tensor,
        goal_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute marginal action probabilities.
        
        Args:
            graph_data: Dictionary from observation_to_graph_data()
            goal_tensors: (num_goals, goal_input_dim) goal encodings
            goal_weights: Optional (num_goals,) goal probabilities
        
        Returns:
            Action probabilities (num_actions,)
        """
        num_goals = goal_tensors.shape[0]
        device = goal_tensors.device
        
        if goal_weights is None:
            goal_weights = torch.ones(num_goals, device=device) / num_goals
        
        # Compute policy for each goal and average
        all_policies = []
        for g in range(num_goals):
            goal_tensor = goal_tensors[g:g+1]  # (1, goal_input_dim)
            q_values = self.q_network.forward(graph_data, goal_tensor)
            policy = self.q_network.get_policy(q_values)  # (1, num_actions)
            all_policies.append(policy)
        
        # Stack: (num_goals, num_actions)
        policies = torch.cat(all_policies, dim=0)
        
        # Weighted average: (num_actions,)
        weights = goal_weights.unsqueeze(-1)  # (num_goals, 1)
        marginal_policy = (policies * weights).sum(dim=0)
        
        return marginal_policy
    
    def compute_marginal(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goals: List[Any],
        goal_weights: Optional[List[float]] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Compute marginal policy for a state.
        
        Args:
            state: Ignored (transport env doesn't have separate state)
            world_model: TransportEnvWrapper instance
            query_agent_idx: Index of query agent
            goals: List of goal objects (TransportGoal or TransportClusterGoal)
            goal_weights: Optional goal probabilities
            device: Torch device
        
        Returns:
            Action probabilities (num_actions,)
        """
        if not goals:
            return torch.ones(self.num_actions, device=device) / self.num_actions
        
        # Encode state using GNN encoder (shared across goals)
        graph_data = observation_to_graph_data(world_model, query_agent_idx, device)
        
        # Encode all goals
        goal_tensors_list = []
        for goal in goals:
            goal_tensor = self.q_network.goal_encoder.encode_goal(goal, device, env=world_model)
            goal_tensors_list.append(goal_tensor)
        
        goal_tensors = torch.cat(goal_tensors_list, dim=0)  # (num_goals, goal_input_dim)
        
        # Weights
        if goal_weights is not None:
            weights = torch.tensor(goal_weights, device=device)
            # Normalize weights
            weights = weights / weights.sum()
        else:
            weights = None
        
        return self.forward(graph_data, goal_tensors, weights)
