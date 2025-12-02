"""
Policy prior network for multigrid environments.

Computes marginal action probabilities by averaging over goals.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ..policy_prior_network import BasePolicyPriorNetwork
from .q_network import MultiGridQNetwork


class MultiGridPolicyPriorNetwork(BasePolicyPriorNetwork):
    """
    Policy prior network for multigrid.
    
    Computes π(a|s) = E_g[π(a|s,g)] by averaging Boltzmann policies
    over sampled goals.
    
    Args:
        q_network: The Q-network to use.
    """
    
    def __init__(self, q_network: MultiGridQNetwork):
        super().__init__(q_network.num_actions)
        self.q_network = q_network
    
    def forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        goal_coords_batch: torch.Tensor,
        interactive_features: torch.Tensor,
        goal_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute marginal action probabilities.
        
        Args:
            grid_tensor: (batch, channels, H, W)
            global_features: (batch, 4)
            agent_features: (batch, agent_input_size)
            goal_coords_batch: (batch, num_goals, 2)
            interactive_features: (batch, interactive_input_size)
            goal_weights: Optional (batch, num_goals) goal probabilities
        
        Returns:
            Action probabilities (batch, num_actions)
        """
        batch_size = grid_tensor.shape[0]
        num_goals = goal_coords_batch.shape[1]
        device = grid_tensor.device
        
        if goal_weights is None:
            goal_weights = torch.ones(batch_size, num_goals, device=device) / num_goals
        
        # Compute policy for each goal and average
        all_policies = []
        for g in range(num_goals):
            goal_coords = goal_coords_batch[:, g, :]
            q_values = self.q_network(
                grid_tensor, global_features, agent_features,
                goal_coords, interactive_features
            )
            policy = self.q_network.get_policy(q_values)
            all_policies.append(policy)
        
        # Stack: (batch, num_goals, num_actions)
        policies = torch.stack(all_policies, dim=1)
        
        # Weighted average: (batch, num_actions)
        weights = goal_weights.unsqueeze(-1)  # (batch, num_goals, 1)
        marginal_policy = (policies * weights).sum(dim=1)
        
        return marginal_policy
    
    def compute_marginal(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: int,
        goals: List[Any],
        goal_weights: Optional[List[float]] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Compute marginal policy for a state.
        
        Args:
            state: Environment state tuple.
            world_model: Environment.
            query_agent_idx: Index of query agent.
            goals: List of goal objects.
            goal_weights: Optional goal probabilities.
            device: Torch device.
        
        Returns:
            Action probabilities (num_actions,)
        """
        if not goals:
            return torch.ones(self.num_actions, device=device) / self.num_actions
        
        # Encode state (shared across goals)
        grid_tensor, global_features = self.q_network.state_encoder.encode_state(
            state, world_model, query_agent_idx, device
        )
        agent_features = self.q_network.agent_encoder.encode_agents(
            state, world_model, query_agent_idx, device
        )
        interactive_features = self.q_network.interactive_encoder.encode_objects(
            state, world_model, device
        )
        
        # Encode goals
        goal_coords_list = []
        for goal in goals:
            coords = self.q_network.goal_encoder.encode_goal(goal, device)
            goal_coords_list.append(coords)
        
        goal_coords_batch = torch.cat(goal_coords_list, dim=0).unsqueeze(0)  # (1, num_goals, 2)
        
        # Weights
        if goal_weights is not None:
            weights = torch.tensor([goal_weights], device=device)
        else:
            weights = None
        
        marginal = self.forward(
            grid_tensor, global_features, agent_features,
            goal_coords_batch, interactive_features, weights
        )
        
        return marginal.squeeze(0)
