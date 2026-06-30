"""
Policy prior network for BushWorld environments (Phase 1, DQN path).

Computes the marginal action probabilities ``π(a|s) = E_g[π(a|s,g)]`` by
averaging the goal-conditioned Boltzmann policies over a set of goals. Mirrors
:class:`~empo.learning_based.multigrid.phase1.policy_prior_network.MultiGridPolicyPriorNetwork`.
"""

from typing import Any, List, Optional

import torch

from empo.learning_based.phase1.policy_prior_network import BasePolicyPriorNetwork
from .q_network import BushWorldQNetwork


class BushWorldPolicyPriorNetwork(BasePolicyPriorNetwork):
    """Marginal policy prior network for BushWorld.

    Args:
        q_network: The :class:`BushWorldQNetwork` to derive policies from.
    """

    def __init__(self, q_network: BushWorldQNetwork):
        super().__init__(q_network)

    def forward(
        self,
        state_input: torch.Tensor,
        goal_coords_batch: torch.Tensor,
        goal_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the marginal policy for a single state over several goals.

        Args:
            state_input: ``(1, input_dim)`` tensorized state.
            goal_coords_batch: ``(num_goals, goal_coord_dim)`` tensorized goals.
            goal_weights: Optional ``(num_goals,)`` goal probabilities.

        Returns:
            Action probabilities of shape ``(1, num_actions)``.
        """
        num_goals = goal_coords_batch.shape[0]
        all_policies = []
        for g in range(num_goals):
            goal_coords = goal_coords_batch[g : g + 1, :]
            q_values = self.q_network._network_forward(state_input, goal_coords)
            all_policies.append(self.q_network.get_policy(q_values))

        # (1, num_goals, num_actions)
        policies = torch.stack(all_policies, dim=1)
        weights = goal_weights.view(1, num_goals) if goal_weights is not None else None
        return self.compute_marginal_from_policies(policies, weights)

    def compute_marginal(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goals: List[Any],
        goal_weights: Optional[List[float]] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Compute the marginal policy for a raw state and a list of goals.

        Returns action probabilities of shape ``(num_actions,)``.
        """
        if not goals:
            return torch.ones(self.num_actions, device=device) / self.num_actions

        state_input = self.q_network.state_encoder.tensorize_state(
            state, world_model, device
        )
        goal_coords_list = [
            self.q_network.goal_encoder.tensorize_goal(goal, device) for goal in goals
        ]
        goal_coords_batch = torch.cat(goal_coords_list, dim=0)

        weights = (
            torch.tensor(goal_weights, device=device, dtype=torch.float32)
            if goal_weights is not None
            else None
        )
        marginal = self.forward(state_input, goal_coords_batch, weights)
        return marginal.squeeze(0)
