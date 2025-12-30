"""
Base neural human policy prior class.
"""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalSampler

from .q_network import BaseQNetwork
from .policy_prior_network import BasePolicyPriorNetwork


class BaseNeuralHumanPolicyPrior(HumanPolicyPrior, ABC):
    """
    Base class for neural network-based human policy priors.
    
    Contains all generic logic for:
    - Computing action probabilities (goal-specific and marginal)
    - Save/load functionality
    - Action encoding validation
    - Policy remapping for transfer
    
    Subclasses implement domain-specific encoding and network creation.
    """
    
    def __init__(
        self,
        q_network: BaseQNetwork,
        policy_network: BasePolicyPriorNetwork,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = 'cpu'
    ):
        self.q_network = q_network
        self.policy_network = policy_network
        self.world_model = world_model
        self.human_agent_indices = human_agent_indices
        self.goal_sampler = goal_sampler
        self.action_encoding = action_encoding or {}
        self.device = device
        self._infeasible_actions_become: Optional[int] = None
        
        self.q_network.to(device)
        self.policy_network.to(device)
    
    def __call__(
        self,
        state: Any,
        agent_idx: int,
        goal: Optional[PossibleGoal] = None
    ) -> Dict[int, float]:
        """
        Compute action probabilities for an agent.
        
        Args:
            state: Current state.
            agent_idx: Index of the agent.
            goal: Optional specific goal. If None, marginalizes over goals.
        
        Returns:
            Dict mapping action index to probability.
        """
        self.q_network.eval()
        
        with torch.no_grad():
            if goal is not None:
                probs = self._compute_goal_specific_policy(state, agent_idx, goal)
            else:
                probs = self._compute_marginal_policy(state, agent_idx)
        
        # Apply action remapping if needed
        if self._infeasible_actions_become is not None:
            probs = self._remap_infeasible_actions(probs)
        
        return {i: probs[i].item() for i in range(len(probs))}
    
    def _compute_goal_specific_policy(
        self,
        state: Any,
        agent_idx: int,
        goal: PossibleGoal
    ) -> torch.Tensor:
        """Compute policy for a specific goal."""
        q_values = self.q_network.forward(
            state, self.world_model, agent_idx, goal, self.device
        )
        return self.q_network.get_policy(q_values).squeeze(0)
    
    @abstractmethod
    def _compute_marginal_policy(
        self,
        state: Any,
        agent_idx: int
    ) -> torch.Tensor:
        """Compute marginal policy over goals. Domain-specific."""
        pass
    
    def _remap_infeasible_actions(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Remap probability mass from infeasible actions.
        
        Actions not in the original action encoding are considered infeasible
        and their probability is redistributed to the fallback action.
        """
        if self._infeasible_actions_become is None:
            return probs
        
        new_probs = probs.clone()
        fallback_action = self._infeasible_actions_become
        
        # Find actions that weren't in original training
        original_actions = set(self.action_encoding.keys())
        for i in range(len(probs)):
            if i not in original_actions:
                new_probs[fallback_action] += new_probs[i]
                new_probs[i] = 0.0
        
        # Renormalize
        total = new_probs.sum()
        if total > 0:
            new_probs = new_probs / total
        
        return new_probs
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Saves network weights and configuration needed for reconstruction.
        """
        config = self.q_network.get_config()
        config['action_encoding'] = self.action_encoding
        config['human_agent_indices'] = self.human_agent_indices
        
        torch.save({
            'config': config,
            'q_network_state_dict': self.q_network.state_dict(),
        }, filepath)
    
    @classmethod
    def _validate_action_encoding(
        cls,
        saved_encoding: Dict[int, str],
        world_model: Any
    ) -> None:
        """Validate that action encodings don't conflict."""
        env_actions = getattr(world_model, 'actions', None)
        if env_actions is None:
            return
        
        # Handle both enum classes and iterables
        if hasattr(env_actions, '__members__'):
            # It's an enum class
            env_encoding = {i: name.lower() for i, name in enumerate(env_actions.__members__.keys())}
        elif hasattr(env_actions, '__iter__'):
            env_encoding = {i: a.name.lower() for i, a in enumerate(env_actions)}
        else:
            return
        
        for action_idx, action_name in saved_encoding.items():
            if action_idx in env_encoding and env_encoding[action_idx] != action_name:
                raise ValueError(
                    f"Action encoding conflict: saved action {action_idx}='{action_name}', "
                    f"environment action {action_idx}='{env_encoding[action_idx]}'"
                )
    
    @classmethod
    @abstractmethod
    def load(
        cls,
        filepath: str,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        infeasible_actions_become: Optional[int] = None,
        device: str = 'cpu'
    ) -> 'BaseNeuralHumanPolicyPrior':
        """
        Load a model from file.
        
        Subclasses implement domain-specific network reconstruction.
        """
        pass
