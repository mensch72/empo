"""
Direct phi network for multigrid environments.

This network directly predicts the marginal policy prior h_phi(state, agent) -> action probabilities
without needing to enumerate or sample goals at inference time.

Training approach:
    The phi network is trained **jointly** with the Q-network in the same training loop.
    At each training step, for a batch of states:
    1. Sample multiple goals from the goal sampler
    2. Compute the Q-network's policy for each goal
    3. Average to get the marginal (stochastic approximation of E_g[π(a|s,g)])
    4. Train phi to match this marginal via cross-entropy loss

    Joint training is preferred over separate distillation because:
    - Both networks see the same state distribution
    - Phi learns from fresh Q-network policies (no lag)
    - Better convergence properties

Mathematical background:
    The Q-network computes goal-conditioned policies:
    h_pi(s, h, g) = softmax(β * h_Q(s, h, g))
    
    The marginal policy prior is:
    h_phi(s, h) = E_g[h_pi(s, h, g)]
    
    This network learns to directly predict h_phi(s, h) without computing the expectation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple

from .state_encoder import MultiGridStateEncoder


class DirectPhiNetwork(nn.Module):
    """
    Direct marginal policy prior network for multigrid.
    
    This network directly predicts h_phi(state, agent) -> action probabilities
    without goal conditioning. It uses the same state encoder architecture as
    the Q-network but has its own trainable parameters.
    
    Training:
        Trained jointly with the Q-network via cross-entropy loss:
        
        L = -sum(target_marginal * log(phi_predicted))
        
        where target_marginal ≈ E_g[softmax(β * Q(s, a, g))] is computed by
        averaging Q-network policies over sampled goals.
    
    Args:
        grid_height: Height of the grid.
        grid_width: Width of the grid.
        num_actions: Number of possible actions.
        num_agents_per_color: Dict mapping color to max agents.
        num_agent_colors: Number of agent colors.
        state_feature_dim: State encoder output dim.
        hidden_dim: Hidden layer dimension.
        max_kill_buttons: Max KillButtons.
        max_pause_switches: Max PauseSwitches.
        max_disabling_switches: Max DisablingSwitches.
        max_control_buttons: Max ControlButtons.
    """
    
    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        num_actions: int,
        num_agents_per_color: Dict[str, int],
        num_agent_colors: int = 7,
        state_feature_dim: int = 256,
        hidden_dim: int = 256,
        max_kill_buttons: int = 4,
        max_pause_switches: int = 4,
        max_disabling_switches: int = 4,
        max_control_buttons: int = 4
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Own state encoder (not shared with Q-network)
        self.state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=num_agent_colors,
            feature_dim=state_feature_dim,
            max_kill_buttons=max_kill_buttons,
            max_pause_switches=max_pause_switches,
            max_disabling_switches=max_disabling_switches,
            max_control_buttons=max_control_buttons
        )
        
        # Policy head: state features -> action logits
        self.policy_head = nn.Sequential(
            nn.Linear(state_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def _network_forward(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Internal: Compute marginal policy prior from pre-encoded tensors.
        
        Args:
            grid_tensor: (batch, channels, H, W)
            global_features: (batch, 4)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Action probabilities (batch, num_actions)
        """
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        logits = self.policy_head(state_features)
        return F.softmax(logits, dim=-1)
    
    def forward_logits(
        self,
        grid_tensor: torch.Tensor,
        global_features: torch.Tensor,
        agent_features: torch.Tensor,
        interactive_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits (before softmax).
        
        Useful for KL divergence computation.
        
        Args:
            grid_tensor: (batch, channels, H, W)
            global_features: (batch, 4)
            agent_features: (batch, agent_input_size)
            interactive_features: (batch, interactive_input_size)
        
        Returns:
            Action logits (batch, num_actions)
        """
        state_features = self.state_encoder(
            grid_tensor, global_features, agent_features, interactive_features
        )
        return self.policy_head(state_features)
    
    def forward(
        self,
        state: Tuple,
        world_model: Any,
        query_agent_idx: int,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode state and compute marginal policy.
        
        Args:
            state: Environment state tuple.
            world_model: Environment.
            query_agent_idx: Index of query agent (unused, for API compatibility).
            device: Torch device.
        
        Returns:
            Action probabilities (1, num_actions)
        """
        # State encoding is agent-agnostic
        grid_tensor, global_features, agent_features, interactive_features = \
            self.state_encoder.tensorize_state(state, world_model, device)
        
        return self._network_forward(
            grid_tensor, global_features, agent_features, interactive_features
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        config = self.state_encoder.get_config()
        config.update({
            'num_actions': self.num_actions,
            'state_feature_dim': self.state_encoder.feature_dim,
            'hidden_dim': self.hidden_dim,
        })
        return config
