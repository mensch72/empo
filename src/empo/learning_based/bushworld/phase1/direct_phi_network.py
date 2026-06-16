"""
Direct phi network for BushWorld environments (Phase 1, DQN path).

This network directly predicts the marginal policy prior
``h_phi(state) -> action probabilities`` without enumerating or sampling goals
at inference time. It is trained jointly with the Q-network to match the
goal-averaged Boltzmann marginal, exactly as in the multigrid implementation
(:class:`~empo.learning_based.multigrid.phase1.direct_phi_network.DirectPhiNetwork`).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..state_encoder import BushWorldStateEncoder


class BushWorldDirectPhiNetwork(nn.Module):
    """Direct marginal policy prior network for BushWorld.

    Uses its own :class:`BushWorldStateEncoder` (not shared with the Q-network)
    followed by a policy head producing action probabilities.

    Args:
        grid_height: Number of rows.
        grid_width: Number of columns.
        B: Maximum bush density (for normalisation).
        num_robots: Number of robot agents.
        max_steps: Episode horizon.
        num_actions: Number of human actions.
        state_feature_dim: State encoder output dim.
        hidden_dim: Hidden layer dimension.
        use_encoders: If ``False``, the encoder runs in identity mode.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robots: int,
        max_steps: int,
        num_actions: int,
        state_feature_dim: int = 128,
        hidden_dim: int = 128,
        use_encoders: bool = True,
    ):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.use_encoders = use_encoders

        self.state_encoder = BushWorldStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            max_steps=max_steps,
            feature_dim=state_feature_dim,
            hidden_dim=hidden_dim,
            use_encoders=use_encoders,
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.state_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward_logits_from_input(self, state_input: torch.Tensor) -> torch.Tensor:
        """Compute action logits from a tensorized state input."""
        return self.policy_head(self.state_encoder(state_input))

    def forward(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Encode ``state`` and return action probabilities ``(1, num_actions)``."""
        state_input = self.state_encoder.tensorize_state(state, world_model, device)
        logits = self.forward_logits_from_input(state_input)
        return F.softmax(logits, dim=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for save/load."""
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "B": self.B,
            "num_robots": self.num_robots,
            "max_steps": self.max_steps,
            "num_actions": self.num_actions,
            "state_feature_dim": self.state_encoder.feature_dim,
            "hidden_dim": self.hidden_dim,
            "use_encoders": self.use_encoders,
        }
