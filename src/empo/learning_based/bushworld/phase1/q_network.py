"""
Q-Network for BushWorld environments (Phase 1, DQN path).

Combines the BushWorld state encoder and goal encoder to predict
goal-conditioned action Q-values ``Q(s, a, g)``. Mirrors
:class:`~empo.learning_based.multigrid.phase1.q_network.MultiGridQNetwork`
but is simplified for BushWorld's flat feature vector (the encoders consume a
single tensor rather than the multigrid 4-tuple of grid/global/agent/interactive
features).
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from empo.learning_based.phase1.q_network import BaseQNetwork
from ..goal_encoder import BushWorldGoalEncoder
from ..state_encoder import BushWorldStateEncoder


class BushWorldQNetwork(BaseQNetwork):
    """Goal-conditioned Q-Network for BushWorld environments.

    Uses :class:`BushWorldStateEncoder` for the state and
    :class:`BushWorldGoalEncoder` for the goal (both cell and rectangle goals
    are supported, since they expose ``target_rect``).

    Args:
        grid_height: Number of rows.
        grid_width: Number of columns.
        B: Maximum bush density (for normalisation).
        num_robots: Number of robot agents (used to split occupancy channels).
        max_steps: Episode horizon (for normalising the step count).
        num_actions: Number of human actions.
        state_feature_dim: State encoder output dim.
        goal_feature_dim: Goal encoder output dim.
        hidden_dim: Hidden layer dimension.
        beta: Boltzmann temperature (theory parameter ``beta_h``).
        feasible_range: Optional ``(a, b)`` tuple for Q-value bounds.
        use_encoders: If ``False``, the encoders run in identity mode.
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
        goal_feature_dim: int = 32,
        hidden_dim: int = 128,
        beta: float = 1.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        use_encoders: bool = True,
    ):
        super().__init__(num_actions, beta, feasible_range)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_robots = num_robots
        self.max_steps = max_steps
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

        self.goal_encoder = BushWorldGoalEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=goal_feature_dim,
            use_encoders=use_encoders,
        )

        combined_dim = self.state_encoder.feature_dim + self.goal_encoder.feature_dim

        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def _network_forward(
        self, state_input: torch.Tensor, goal_coords: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-values from pre-tensorized state and goal inputs."""
        state_features = self.state_encoder(state_input)
        goal_emb = self.goal_encoder(goal_coords)
        combined = torch.cat([state_features, goal_emb], dim=1)
        q_values = self.q_head(combined)
        return self.apply_soft_clamp(q_values)

    def forward(
        self,
        state: Any,
        world_model: Any,
        query_agent_idx: int,
        goal: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Encode ``state``/``goal`` and return Q-values of shape ``(1, num_actions)``.

        ``query_agent_idx`` is accepted for API compatibility; the BushWorld
        state encoding is agent-agnostic (the goal already identifies the human).
        """
        state_input = self.state_encoder.tensorize_state(state, world_model, device)
        goal_coords = self.goal_encoder.tensorize_goal(goal, device)
        return self._network_forward(state_input, goal_coords)

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
            "goal_feature_dim": self.goal_encoder.feature_dim,
            "hidden_dim": self.hidden_dim,
            "beta": self.beta,
            "feasible_range": self.feasible_range,
            "use_encoders": self.use_encoders,
        }
