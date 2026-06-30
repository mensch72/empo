"""
BushWorld-specific Intrinsic Robot Reward Network for Phase 2.

Implements U_r(s) from equation (8) for BushWorld. Predicts ``log(y - 1)`` for
numerical stability, then ``y = 1 + exp(log(y-1))`` and ``U_r = -y^η``.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...phase2.intrinsic_reward_network import BaseIntrinsicRewardNetwork
from ..state_encoder import BushWorldStateEncoder


class BushWorldIntrinsicRewardNetwork(BaseIntrinsicRewardNetwork):
    """Intrinsic reward network U_r for BushWorld."""

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robots: int,
        max_steps: int,
        state_feature_dim: int = 128,
        hidden_dim: int = 128,
        xi: float = 1.0,
        eta: float = 1.1,
        use_encoders: bool = True,
        state_encoder: Optional[BushWorldStateEncoder] = None,
    ):
        super().__init__(xi=xi, eta=eta)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_robots = num_robots
        self.max_steps = max_steps
        self.state_feature_dim = state_feature_dim
        self.hidden_dim = hidden_dim
        self.use_encoders = use_encoders

        self.state_encoder = state_encoder or BushWorldStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            max_steps=max_steps,
            feature_dim=state_feature_dim,
            hidden_dim=hidden_dim,
            use_encoders=use_encoders,
        )

        self.y_head = nn.Sequential(
            nn.Linear(self.state_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _network_forward(self, state_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.state_encoder(state_input)
        log_y_minus_1 = self.y_head(features).squeeze(-1)
        y = self.log_y_minus_1_to_y(log_y_minus_1)
        u_r = self.y_to_u_r(y)
        return y, u_r

    def forward(self, state: Any, world_model: Any, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        state_input = self.state_encoder.tensorize_state(state, world_model, device)
        return self._network_forward(state_input)

    def forward_batch(
        self, states: List[Any], world_model: Any, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat(
            [self.state_encoder.tensorize_state(s, world_model, device) for s in states],
            dim=0,
        )
        return self._network_forward(inputs)

    def get_config(self) -> Dict[str, Any]:
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "B": self.B,
            "num_robots": self.num_robots,
            "max_steps": self.max_steps,
            "state_feature_dim": self.state_feature_dim,
            "hidden_dim": self.hidden_dim,
            "xi": self.xi,
            "eta": self.eta,
            "use_encoders": self.use_encoders,
            "state_encoder_config": self.state_encoder.get_config(),
        }
