"""
BushWorld-specific Robot State Value Network for Phase 2.

Implements V_r(s) from equation (9) for BushWorld. Predicts a strictly negative
value via ``ensure_negative``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ...phase2.robot_value_network import BaseRobotValueNetwork
from ..state_encoder import BushWorldStateEncoder


class BushWorldRobotValueNetwork(BaseRobotValueNetwork):
    """Robot value network V_r for BushWorld."""

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robots: int,
        max_steps: int,
        state_feature_dim: int = 128,
        hidden_dim: int = 128,
        gamma_r: float = 0.99,
        use_encoders: bool = True,
        state_encoder: Optional[BushWorldStateEncoder] = None,
        use_z_space: bool = False,
        eta: float = 1.1,
        xi: float = 1.0,
    ):
        super().__init__(gamma_r=gamma_r, use_z_space=use_z_space, eta=eta, xi=xi)
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

        self.value_head = nn.Sequential(
            nn.Linear(self.state_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _network_forward(self, state_input: torch.Tensor) -> torch.Tensor:
        features = self.state_encoder(state_input)
        raw_value = self.value_head(features).squeeze(-1)
        return self.ensure_negative(raw_value)

    def forward(self, state: Any, world_model: Any, device: str = "cpu") -> torch.Tensor:
        state_input = self.state_encoder.tensorize_state(state, world_model, device)
        return self._network_forward(state_input)

    def forward_batch(self, states: List[Any], world_model: Any, device: str = "cpu") -> torch.Tensor:
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
            "gamma_r": self.gamma_r,
            "use_encoders": self.use_encoders,
            "use_z_space": self.use_z_space,
            "eta": self.eta,
            "xi": self.xi,
            "state_encoder_config": self.state_encoder.get_config(),
        }
