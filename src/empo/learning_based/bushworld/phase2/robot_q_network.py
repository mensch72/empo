"""
BushWorld-specific Robot Q-Network for Phase 2.

Implements Q_r(s, a_r) from equation (4) for BushWorld. Uses a single
:class:`BushWorldStateEncoder` followed by an MLP head over the joint robot
action space. Mirrors the multigrid network but simplified for BushWorld's
flat feature vector (no separate "own" encoder is needed because BushWorld
networks do not share encoders).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...phase2.robot_q_network import BaseRobotQNetwork
from ..state_encoder import BushWorldStateEncoder


class BushWorldRobotQNetwork(BaseRobotQNetwork):
    """Robot Q-Network for BushWorld environments."""

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robot_actions: int,
        num_robots: int,
        num_humans: int,
        max_steps: int,
        state_feature_dim: int = 128,
        hidden_dim: int = 128,
        beta_r: float = 10.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        use_encoders: bool = True,
        state_encoder: Optional[BushWorldStateEncoder] = None,
    ):
        super().__init__(
            num_actions=num_robot_actions,
            num_robots=num_robots,
            beta_r=beta_r,
            feasible_range=feasible_range,
        )
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_humans = num_humans
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim
        self.state_feature_dim = state_feature_dim
        self.use_encoders = use_encoders

        if state_encoder is not None:
            self.state_encoder = state_encoder
        else:
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

        self.q_head = nn.Sequential(
            nn.Linear(self.state_encoder.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_action_combinations),
        )

    def _network_forward(self, state_input: torch.Tensor) -> torch.Tensor:
        features = self.state_encoder(state_input)
        raw_q = self.q_head(features)
        return self.ensure_negative(raw_q)

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
            "num_robot_actions": self.num_actions,
            "num_robots": self.num_robots,
            "num_humans": self.num_humans,
            "max_steps": self.max_steps,
            "state_feature_dim": self.state_feature_dim,
            "hidden_dim": self.hidden_dim,
            "beta_r": self.beta_r,
            "feasible_range": self.feasible_range,
            "use_encoders": self.use_encoders,
            "state_encoder_config": self.state_encoder.get_config(),
        }
