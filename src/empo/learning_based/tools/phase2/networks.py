"""
Tools-specific Phase 2 (DQN) neural networks.

Re-exports the auxiliary networks already defined for the PPO path
(``ToolsHumanGoalAchievementNetwork``, ``ToolsAggregateGoalAbilityNetwork``,
``ToolsIntrinsicRewardNetwork``) and adds ``ToolsRobotQNetwork`` and
``ToolsRobotValueNetwork`` for the DQN training loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from empo.learning_based.phase2.robot_q_network import BaseRobotQNetwork
from empo.learning_based.phase2.robot_value_network import BaseRobotValueNetwork
from empo.learning_based.tools.state_encoder import ToolsStateEncoder

# Re-export PPO-path auxiliary networks — they subclass the same base
# classes expected by Phase2Networks.
from empo.learning_based.tools.phase2_ppo.networks import (  # noqa: F401
    ToolsHumanGoalAchievementNetwork,
    ToolsAggregateGoalAbilityNetwork,
    ToolsIntrinsicRewardNetwork,
)

# -------------------------------------------------------------------
# Q_r  (equation 4)
# -------------------------------------------------------------------


class ToolsRobotQNetwork(BaseRobotQNetwork):
    """Robot Q-network for the tools environment.

    Uses a shared ``ToolsStateEncoder`` (frozen during Q_r updates) plus
    an ``own`` encoder whose gradients flow through Q_r's loss.

    Parameters
    ----------
    state_encoder : ToolsStateEncoder
        Shared encoder (trained only by V_h^e / X_h).
    num_robot_actions : int
        Actions available to each robot.
    num_robots : int
        Number of robots in the fleet.
    hidden_dim : int
        MLP hidden width.
    beta_r : float
        Power-law policy exponent.
    """

    def __init__(
        self,
        state_encoder: ToolsStateEncoder,
        num_robot_actions: int,
        num_robots: int = 1,
        hidden_dim: int = 128,
        beta_r: float = 10.0,
        feasible_range: Optional[Tuple[float, float]] = None,
        use_z_space: bool = False,
        eta: float = 1.1,
        xi: float = 1.0,
    ):
        super().__init__(
            num_actions=num_robot_actions,
            num_robots=num_robots,
            beta_r=beta_r,
            feasible_range=feasible_range,
            use_z_space=use_z_space,
            eta=eta,
            xi=xi,
        )
        self.state_encoder = state_encoder
        self.hidden_dim = hidden_dim

        # Own encoder (trained with Q_r loss)
        self.own_state_encoder = ToolsStateEncoder(
            n_agents=state_encoder.n_agents,
            n_tools=state_encoder.n_tools,
            max_steps=state_encoder.max_steps,
            feature_dim=state_encoder.feature_dim,
            hidden_dim=hidden_dim,
        )

        combined_dim = state_encoder.feature_dim + self.own_state_encoder.feature_dim
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_action_combinations),
        )

    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        # Shared encoder — detached so Q_r loss doesn't update it
        x = self.state_encoder.tensorize_state(state, world_model, device=device)
        shared_feat = self.state_encoder(x).detach()
        # Own encoder — gradient flows
        own_x = self.own_state_encoder.tensorize_state(
            state, world_model, device=device
        )
        own_feat = self.own_state_encoder(own_x)
        combined = torch.cat([shared_feat, own_feat], dim=-1)
        raw_q = self.q_head(combined)
        return self.ensure_negative(raw_q)

    def forward_batch(
        self,
        states: List[Any],
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Batch forward: vectorise states."""
        shared_list, own_list = [], []
        for state in states:
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            shared_list.append(self.state_encoder(x).detach().squeeze(0))
            own_x = self.own_state_encoder.tensorize_state(
                state, world_model, device=device
            )
            own_list.append(self.own_state_encoder(own_x).squeeze(0))
        shared = torch.stack(shared_list)
        own = torch.stack(own_list)
        combined = torch.cat([shared, own], dim=-1)
        return self.ensure_negative(self.q_head(combined))

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_actions": self.num_actions,
            "num_robots": self.num_robots,
            "hidden_dim": self.hidden_dim,
            "beta_r": self.beta_r,
        }


# -------------------------------------------------------------------
# V_r  (equation 9)
# -------------------------------------------------------------------


class ToolsRobotValueNetwork(BaseRobotValueNetwork):
    """Robot value network for the tools environment.

    Parameters
    ----------
    state_encoder : ToolsStateEncoder
        Shared encoder.
    hidden_dim : int
        MLP hidden width.
    gamma_r : float
        Robot discount factor.
    """

    def __init__(
        self,
        state_encoder: ToolsStateEncoder,
        hidden_dim: int = 128,
        gamma_r: float = 0.99,
        use_z_space: bool = False,
        eta: float = 1.1,
        xi: float = 1.0,
    ):
        super().__init__(
            gamma_r=gamma_r,
            use_z_space=use_z_space,
            eta=eta,
            xi=xi,
        )
        self.state_encoder = state_encoder
        self.hidden_dim = hidden_dim

        # Own encoder (trained with V_r loss)
        self.own_state_encoder = ToolsStateEncoder(
            n_agents=state_encoder.n_agents,
            n_tools=state_encoder.n_tools,
            max_steps=state_encoder.max_steps,
            feature_dim=state_encoder.feature_dim,
            hidden_dim=hidden_dim,
        )

        combined_dim = state_encoder.feature_dim + self.own_state_encoder.feature_dim
        self.v_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: Any,
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        x = self.state_encoder.tensorize_state(state, world_model, device=device)
        shared_feat = self.state_encoder(x).detach()
        own_x = self.own_state_encoder.tensorize_state(
            state, world_model, device=device
        )
        own_feat = self.own_state_encoder(own_x)
        combined = torch.cat([shared_feat, own_feat], dim=-1)
        raw_v = self.v_head(combined)
        return self.ensure_negative(raw_v)

    def ensure_negative(self, raw: torch.Tensor) -> torch.Tensor:
        """Map unbounded output to (-inf, 0)."""
        return -torch.nn.functional.softplus(raw)

    def forward_batch(
        self,
        states: List[Any],
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Batch forward: vectorise states."""
        shared_list, own_list = [], []
        for state in states:
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            shared_list.append(self.state_encoder(x).detach().squeeze(0))
            own_x = self.own_state_encoder.tensorize_state(
                state, world_model, device=device
            )
            own_list.append(self.own_state_encoder(own_x).squeeze(0))
        shared = torch.stack(shared_list)
        own = torch.stack(own_list)
        combined = torch.cat([shared, own], dim=-1)
        return self.ensure_negative(self.v_head(combined)).squeeze(-1)

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "gamma_r": self.gamma_r,
        }
