"""
Factory for creating Tools PPO Phase 2 networks.

Provides :func:`create_tools_ppo_networks` which builds a
:class:`ToolsStateEncoder`, an :class:`EMPOActorCritic`, and
MLP-based auxiliary networks (V_h^e, X_h, U_r) wired to share
the same state encoder.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.trainer import PPOAuxiliaryNetworks

from empo.learning_based.phase2.human_goal_ability import (
    BaseHumanGoalAchievementNetwork,
)
from empo.learning_based.phase2.aggregate_goal_ability import (
    BaseAggregateGoalAbilityNetwork,
)
from empo.learning_based.phase2.intrinsic_reward_network import (
    BaseIntrinsicRewardNetwork,
)

from empo.learning_based.tools.state_encoder import ToolsStateEncoder

# ======================================================================
# MLP auxiliary networks for the tools environment
# ======================================================================


class ToolsHumanGoalAchievementNetwork(BaseHumanGoalAchievementNetwork):
    """V_h^e for the tools environment — MLP-based.

    Encodes ``(state_features, goal_type, agent_idx, tool_idx)`` and
    predicts V_h^e ∈ [0, 1].
    """

    def __init__(
        self,
        state_encoder: ToolsStateEncoder,
        n_agents: int,
        n_tools: int,
        hidden_dim: int = 128,
        gamma_h: float = 0.99,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(gamma_h=gamma_h, feasible_range=feasible_range)
        self.state_encoder = state_encoder
        self.n_agents = n_agents
        self.n_tools = n_tools
        # goal encoding: 3 (goal type one-hot) + n_agents + n_tools
        goal_dim = 3 + n_agents + n_tools
        # agent encoding: n_agents (one-hot)
        agent_dim = n_agents
        input_dim = state_encoder.feature_dim + goal_dim + agent_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_goal(self, goal: Any, device: str | torch.device) -> torch.Tensor:
        """Encode a tools goal as a flat vector."""
        from empo.world_specific_helpers.tools import (
            HoldGoal,
            WorkbenchGoal,
            IdleGoal,
        )

        goal_type = torch.zeros(3, device=device)
        agent_onehot = torch.zeros(self.n_agents, device=device)
        tool_onehot = torch.zeros(self.n_tools, device=device)
        if isinstance(goal, HoldGoal):
            goal_type[0] = 1.0
            agent_onehot[goal.agent_idx] = 1.0
            tool_onehot[goal.tool_idx] = 1.0
        elif isinstance(goal, WorkbenchGoal):
            goal_type[1] = 1.0
            agent_onehot[goal.agent_idx] = 1.0
            tool_onehot[goal.tool_idx] = 1.0
        elif isinstance(goal, IdleGoal):
            goal_type[2] = 1.0
            agent_onehot[goal.agent_idx] = 1.0
        return torch.cat([goal_type, agent_onehot, tool_onehot])

    def forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
        goal_feat = self._encode_goal(goal, device=device)
        agent_onehot = torch.zeros(self.n_agents, device=device)
        agent_onehot[human_agent_idx] = 1.0
        combined = torch.cat([state_feat, goal_feat, agent_onehot]).unsqueeze(0)
        raw = self.head(combined)
        return self.apply_clamp(raw.squeeze(0))

    def forward_batch(
        self,
        states: List[Any],
        goals: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Batch forward: vectorise states/goals/humans."""
        feats = []
        for state, goal, h_idx in zip(states, goals, human_indices):
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
            goal_feat = self._encode_goal(goal, device=device)
            agent_oh = torch.zeros(self.n_agents, device=device)
            agent_oh[h_idx] = 1.0
            feats.append(torch.cat([state_feat, goal_feat, agent_oh]))
        batch = torch.stack(feats)  # (B, input_dim)
        raw = self.head(batch)  # (B, 1)
        return self.apply_clamp(raw.squeeze(-1))

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "n_tools": self.n_tools,
            "hidden_dim": self.head[0].in_features,
            "gamma_h": self.gamma_h,
        }


class ToolsAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """X_h for the tools environment — MLP-based.

    Encodes ``(state_features, agent_idx)`` and predicts X_h ∈ (0, 1].
    """

    def __init__(
        self,
        state_encoder: ToolsStateEncoder,
        n_agents: int,
        hidden_dim: int = 128,
        zeta: float = 2.0,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        self.state_encoder = state_encoder
        self.n_agents = n_agents
        input_dim = state_encoder.feature_dim + n_agents
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        with torch.no_grad():
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
        agent_onehot = torch.zeros(self.n_agents, device=device)
        agent_onehot[human_agent_idx] = 1.0
        combined = torch.cat([state_feat, agent_onehot]).unsqueeze(0)
        raw = self.head(combined)
        return self.apply_clamp(raw.squeeze(0))

    def forward_batch(
        self,
        states: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Batch forward: vectorise states/humans."""
        feats = []
        for state, h_idx in zip(states, human_indices):
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
            agent_oh = torch.zeros(self.n_agents, device=device)
            agent_oh[h_idx] = 1.0
            feats.append(torch.cat([state_feat, agent_oh]))
        batch = torch.stack(feats)
        raw = self.head(batch)
        return self.apply_clamp(raw.squeeze(-1))

    def get_config(self) -> Dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "hidden_dim": self.head[0].in_features,
            "zeta": self.zeta,
        }


class ToolsIntrinsicRewardNetwork(BaseIntrinsicRewardNetwork):
    """U_r for the tools environment — MLP-based.

    Encodes ``state_features`` and predicts (y, U_r).
    """

    def __init__(
        self,
        state_encoder: ToolsStateEncoder,
        hidden_dim: int = 128,
        xi: float = 1.0,
        eta: float = 1.1,
    ):
        super().__init__(xi=xi, eta=eta)
        self.state_encoder = state_encoder
        self.head = nn.Sequential(
            nn.Linear(state_encoder.feature_dim, hidden_dim),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
        y_raw = self.head(state_feat.unsqueeze(0)).squeeze(0)
        # y should be positive (mean of X_h^{-xi})
        y = torch.clamp(y_raw, min=1e-6)
        u_r = -(y**self.eta)
        return y, u_r

    def forward_batch(
        self,
        states: List[Any],
        world_model: Any,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch forward: vectorise states."""
        feats = []
        for state in states:
            x = self.state_encoder.tensorize_state(state, world_model, device=device)
            state_feat = self.state_encoder(x).squeeze(0)
            feats.append(state_feat)
        batch = torch.stack(feats)
        y_raw = self.head(batch).squeeze(-1)
        y = torch.clamp(y_raw, min=1e-6)
        u_r = -(y**self.eta)
        return y, u_r

    def get_config(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.head[0].in_features,
            "xi": self.xi,
            "eta": self.eta,
        }


# ======================================================================
# Factory
# ======================================================================


def create_tools_ppo_networks(
    env: Any,
    config: PPOPhase2Config,
    *,
    feature_dim: int = 64,
    hidden_dim: int = 128,
    use_x_h: bool = True,
    use_u_r: bool = False,
    device: str = "cpu",
) -> Tuple[EMPOActorCritic, PPOAuxiliaryNetworks, ToolsStateEncoder]:
    """Create all networks for Tools PPO Phase 2 training.

    Parameters
    ----------
    env : ToolsWorldModel
        A tools world model instance.
    config : PPOPhase2Config
        PPO Phase 2 configuration.
    feature_dim : int
        State encoder output dimension.
    hidden_dim : int
        Hidden-layer width for MLPs.
    use_x_h : bool
        Whether to create the X_h network.
    use_u_r : bool
        Whether to create the U_r network.
    device : str
        Torch device string.

    Returns
    -------
    actor_critic : EMPOActorCritic
    auxiliary_networks : PPOAuxiliaryNetworks
    state_encoder : ToolsStateEncoder
    """
    n_agents = env.n_agents
    n_tools = env.n_tools
    max_steps = env.max_steps

    # State encoder
    state_encoder = ToolsStateEncoder(
        n_agents=n_agents,
        n_tools=n_tools,
        max_steps=max_steps,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    # Actor-critic
    actor_critic = EMPOActorCritic(
        state_encoder=None,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
        num_robots=config.num_robots,
        obs_dim=state_encoder.feature_dim,
    ).to(device)

    # V_h^e
    v_h_e = ToolsHumanGoalAchievementNetwork(
        state_encoder=state_encoder,
        n_agents=n_agents,
        n_tools=n_tools,
        hidden_dim=hidden_dim,
        gamma_h=config.gamma_h,
    ).to(device)

    # X_h (optional)
    x_h: Optional[ToolsAggregateGoalAbilityNetwork] = None
    if use_x_h:
        x_h = ToolsAggregateGoalAbilityNetwork(
            state_encoder=state_encoder,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            zeta=config.zeta,
        ).to(device)

    # U_r (optional)
    u_r: Optional[ToolsIntrinsicRewardNetwork] = None
    if use_u_r:
        u_r = ToolsIntrinsicRewardNetwork(
            state_encoder=state_encoder,
            hidden_dim=hidden_dim,
            xi=config.xi,
            eta=config.eta,
        ).to(device)

    auxiliary_networks = PPOAuxiliaryNetworks(
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
    )

    return actor_critic, auxiliary_networks, state_encoder
