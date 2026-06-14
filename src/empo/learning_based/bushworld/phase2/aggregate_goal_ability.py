"""
BushWorld-specific Aggregate Goal Achievement Network for Phase 2.

Implements X_h(s) from equation (7) for BushWorld. Combines state and
agent-identity features and predicts a value in (0, 1].
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...phase2.aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from ..agent_encoder import BushWorldAgentEncoder
from ..state_encoder import BushWorldStateEncoder


class BushWorldAggregateGoalAbilityNetwork(BaseAggregateGoalAbilityNetwork):
    """Aggregate goal-ability network X_h for BushWorld."""

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        B: int,
        num_robots: int,
        num_agents: int,
        max_steps: int,
        state_feature_dim: int = 128,
        agent_feature_dim: int = 16,
        hidden_dim: int = 128,
        zeta: float = 2.0,
        feasible_range: Tuple[float, float] = (0.0, 1.0),
        use_encoders: bool = True,
        state_encoder: Optional[BushWorldStateEncoder] = None,
        agent_encoder: Optional[BushWorldAgentEncoder] = None,
    ):
        super().__init__(zeta=zeta, feasible_range=feasible_range)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.B = B
        self.num_robots = num_robots
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.state_feature_dim = state_feature_dim
        self.agent_feature_dim = agent_feature_dim
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
        self.agent_encoder = agent_encoder or BushWorldAgentEncoder(
            num_agents=num_agents,
            grid_height=grid_height,
            grid_width=grid_width,
            output_dim=agent_feature_dim,
            use_encoders=use_encoders,
        )

        combined_dim = self.state_encoder.feature_dim + self.agent_encoder.output_dim
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _network_forward(self, state_input: torch.Tensor, agent_input: torch.Tensor) -> torch.Tensor:
        state_features = self.state_encoder(state_input)
        agent_features = self.agent_encoder(agent_input)
        combined = torch.cat([state_features, agent_features], dim=-1)
        raw_value = self.value_head(combined).squeeze(-1)
        return self.apply_clamp(raw_value)

    def forward(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        state_input = self.state_encoder.tensorize_state(state, world_model, device)
        agent_input = self.agent_encoder.tensorize_single(
            human_agent_idx, state, world_model, device
        )
        return self._network_forward(state_input, agent_input)

    def forward_batch(
        self,
        states: List[Any],
        human_indices: List[int],
        world_model: Any,
        device: str = "cpu",
    ) -> torch.Tensor:
        state_input = torch.cat(
            [self.state_encoder.tensorize_state(s, world_model, device) for s in states],
            dim=0,
        )
        agent_input = torch.cat(
            [
                self.agent_encoder.tensorize_single(h, s, world_model, device)
                for s, h in zip(states, human_indices)
            ],
            dim=0,
        )
        return self._network_forward(state_input, agent_input)

    def get_config(self) -> Dict[str, Any]:
        return {
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "B": self.B,
            "num_robots": self.num_robots,
            "num_agents": self.num_agents,
            "max_steps": self.max_steps,
            "state_feature_dim": self.state_feature_dim,
            "agent_feature_dim": self.agent_feature_dim,
            "hidden_dim": self.hidden_dim,
            "zeta": self.zeta,
            "feasible_range": self.feasible_range,
            "use_encoders": self.use_encoders,
            "state_encoder_config": self.state_encoder.get_config(),
            "agent_encoder_config": self.agent_encoder.get_config(),
        }
