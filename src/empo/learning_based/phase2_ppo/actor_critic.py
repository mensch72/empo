"""
EMPO Actor-Critic network for PPO-based Phase 2 training.

Provides an explicit (π_r, V_r) actor-critic that replaces the implicit
Q_r → π_r power-law softmax policy derivation used in the DQN path.

The actor produces logits over the joint robot action space (|A|^N),
and the critic estimates V_r(s) = E[Σ_t γ^t U_r(s_t) | s_0 = s, π_r].

This module does NOT modify any existing network classes.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class EMPOActorCritic(nn.Module):
    """Combined actor-critic for PufferLib/CleanRL PPO training.

    Architecture::

        state → state_encoder → shared_features
                                   ├─ actor_head → action logits
                                   └─ critic_head → V_r(s)

    If ``state_encoder`` is ``None`` the actor and critic operate
    directly on a flat observation vector (useful for testing / simple envs).

    Parameters
    ----------
    state_encoder : nn.Module or None
        Shared state encoder.  Must expose an ``output_dim`` attribute
        indicating the dimensionality of its output vector.  May be shared
        with auxiliary networks (V_h^e) when ``use_shared_encoder=True``
        in the config.
    hidden_dim : int
        Width of hidden layers in actor / critic heads.
    num_actions : int
        Number of actions available to a single robot.
    num_robots : int
        Number of robots whose joint action is output.  The actor produces
        logits over ``num_actions ** num_robots`` joint actions.
    obs_dim : int or None
        Required when ``state_encoder is None``.  Dimensionality of the
        raw observation vector fed directly to actor/critic heads.
    """

    def __init__(
        self,
        state_encoder: Optional[nn.Module],
        hidden_dim: int,
        num_actions: int,
        num_robots: int = 1,
        obs_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.num_actions = num_actions
        self.num_robots = num_robots
        self.num_joint_actions = num_actions ** num_robots

        if state_encoder is not None:
            input_dim: int = state_encoder.output_dim  # type: ignore[union-attr]
        elif obs_dim is not None:
            input_dim = obs_dim
        else:
            raise ValueError(
                "Either state_encoder or obs_dim must be provided."
            )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_joint_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Run the shared encoder (or identity when no encoder)."""
        if self.state_encoder is not None:
            return self.state_encoder(obs)
        return obs

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, value)``."""
        features = self._encode(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V_r(s) estimate (scalar per batch element)."""
        features = self._encode(obs)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO-compatible interface.

        Parameters
        ----------
        obs : Tensor
            Observation batch, shape ``(B, *obs_shape)``.
        action : Tensor or None
            If provided, evaluate log-prob and entropy for these actions.
            If ``None``, sample new actions from the policy.

        Returns
        -------
        action : Tensor, shape (B,)
            Sampled or provided actions.
        log_prob : Tensor, shape (B,)
            Log-probability of the returned action.
        entropy : Tensor, shape (B,)
            Entropy of the policy distribution.
        value : Tensor, shape (B,)
            Critic's V_r(s) estimate.
        """
        features = self._encode(obs)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def action_index_to_tuple(self, index: int) -> Tuple[int, ...]:
        """Map a flat joint-action index back to per-robot actions."""
        actions = []
        remaining = index
        for _ in range(self.num_robots):
            actions.append(remaining % self.num_actions)
            remaining //= self.num_actions
        return tuple(actions)

    def action_tuple_to_index(self, actions: Tuple[int, ...]) -> int:
        """Map per-robot actions to a flat joint-action index."""
        idx = 0
        for i, a in enumerate(actions):
            idx += a * (self.num_actions ** i)
        return idx
