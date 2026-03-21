"""
EMPO Actor-Critic network for PufferLib PPO-based Phase 2 training.

Provides an explicit (π_r, V_r) actor-critic that replaces the implicit
Q_r → π_r power-law softmax policy derivation used in the DQN path.

The actor produces logits over the joint robot action space (|A|^N),
and the critic estimates V_r(s) = E[Σ_t γ^t U_r(s_t) | s_0 = s, π_r].

**PufferLib convention**: The ``forward()`` method returns
``(logits, value)`` — a ``(batch, num_joint_actions)`` logit tensor
and a ``(batch, 1)`` value tensor.  This is the interface that
``pufferlib.pufferl.PuffeRL`` calls during ``evaluate()`` and
``train()``.

This module does NOT modify any existing network classes.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import pufferlib.pytorch

    def _layer_init(
        layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
    ) -> nn.Linear:
        """Orthogonal init following PufferLib / CleanRL convention."""
        return pufferlib.pytorch.layer_init(layer, std=std, bias_const=bias_const)

except ImportError:

    def _layer_init(  # type: ignore[misc]
        layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
    ) -> nn.Linear:
        """Fallback when pufferlib is not installed (unit-test mode)."""
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


class EMPOActorCritic(nn.Module):
    """Combined actor-critic for PufferLib PPO training.

    Architecture::

        observations → encoder → shared_features
                                    ├─ actor_head → action logits
                                    └─ critic_head → V_r(s)

    If ``state_encoder`` is ``None`` the actor and critic operate
    directly on a flat observation vector (useful for testing / simple envs).

    PufferLib integration
    ---------------------
    * ``forward(observations, state=None)`` returns ``(logits, value)``
      matching PufferLib's expected policy interface.
    * ``encode_observations`` and ``decode_actions`` are provided so
      that the policy can be wrapped with ``pufferlib.models.LSTMWrapper``
      if recurrence is desired.

    Parameters
    ----------
    state_encoder : nn.Module or None
        Shared state encoder.  Must expose an ``output_dim`` attribute
        indicating the dimensionality of its output vector.
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
        self.hidden_size = hidden_dim  # PufferLib LSTMWrapper reads this

        if state_encoder is not None:
            input_dim: int = state_encoder.output_dim  # type: ignore[union-attr]
        elif obs_dim is not None:
            input_dim = obs_dim
        else:
            raise ValueError(
                "Either state_encoder or obs_dim must be provided."
            )

        self.encoder = nn.Sequential(
            _layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, self.num_joint_actions), std=0.01),
        )

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    # PufferLib policy interface
    # ------------------------------------------------------------------

    def forward(
        self, observations: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PufferLib-compatible forward: return ``(logits, value)``.

        Parameters
        ----------
        observations : Tensor, (B, *obs_shape)
        state : dict or None
            Recurrent state (unused unless wrapped with LSTMWrapper).

        Returns
        -------
        logits : Tensor, (B, num_joint_actions)
        value : Tensor, (B, 1)
        """
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward_eval(
        self, observations: torch.Tensor, state: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PufferLib calls ``forward_eval`` during ``evaluate()``."""
        return self.forward(observations, state)

    def encode_observations(
        self, observations: torch.Tensor, state: Optional[dict] = None
    ) -> torch.Tensor:
        """Encode observations into hidden state.

        Separated from ``decode_actions`` so that PufferLib's
        ``LSTMWrapper`` can inject a recurrent cell between them.
        """
        batch_size = observations.shape[0]
        x = observations.reshape(batch_size, -1).float()
        if self.state_encoder is not None:
            x = self.state_encoder(x)
        return self.encoder(x)

    def decode_actions(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden state into action logits and value."""
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

    # ------------------------------------------------------------------
    # Convenience (kept for backward compat with env_wrapper / tests)
    # ------------------------------------------------------------------

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V_r(s) estimate (scalar per batch element)."""
        hidden = self.encode_observations(obs)
        return self.critic(hidden).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy PPO-compatible interface.

        Returns ``(action, log_prob, entropy, value)``.
        """
        from torch.distributions import Categorical

        hidden = self.encode_observations(obs)
        logits, value = self.decode_actions(hidden)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)

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
