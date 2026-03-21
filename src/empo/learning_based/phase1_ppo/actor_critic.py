"""
Goal-conditioned actor-critic for PufferLib PPO-based Phase 1 training.

Provides a goal-conditioned (π_h, V_h) actor-critic that approximates the
human policy prior π_h(a|s,g) directly, replacing the implicit Q-value-
derived Boltzmann softmax policy from the DQN path.

The actor receives concatenated state and goal features and produces logits
over human actions.  The critic estimates V_h(s,g) — the expected discounted
return for goal *g* under the learned policy.

**PufferLib convention**: The ``forward()`` method returns
``(logits, value)`` — a ``(batch, num_actions)`` logit tensor and a
``(batch, 1)`` value tensor, matching PufferLib's expected policy interface.

This module does NOT modify any existing Phase 1 network classes.
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


class GoalConditionedActorCritic(nn.Module):
    """Goal-conditioned actor-critic for PufferLib PPO Phase 1 training.

    Architecture::

        observations (state_features ⊕ goal_features)
            → encoder → shared_features
                ├─ actor_head → action logits (num_actions)
                └─ critic_head → V_h(s, g)

    The observation vector is the concatenation of state features and
    goal features.  Both the state encoder and goal encoder are applied
    outside this network (in the environment wrapper), so the actor-critic
    operates on pre-encoded flat vectors.

    PufferLib integration
    ---------------------
    * ``forward(observations, state=None)`` returns ``(logits, value)``
      matching PufferLib's expected policy interface.
    * ``encode_observations`` and ``decode_actions`` are provided so
      that the policy can be wrapped with ``pufferlib.models.LSTMWrapper``
      if recurrence is desired.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the concatenated (state_features ⊕ goal_features)
        observation vector.
    hidden_dim : int
        Width of hidden layers in actor / critic heads.
    num_actions : int
        Number of actions available to the human agent.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_actions: int,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_dim  # PufferLib LSTMWrapper reads this

        self.encoder = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
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
        observations : Tensor, (B, obs_dim)
            Concatenated state + goal features.
        state : dict or None
            Recurrent state (unused unless wrapped with LSTMWrapper).

        Returns
        -------
        logits : Tensor, (B, num_actions)
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
        return self.encoder(x)

    def decode_actions(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode hidden state into action logits and value."""
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V_h(s, g) estimate (scalar per batch element)."""
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
