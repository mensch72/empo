"""
BushWorld-specific ``EMPOWorldModelEnv`` subclass for PPO Phase 2 training.

Provides :class:`BushWorldWorldModelEnv`, which overrides
:meth:`~EMPOWorldModelEnv._state_to_obs` with a real observation encoder
backed by :class:`BushWorldStateEncoder`.

Unlike the MultiGrid wrapper, BushWorld implements the standard Gymnasium API
(``reset(seed, options)`` and a 5-tuple ``step``), so the base class
``reset``/``step`` implementations work unchanged — only the observation
encoding needs to be provided here.

This module does NOT modify any code in ``learning_based/bushworld/phase2/``.
"""

from __future__ import annotations

from typing import Any, Callable, List

import numpy as np
import torch
from gymnasium import spaces

from empo.learning_based.phase2_ppo.env_wrapper import EMPOWorldModelEnv
from empo.learning_based.bushworld.state_encoder import BushWorldStateEncoder


class BushWorldWorldModelEnv(EMPOWorldModelEnv):
    """BushWorld PPO environment wrapper with real state-to-observation encoding.

    Extends :class:`EMPOWorldModelEnv` by implementing ``_state_to_obs``
    using a :class:`BushWorldStateEncoder` to convert the raw world-model
    state tuple into a flat ``float32`` observation vector suitable for the
    PPO actor-critic.

    Parameters
    ----------
    world_model : BushWorld (WorldModel)
        The BushWorld world model.
    human_policy_prior : callable
        ``(state, h_idx, goal, world_model) → action_probs``.
    goal_sampler : callable
        ``(state, h_idx) → (goal, weight)``.
    human_agent_indices : list[int]
        Indices of human agents.
    robot_agent_indices : list[int]
        Indices of robot agents.
    config : PPOPhase2Config
        PPO Phase 2 configuration.
    state_encoder : BushWorldStateEncoder
        Pre-constructed state encoder.  The encoder's ``feature_dim``
        determines the observation dimensionality.
    auxiliary_networks : object or None
        Frozen auxiliary networks for U_r computation.
    """

    def __init__(
        self,
        world_model: Any,
        human_policy_prior: Callable,
        goal_sampler: Callable,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        config: Any,
        state_encoder: BushWorldStateEncoder,
        auxiliary_networks: Any = None,
    ):
        self._state_encoder = state_encoder

        obs_dim = state_encoder.feature_dim
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        super().__init__(
            world_model=world_model,
            human_policy_prior=human_policy_prior,
            goal_sampler=goal_sampler,
            human_agent_indices=human_agent_indices,
            robot_agent_indices=robot_agent_indices,
            config=config,
            auxiliary_networks=auxiliary_networks,
            observation_space=observation_space,
            obs_dim=obs_dim,
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: Any) -> np.ndarray:
        """Convert a BushWorld world-model state to a flat observation.

        Uses the ``BushWorldStateEncoder`` to:
        1. Tensorize the raw state into a single ``(1, input_dim)`` tensor.
        2. Forward through the encoder network to get a feature vector.
        3. Return as a flat ``float32`` numpy array.

        The encoder forward pass is run without gradient tracking (this
        is an environment-side operation; gradients for the encoder flow
        through the auxiliary-network training path instead).
        """
        encoder_device = next(self._state_encoder.parameters(), None)
        encoder_device = (
            encoder_device.device if encoder_device is not None else torch.device("cpu")
        )
        with torch.no_grad():
            state_t = self._state_encoder.tensorize_state(
                state, self.world_model, device=encoder_device
            )
            features = self._state_encoder(state_t)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
