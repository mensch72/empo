"""
Tools-specific ``EMPOWorldModelEnv`` subclass for PPO Phase 2 training.

Overrides :meth:`~EMPOWorldModelEnv._state_to_obs` with a
:class:`ToolsStateEncoder` to convert the raw tools state tuple
into a flat observation vector.
"""

from __future__ import annotations

from typing import Any, Callable, List

import numpy as np
import torch
from gymnasium import spaces

from empo.learning_based.phase2_ppo.env_wrapper import EMPOWorldModelEnv
from empo.learning_based.tools.state_encoder import ToolsStateEncoder


class ToolsWorldModelEnv(EMPOWorldModelEnv):
    """Tools PPO environment wrapper with MLP-based state encoding.

    Parameters
    ----------
    world_model : ToolsWorldModel
        The tools world model.
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
    state_encoder : ToolsStateEncoder
        Pre-constructed state encoder.
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
        state_encoder: ToolsStateEncoder,
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
        """Convert a tools world-model state to a flat observation."""
        encoder_device = next(self._state_encoder.parameters()).device
        with torch.no_grad():
            x = self._state_encoder.tensorize_state(
                state, self.world_model, device=encoder_device
            )
            features = self._state_encoder(x)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
