"""
BushWorld-specific ``Phase1PPOEnv`` subclass for PPO Phase 1 training.

Provides :class:`BushWorldPhase1PPOEnv`, which overrides
:meth:`~empo.learning_based.phase1_ppo.env_wrapper.Phase1PPOEnv._state_to_obs`
with a real observation encoder backed by :class:`BushWorldStateEncoder` and
:class:`BushWorldGoalEncoder`.

BushWorld implements the standard Gymnasium ``reset(seed, options)`` /
``step(actions) -> 5-tuple`` API, so — unlike the multigrid wrapper — there is
no need to override ``reset``/``step`` for legacy compatibility.

This module does NOT modify any code in ``learning_based/bushworld/phase1/``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from empo.learning_based.phase1_ppo.env_wrapper import Phase1PPOEnv
from empo.learning_based.bushworld.goal_encoder import BushWorldGoalEncoder
from empo.learning_based.bushworld.state_encoder import BushWorldStateEncoder


class BushWorldPhase1PPOEnv(Phase1PPOEnv):
    """BushWorld PPO Phase 1 environment wrapper with real state+goal encoding.

    The observation is the concatenation of:
    - State features: ``state_encoder.forward(tensorize_state(state))``
    - Goal features: ``goal_encoder.forward(tensorize_goal(goal))``

    Parameters
    ----------
    world_model : BushWorld (WorldModel)
        The BushWorld world model.
    goal_sampler : callable
        ``(state, h_idx) → (goal, weight)``.
    training_human_index : int
        Index of the human agent being trained.
    other_agent_policies : dict[int, callable]
        Policies for all other agents (must cover every non-training agent).
    config : PPOPhase1Config
        PPO Phase 1 configuration.
    state_encoder : BushWorldStateEncoder
        Pre-constructed state encoder.
    goal_encoder : BushWorldGoalEncoder
        Pre-constructed goal encoder.
    """

    def __init__(
        self,
        world_model: Any,
        goal_sampler: Callable,
        training_human_index: int,
        other_agent_policies: Dict[int, Callable],
        config: Any,
        state_encoder: BushWorldStateEncoder,
        goal_encoder: BushWorldGoalEncoder,
    ):
        self._state_encoder = state_encoder
        self._goal_encoder = goal_encoder

        obs_dim = state_encoder.feature_dim + goal_encoder.feature_dim

        super().__init__(
            world_model=world_model,
            goal_sampler=goal_sampler,
            training_human_index=training_human_index,
            other_agent_policies=other_agent_policies,
            config=config,
            obs_dim=obs_dim,
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: Any, goal: Any) -> np.ndarray:
        """Convert state + goal to a flat observation via the encoders."""
        try:
            encoder_device = next(self._state_encoder.parameters()).device
        except StopIteration:
            try:
                encoder_device = next(self._state_encoder.buffers()).device
            except StopIteration:
                encoder_device = torch.device("cpu")

        with torch.no_grad():
            state_input = self._state_encoder.tensorize_state(
                state, self.world_model, device=encoder_device
            )
            state_features = self._state_encoder(state_input)

            goal_tensor = self._goal_encoder.tensorize_goal(goal, device=encoder_device)
            goal_features = self._goal_encoder(goal_tensor)

            obs = torch.cat(
                [state_features.squeeze(0), goal_features.squeeze(0)], dim=-1
            )
        return obs.cpu().numpy().astype(np.float32)
