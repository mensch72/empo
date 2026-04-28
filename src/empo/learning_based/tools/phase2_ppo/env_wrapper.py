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
    # Human action sampling (per-agent action counts)
    # ------------------------------------------------------------------

    def _sample_human_actions(self, state):
        """Sample human actions using per-agent action counts.

        The base class uses ``config.num_actions`` (the robot's count) for
        the no-goal fallback, but tools agents may have different numbers
        of valid actions.  This override reads the per-agent counts from
        the world model.
        """
        all_indices = self.human_agent_indices + self.robot_agent_indices
        n_agents = max(all_indices) + 1 if all_indices else 0
        actions = [0] * n_agents
        per_agent = self.world_model.n_actions_per_agent
        for h_idx in self.human_agent_indices:
            goal = self._goals.get(h_idx)
            if goal is None:
                actions[h_idx] = int(self._py_rng.randint(per_agent[h_idx]))
            else:
                probs = self.human_policy_prior(state, h_idx, goal, self.world_model)
                actions[h_idx] = int(
                    self._py_rng.choice(len(probs), p=np.asarray(probs))
                )
        return actions

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
