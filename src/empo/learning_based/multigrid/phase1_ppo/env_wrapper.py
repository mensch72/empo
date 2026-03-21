"""
MultiGrid-specific ``Phase1PPOEnv`` subclass for PPO Phase 1 training.

Provides :class:`MultiGridPhase1PPOEnv`, which overrides
:meth:`~Phase1PPOEnv._state_to_obs` with a real observation encoder
backed by :class:`MultiGridStateEncoder` and :class:`MultiGridGoalEncoder`.

This module does NOT modify any code in ``learning_based/multigrid/phase1/``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
from gymnasium import spaces

from empo.learning_based.phase1_ppo.env_wrapper import Phase1PPOEnv
from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.goal_encoder import MultiGridGoalEncoder


class MultiGridPhase1PPOEnv(Phase1PPOEnv):
    """MultiGrid PPO Phase 1 environment wrapper with real state+goal encoding.

    Extends :class:`Phase1PPOEnv` by implementing ``_state_to_obs``
    using a :class:`MultiGridStateEncoder` and :class:`MultiGridGoalEncoder`
    to convert the raw world-model state and goal into a flat ``float32``
    observation vector suitable for the PPO actor-critic.

    The observation is the concatenation of:
    - State features: ``state_encoder.forward(tensorize_state(state))``
    - Goal features: ``goal_encoder.forward(tensorize_goal(goal))``

    Parameters
    ----------
    world_model : MultiGridEnv (WorldModel)
        The MultiGrid world model.
    goal_sampler : callable
        ``(state, h_idx) → (goal, weight)``.
    training_human_index : int
        Index of the human agent being trained.
    other_agent_policies : dict[int, callable]
        Policies for all other agents.
    config : PPOPhase1Config
        PPO Phase 1 configuration.
    state_encoder : MultiGridStateEncoder
        Pre-constructed state encoder.
    goal_encoder : MultiGridGoalEncoder
        Pre-constructed goal encoder.
    """

    def __init__(
        self,
        world_model: Any,
        goal_sampler: Callable,
        training_human_index: int,
        other_agent_policies: Dict[int, Callable],
        config: Any,
        state_encoder: MultiGridStateEncoder,
        goal_encoder: MultiGridGoalEncoder,
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
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment.

        Overrides the base class to handle the fact that
        ``MultiGridEnv.reset()`` does not accept ``seed`` or ``options``
        keyword arguments.
        """
        import gymnasium

        # Let Gymnasium set np_random from the seed
        gymnasium.Env.reset(self, seed=seed, options=options)

        # Seed internal RNG
        self._py_rng = np.random.RandomState(self.np_random.integers(0, 2**31))

        # Reset world model (MultiGridEnv.reset takes no kwargs)
        self.world_model.reset()
        state = self.world_model.get_state()

        # Sample a goal for this episode
        self._current_goal, self._current_goal_weight = self.goal_sampler(
            state, self.training_human_index
        )
        self._step_count = 0

        obs = self._state_to_obs(state, self._current_goal)
        info: Dict[str, Any] = {}
        return obs, info

    def _step_world_model(self, joint_action):
        """Step the world model and return a 5-tuple.

        ``MultiGridEnv.step()`` returns the old Gym API 4-tuple
        ``(obs, reward, done, info)`` without a separate ``truncated``
        flag.  This helper normalises to the Gymnasium 5-tuple.
        """
        result = self.world_model.step(joint_action)
        if len(result) == 5:
            return result
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info

    def step(self, action):
        """Step the environment.

        Overrides the base class to handle the old Gym 4-tuple from
        ``MultiGridEnv.step()``.
        """
        state = self.world_model.get_state()

        # Build joint action
        joint_action = self._build_joint_action(state, int(action))

        # Step world model (handles old Gym 4-tuple)
        _, env_reward, terminated, truncated, _wm_info = self._step_world_model(
            joint_action
        )

        # Interpret legacy MultiGrid `done` as truncation when it's a time limit
        wm_step_count = getattr(self.world_model, "step_count", None)
        wm_max_steps = getattr(self.world_model, "max_steps", None)
        if (
            truncated is False
            and terminated
            and wm_step_count is not None
            and wm_max_steps is not None
            and wm_step_count >= wm_max_steps
        ):
            terminated = False
            truncated = True

        next_state = self.world_model.get_state()
        self._step_count += 1

        # Truncate at episode length limit
        if self._step_count >= self.config.steps_per_episode and not terminated:
            truncated = True

        # Compute reward: binary goal achievement
        goal_achieved = float(self._current_goal.is_achieved(next_state))
        reward = goal_achieved

        # Optional reward shaping
        if self.config.reward_shaping_coef > 0.0:
            shaping = self._compute_reward_shaping(state, next_state)
            reward += self.config.reward_shaping_coef * shaping

        # Goal resampling mid-episode
        if self.config.goal_resample_prob > 0.0:
            if self._py_rng.random() < self.config.goal_resample_prob:
                self._current_goal, self._current_goal_weight = self.goal_sampler(
                    next_state, self.training_human_index
                )

        obs = self._state_to_obs(next_state, self._current_goal)
        info: Dict[str, Any] = {
            "goal_achieved": goal_achieved,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: Any, goal: Any) -> np.ndarray:
        """Convert state + goal to a flat observation via encoders.

        Concatenates:
        - State features from MultiGridStateEncoder
        - Goal features from MultiGridGoalEncoder
        """
        encoder_device = next(self._state_encoder.parameters()).device
        with torch.no_grad():
            # Encode state
            grid_t, glob_f, agent_f, inter_f = self._state_encoder.tensorize_state(
                state, self.world_model, device=encoder_device
            )
            state_features = self._state_encoder(grid_t, glob_f, agent_f, inter_f)

            # Encode goal
            goal_tensor = self._goal_encoder.tensorize_goal(
                goal, device=encoder_device
            )
            goal_features = self._goal_encoder(goal_tensor)

            # Concatenate
            obs = torch.cat(
                [state_features.squeeze(0), goal_features.squeeze(0)], dim=-1
            )
        return obs.cpu().numpy().astype(np.float32)
