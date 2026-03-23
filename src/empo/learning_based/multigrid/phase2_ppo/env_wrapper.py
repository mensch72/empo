"""
MultiGrid-specific ``EMPOWorldModelEnv`` subclass for PPO Phase 2 training.

Provides :class:`MultiGridWorldModelEnv`, which overrides
:meth:`~EMPOWorldModelEnv._state_to_obs` with a real observation encoder
backed by :class:`MultiGridStateEncoder`.

This module does NOT modify any code in ``learning_based/multigrid/phase2/``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces

from empo.learning_based.phase2_ppo.env_wrapper import EMPOWorldModelEnv
from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder


class MultiGridWorldModelEnv(EMPOWorldModelEnv):
    """MultiGrid PPO environment wrapper with real state-to-observation encoding.

    Extends :class:`EMPOWorldModelEnv` by implementing ``_state_to_obs``
    using a :class:`MultiGridStateEncoder` to convert the raw world-model
    state tuple into a flat ``float32`` observation vector suitable for the
    PPO actor-critic.

    Parameters
    ----------
    world_model : MultiGridEnv (WorldModel)
        The MultiGrid world model.
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
    state_encoder : MultiGridStateEncoder
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
        state_encoder: MultiGridStateEncoder,
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
        keyword arguments.  Seeding is handled by the Gymnasium base class
        (``super().reset(seed=...)``), and the world model is reset via
        its no-argument ``reset()`` method.
        """
        import gymnasium

        # Let Gymnasium set np_random from the seed
        gymnasium.Env.reset(self, seed=seed, options=options)

        # Seed internal RNG from Gymnasium's np_random
        self._py_rng = np.random.RandomState(self.np_random.integers(0, 2**31))

        # Reset world model (MultiGridEnv.reset takes no kwargs)
        self.world_model.reset()
        state = self.world_model.get_state()

        # Sample initial goals for each human
        self._resample_goals(state)
        self._step_count = 0
        self._aux_buffer.clear()

        obs = self._state_to_obs(state)
        info: Dict[str, Any] = {"state": state}
        return obs, info

    def _step_world_model(self, joint_action):
        """Step the world model and return a 5-tuple.

        ``MultiGridEnv.step()`` returns the old Gym API 4-tuple
        ``(obs, reward, done, info)`` without a separate ``truncated``
        flag.  This helper normalises the return value to the Gymnasium
        5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        result = self.world_model.step(joint_action)
        if len(result) == 5:
            return result
        # Old-style 4-tuple: treat done as terminated, truncated=False
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info

    def step(self, action):
        """Step the environment.

        Overrides the base class to handle the fact that
        ``MultiGridEnv.step()`` returns the old Gym 4-tuple instead of
        the Gymnasium 5-tuple.
        """
        from empo.learning_based.phase2_ppo.env_wrapper import _flat_index_to_tuple

        # -- Capture pre-transition state for U_r(s_t) --
        pre_state = self.world_model.get_state()

        # -- Sample human actions from policy prior --
        human_actions = self._sample_human_actions(pre_state)

        # -- Build joint action and step the world model --
        joint_action = self._build_joint_action(action, human_actions)
        _, env_reward, terminated, truncated, _wm_info = self._step_world_model(
            joint_action
        )

        # -- EMPO finite-horizon interpretation --
        # EMPO uses backward induction over a finite horizon (max_steps).
        # When the world_model signals `done` because it hit its own
        # max-steps limit, we keep this as a genuine terminal state
        # (no value bootstrapping), matching BI's finite-horizon V_r.

        next_state = self.world_model.get_state()
        self._step_count += 1

        # -- Terminate if episode exceeds maximum length (wrapper-level cap) --
        # Treated as true termination (not truncation) because EMPO's
        # backward induction computes V_r over a finite horizon.
        if self._step_count >= self.config.steps_per_episode:
            terminated = True

        # -- Compute intrinsic reward U_r(s_t) at pre-transition state --
        u_r = self._compute_u_r(pre_state)

        # Normalise into [-1, ≈0] so PufferLib's clamp(r, -1, 1) is benign.
        u_r = u_r / self._u_r_scale

        # -- Goal resampling (stochastic, using seeded RNG) --
        if self._py_rng.random() < self.config.goal_resample_prob:
            self._resample_goals(next_state)

        # -- Compute transition probabilities for auxiliary training --
        if getattr(self.config, "compute_transition_probs", False):
            transition_probs = self._compute_transition_probs(pre_state, human_actions)
        else:
            transition_probs = None

        obs = self._state_to_obs(next_state)

        info: Dict[str, Any] = {
            "env_reward": (
                float(np.sum(env_reward))
                if hasattr(env_reward, "__len__")
                else float(env_reward)
            ),
            "u_r": u_r,
        }

        # Decode flat joint-action index to per-robot action tuple
        num_robots = len(self.robot_agent_indices)
        if num_robots == 1:
            robot_action_tuple = (int(action),)
        else:
            robot_action_tuple = _flat_index_to_tuple(
                int(action), self.config.num_actions, num_robots
            )

        self._aux_buffer.append(
            {
                "state": pre_state,
                "next_state": next_state,
                "goals": dict(self._goals),
                "goal_weights": dict(self._goal_weights),
                "human_actions": human_actions,
                "transition_probs": transition_probs,
                "robot_action": robot_action_tuple,
                "terminated": terminated,
                "truncated": truncated,
                "terminal": bool(terminated or truncated),
            }
        )
        return obs, u_r, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: Any) -> np.ndarray:
        """Convert a MultiGrid world-model state to a flat observation.

        Uses the ``MultiGridStateEncoder`` to:
        1. Tensorize the raw state (grid + global + agent + interactive features).
        2. Forward through the encoder network to get a feature vector.
        3. Return as a flat ``float32`` numpy array.

        The encoder forward pass is run without gradient tracking (this
        is an environment-side operation; gradients for the encoder flow
        through the auxiliary-network training path instead).
        """
        # Ensure tensorization and forward happen on the same device as
        # the encoder so that there is no device-mismatch when the
        # encoder lives on CUDA.
        encoder_device = next(self._state_encoder.parameters()).device
        with torch.no_grad():
            grid_t, glob_f, agent_f, inter_f = self._state_encoder.tensorize_state(
                state, self.world_model, device=encoder_device
            )
            features = self._state_encoder(grid_t, glob_f, agent_f, inter_f)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
