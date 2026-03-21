"""
Gymnasium-compatible environment wrapper for PPO-based Phase 1 training.

``Phase1PPOEnv`` wraps a :class:`WorldModel` so that:

1. Each episode starts by sampling a goal for the training human agent from
   the goal sampler.  The goal is included in the observation as additional
   features (goal-conditioned observation).
2. All other agents (humans and robots) are controlled by provided policy
   callables.
3. The reward is ``goal.is_achieved(state)`` — a binary 0/1 signal — plus
   an optional distance-based reward shaping term.
4. The environment follows the Gymnasium 5-tuple API.

This module does NOT modify any existing environment or wrapper code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces


class Phase1PPOEnv(gymnasium.Env):
    """PufferLib-compatible wrapper around a :class:`WorldModel` for Phase 1.

    The environment trains a goal-conditioned policy for a single human agent.
    A new goal is sampled at the start of each episode (and optionally
    mid-episode with probability ``config.goal_resample_prob``).  The
    observation is the concatenation of encoded state features and encoded
    goal features.

    Parameters
    ----------
    world_model : WorldModel
        The EMPO world model (must implement ``get_state``, ``set_state``,
        ``step``, ``reset``).
    goal_sampler : callable
        ``(state, human_idx) → (goal, weight)`` — samples a possible goal
        and its weight for the given human.
    training_human_index : int
        Index of the human agent being trained.
    other_agent_policies : dict[int, callable]
        ``{agent_idx: (state, agent_idx) → action}`` — policies for all
        other agents (humans and robots).  Each callable takes the state
        and agent index and returns an integer action.
    config : PPOPhase1Config
        PPO Phase 1 configuration.
    obs_dim : int
        Dimensionality of the concatenated (state + goal) observation.
    state_to_obs : callable or None
        ``(state, goal) → np.ndarray`` — converts a state and goal to a
        flat observation vector.  Must be provided (the base implementation
        raises ``NotImplementedError``).
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        world_model: Any,
        goal_sampler: Callable,
        training_human_index: int,
        other_agent_policies: Dict[int, Callable],
        config: Any,
        obs_dim: int,
        state_to_obs: Optional[Callable] = None,
    ):
        super().__init__()
        self.world_model = world_model
        self.goal_sampler = goal_sampler
        self.training_human_index = training_human_index
        self.other_agent_policies = dict(other_agent_policies)
        self.config = config
        self._obs_dim = obs_dim
        self._state_to_obs_fn = state_to_obs

        self.action_space = spaces.Discrete(config.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._current_goal: Any = None
        self._current_goal_weight: float = 1.0
        self._step_count: int = 0

        # Seeded RNG (set from Gymnasium's np_random during reset())
        self._py_rng: np.random.RandomState = np.random.RandomState(0)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self._py_rng = np.random.RandomState(self.np_random.integers(0, 2**31))
        self.world_model.reset(seed=seed, options=options)
        state = self.world_model.get_state()

        # Sample a goal for this episode
        self._current_goal, self._current_goal_weight = self.goal_sampler(
            state, self.training_human_index
        )
        self._step_count = 0

        obs = self._state_to_obs(state, self._current_goal)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        state = self.world_model.get_state()

        # Build joint action: training human uses the PPO action,
        # all other agents use their provided policies.
        joint_action = self._build_joint_action(state, int(action))

        # Step the world model
        _, _, terminated, truncated, _wm_info = self.world_model.step(joint_action)

        next_state = self.world_model.get_state()
        self._step_count += 1

        # Truncate at episode length limit
        if self._step_count >= self.config.steps_per_episode:
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_joint_action(self, state: Any, human_action: int) -> List[int]:
        """Build a joint action for all agents.

        The training human uses the provided ``human_action``.  All other
        agents use their respective policies from ``other_agent_policies``.
        """
        all_indices = [self.training_human_index] + list(
            self.other_agent_policies.keys()
        )
        n_agents = max(all_indices) + 1 if all_indices else 1
        actions: List[int] = [0] * n_agents
        actions[self.training_human_index] = human_action

        for agent_idx, policy_fn in self.other_agent_policies.items():
            actions[agent_idx] = int(policy_fn(state, agent_idx))
        return actions

    def _state_to_obs(self, state: Any, goal: Any) -> np.ndarray:
        """Convert state and goal to observation vector.

        Delegates to the ``state_to_obs`` callable provided at construction.
        Subclasses may override this instead of providing a callable.
        """
        if self._state_to_obs_fn is not None:
            return self._state_to_obs_fn(state, goal)
        raise NotImplementedError(
            "Phase1PPOEnv._state_to_obs() must be overridden or a "
            "'state_to_obs' callable must be provided at construction."
        )

    def _compute_reward_shaping(self, state: Any, next_state: Any) -> float:
        """Compute optional reward shaping (default: 0.0).

        Subclasses may override to provide domain-specific distance-based
        reward shaping (e.g. negative distance to goal position).
        """
        return 0.0
