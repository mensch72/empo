"""
Gymnasium-compatible environment wrapper for PPO-based Phase 2 training.

``EMPOMultiGridEnv`` wraps a :class:`WorldModel` so that:

1. Each ``step()`` simulates the human agents using a policy prior,
   samples goals, and advances the world model — only the robot action(s)
   are exposed to the RL agent.
2. The reward returned is the intrinsic EMPO reward **U_r(s_t)** evaluated
   at the *pre-transition* state (consistent with the Bellman equation
   V_r(s) = U_r(s) + γ E[V_r(s')]; see Appendix B of the migration plan).
3. Rich auxiliary data (states, goals, human actions, transition
   probabilities) is packed into the ``info`` dict so the PPO training loop
   can extract it and train the auxiliary networks (V_h^e, X_h, U_r).

This module does NOT modify any existing environment or wrapper code.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces


def _flat_index_to_tuple(
    index: int, num_actions: int, num_robots: int
) -> Tuple[int, ...]:
    """Convert a flat joint-action index to a per-robot action tuple.

    Inverse of :meth:`EMPOActorCritic.action_tuple_to_index` (defined in
    ``actor_critic.py``).  Uses little-endian base-``num_actions`` expansion::

        index = a_0 + a_1 * num_actions + a_2 * num_actions^2 + ...
    """
    actions = []
    remaining = index
    for _ in range(num_robots):
        actions.append(remaining % num_actions)
        remaining //= num_actions
    return tuple(actions)


class EMPOMultiGridEnv(gymnasium.Env):
    """PufferLib-compatible wrapper around a :class:`WorldModel`.

    Parameters
    ----------
    world_model : WorldModel
        The EMPO world model (must implement ``get_state``, ``set_state``,
        ``step``, ``reset``, ``transition_probabilities``).
    human_policy_prior : callable
        ``(state, human_idx, goal, world_model) → action_probs`` — returns a
        distribution over actions for the given human and goal.
    goal_sampler : callable
        ``(state, human_idx) → (goal, weight)`` — samples a possible goal and
        its weight for the given human.
    human_agent_indices : list[int]
        Indices of human agents in the world model.
    robot_agent_indices : list[int]
        Indices of robot agents in the world model.
    config : PPOPhase2Config
        PPO Phase 2 configuration.
    auxiliary_networks : object or None
        Frozen auxiliary networks used to compute U_r(s).  When ``None`` the
        environment returns zero reward (useful for warm-up / testing).
    observation_space : gymnasium.Space or None
        Override the observation space.  When ``None`` a flat ``Box`` of
        dimension ``obs_dim`` is used (caller must set ``obs_dim``).
    obs_dim : int
        Flat observation dimensionality (used when ``observation_space`` is
        not provided).  Default 64.
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        world_model: Any,
        human_policy_prior: Callable,
        goal_sampler: Callable,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        config: Any,
        auxiliary_networks: Any = None,
        observation_space: Optional[gymnasium.Space] = None,
        obs_dim: int = 64,
    ):
        super().__init__()
        self.world_model = world_model
        self.human_policy_prior = human_policy_prior
        self.goal_sampler = goal_sampler
        self.human_agent_indices = list(human_agent_indices)
        self.robot_agent_indices = list(robot_agent_indices)
        self.config = config
        self.auxiliary_networks = auxiliary_networks

        num_robots = len(self.robot_agent_indices)
        num_actions = config.num_actions

        # Action space: Discrete for single robot, MultiDiscrete for multiple
        if num_robots == 1:
            self.action_space = spaces.Discrete(num_actions)
        else:
            self.action_space = spaces.MultiDiscrete(
                [num_actions] * num_robots
            )

        # Observation space
        if observation_space is not None:
            self.observation_space = observation_space
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # Current goals for each human
        self._goals: Dict[int, Any] = {}
        self._goal_weights: Dict[int, float] = {}
        self._step_count: int = 0

        # Seeded RNG for reproducibility (set from Gymnasium's np_random
        # during reset(); initial value is a throwaway that is never used
        # because reset() must be called before step()).
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
        super().reset(seed=seed)
        # Seed the internal RNG from Gymnasium's np_random
        self._py_rng = np.random.RandomState(
            self.np_random.integers(0, 2**31)
        )
        self.world_model.reset(seed=seed)
        state = self.world_model.get_state()

        # Sample initial goals for each human
        self._resample_goals(state)
        self._step_count = 0

        obs = self._state_to_obs(state)
        info: Dict[str, Any] = {"state": state}
        return obs, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # -- Capture pre-transition state for U_r(s_t) --
        pre_state = self.world_model.get_state()

        # -- Sample human actions from policy prior --
        human_actions = self._sample_human_actions(pre_state)

        # -- Build joint action and step the world model --
        joint_action = self._build_joint_action(action, human_actions)
        _, env_reward, terminated, truncated, wm_info = self.world_model.step(
            joint_action
        )

        next_state = self.world_model.get_state()
        self._step_count += 1

        # -- Truncate if episode exceeds maximum length --
        if self._step_count >= self.config.steps_per_episode:
            truncated = True

        # -- Compute intrinsic reward U_r(s_t) at pre-transition state --
        u_r = self._compute_u_r(pre_state)

        # -- Goal resampling (stochastic, using seeded RNG) --
        if self._py_rng.random() < self.config.goal_resample_prob:
            self._resample_goals(next_state)

        # -- Compute transition probabilities for auxiliary training --
        transition_probs = self._compute_transition_probs(
            pre_state, human_actions
        )

        obs = self._state_to_obs(next_state)
        info: Dict[str, Any] = {
            "state": pre_state,
            "next_state": next_state,
            "goals": dict(self._goals),
            "goal_weights": dict(self._goal_weights),
            "human_actions": human_actions,
            "transition_probs": transition_probs,
            "env_reward": env_reward,
            "u_r": u_r,
        }
        return obs, u_r, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resample_goals(self, state: Any) -> None:
        """(Re-)sample goals for each human from the goal sampler."""
        for h_idx in self.human_agent_indices:
            goal, weight = self.goal_sampler(state, h_idx)
            self._goals[h_idx] = goal
            self._goal_weights[h_idx] = weight

    def _sample_human_actions(self, state: Any) -> List[int]:
        """Sample an action for every agent; humans use the policy prior.

        The joint-action list is sized by ``max(all_indices) + 1`` so that
        non-contiguous agent indices are handled correctly (matching the
        DQN Phase 2 trainer convention).
        """
        all_indices = self.human_agent_indices + self.robot_agent_indices
        n_agents = max(all_indices) + 1 if all_indices else 0
        actions: List[int] = [0] * n_agents
        for h_idx in self.human_agent_indices:
            goal = self._goals.get(h_idx)
            if goal is None:
                # Uniform random if no goal assigned yet
                actions[h_idx] = int(
                    self._py_rng.randint(self.config.num_actions)
                )
            else:
                probs = self.human_policy_prior(
                    state, h_idx, goal, self.world_model
                )
                actions[h_idx] = int(
                    self._py_rng.choice(len(probs), p=np.asarray(probs))
                )
        return actions

    def _build_joint_action(
        self, robot_action: Any, human_actions: List[int]
    ) -> List[int]:
        """Merge the robot action(s) into the human-action list."""
        joint = list(human_actions)
        if isinstance(self.action_space, spaces.Discrete):
            assert len(self.robot_agent_indices) == 1
            joint[self.robot_agent_indices[0]] = int(robot_action)
        else:
            for i, r_idx in enumerate(self.robot_agent_indices):
                joint[r_idx] = int(robot_action[i])
        return joint

    def _compute_u_r(self, state: Any) -> float:
        """Compute U_r(s) using auxiliary networks (or return 0.0).

        X_h values are floored at ``_X_H_MIN`` (1e-3) to prevent
        numerical instability in the X_h^{-ξ} exponentiation.
        """
        if self.auxiliary_networks is None:
            return 0.0

        import torch

        # Floor value matching trainer.py
        _X_H_MIN = 1e-3

        with torch.no_grad():
            nets = self.auxiliary_networks
            # Option A: dedicated U_r network
            if hasattr(nets, "u_r") and nets.u_r is not None:
                _, u_r = nets.u_r.forward(
                    state, self.world_model, "cpu"
                )
                return float(u_r.item())
            # Option B: compute from X_h values directly (eq. 8)
            if hasattr(nets, "x_h") and nets.x_h is not None:
                x_h_vals = []
                for h_idx in self.human_agent_indices:
                    x_h = nets.x_h.forward(
                        state, self.world_model, h_idx, "cpu"
                    )
                    x_h_vals.append(max(float(x_h.item()), _X_H_MIN))
                if x_h_vals:
                    y = float(
                        np.mean(
                            [x ** (-self.config.xi) for x in x_h_vals]
                        )
                    )
                    return -(y ** self.config.eta)
        return 0.0

    def _compute_transition_probs(
        self, state: Any, human_actions: List[int]
    ) -> Dict[int, list]:
        """Compute per-robot-action transition probabilities.

        Returns ``{action_index: [(prob, next_state), ...]}``.
        Handles both ``Discrete`` and ``MultiDiscrete`` action spaces.

        For ``MultiDiscrete`` (multi-robot) the flat joint-action index is
        converted to a per-robot action tuple before calling
        ``_build_joint_action``, which expects an indexable sequence.
        """
        if isinstance(self.action_space, spaces.Discrete):
            num_robot_actions = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            num_robot_actions = int(np.prod(self.action_space.nvec))
        else:
            return {}

        num_actions = self.config.num_actions
        num_robots = len(self.robot_agent_indices)

        probs_by_action: Dict[int, list] = {}
        for a_r_idx in range(num_robot_actions):
            # For MultiDiscrete, convert flat index to per-robot action tuple
            if isinstance(self.action_space, spaces.MultiDiscrete):
                robot_action: Any = _flat_index_to_tuple(
                    a_r_idx, num_actions, num_robots
                )
            else:
                robot_action = a_r_idx
            joint = self._build_joint_action(robot_action, human_actions)
            trans = self.world_model.transition_probabilities(state, joint)
            if trans is not None:
                probs_by_action[a_r_idx] = trans
        return probs_by_action

    def _state_to_obs(self, state: Any) -> np.ndarray:
        """Convert a world-model state to a flat observation.

        Subclasses (e.g. the MultiGrid-specific wrapper) should override
        this to produce a proper observation using the environment's encoder.
        The base implementation returns a zero vector of the right shape.
        """
        return np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
