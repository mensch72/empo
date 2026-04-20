"""
Gymnasium-compatible environment wrapper for PPO-based Phase 2 training.

``EMPOWorldModelEnv`` wraps a :class:`WorldModel` so that:

1. Each ``step()`` simulates the human agents using a policy prior,
   samples goals, and advances the world model — only the robot action(s)
   are exposed to the RL agent.
2. The reward returned is the intrinsic EMPO reward **U_r(s_t)** evaluated
   at the *pre-transition* state (consistent with the Bellman equation
   V_r(s) = U_r(s) + γ E[V_r(s')]; see Appendix B of the migration plan).
3. The Gymnasium ``info`` dict returned by each ``step()`` is restricted
   to numeric scalar values suitable for logging/monitoring. Rich auxiliary
   transition data (states, goals, human actions, transition probabilities,
   etc.) is stored internally in ``_aux_buffer`` so the PPO training loop
   can consume it when training the auxiliary networks (V_h^e, X_h, U_r).

This module does NOT modify any existing environment or wrapper code.
"""

from __future__ import annotations

from collections import deque
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


class EMPOWorldModelEnv(gymnasium.Env):
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

        # Action space: always flat Discrete over joint robot actions.
        # For multi-robot, the flat index is decoded to per-robot actions
        # inside _build_joint_action() via _flat_index_to_tuple().
        # This matches EMPOActorCritic which outputs logits of shape
        # (batch, num_actions ** num_robots) — a single Discrete head.
        num_joint_actions = num_actions**num_robots
        self.action_space = spaces.Discrete(num_joint_actions)

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

        # Ring buffer for auxiliary transition data collected during step().
        # The trainer reads this directly (via vecenv env references) after
        # each PufferLib evaluate() call, bypassing PufferLib's info
        # aggregation which cannot handle non-numeric values.
        # Size is rollout_length + 1 to accommodate edge cases where an
        # episode boundary adds one extra transition before the trainer
        # drains the buffer.
        self._aux_buffer: deque = deque(maxlen=config.ppo_rollout_length + 1)

        # Seeded RNG for reproducibility (set from Gymnasium's np_random
        # during reset(); initial value is a throwaway that is never used
        # because reset() must be called before step()).
        self._py_rng: np.random.RandomState = np.random.RandomState(0)

        # U_r normalisation: dividing by a scale factor normalises the
        # variance of U_r for stable PPO training.  PufferLib's hard reward
        # clamp has been patched out (see vendor/pufferlib/pufferl.py).
        # Prefer config.u_r_scale (empirical, from calibration) over
        # the conservative theoretical upper-bound.
        if config.u_r_scale is not None:
            self._u_r_scale: float = config.u_r_scale
        else:
            if config.use_simplified_x_h:
                # Simplified mode has X_h >= 1 so y = mean(X_h^-xi) <= 1,
                # hence U_r = -(y^eta) is naturally in [-1, 0].
                self._u_r_scale = 1.0
            else:
                _X_H_MIN = 1e-3
                self._u_r_scale = (_X_H_MIN ** (-config.xi)) ** config.eta

        # Running scale normalization for U_r:
        # - During warmup: collect U_r samples to estimate std
        # - After warmup: freeze std and scale U_r → U_r / (σ + ε)
        # This preserves the sign of U_r (non-positive by construction).
        self._u_r_frozen: bool = False  # True after warmup ends
        self._u_r_std: float = 1.0
        self._u_r_warmup_buffer: deque = deque(
            maxlen=config.ppo_rollout_length * 100
        )  # buffer ~100 rollouts

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
        # Seed the internal RNG from Gymnasium's np_random
        self._py_rng = np.random.RandomState(self.np_random.integers(0, 2**31))
        self.world_model.reset(seed=seed, options=options)
        state = self.world_model.get_state()

        # Sample initial goals for each human
        self._resample_goals(state)
        self._step_count = 0
        self._aux_buffer.clear()

        obs = self._state_to_obs(state)
        info: Dict[str, Any] = {"state": state}
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # -- Capture pre-transition state for U_r(s_t) --
        pre_state = self.world_model.get_state()

        # -- Sample human actions from policy prior --
        human_actions = self._sample_human_actions(pre_state)

        # -- Build joint action and step the world model --
        joint_action = self._build_joint_action(action, human_actions)
        _, env_reward, terminated, truncated, _wm_info = self.world_model.step(
            joint_action
        )

        next_state = self.world_model.get_state()
        self._step_count += 1

        # -- Truncate if episode exceeds maximum length --
        # Only mark truncated when the world model hasn't already terminated the
        # episode, so terminated and truncated remain mutually exclusive.
        if self._step_count >= self.config.steps_per_episode and not terminated:
            truncated = True

        # -- Compute intrinsic reward U_r(s_t) at pre-transition state --
        u_r_raw = self._compute_u_r(pre_state)

        # Collect statistics during warmup; scale by std after freezing.
        if not self._u_r_frozen:
            self._u_r_warmup_buffer.append(u_r_raw)
            u_r = u_r_raw  # Raw U_r during warmup
            # Scale during warmup (only scale, no variance normalization yet)
            u_r = u_r / self._u_r_scale
        else:
            # After warmup: std-only scaling (no centering).
            u_r = u_r_raw / (self._u_r_std + 1e-8)

        # Distinguish theory quantity from PPO training reward.
        # - u_r_raw: theoretical U_r(s) (must remain non-positive)
        # - u_r: reward returned to PPO (scaled for stable optimization)
        u_r_ppo = u_r

        # -- Goal resampling (stochastic, using seeded RNG) --
        if self._py_rng.random() < self.config.goal_resample_prob:
            self._resample_goals(next_state)

        # -- Compute transition probabilities for auxiliary training --
        # This can be expensive for multi-robot environments (enumerates
        # |A|^N joint actions).  Disable via config.compute_transition_probs
        # when auxiliary training does not need model-based targets.
        if getattr(self.config, "compute_transition_probs", False):
            transition_probs = self._compute_transition_probs(pre_state, human_actions)
        else:
            transition_probs = None

        obs = self._state_to_obs(next_state)

        # ``info`` must only contain scalar-numeric values because PufferLib's
        # Serial backend aggregates infos via ``np.mean`` over each key.
        # Rich auxiliary data is instead stored in ``_aux_buffer`` below.
        info: Dict[str, Any] = {
            "env_reward": env_reward,
            "u_r": u_r_raw,
            "u_r_ppo": u_r_ppo,
            # Dashboard prints 3 decimals; this reveals proximity to -1.0.
            "u_r_plus1": (u_r_raw + 1.0),
        }

        # Decode flat joint-action index to per-robot action tuple for
        # Phase2Transition.robot_action compatibility.
        num_robots = len(self.robot_agent_indices)
        if num_robots == 1:
            robot_action_tuple = (int(action),)
        else:
            robot_action_tuple = _flat_index_to_tuple(
                int(action), self.config.num_actions, num_robots
            )

        # Store auxiliary data for trainer to read directly (bypasses
        # PufferLib info aggregation which only handles numeric scalars).
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
        return obs, u_r_ppo, terminated, truncated, info

    # ------------------------------------------------------------------
    # Variance normalization control
    # ------------------------------------------------------------------

    def freeze_u_r_normalization(self) -> None:
        """Freeze U_r scale statistics and switch to std-scaled rewards.

        Called by trainer after warmup phase ends. Estimates std from
        collected samples, then applies U_r / (σ + ε) to all subsequent
        U_r values.
        """
        if self._u_r_frozen:
            return  # Already frozen

        if len(self._u_r_warmup_buffer) > 0:
            u_r_array = np.array(list(self._u_r_warmup_buffer))
            self._u_r_std = float(np.std(u_r_array))
            if self._u_r_std <= 0.0:
                self._u_r_std = 1.0
        else:
            # No samples collected (unusual), use defaults
            self._u_r_std = 1.0

        self._u_r_frozen = True
        self._u_r_warmup_buffer.clear()  # Free memory

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
                actions[h_idx] = int(self._py_rng.randint(self.config.num_actions))
            else:
                probs = self.human_policy_prior(state, h_idx, goal, self.world_model)
                actions[h_idx] = int(
                    self._py_rng.choice(len(probs), p=np.asarray(probs))
                )
        return actions

    def _build_joint_action(
        self, robot_action: Any, human_actions: List[int]
    ) -> List[int]:
        """Merge the robot action(s) into the human-action list.

        ``robot_action`` is always a flat integer index into the joint
        robot action space (Discrete of size num_actions ** num_robots).
        For multi-robot, it is decoded via ``_flat_index_to_tuple()``.
        """
        joint = list(human_actions)
        num_robots = len(self.robot_agent_indices)
        if num_robots == 1:
            joint[self.robot_agent_indices[0]] = int(robot_action)
        else:
            per_robot = _flat_index_to_tuple(
                int(robot_action), self.config.num_actions, num_robots
            )
            for i, r_idx in enumerate(self.robot_agent_indices):
                joint[r_idx] = per_robot[i]
        return joint

    def _compute_u_r(self, state: Any) -> float:
        """Compute U_r(s) using auxiliary networks (or return 0.0).

        When frozen target copies (``*_target``) are available (created by
        :meth:`PPOPhase2Trainer.freeze_auxiliary_networks`), they are used
        for reward computation so that the intrinsic reward is stationary
        within a rollout.  Falls back to the online networks when targets
        are absent.

        X_h values are lower-bounded to prevent numerical instability in
        the X_h^{-ξ} exponentiation:
        - Standard mode: floor at 1e-3.
        - Simplified mode: floor at 1.0 (no upper clamp).

        The device is inferred from the auxiliary network parameters
        to avoid device-mismatch errors when running on CUDA.
        """
        if self.auxiliary_networks is None:
            return 0.0

        import torch

        # Lower-bound values matching trainer.py semantics.
        x_h_floor = 1.0 if self.config.use_simplified_x_h else 1e-3

        nets = self.auxiliary_networks

        # Prefer frozen target networks for stationary reward during rollouts.
        u_r_net = getattr(nets, "u_r_target", None) or getattr(nets, "u_r", None)
        x_h_net = getattr(nets, "x_h_target", None) or getattr(nets, "x_h", None)
        # v_h_e is not used directly for reward computation (only U_r and
        # X_h are), but we include it in the device probe below as a
        # fallback when neither U_r nor X_h has parameters.
        v_h_e_net = getattr(nets, "v_h_e_target", None) or getattr(nets, "v_h_e", None)

        # Infer device from aux network parameters, defaulting to cpu
        device = "cpu"
        for net in [u_r_net, x_h_net, v_h_e_net]:
            if net is not None:
                try:
                    device = str(next(net.parameters()).device)
                    break
                except StopIteration:
                    pass

        with torch.no_grad():
            # Option A: dedicated U_r network
            if u_r_net is not None:
                _, u_r = u_r_net(state, self.world_model, device)
                # U_r must be non-positive by construction.
                return min(float(u_r.item()), 0.0)
            # Option B: compute from X_h values directly (eq. 8)
            if x_h_net is not None:
                x_h_vals = []
                for h_idx in self.human_agent_indices:
                    x_h = x_h_net(state, self.world_model, h_idx, device)
                    x_h_vals.append(max(float(x_h.item()), x_h_floor))
                if x_h_vals:
                    y = float(np.mean([x ** (-self.config.xi) for x in x_h_vals]))
                    u_r = -(y**self.config.eta)
                    # Guard against numerical/model drift to positive values.
                    return min(u_r, 0.0)
        return 0.0

    def _compute_transition_probs(
        self, state: Any, human_actions: List[int]
    ) -> Dict[int, list]:
        """Compute per-robot-action transition probabilities.

        Returns ``{action_index: [(prob, next_state), ...]}``.

        The action space is always flat ``Discrete(num_actions ** num_robots)``
        so each ``a_r_idx`` is a flat joint-action index.  For multi-robot
        envs, ``_build_joint_action()`` internally decodes it to per-robot
        actions via ``_flat_index_to_tuple()``.
        """
        num_robot_actions = self.action_space.n

        probs_by_action: Dict[int, list] = {}
        for a_r_idx in range(num_robot_actions):
            joint = self._build_joint_action(a_r_idx, human_actions)
            trans = self.world_model.transition_probabilities(state, joint)
            if trans is not None:
                probs_by_action[a_r_idx] = trans
        return probs_by_action

    def _state_to_obs(self, state: Any) -> np.ndarray:
        """Convert a world-model state to a flat observation.

        Subclasses (e.g. the MultiGrid-specific wrapper) must override
        this to produce a proper observation using the world_model's encoder
        or raw observations.

        The base implementation deliberately raises ``NotImplementedError`` to
        avoid silently training PPO on constant (zero) observations when
        ``EMPOWorldModelEnv`` is used directly.  Callers should either
        subclass ``EMPOWorldModelEnv`` and implement a real encoder here, or
        extend the constructor to accept an explicit observation function.
        """
        raise NotImplementedError(
            "EMPOWorldModelEnv._state_to_obs() must be overridden to "
            "convert world_model states to observations. "
            "Provide an encoder (e.g., via a subclass) instead of "
            "relying on the base implementation."
        )


# Backward-compatible alias (the class was previously named EMPOMultiGridEnv
# even though it is not MultiGrid-specific).
EMPOMultiGridEnv = EMPOWorldModelEnv
