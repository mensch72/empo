"""
Configuration for PPO-based Phase 2 training.

This is a **standalone** config class — it does NOT inherit from or modify
the existing ``Phase2Config``.  Shared theory parameters (γ_r, γ_h, ζ, ξ, η)
are intentionally duplicated to avoid coupling between the two code paths.
"""

from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class PPOPhase2Config:
    """
    Configuration for PPO-based Phase 2 robot policy approximation.

    This config is fully independent of :class:`Phase2Config` (DQN path).
    Theory parameters that appear in both configs are duplicated on purpose
    so that the PPO path can be modified without risk to the DQN path.

    Theory parameters
    -----------------
    gamma_r : float
        Robot discount factor.
    gamma_h : float
        Human discount factor (for V_h^e computation, eq. 6).
    zeta : float
        Risk/reliability preference parameter (ζ ≥ 1, 1 = neutral, eq. 7).
    xi : float
        Inter-human inequality aversion (ξ ≥ 1, eq. 8).
    eta : float
        Intertemporal inequality aversion (η ≥ 1, 1 = neutral, eq. 8).

    PPO hyper-parameters
    --------------------
    ppo_rollout_length : int
        Number of environment steps per PPO rollout.
    ppo_num_minibatches : int
        Number of mini-batches per PPO update epoch.
    ppo_update_epochs : int
        Number of epochs per PPO update.
    ppo_clip_coef : float
        PPO surrogate-objective clipping coefficient (ε).
    ppo_vf_coef : float
        Value-function loss coefficient.
    ppo_max_grad_norm : float
        Maximum gradient norm for gradient clipping.
    ppo_gae_lambda : float
        GAE λ for advantage estimation.
    lr_ppo : float
        Learning rate for the PPO actor-critic.
    num_envs : int
        Number of vectorised environments.
    num_ppo_iterations : int
        Total PPO update iterations (outer loop count).

    Auxiliary-network training
    -------------------------
    lr_v_h_e, lr_x_h, lr_u_r : float
        Learning rates for the auxiliary networks.
    aux_training_steps_per_iteration : int
        Gradient steps on auxiliary networks per PPO iteration.
    aux_buffer_size : int
        Capacity of the replay buffer used for auxiliary training.
    reward_freeze_interval : int
        Re-freeze auxiliary networks every *N* PPO iterations.
    batch_size : int
        Batch size for auxiliary-network gradient steps.

    Warm-up schedule (training steps = gradient updates)
    ---------------------------------------------------
    warmup_v_h_e_steps : int
        Stage 0 → 1 boundary: train V_h^e only.
    warmup_x_h_steps : int
        Stage 1 → 2 boundary: train V_h^e + X_h.
    warmup_u_r_steps : int
        Stage 2 → 3 boundary: train V_h^e + X_h + U_r.

    Entropy schedule
    ----------------
    ppo_ent_coef_start : float
        Initial entropy coefficient (high → exploratory).
    ppo_ent_coef_end : float
        Final entropy coefficient.
    ppo_ent_anneal_steps : int
        Training steps over which to anneal the entropy coefficient.

    Network architecture
    --------------------
    hidden_dim : int
        Hidden-layer width for actor, critic, and auxiliary MLP heads.
    use_shared_encoder : bool
        Whether actor-critic and V_h^e share a state encoder.

    Environment
    -----------
    steps_per_episode : int
        Maximum environment steps per episode.
    num_actions : int
        Number of actions available to a single robot.
    num_robots : int
        Number of robots in the environment.

    Optional flags
    --------------
    use_z_space_transform : bool
        Apply z-space value transforms to auxiliary targets.
    """

    # ── Theory parameters (duplicated from DQN config on purpose) ────────
    gamma_r: float = 0.99
    gamma_h: float = 0.99
    zeta: float = 2.0
    xi: float = 1.0
    eta: float = 1.1
    steps_per_episode: int = 50

    # ── PPO hyper-parameters ─────────────────────────────────────────────
    ppo_rollout_length: int = 128
    ppo_num_minibatches: int = 4
    ppo_update_epochs: int = 4
    ppo_clip_coef: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95
    lr_ppo: float = 3e-4
    num_envs: int = 16
    num_ppo_iterations: int = 10_000

    # ── Auxiliary-network training ───────────────────────────────────────
    lr_v_h_e: float = 1e-4
    lr_x_h: float = 1e-4
    lr_u_r: float = 1e-4
    aux_training_steps_per_iteration: int = 10
    aux_buffer_size: int = 50_000
    reward_freeze_interval: int = 1
    batch_size: int = 256

    # ── Warm-up schedule (measured in training steps / gradient updates) ─
    warmup_v_h_e_steps: int = 5_000
    warmup_x_h_steps: int = 7_500
    warmup_u_r_steps: int = 10_000

    # ── Entropy schedule ─────────────────────────────────────────────────
    ppo_ent_coef_start: float = 0.1
    ppo_ent_coef_end: float = 0.01
    ppo_ent_anneal_steps: int = 10_000

    # ── Network architecture ─────────────────────────────────────────────
    hidden_dim: int = 256
    use_shared_encoder: bool = True

    # ── Environment ──────────────────────────────────────────────────────
    num_actions: int = 7
    num_robots: int = 1

    # ── Optional flags ───────────────────────────────────────────────────
    use_z_space_transform: bool = False

    # ── Simplified goal-agnostic X_h computation ─────────────────────────
    # When use_simplified_x_h=True, X_h is computed via the goal-agnostic
    # recursion X_h(s) = 1 + gamma_h^zeta * sum_{s'} q_h(s,s')^zeta * X_h(s')
    # where q_h(s,s') = max_{a_h} P(s'|s, a_h, pi_{-h}).
    # This bypasses V_h^e entirely.  X_h >= 1 for all states (terminal X_h = 1).
    # Bounded rationality: when x_h_epsilon_h > 0, q_h mixes best human action
    # with a uniform prior:
    #   q_h = (1-eps)*max_{a_h} P(...) + eps*mean_{a_h} P(...)
    use_simplified_x_h: bool = False
    x_h_epsilon_h: float = 0.0

    # ── Auxiliary-network regularisation ──────────────────────────────────
    v_h_e_weight_decay: float = 1e-4
    x_h_weight_decay: float = 1e-4
    u_r_weight_decay: float = 1e-4
    v_h_e_grad_clip: Optional[float] = 10.0
    x_h_grad_clip: Optional[float] = 10.0
    u_r_grad_clip: Optional[float] = 10.0

    # ── Goal resampling ──────────────────────────────────────────────────
    goal_resample_prob: float = 0.01

    # ── PufferLib runtime settings ───────────────────────────────────────
    device: str = "cpu"
    seed: int = 1

    # ── Logging ───────────────────────────────────────────────────────────
    tensorboard_dir: Optional[str] = None
    log_interval: int = 1

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_interval: int = 0  # 0 = disabled; otherwise save every N PPO iters
    checkpoint_dir: Optional[str] = None

    # ── Reward scaling ─────────────────────────────────────────────────────
    # U_r is divided by ``u_r_scale`` before being passed to PufferLib as
    # the PPO reward.  This prevents PufferLib's hard ``clamp(r, -1, 1)``
    # from destroying the reward signal.  If ``None``, the env wrapper
    # falls back to a conservative theoretical upper-bound.  For better
    # signal, set this to the empirical ``max(|U_r|)`` observed during a
    # short calibration phase (see ``PPOPhase2Trainer.calibrate_reward_scale``).
    u_r_scale: Optional[float] = None

    # ── Performance flags ─────────────────────────────────────────────────
    # Transition-probability computation on every env step is expensive for
    # multi-robot (|A|^N) and currently unused by train_auxiliary_step().
    # Default to False; enable when model-based / policy-weighted targets
    # are needed.
    compute_transition_probs: bool = False

    def __post_init__(self) -> None:
        if self.zeta < 1.0:
            raise ValueError(f"zeta must be >= 1.0, got {self.zeta}")
        if self.xi < 1.0:
            raise ValueError(f"xi must be >= 1.0, got {self.xi}")
        if self.eta < 1.0:
            raise ValueError(f"eta must be >= 1.0, got {self.eta}")
        if self.gamma_r < 0.0 or self.gamma_r > 1.0:
            raise ValueError(f"gamma_r must be in [0, 1], got {self.gamma_r}")
        if self.gamma_h < 0.0 or self.gamma_h > 1.0:
            raise ValueError(f"gamma_h must be in [0, 1], got {self.gamma_h}")
        # Warm-up thresholds must be non-decreasing (cumulative).
        if not (
            self.warmup_v_h_e_steps <= self.warmup_x_h_steps <= self.warmup_u_r_steps
        ):
            raise ValueError(
                "Warm-up thresholds must be non-decreasing: "
                f"warmup_v_h_e_steps={self.warmup_v_h_e_steps} <= "
                f"warmup_x_h_steps={self.warmup_x_h_steps} <= "
                f"warmup_u_r_steps={self.warmup_u_r_steps}"
            )
        if self.u_r_scale is not None and self.u_r_scale <= 0:
            raise ValueError(
                f"u_r_scale must be > 0 (divides U_r rewards; zero/negative "
                f"would cause division-by-zero or sign flips), got {self.u_r_scale}"
            )
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {self.log_interval}")

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_total_warmup_steps(self) -> int:
        """Total training steps consumed by warm-up (stages 0-2).

        The warm-up stage boundaries are cumulative thresholds:
        - Stage 0 → 1: ``warmup_v_h_e_steps``
        - Stage 1 → 2: ``warmup_x_h_steps``
        - Stage 2 → end: ``warmup_u_r_steps`` (the final threshold)

        So ``warmup_u_r_steps`` is the cumulative total of all warm-up.
        """
        return self.warmup_u_r_steps

    def is_in_warmup(self, training_step: int) -> bool:
        """Return ``True`` while auxiliary-only warm-up is active."""
        return training_step < self.warmup_u_r_steps

    def get_warmup_stage(self, training_step: int) -> int:
        """Numeric warm-up stage at *training_step*.

        Stages::

            0: V_h^e only            (0 ≤ step < warmup_v_h_e_steps)
            1: V_h^e + X_h           (warmup_v_h_e_steps ≤ step < warmup_x_h_steps)
            2: V_h^e + X_h + U_r     (warmup_x_h_steps ≤ step < warmup_u_r_steps)
            3: Full PPO training     (step ≥ warmup_u_r_steps)
        """
        if training_step < self.warmup_v_h_e_steps:
            return 0
        if training_step < self.warmup_x_h_steps:
            return 1
        if training_step < self.warmup_u_r_steps:
            return 2
        return 3

    _WARMUP_STAGE_NAMES = {
        0: "V_h^e only",
        1: "V_h^e + X_h",
        2: "V_h^e + X_h + U_r",
        3: "full PPO",
    }

    def get_warmup_stage_name(self, training_step: int) -> str:
        """Human-readable name for the current warm-up stage."""
        return self._WARMUP_STAGE_NAMES[self.get_warmup_stage(training_step)]

    def get_active_aux_networks(self, training_step: int) -> Set[str]:
        """Set of auxiliary-network names to train at *training_step*.

        V_h^e is always active; X_h and U_r are added according to the
        staged warm-up schedule.
        """
        active: Set[str] = {"v_h_e"}
        if training_step >= self.warmup_v_h_e_steps:
            active.add("x_h")
        if training_step >= self.warmup_x_h_steps:
            active.add("u_r")
        return active

    def get_entropy_coef(self, training_step: int) -> float:
        """Linearly-annealed entropy coefficient.

        The trainer mutates ``pufferl.config['ent_coef']`` before each
        ``pufferl.train()`` call so the annealed value is used.
        """
        if training_step >= self.ppo_ent_anneal_steps:
            return self.ppo_ent_coef_end
        frac = training_step / max(1, self.ppo_ent_anneal_steps)
        return self.ppo_ent_coef_start + frac * (
            self.ppo_ent_coef_end - self.ppo_ent_coef_start
        )

    @property
    def num_joint_actions(self) -> int:
        """Total joint robot actions: |A|^N."""
        return self.num_actions**self.num_robots

    def to_pufferlib_config(self) -> dict:
        """Build the flat config dict expected by ``pufferlib.pufferl.PuffeRL``.

        PuffeRL reads a plain dict with string keys.  This method maps
        EMPO PPO config fields to PufferLib's expected keys.
        """
        batch_size = self.num_envs * self.ppo_rollout_length
        return {
            "batch_size": batch_size,
            "bptt_horizon": self.ppo_rollout_length,
            "minibatch_size": max(1, batch_size // max(1, self.ppo_num_minibatches)),
            "max_minibatch_size": max(
                1, batch_size // max(1, self.ppo_num_minibatches)
            ),
            "update_epochs": self.ppo_update_epochs,
            "gamma": self.gamma_r,
            "gae_lambda": self.ppo_gae_lambda,
            "clip_coef": self.ppo_clip_coef,
            "vf_coef": self.ppo_vf_coef,
            "vf_clip_coef": 10.0,
            "ent_coef": self.ppo_ent_coef_start,
            "max_grad_norm": self.ppo_max_grad_norm,
            "learning_rate": self.lr_ppo,
            "anneal_lr": False,
            "min_lr_ratio": 0.0,
            "device": self.device,
            "seed": self.seed,
            "torch_deterministic": False,
            "total_timesteps": self.num_ppo_iterations * batch_size,
            "compile": False,
            "compile_mode": "reduce-overhead",
            "use_rnn": False,
            "cpu_offload": False,
            "checkpoint_interval": 1_000_000,
            "data_dir": "experiments",
            "env": "empo_phase2",
            "precision": "float32",
            "optimizer": "adam",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-5,
            "prio_alpha": 0.0,
            "prio_beta0": 1.0,
            "vtrace_rho_clip": 1.0,
            "vtrace_c_clip": 1.0,
            "neptune": False,
            "wandb": False,
        }
