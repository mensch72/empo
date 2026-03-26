"""
Configuration for PPO-based Phase 1 training.

This is a **standalone** config class — it does NOT inherit from or modify
the existing Phase 1 DQN config.  Shared theory parameters (γ_h, β_h) are
intentionally duplicated to avoid coupling between the two code paths.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOPhase1Config:
    """
    Configuration for PPO-based Phase 1 goal-conditioned human policy prior
    approximation.

    Phase 1 is simpler than Phase 2:

    - No auxiliary networks (no V_h^e, X_h, U_r).
    - No warm-up stages.
    - Simple binary reward: ``goal.is_achieved(state)`` returns 0 or 1.
    - The policy is goal-conditioned: π_h(a|s,g).
    - The value function is goal-conditioned: V_h(s,g).

    This config is fully independent of the DQN-based Phase 1 config.
    Theory parameters that appear in both configs are duplicated on purpose
    so that the PPO path can be modified without risk to the DQN path.

    Theory parameters
    -----------------
    gamma_h : float
        Human discount factor.
    beta_h : float
        Boltzmann temperature for goal-conditioned policy (theory parameter,
        NOT a hyperparameter).

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
    lr : float
        Learning rate for the PPO actor-critic.
    num_envs : int
        Number of vectorised environments.
    num_ppo_iterations : int
        Total PPO update iterations (outer loop count).

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
        Hidden-layer width for actor and critic MLP heads.

    Environment
    -----------
    steps_per_episode : int
        Maximum environment steps per episode.
    num_actions : int
        Number of actions available to the human agent.
    num_humans : int
        Number of humans in the environment.

    Goal sampling
    -------------
    goal_resample_prob : float
        Probability of resampling the goal mid-episode (0 = fixed per
        episode).

    PufferLib runtime
    -----------------
    device : str
        Torch device for training (``"cpu"`` or ``"cuda"``).
    seed : int
        Random seed for reproducibility.

    Logging
    -------
    tensorboard_dir : Optional[str]
        Directory for TensorBoard logs (``None`` = disabled).
    log_interval : int
        Log metrics every *N* PPO iterations.

    Checkpointing
    -------------
    checkpoint_interval : int
        Save a checkpoint every *N* PPO iterations (0 = disabled).
    checkpoint_dir : Optional[str]
        Directory for checkpoint files (``None`` = disabled).

    Reward shaping
    --------------
    reward_shaping_coef : float
        Coefficient for distance-based reward shaping (0 = disabled, only
        binary ``goal.is_achieved``).
    """

    # ── Theory parameters (duplicated from DQN config on purpose) ────────
    gamma_h: float = 0.99
    beta_h: float = 1.0

    # ── PPO hyper-parameters ─────────────────────────────────────────────
    ppo_rollout_length: int = 128
    ppo_num_minibatches: int = 4
    ppo_update_epochs: int = 4
    ppo_clip_coef: float = 0.2
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_gae_lambda: float = 0.95
    lr: float = 3e-4
    num_envs: int = 16
    num_ppo_iterations: int = 10_000

    # ── Entropy schedule ─────────────────────────────────────────────────
    ppo_ent_coef_start: float = 0.1
    ppo_ent_coef_end: float = 0.01
    ppo_ent_anneal_steps: int = 10_000

    # ── Network architecture ─────────────────────────────────────────────
    hidden_dim: int = 256

    # ── Environment ──────────────────────────────────────────────────────
    steps_per_episode: int = 50
    num_actions: int = 7
    num_humans: int = 1

    # ── Goal sampling ────────────────────────────────────────────────────
    goal_resample_prob: float = 0.0

    # ── PufferLib runtime settings ───────────────────────────────────────
    device: str = "cpu"
    seed: int = 1

    # ── Logging ───────────────────────────────────────────────────────────
    tensorboard_dir: Optional[str] = None
    log_interval: int = 1

    # ── Checkpointing ────────────────────────────────────────────────────
    checkpoint_interval: int = 0  # 0 = disabled; otherwise save every N PPO iters
    checkpoint_dir: Optional[str] = None

    # ── Reward shaping ───────────────────────────────────────────────────
    reward_shaping_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.gamma_h < 0.0 or self.gamma_h > 1.0:
            raise ValueError(f"gamma_h must be in [0, 1], got {self.gamma_h}")
        if self.beta_h <= 0.0:
            raise ValueError(f"beta_h must be > 0, got {self.beta_h}")
        if self.log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {self.log_interval}")

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_entropy_coef(self, training_step: int) -> float:
        """Linearly-annealed entropy coefficient.

        .. note::

           PufferLib's ``PuffeRL`` does not support per-iteration entropy
           coefficient updates.  ``to_pufferlib_config()`` sets a fixed
           ``ent_coef`` equal to ``ppo_ent_coef_start``.  This method is
           provided for callers that implement custom training loops
           outside PufferLib, or for future PufferLib versions that expose
           a mutable ``ent_coef`` field.
        """
        if training_step >= self.ppo_ent_anneal_steps:
            return self.ppo_ent_coef_end
        frac = training_step / max(1, self.ppo_ent_anneal_steps)
        return self.ppo_ent_coef_start + frac * (
            self.ppo_ent_coef_end - self.ppo_ent_coef_start
        )

    def to_pufferlib_config(self) -> dict:
        """Build the flat config dict expected by ``pufferlib.pufferl.PuffeRL``.

        PuffeRL reads a plain dict with string keys.  This method maps
        EMPO PPO config fields to PufferLib's expected keys.
        """
        batch_size = self.num_envs * self.ppo_rollout_length
        minibatch_size = max(1, batch_size // max(1, self.ppo_num_minibatches))
        return {
            "batch_size": batch_size,
            "bptt_horizon": self.ppo_rollout_length,
            "minibatch_size": minibatch_size,
            "max_minibatch_size": minibatch_size,
            "update_epochs": self.ppo_update_epochs,
            "gamma": self.gamma_h,
            "gae_lambda": self.ppo_gae_lambda,
            "clip_coef": self.ppo_clip_coef,
            "vf_coef": self.ppo_vf_coef,
            "vf_clip_coef": 10.0,
            "ent_coef": self.ppo_ent_coef_start,
            "max_grad_norm": self.ppo_max_grad_norm,
            "learning_rate": self.lr,
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
            "env": "empo_phase1",
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
