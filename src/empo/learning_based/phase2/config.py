"""
Configuration for Phase 2 robot policy learning.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Set
import math
import os


@dataclass
class Phase2Config:
    """
    Configuration for Phase 2 training.
    
    Phase 2 learns the robot policy to softly maximize aggregate human power
    as defined in equations (4)-(9) of the EMPO paper.
    
    Warm-up Phase:
        Training proceeds in stages to break mutual network dependencies:
        1. V_h^e only (training steps 0 to warmup_v_h_e_steps)
        2. V_h^e + X_h (training steps warmup_v_h_e_steps to warmup_x_h_steps)
        3. V_h^e + X_h + U_r (training steps warmup_x_h_steps to warmup_u_r_steps)
        4. V_h^e + X_h + U_r + Q_r (training steps warmup_u_r_steps to warmup_q_r_steps)
        5. All networks including V_r if enabled (after warmup_q_r_steps)
        
        During warm-up, beta_r=0 (uniform random robot policy).
        After warm-up, beta_r ramps up to nominal value over beta_r_rampup_steps.
        Learning rates are constant during warm-up, then decay as 1/sqrt(t).
        
        Note: Warmup stages are measured in training steps (gradient updates), not environment steps.
    
    Attributes:
        gamma_r: Robot discount factor.
        gamma_h: Human discount factor (for V_h^e computation).
        zeta: Risk/reliability preference parameter (ζ >= 1, 1 = neutral).
        xi: Inter-human inequality aversion (ξ >= 1 to protect last bit of power).
        eta: Additional intertemporal inequality aversion (η >= 1, 1 = neutral).
        beta_r: Robot power-law policy exponent (< ∞ to prevent overoptimization risks).
        epsilon_r_start: Initial robot exploration rate.
        epsilon_r_end: Final robot exploration rate.
        epsilon_r_decay_steps: Steps over which robot epsilon decays.
        epsilon_h_start: Initial human exploration rate.
        epsilon_h_end: Final human exploration rate.
        epsilon_h_decay_steps: Steps over which human epsilon decays.
        lr_q_r: Learning rate for Q_r network.
        lr_v_r: Learning rate for V_r network.
        lr_v_h_e: Learning rate for V_h^e network.
        lr_x_h: Learning rate for X_h network.
        lr_u_r: Learning rate for U_r network.
        q_r_target_update_interval: Training steps between Q_r target network updates.
        v_r_target_update_interval: Training steps between V_r target network updates.
        v_h_e_target_update_interval: Training steps between V_h^e target network updates.
        x_h_target_update_interval: Training steps between X_h target network updates.
        u_r_target_update_interval: Training steps between U_r target network updates.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        num_training_steps: Total training steps (gradient updates) - fundamental time unit.
        steps_per_episode: Environment steps per episode (affects data collection frequency only).
        training_steps_per_env_step: Training steps (gradient updates) per environment step (can be fractional).
        goal_resample_prob: Probability of resampling goals each step.
        hidden_dim: Hidden layer dimension for networks.
        state_feature_dim: State encoder output dimension.
        goal_feature_dim: Goal encoder output dimension.
        agent_embedding_dim: Agent identity encoder: index embedding dimension.
        agent_position_feature_dim: Agent identity encoder: position encoding output dimension.
        agent_feature_dim: Agent identity encoder: agent feature encoding output dimension.
    """
    
    # Discount factors
    gamma_r: float = 0.99
    gamma_h: float = 0.99
    
    # Power metric parameters (from paper)
    zeta: float = 2.0    # ζ - risk/reliability preference (>=1, 1 = neutral)
    xi: float = 1.0      # ξ - inter-human inequality aversion (>=1)
    eta: float = 1.1     # η - intertemporal inequality aversion (>=1, 1 = neutral)
    
    # Robot policy
    beta_r: float = 10.0  # Power-law policy exponent (nominal value after warm-up)
    
    # Robot exploration (in addition to power-law policy randomization)
    epsilon_r_start: float = 1.0
    epsilon_r_end: float = 0.01
    epsilon_r_decay_steps: int = 10000
    
    # Human exploration (analogous to robot exploration)
    epsilon_h_start: float = 1.0
    epsilon_h_end: float = 0.01
    epsilon_h_decay_steps: int = 10000
    
    # =========================================================================
    # Curiosity-driven exploration (RND - Random Network Distillation)
    # =========================================================================
    # RND provides intrinsic motivation to explore novel states by using
    # prediction error as a novelty signal. A trainable predictor network
    # tries to match a fixed random target network. High prediction error
    # indicates novel states, providing an exploration bonus.
    #
    # When enabled, curiosity affects action selection:
    # - During epsilon exploration: actions weighted by expected novelty
    # - During policy selection: multiplicative scaling of Q-values:
    #     Q_effective = Q * exp(-bonus_coef * novelty)
    #   This preserves Q < 0 required by the power-law policy.
    use_rnd: bool = False                    # Enable RND curiosity exploration
    rnd_feature_dim: int = 64                # Output dimension of RND networks
    rnd_hidden_dim: int = 256                # Hidden layer dimension for RND networks
    rnd_bonus_coef_r: float = 0.1            # Robot curiosity bonus coefficient
    rnd_bonus_coef_h: float = 0.1            # Human curiosity bonus coefficient (for human action RND)
    lr_rnd: float = 1e-4                     # Learning rate for RND predictor
    rnd_weight_decay: float = 1e-4           # Weight decay for RND predictor
    rnd_grad_clip: Optional[float] = 10.0    # Gradient clipping for RND
    normalize_rnd: bool = True               # Normalize novelty by running mean/std
    rnd_normalization_decay: float = 0.99    # EMA decay for normalization stats
    
    # Human Action RND: Separate RND module for human action exploration.
    # Unlike the state-based RND for robots, this outputs per-action novelty
    # scores for (state, human_identity, action) tuples, encouraging each human
    # to try actions they haven't taken in similar states.
    use_human_action_rnd: bool = False       # Enable human action RND (requires use_rnd=True)
    
    # =========================================================================
    # Curiosity-driven exploration (Count-based - for tabular/lookup table mode)
    # =========================================================================
    # Count-based curiosity maintains state visit counts and provides exploration
    # bonuses based on inverse visit frequency. This is simpler than RND and
    # works well with lookup table networks where states are already hashable.
    #
    # Bonus formulas:
    # - Simple (use_ucb_bonus=False): bonus = scale / sqrt(visits + 1)
    # - UCB-style (use_ucb_bonus=True): bonus = scale * sqrt(log(total) / (visits + 1))
    #
    # When enabled, curiosity affects action selection similarly to RND:
    # - During epsilon exploration: actions weighted by expected novelty
    # - During policy selection: multiplicative scaling of Q-values
    #
    # Note: Only one of use_rnd or use_count_based_curiosity should be enabled.
    # Count-based curiosity is recommended for lookup table mode (use_lookup_tables=True).
    # RND is recommended for neural network mode.
    use_count_based_curiosity: bool = False  # Enable count-based curiosity exploration
    count_curiosity_scale: float = 1.0       # Scale factor for curiosity bonus
    count_curiosity_use_ucb: bool = False    # Use UCB-style bonus vs simple 1/sqrt(n)
    count_curiosity_bonus_coef_r: float = 0.1  # Robot curiosity bonus coefficient
    count_curiosity_bonus_coef_h: float = 0.1  # Human curiosity bonus coefficient
    
    # Learning rates (base rates, may be modified by schedule)
    lr_q_r: float = 1e-4
    lr_v_r: float = 1e-4
    lr_v_h_e: float = 1e-3
    lr_x_h: float = 1e-4
    lr_u_r: float = 1e-4
    
    # =========================================================================
    # Warm-up phase configuration
    # =========================================================================
    # Warm-up proceeds in stages. Each parameter specifies DURATION of that stage:
    # Stage 1: Only V_h^e (warmup_v_h_e_steps duration)
    # Stage 2: V_h^e + X_h (warmup_x_h_steps duration) - SKIPPED if x_h_use_network=False
    # Stage 3: V_h^e + X_h + U_r (warmup_u_r_steps duration) - SKIPPED if u_r_use_network=False
    # Stage 4: V_h^e + X_h + (U_r) + Q_r (warmup_q_r_steps duration)
    # Stage 5: V_h^e + X_h + (U_r) + Q_r + V_r (warmup_v_r_steps duration) - SKIPPED if v_r_use_network=False
    # Stage 6: All networks with beta_r ramp-up (beta_r_rampup_steps duration)
    # Stage 7: Full training with LR decay (remainder)
    # 
    # NOTE: warmup_x_h_steps is set to 0 if x_h_use_network=False (in __post_init__)
    # NOTE: warmup_u_r_steps is set to 0 if u_r_use_network=False (in __post_init__)
    # NOTE: warmup_v_r_steps is set to 0 if v_r_use_network=False (in __post_init__)
    warmup_v_h_e_steps: int = 1e4   # Duration of V_h^e-only stage
    warmup_x_h_steps: int = 1e4     # Duration of V_h^e + X_h stage (0 if x_h_use_network=False)
    warmup_u_r_steps: int = 5e3     # Duration of V_h^e + X_h + U_r stage (0 if u_r_use_network=False)
    warmup_q_r_steps: int = 1e4     # Duration of V_h^e + X_h + (U_r) + Q_r stage
    warmup_v_r_steps: int = 5e3     # Duration of + V_r stage (0 if v_r_use_network=False)
    
    # Beta_r schedule: ramps from 0 to beta_r over this many steps after warm-up ends
    beta_r_rampup_steps: int = 2e4
    
    # Learning rate schedule after warm-up
    # After warm-up, use 1/sqrt(t) decay: lr(t) = lr_base * sqrt(warmup) / sqrt(t)
    # This is a compromise between 1/t (for expectations) and constant (for Q-learning)
    use_sqrt_lr_decay: bool = True
    
    # Fraction of total training steps to keep LR constant before starting decay.
    # 0.0 = decay immediately after warm-up (original behavior)
    # 0.7 = constant LR until 70% of total steps, then decay
    # This allows the network to learn the value function structure before fine-tuning.
    # Recommended: 0.7 for most cases.
    lr_constant_fraction: float = 0.7
    
    # If True, use 1/t decay (instead of 1/sqrt(t)) after the constant phase.
    # 1/t is theoretically correct for converging to expected values (Robbins-Monro).
    # Only applies after lr_constant_fraction of training is complete.
    constant_lr_then_1_over_t: bool = True
    
    # =========================================================================
    # Z-space transformation for Q_r, V_r, U_r networks
    # =========================================================================
    # When enabled, networks predict z = f(Q) = (-Q)^{-1/(ηξ)} ∈ (0, 1]
    # instead of Q directly. This makes it easier for networks to represent
    # values across orders of magnitude (e.g., Q from -1 to -1000).
    #
    # Loss function depends on training phase:
    # - During constant LR phase: MSE in z-space (balanced gradients)
    # - During 1/t decay phase: MSE in Q-space (Robbins-Monro convergence)
    #
    # See docs/plans/learning_qr_scale.md for full rationale.
    use_z_space_transform: bool = False  # Enable z-space transformation
    
    # Loss function mode when z-space transform is enabled:
    # - True: Use z-based loss in constant LR phase, Q-based loss in decay phase (legacy)
    # - False (default): Use Q-based loss throughout (recommended for faster outlier correction)
    # Only has effect when use_z_space_transform=True.
    # See docs/VALUE_TRANSFORMATIONS.md for rationale.
    use_z_based_loss: bool = False
    
    # Target network updates
    q_r_target_update_interval: int = 100
    v_r_target_update_interval: int = 100
    v_h_e_target_update_interval: int = 100
    x_h_target_update_interval: int = 100
    u_r_target_update_interval: int = 100
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    x_h_batch_size: Optional[int] = None  # Larger batch for X_h (None = use batch_size)
    
    # Training
    num_training_steps: int = 1e5  # Total training steps (gradient updates)
    steps_per_episode: int = 50
    
    # Env-to-training step ratio (for sync mode)
    # Can specify either:
    #   - training_steps_per_env_step: gradient updates per env step (default 1.0)
    #   - env_steps_per_training_step: env steps between gradient updates (alternative)
    # If both specified, training_steps_per_env_step takes precedence.
    # Examples:
    #   training_steps_per_env_step=4.0 → 4 gradient updates per env step
    #   training_steps_per_env_step=0.1 → train every 10 env steps
    #   env_steps_per_training_step=10 → same as training_steps_per_env_step=0.1
    training_steps_per_env_step: float = 1.0
    env_steps_per_training_step: Optional[float] = None  # Alternative way to specify ratio
    
    # Goal resampling
    goal_resample_prob: float = 0.01
    
    # U_r loss computation: number of humans to sample (None = all humans)
    u_r_sample_humans: Optional[int] = None
    
    # X_h loss computation: number of human-goal pairs to sample (None = all from transition)
    x_h_sample_humans: Optional[int] = None
    
    # Regularization options for all networks (weight decay, gradient clipping, dropout)
    # These help with training stability, especially for high-variance learning.
    # Weight decay (L2 regularization) for each network's optimizer
    q_r_weight_decay: float = 1e-4
    v_r_weight_decay: float = 1e-4
    v_h_e_weight_decay: float = 1e-4
    x_h_weight_decay: float = 1e-4
    u_r_weight_decay: float = 1e-4
    
    # Max gradient norm for each network (0 or None to disable clipping)
    # If auto_grad_clip is True, these values are scaled by the learning rate.
    q_r_grad_clip: Optional[float] = 10.0
    v_r_grad_clip: Optional[float] = 10.0
    v_h_e_grad_clip: Optional[float] = 10.0
    x_h_grad_clip: Optional[float] = 10.0
    u_r_grad_clip: Optional[float] = 10.0
    
    # Automatic gradient clipping: if True, the effective grad clip is scaled by
    # learning rate to keep step sizes bounded regardless of LR magnitude.
    # Effective clip = grad_clip * lr / auto_grad_clip_reference_lr
    # This means at reference_lr, you get exactly grad_clip; at higher LR you get
    # proportionally larger clips (to allow similar step sizes).
    auto_grad_clip: bool = True
    auto_grad_clip_reference_lr: float = 1e-4  # Reference LR for scaling
    
    # Dropout rate for hidden layers (not input/output) of each network
    q_r_dropout: float = 0.5
    v_r_dropout: float = 0.5
    v_h_e_dropout: float = 0.5
    x_h_dropout: float = 0.5
    u_r_dropout: float = 0.5
    
    # V_r computation mode: if False (default), compute V_r directly from U_r and Q_r
    # instead of using a separate network. This reduces complexity since V_r = U_r + π_r · Q_r.
    v_r_use_network: bool = False
    
    # U_r computation mode: if False (default), compute U_r directly from X_h values
    # instead of using a separate network. This reduces complexity and one source of
    # error since U_r = -(E_h[X_h^{-ξ}])^η can be computed exactly from X_h.
    u_r_use_network: bool = False
    
    # X_h computation mode: if False, compute X_h directly from V_h^e samples
    # instead of using a separate network. This reduces complexity and one source of
    # error since X_h = E_g[V_h^e(s, g)^ζ] can be computed exactly from V_h^e samples.
    # When False, the goal sampler is used at each step to sample goals and compute X_h.
    x_h_use_network: bool = True
    
    # Whether to include step count (remaining time) in state encoding.
    # Set to False to verify that identical grid states get identical values.
    include_step_count: bool = True
    
    # Profiling: if True, collect timing statistics for batched computation stages.
    # This adds minimal overhead but provides detailed breakdown of where time is spent.
    # Results are printed periodically and at the end of training.
    profile_batching: bool = False
    profile_batching_interval: int = 100  # Print stats every N training steps
    
    # Debugging: if False, encoder networks' forward functions become identity functions
    # (flatten+pad/truncate to match output dimension). Tensorizers remain unchanged.
    # This is useful for debugging to isolate whether problems come from encoders.
    use_encoders: bool = True
    
    # =========================================================================
    # Lookup table networks (tabular representations)
    # =========================================================================
    # If True, use lookup table (dictionary) implementations for some/all networks
    # instead of neural networks. Useful for small state spaces, debugging,
    # and interpretability. Each network can be selectively enabled.
    # 
    # Benefits: No function approximation error, guaranteed convergence, interpretability.
    # Drawbacks: Memory scales with state space size, no generalization to unseen states.
    #
    # When use_lookup_tables=True, the following flags control which networks use tables:
    # - use_lookup_q_r: Q_r(s, a_r) - robot Q-values
    # - use_lookup_v_r: V_r(s) - robot value function (only if v_r_use_network=True)
    # - use_lookup_v_h_e: V_h^e(s, g_h) - human goal achievement probability
    # - use_lookup_x_h: X_h(s) - aggregate human goal ability (only if x_h_use_network=True)
    # - use_lookup_u_r: U_r(s) - intrinsic reward (only if u_r_use_network=True)
    use_lookup_tables: bool = False
    use_lookup_q_r: bool = True      # Use lookup table for Q_r
    use_lookup_v_r: bool = True      # Use lookup table for V_r (if v_r_use_network=True)
    use_lookup_v_h_e: bool = True    # Use lookup table for V_h^e
    use_lookup_x_h: bool = True      # Use lookup table for X_h (if x_h_use_network=True)
    use_lookup_u_r: bool = True      # Use lookup table for U_r (if u_r_use_network=True)
    
    # Default values for lookup table entries (used when state is first seen)
    lookup_default_q_r: float = -1.0     # Q_r < 0 (negative value)
    lookup_default_v_r: float = -1.0     # V_r < 0 (negative value)
    lookup_default_v_h_e: float = 0.0    # V_h^e ∈ [0, 1] (probability) - 0 = pessimistic default
    lookup_default_x_h: float = 1e-10    # X_h ∈ (0, 1] (aggregate ability)
    lookup_default_y: float = 2.0        # y >= 1 (intermediate for U_r)
    
    # =========================================================================
    # Adaptive per-entry learning rate for lookup tables
    # =========================================================================
    # When enabled, each lookup table entry uses a learning rate of 1/n where n
    # is the number of updates to that entry. This makes each entry converge to
    # the exact arithmetic mean of all target values it has seen.
    #
    # Implementation: Before optimizer.step(), gradients are scaled by 1/update_count
    # for each entry. The base learning rate should be set to 1.0 for this mode.
    #
    # This generalizes to neural networks via uncertainty-weighted learning rates:
    # instead of 1/update_count, use an uncertainty estimate (e.g., ensemble variance).
    #
    # Robbins-Monro condition: For convergence to true expectation, we need
    # sum(lr) = ∞ and sum(lr²) < ∞. The 1/n schedule satisfies this exactly.
    lookup_use_adaptive_lr: bool = False  # Enable per-entry adaptive learning rate
    lookup_adaptive_lr_min: float = 1e-6  # Minimum learning rate (prevents 1/∞ = 0)
    
    # =========================================================================
    # RND-based adaptive learning rate (neural network mode)
    # =========================================================================
    # When enabled in neural network mode with RND, gradients are scaled by the
    # RND prediction error (MSE) as a proxy for uncertainty. States with high
    # RND error (novel/uncertain) get larger effective learning rates.
    #
    # This is a SPECULATIVE approach (not published in literature) that assumes
    # RND novelty correlates with value estimate uncertainty. The RND MSE is
    # already a squared quantity, analogous to variance, so lr ∝ rnd_error.
    #
    # When both use_lookup_tables and rnd_use_adaptive_lr are set:
    # - Lookup tables use 1/n update counts
    # - Neural networks use RND-based scaling  
    # If use_rnd=False but rnd_use_adaptive_lr=True, a warning is issued.
    #
    # See docs/ADAPTIVE_LEARNING.md for theoretical background.
    rnd_use_adaptive_lr: bool = False  # Enable RND-based adaptive learning rate
    rnd_adaptive_lr_scale: float = 1.0  # Multiplier for RND-based LR scaling
    rnd_adaptive_lr_min: float = 0.1  # Minimum LR multiplier (prevents vanishing updates)
    rnd_adaptive_lr_max: float = 10.0  # Maximum LR multiplier (prevents exploding updates)
    
    def __post_init__(self):
        """Compute cumulative warmup thresholds and apply network flags."""
        # Override X_h warmup duration to 0 if not using X_h network
        if not self.x_h_use_network:
            self.warmup_x_h_steps = 0
        
        # Override U_r warmup duration to 0 if not using U_r network
        if not self.u_r_use_network:
            self.warmup_u_r_steps = 0
        
        # Override V_r warmup duration to 0 if not using V_r network
        if not self.v_r_use_network:
            self.warmup_v_r_steps = 0
        
        # Compute cumulative thresholds from per-stage durations
        # These are used internally for all step-based comparisons
        self._warmup_v_h_e_end = self.warmup_v_h_e_steps
        self._warmup_x_h_end = self._warmup_v_h_e_end + self.warmup_x_h_steps
        self._warmup_u_r_end = self._warmup_x_h_end + self.warmup_u_r_steps
        self._warmup_q_r_end = self._warmup_u_r_end + self.warmup_q_r_steps
        self._warmup_v_r_end = self._warmup_q_r_end + self.warmup_v_r_steps
        
        # Validate lookup table settings
        if self.use_lookup_tables:
            # Warn if using lookup tables with networks that aren't enabled
            if self.use_lookup_v_r and not self.v_r_use_network:
                warnings.warn(
                    "use_lookup_v_r=True but v_r_use_network=False. "
                    "V_r lookup table will not be used since V_r is computed directly from U_r and Q_r.",
                    UserWarning,
                    stacklevel=2
                )
            if self.use_lookup_u_r and not self.u_r_use_network:
                warnings.warn(
                    "use_lookup_u_r=True but u_r_use_network=False. "
                    "U_r lookup table will not be used since U_r is computed directly from X_h.",
                    UserWarning,
                    stacklevel=2
                )
            if self.use_lookup_x_h and not self.x_h_use_network:
                warnings.warn(
                    "use_lookup_x_h=True but x_h_use_network=False. "
                    "X_h lookup table will not be used since X_h is computed directly from V_h^e samples.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Validate curiosity settings
        if self.use_rnd and self.use_count_based_curiosity:
            warnings.warn(
                "Both use_rnd and use_count_based_curiosity are enabled. "
                "Only one curiosity method should be used at a time. "
                "For lookup table mode, prefer count-based curiosity. "
                "For neural network mode, prefer RND.",
                UserWarning,
                stacklevel=2
            )
        
        # Recommend count-based curiosity for lookup tables
        if self.use_lookup_tables and self.use_rnd and not self.use_count_based_curiosity:
            warnings.warn(
                "RND is enabled with lookup tables. Consider using count-based curiosity "
                "(use_count_based_curiosity=True) instead, which is simpler and more "
                "appropriate for tabular settings.",
                UserWarning,
                stacklevel=2
            )
        
        # Validate z-based loss settings
        if self.use_z_based_loss and not self.use_z_space_transform:
            warnings.warn(
                "use_z_based_loss=True but use_z_space_transform=False. "
                "Z-based loss requires z-space transformation to be enabled. "
                "Setting use_z_based_loss=False.",
                UserWarning,
                stacklevel=2
            )
            self.use_z_based_loss = False
        
        # Validate RND-based adaptive learning rate settings
        if self.rnd_use_adaptive_lr and not self.use_rnd:
            warnings.warn(
                "rnd_use_adaptive_lr=True but use_rnd=False. "
                "RND-based adaptive learning rate requires use_rnd=True. "
                "The adaptive learning rate will not be applied.",
                UserWarning,
                stacklevel=2
            )
        
        if self.rnd_use_adaptive_lr and self.use_lookup_tables:
            # Both can coexist: lookup tables use 1/n, neural networks use RND
            pass  # No warning needed, they handle different network types
        
        # Handle env_steps_per_training_step as an alternative way to specify the ratio
        # If env_steps_per_training_step is set, convert to training_steps_per_env_step
        if self.env_steps_per_training_step is not None:
            # Only override if user explicitly set env_steps_per_training_step
            # and training_steps_per_env_step is at default value
            if self.training_steps_per_env_step == 1.0:
                self.training_steps_per_env_step = 1.0 / self.env_steps_per_training_step
    
    # Model-based targets: if True (default), use transition_probabilities() to compute
    # expected V(s') over all possible successor states instead of using single samples.
    # This is analogous to Expected SARSA vs SARSA.
    # Benefits: (1) Lower variance, (2) Actions with same successor get same Q-value,
    # (3) Updates ALL action Q-values per state, not just the taken action.
    # Requires the environment to provide a transition_probabilities() method.
    use_model_based_targets: bool = True
    
    # =========================================================================
    # Async training configuration (actor-learner architecture)
    # =========================================================================
    # When enabled, data collection runs in separate processes while the learner
    # trains on GPU. This allows CPU-bound environment stepping and transition
    # probability computation to overlap with GPU-bound training.
    async_training: bool = False
    
    # Number of parallel actor processes for data collection.
    # Each actor has its own environment copy and runs with a frozen policy.
    # More actors = more diverse data, but often 1 actor is fast enough.
    num_actors: int = 1
    
    # Steps between syncing frozen policy from learner to actors.
    # Lower = more on-policy but more sync overhead.
    # Higher = more off-policy but more efficient.
    actor_sync_freq: int = 100
    
    # Steps between syncing RND networks from learner to actors.
    # RND novelty changes rapidly as the predictor learns, so we need
    # more frequent sync than policy to keep exploration accurate.
    # Set to 0 to use actor_sync_freq for RND as well.
    rnd_sync_freq: int = 10
    
    # Minimum transitions in buffer before training starts (for async mode).
    # Ensures actors have collected enough initial data.
    async_min_buffer_size: int = 1000
    
    # Maximum ratio of env steps to training steps before actors pause.
    # Prevents actors from getting too far ahead of the learner.
    # Set to None to disable throttling.
    max_env_steps_per_training_step: Optional[float] = 10.0
    
    # Queue size for actor-to-learner transition queue.
    # Should be large enough to buffer actor output during learner training.
    async_queue_size: int = 10000
    
    # =========================================================================
    # Checkpoint and Memory Monitoring
    # =========================================================================
    # Checkpoint configuration for saving training state periodically.
    # Memory monitoring to prevent OOM crashes that could lose training progress.
    
    # Save checkpoint every N training steps (0 to disable periodic checkpoints).
    # Checkpoints are always saved on Ctrl-C interrupt or memory limit breach.
    # Recommended: Set to 10000-50000 for long runs to avoid losing progress.
    checkpoint_interval: int = 10000
    
    # Minimum free memory as a fraction of total system memory (0.0-1.0).
    # When free memory falls below this threshold:
    # 1. Training pauses for memory_pause_duration seconds
    # 2. If still low, training stops gracefully (like Ctrl-C) and saves checkpoint.
    # Set to 0.0 to disable memory monitoring.
    # Recommended: 0.1 (10%) to leave headroom for other processes.
    min_free_memory_fraction: float = 0.1
    
    # How often to check memory usage (in training steps).
    # Lower values catch memory issues faster but add overhead.
    # Recommended: 100-1000 depending on training speed.
    memory_check_interval: int = 100
    
    # How long to pause (seconds) when memory is low before checking again.
    # If memory is still low after this pause, training stops.
    # Recommended: 60 seconds to give other processes time to release memory.
    memory_pause_duration: float = 60.0
    
    # Network architecture
    hidden_dim: int = 256
    state_feature_dim: int = 256
    goal_feature_dim: int = 64              # Goal encoder output dimension
    agent_embedding_dim: int = 16           # Agent identity encoder: index embedding dimension
    agent_position_feature_dim: int = 32    # Agent identity encoder: position encoding output dimension
    agent_feature_dim: int = 32             # Agent identity encoder: agent feature encoding output dimension
    
    def get_epsilon_r(self, step: int) -> float:
        """Get robot epsilon value for given training step."""
        if step >= self.epsilon_r_decay_steps:
            return self.epsilon_r_end
        
        # Linear decay
        decay_rate = (self.epsilon_r_start - self.epsilon_r_end) / self.epsilon_r_decay_steps
        return self.epsilon_r_start - decay_rate * step
    
    def get_epsilon_h(self, step: int) -> float:
        """Get human epsilon value for given training step."""
        if step >= self.epsilon_h_decay_steps:
            return self.epsilon_h_end
        
        # Linear decay
        decay_rate = (self.epsilon_h_start - self.epsilon_h_end) / self.epsilon_h_decay_steps
        return self.epsilon_h_start - decay_rate * step
    
    def get_effective_grad_clip(self, network_name: str, current_lr: float) -> Optional[float]:
        """
        Get effective gradient clipping value, optionally scaled by learning rate.
        
        When auto_grad_clip is True, the clip value is scaled proportionally to
        the learning rate, so that step_size = lr * grad stays bounded even when
        gradients are large but LR is small (and vice versa).
        
        Formula: effective_clip = base_clip * (current_lr / reference_lr)
        
        This ensures gradient clipping doesn't become overly restrictive at small
        learning rates (which would cause very slow training) or too permissive at
        large learning rates (which would cause instability).
        
        Args:
            network_name: One of 'v_h_e', 'x_h', 'u_r', 'q_r', 'v_r'
            current_lr: Current learning rate for the network.
            
        Returns:
            Effective gradient clip value, or None if clipping is disabled.
        """
        base_clips = {
            'v_h_e': self.v_h_e_grad_clip,
            'x_h': self.x_h_grad_clip,
            'u_r': self.u_r_grad_clip,
            'q_r': self.q_r_grad_clip,
            'v_r': self.v_r_grad_clip,
        }
        
        base_clip = base_clips.get(network_name)
        if base_clip is None or base_clip <= 0:
            return None
        
        if not self.auto_grad_clip:
            return base_clip
        
        # Scale by learning rate ratio
        lr_ratio = current_lr / self.auto_grad_clip_reference_lr
        return base_clip * lr_ratio
    
    # =========================================================================
    # Warm-up phase methods
    # =========================================================================
    
    def get_total_warmup_steps(self) -> int:
        """Get total number of warm-up steps (including beta_r ramp-up)."""
        return self._warmup_v_r_end + self.beta_r_rampup_steps
    
    def is_in_decay_phase(self, step: int) -> bool:
        """
        Check if we're in the late decay phase (1/t or 1/sqrt(t) decay).
        
        This is the phase where we want to use Q-space loss for Robbins-Monro
        convergence to true expected values.
        
        Args:
            step: Current training step.
            
        Returns:
            True if we're past lr_constant_fraction of training and 
            constant_lr_then_1_over_t is enabled.
        """
        full_warmup_end = self._warmup_v_r_end + self.beta_r_rampup_steps
        decay_start_step = max(
            full_warmup_end,
            int(self.lr_constant_fraction * self.num_training_steps)
        )
        return step >= decay_start_step and self.constant_lr_then_1_over_t
    
    def should_use_z_loss(self, step: int) -> bool:
        """
        Check if z-space loss should be used at the given step.
        
        Z-space loss is only used when:
        1. use_z_space_transform=True (z-space predictions enabled)
        2. use_z_based_loss=True (legacy mode for z-based loss in constant LR phase)
        3. Not in decay phase (switch to Q-space loss for Robbins-Monro)
        
        When use_z_based_loss=False (default), Q-space loss is used throughout,
        which corrects large outliers faster while still benefiting from the
        bounded network output range that z-space predictions provide.
        
        Args:
            step: Current training step.
            
        Returns:
            True if z-space MSE loss should be used, False for Q-space MSE loss.
        """
        if not self.use_z_space_transform:
            return False
        if not self.use_z_based_loss:
            return False
        # In legacy mode, use z-loss only during constant LR phase
        return not self.is_in_decay_phase(step)
    
    def is_in_warmup(self, step: int) -> bool:
        """Check if we're still in the warm-up phase (before all networks active)."""
        return step < self._warmup_v_r_end
    
    def is_in_rampup(self, step: int) -> bool:
        """Check if we're in the beta_r ramp-up phase."""
        return self._warmup_v_r_end <= step < self._warmup_v_r_end + self.beta_r_rampup_steps
    
    def is_fully_trained(self, step: int) -> bool:
        """Check if we're past all warmup/rampup phases (LR decay starts here)."""
        return step >= self._warmup_v_r_end + self.beta_r_rampup_steps
    
    def get_active_networks(self, step: int) -> Set[str]:
        """
        Get the set of networks that should be trained at the given step.
        
        Args:
            step: Current training step.
            
        Returns:
            Set of network names to train: subset of {'v_h_e', 'x_h', 'u_r', 'q_r', 'v_r', 'rnd'}
        """
        active = set()
        
        # V_h^e is always active (it's the foundation)
        active.add('v_h_e')
        
        # RND is always active when enabled (aids exploration from the start)
        if self.use_rnd:
            active.add('rnd')
        
        # X_h starts after V_h^e warmup (only if using network mode)
        if step >= self._warmup_v_h_e_end and self.x_h_use_network:
            active.add('x_h')
        
        # U_r starts after X_h warmup (only if using network mode)
        if step >= self._warmup_x_h_end and self.u_r_use_network:
            active.add('u_r')
        
        # Q_r starts after U_r warmup (U_r duration is 0 if not using U_r network)
        if step >= self._warmup_u_r_end:
            active.add('q_r')
        
        # V_r starts after Q_r warmup (only if using network mode)
        # Note: uses _warmup_q_r_end, then runs for warmup_v_r_steps before ramp-up
        if step >= self._warmup_q_r_end and self.v_r_use_network:
            active.add('v_r')
        
        return active
    
    def get_effective_beta_r(self, step: int) -> float:
        """
        Get effective beta_r for the given step.
        
        During warm-up: beta_r = 0 (uniform random policy)
        After warm-up: beta_r ramps from 0 to nominal value using sigmoid curve
        
        The sigmoid provides smooth transition: slow start, fast middle, slow end.
        Uses sigmoid((t - t0) / τ) scaled to [0, beta_r] where t0 is the midpoint.
        
        Args:
            step: Current training step.
            
        Returns:
            Effective beta_r value.
        """
        warmup_end = self._warmup_v_r_end
        
        if step < warmup_end:
            # During warm-up: uniform random policy
            return 0.0
        
        # After warm-up: sigmoidal ramp up beta_r
        steps_after_warmup = step - warmup_end
        
        if self.beta_r_rampup_steps <= 0:
            # No ramp-up: immediate switch to nominal beta_r
            return self.beta_r
        
        if steps_after_warmup >= self.beta_r_rampup_steps * 3:
            # Well past ramp-up: use nominal beta_r (sigmoid is ~1 at x=3)
            return self.beta_r
        
        # Sigmoidal ramp: sigmoid(6 * (t/T - 0.5)) maps [0, T] to roughly [0, 1]
        # At t=0: sigmoid(-3) ≈ 0.047
        # At t=T/2: sigmoid(0) = 0.5
        # At t=T: sigmoid(3) ≈ 0.953
        # We shift and scale to get closer to [0, 1]
        x = 6.0 * (steps_after_warmup / self.beta_r_rampup_steps - 0.5)
        sigmoid_value = 1.0 / (1.0 + math.exp(-x))
        
        # Scale sigmoid from [0.047, 0.953] to [0, 1]
        # sigmoid(-3) ≈ 0.0474, sigmoid(3) ≈ 0.9526
        min_sig = 1.0 / (1.0 + math.exp(3))  # ≈ 0.0474
        max_sig = 1.0 / (1.0 + math.exp(-3))  # ≈ 0.9526
        normalized = (sigmoid_value - min_sig) / (max_sig - min_sig)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
        
        return self.beta_r * normalized
    
    def get_learning_rate(self, network_name: str, step: int, update_count: int) -> float:
        """
        Get learning rate for a network at the given step.
        
        Schedule:
        1. During warm-up: constant learning rate
        2. After warm-up until lr_constant_fraction of total steps: constant LR
        3. After lr_constant_fraction: 1/sqrt(t) or 1/t decay
        
        The constant phase allows the network to learn the value function structure
        before fine-tuning. The late decay phase (1/t) satisfies Robbins-Monro
        conditions for converging to true expected values.
        
        Also respects legacy 1/t decay settings for X_h and U_r if enabled.
        
        Args:
            network_name: One of 'v_h_e', 'x_h', 'u_r', 'q_r', 'v_r'
            step: Current training step.
            update_count: Number of updates for this specific network.
            
        Returns:
            Learning rate for the network.
        """
        # Get base learning rate
        base_lr = {
            'v_h_e': self.lr_v_h_e,
            'x_h': self.lr_x_h,
            'u_r': self.lr_u_r,
            'q_r': self.lr_q_r,
            'v_r': self.lr_v_r,
        }.get(network_name, 1e-4)
        
        # During warm-up and beta_r ramp-up: constant learning rate
        full_warmup_end = self._warmup_v_r_end + self.beta_r_rampup_steps
        if step < full_warmup_end:
            return base_lr
        
        # Compute when decay should start (based on lr_constant_fraction)
        decay_start_step = max(
            full_warmup_end,
            int(self.lr_constant_fraction * self.num_training_steps)
        )
        
        # If LR decay is disabled or we haven't reached decay start, return constant
        if not self.use_sqrt_lr_decay or step < decay_start_step:
            return base_lr
        
        # After decay start: apply decay schedule
        # Use lr = base_lr * decay_start_step / step to ensure continuity at decay_start_step
        # At step = decay_start_step: lr = base_lr (continuous with constant phase)
        # As step increases: lr decays smoothly proportional to 1/step
        
        if self.constant_lr_then_1_over_t:
            # 1/t decay: lr = base_lr * decay_start_step / step
            return base_lr * decay_start_step / step
        else:
            # 1/sqrt(t) decay: lr = base_lr * sqrt(decay_start_step) / sqrt(step)
            return base_lr * math.sqrt(decay_start_step / step)
    
    def get_warmup_stage(self, step: int) -> int:
        """
        Get numeric warm-up stage (0-6).
        
        When x_h_use_network=True, u_r_use_network=True, v_r_use_network=True:
            0: Stage 1 - V_h^e only
            1: Stage 2 - V_h^e + X_h
            2: Stage 3 - V_h^e + X_h + U_r
            3: Stage 4 - V_h^e + X_h + U_r + Q_r
            4: Stage 5 - V_h^e + X_h + U_r + Q_r + V_r
            5: Post-warmup (beta_r ramping)
            6: Post-warmup (beta_r at nominal)
        
        When x_h_use_network=False (X_h stage skipped, warmup_x_h_steps=0):
            0: Stage 1 - V_h^e only
            3: Stage 2 - V_h^e + Q_r (X_h computed from V_h^e samples)
            5: Post-warmup (beta_r ramping)
            6: Post-warmup (beta_r at nominal)
        
        When u_r_use_network=False (U_r stage skipped, warmup_u_r_steps=0):
            0: Stage 1 - V_h^e only
            1: Stage 2 - V_h^e + X_h
            3: Stage 3 - V_h^e + X_h + Q_r
            5: Post-warmup (beta_r ramping)
            6: Post-warmup (beta_r at nominal)
        
        When v_r_use_network=False (default, V_r stage skipped, warmup_v_r_steps=0):
            Stages 0-3 as above, then stage 5 (beta_r ramping), then stage 6 (full training)
        """
        if step < self._warmup_v_h_e_end:
            return 0  # V_h^e only
        elif step < self._warmup_x_h_end:
            # This branch only reached if x_h_use_network=True (else warmup_x_h_steps=0)
            return 1  # + X_h
        elif step < self._warmup_u_r_end:
            # This branch only reached if u_r_use_network=True (else warmup_u_r_steps=0)
            return 2  # + U_r (training U_r before Q_r)
        elif step < self._warmup_q_r_end:
            return 3  # + Q_r
        elif step < self._warmup_v_r_end:
            # This branch only reached if v_r_use_network=True (else warmup_v_r_steps=0)
            return 4  # + V_r
        elif step < self._warmup_v_r_end + self.beta_r_rampup_steps:
            return 5  # beta_r ramping
        else:
            return 6  # full training with LR decay
    
    def get_warmup_stage_name(self, step: int) -> str:
        """Get human-readable name of current warm-up stage."""
        stage = self.get_warmup_stage(step)
        
        if self.x_h_use_network and self.u_r_use_network and self.v_r_use_network:
            names = {
                0: "Stage 1: V_h^e only",
                1: "Stage 2: V_h^e + X_h",
                2: "Stage 3: V_h^e + X_h + U_r",
                3: "Stage 4: V_h^e + X_h + U_r + Q_r",
                4: "Stage 5: V_h^e + X_h + U_r + Q_r + V_r",
                5: "β_r ramping",
                6: "Full training",
            }
        elif self.x_h_use_network and self.u_r_use_network and not self.v_r_use_network:
            names = {
                0: "Stage 1: V_h^e only",
                1: "Stage 2: V_h^e + X_h",
                2: "Stage 3: V_h^e + X_h + U_r",
                3: "Stage 4: V_h^e + X_h + U_r + Q_r",
                5: "β_r ramping",
                6: "Full training",
            }
        elif self.x_h_use_network and not self.u_r_use_network:
            # U_r computed directly from X_h, not trained (U_r stage skipped)
            names = {
                0: "Stage 1: V_h^e only",
                1: "Stage 2: V_h^e + X_h",
                3: "Stage 3: V_h^e + X_h + Q_r",
                5: "β_r ramping",
                6: "Full training",
            }
        else:
            # X_h computed directly from V_h^e samples, not trained (X_h stage skipped)
            names = {
                0: "Stage 1: V_h^e only",
                3: "Stage 2: V_h^e + Q_r",
                5: "β_r ramping",
                6: "Full training",
            }
        return names.get(stage, "Unknown")
    
    def get_stage_transition_steps(self) -> list:
        """
        Get list of steps where warm-up stage transitions occur.
        
        Returns:
            List of (step, stage_name) tuples for each transition.
        """
        transitions = []
        if self._warmup_v_h_e_end > 0:
            if self.x_h_use_network:
                transitions.append((self._warmup_v_h_e_end, "X_h starts"))
            else:
                transitions.append((self._warmup_v_h_e_end, "Q_r starts (X_h computed from V_h^e)"))
        
        if self.x_h_use_network:
            if self.u_r_use_network:
                # With U_r network: X_h -> U_r -> Q_r
                if self._warmup_x_h_end > self._warmup_v_h_e_end:
                    transitions.append((self._warmup_x_h_end, "U_r starts"))
                if self._warmup_u_r_end > self._warmup_x_h_end:
                    transitions.append((self._warmup_u_r_end, "Q_r starts"))
            else:
                # Without U_r network: X_h -> Q_r (skip U_r stage, _warmup_u_r_end == _warmup_x_h_end)
                if self._warmup_x_h_end > self._warmup_v_h_e_end:
                    transitions.append((self._warmup_x_h_end, "Q_r starts"))
        
        if self._warmup_q_r_end > self._warmup_u_r_end:
            if self.v_r_use_network:
                transitions.append((self._warmup_q_r_end, "V_r starts"))
            else:
                transitions.append((self._warmup_q_r_end, "Warmup ends"))
        
        if self.v_r_use_network and self._warmup_v_r_end > self._warmup_q_r_end:
            transitions.append((self._warmup_v_r_end, "Warmup ends"))
        
        if self.beta_r_rampup_steps > 0:
            transitions.append((self._warmup_v_r_end + self.beta_r_rampup_steps, "β_r ramp complete"))
        return transitions
    
    def get_stages_info(self) -> list:
        """
        Get list of all training stages with their durations and cumulative end steps.
        
        Returns:
            List of dicts with keys: 'stage_num', 'name', 'duration', 'end_step', 'networks'
            Stages with duration 0 (skipped) are not included.
        """
        stages = []
        
        # Stage 0: V_h^e only
        if self.warmup_v_h_e_steps > 0:
            stages.append({
                'stage_num': 0,
                'name': 'V_h^e only',
                'duration': int(self.warmup_v_h_e_steps),
                'end_step': int(self._warmup_v_h_e_end),
                'networks': ['v_h_e'],
            })
        
        # Stage 1: V_h^e + X_h (only if x_h_use_network=True)
        if self.x_h_use_network and self.warmup_x_h_steps > 0:
            stages.append({
                'stage_num': 1,
                'name': 'V_h^e + X_h',
                'duration': int(self.warmup_x_h_steps),
                'end_step': int(self._warmup_x_h_end),
                'networks': ['v_h_e', 'x_h'],
            })
        
        # Stage 2: V_h^e + X_h + U_r (only if u_r_use_network=True)
        if self.u_r_use_network and self.warmup_u_r_steps > 0:
            stages.append({
                'stage_num': 2,
                'name': 'V_h^e + X_h + U_r',
                'duration': int(self.warmup_u_r_steps),
                'end_step': int(self._warmup_u_r_end),
                'networks': ['v_h_e', 'x_h', 'u_r'],
            })
        
        # Stage 3: + Q_r
        if self.warmup_q_r_steps > 0:
            networks = ['v_h_e', 'q_r']
            if self.x_h_use_network:
                networks.insert(1, 'x_h')
            if self.u_r_use_network:
                networks.insert(-1, 'u_r')
            stages.append({
                'stage_num': 3,
                'name': '+ Q_r',
                'duration': int(self.warmup_q_r_steps),
                'end_step': int(self._warmup_q_r_end),
                'networks': networks,
            })
        
        # Stage 4: + V_r (only if v_r_use_network=True)
        if self.v_r_use_network and self.warmup_v_r_steps > 0:
            networks = ['v_h_e', 'q_r', 'v_r']
            if self.x_h_use_network:
                networks.insert(1, 'x_h')
            if self.u_r_use_network:
                networks.insert(-2, 'u_r')
            stages.append({
                'stage_num': 4,
                'name': '+ V_r',
                'duration': int(self.warmup_v_r_steps),
                'end_step': int(self._warmup_v_r_end),
                'networks': networks,
            })
        
        # Stage 5: β_r ramp-up
        if self.beta_r_rampup_steps > 0:
            networks = ['v_h_e', 'q_r']
            if self.x_h_use_network:
                networks.insert(1, 'x_h')
            if self.u_r_use_network:
                networks.insert(-1, 'u_r')
            if self.v_r_use_network:
                networks.append('v_r')
            stages.append({
                'stage_num': 5,
                'name': 'β_r ramp-up',
                'duration': int(self.beta_r_rampup_steps),
                'end_step': int(self._warmup_v_r_end + self.beta_r_rampup_steps),
                'networks': networks,
            })
        
        # Stage 6: Full training
        full_warmup_end = int(self._warmup_v_r_end + self.beta_r_rampup_steps)
        remaining = int(self.num_training_steps) - full_warmup_end
        if remaining > 0:
            networks = ['v_h_e', 'q_r']
            if self.x_h_use_network:
                networks.insert(1, 'x_h')
            if self.u_r_use_network:
                networks.insert(-1, 'u_r')
            if self.v_r_use_network:
                networks.append('v_r')
            stages.append({
                'stage_num': 6,
                'name': 'Full training',
                'duration': remaining,
                'end_step': int(self.num_training_steps),
                'networks': networks,
            })
        
        return stages
    
    def format_stages_table(self) -> str:
        """
        Get a formatted ASCII table of all training stages.
        
        Returns:
            String with formatted table showing stage names, durations, and end steps.
        """
        stages = self.get_stages_info()
        if not stages:
            return "No stages configured."
        
        # Build table
        lines = []
        lines.append("Training Stages:")
        lines.append("-" * 70)
        lines.append(f"{'Stage':<6} {'Name':<25} {'Duration':>12} {'End Step':>12} {'Networks'}")
        lines.append("-" * 70)
        
        for s in stages:
            networks_str = ', '.join(s['networks'])
            lines.append(
                f"{s['stage_num']:<6} {s['name']:<25} {s['duration']:>12,} {s['end_step']:>12,} {networks_str}"
            )
        
        lines.append("-" * 70)
        total_warmup = int(self._warmup_v_r_end + self.beta_r_rampup_steps)
        lines.append(f"Total warmup (before LR decay): {total_warmup:,} steps")
        lines.append(f"Total training steps: {int(self.num_training_steps):,}")
        
        return '\n'.join(lines)
    
    def get_stage_duration(self, stage_num: int) -> int:
        """
        Get the duration in steps for a specific stage number.
        
        Args:
            stage_num: Stage number (0-6)
            
        Returns:
            Duration in training steps, or 0 if stage is skipped.
        """
        if stage_num == 0:
            return int(self.warmup_v_h_e_steps)
        elif stage_num == 1:
            return int(self.warmup_x_h_steps) if self.x_h_use_network else 0
        elif stage_num == 2:
            return int(self.warmup_u_r_steps) if self.u_r_use_network else 0
        elif stage_num == 3:
            return int(self.warmup_q_r_steps)
        elif stage_num == 4:
            return int(self.warmup_v_r_steps) if self.v_r_use_network else 0
        elif stage_num == 5:
            return int(self.beta_r_rampup_steps)
        elif stage_num == 6:
            full_warmup_end = int(self._warmup_v_r_end + self.beta_r_rampup_steps)
            return max(0, int(self.num_training_steps) - full_warmup_end)
        else:
            return 0
    
    # =========================================================================
    # Lookup table helper methods
    # =========================================================================
    
    def should_use_lookup_table(self, network_name: str) -> bool:
        """
        Check if a specific network should use lookup table implementation.
        
        Args:
            network_name: One of 'q_r', 'v_r', 'v_h_e', 'x_h', 'u_r'
            
        Returns:
            True if lookup table should be used for this network.
        """
        if not self.use_lookup_tables:
            return False
        
        lookup_flags = {
            'q_r': self.use_lookup_q_r,
            'v_r': self.use_lookup_v_r and self.v_r_use_network,  # Only if V_r network is used
            'v_h_e': self.use_lookup_v_h_e,
            'x_h': self.use_lookup_x_h and self.x_h_use_network,  # Only if X_h network is used
            'u_r': self.use_lookup_u_r and self.u_r_use_network,  # Only if U_r network is used
        }
        return lookup_flags.get(network_name, False)
    
    def get_lookup_default(self, network_name: str) -> float:
        """
        Get default value for lookup table entries for a network.
        
        Args:
            network_name: One of 'q_r', 'v_r', 'v_h_e', 'x_h', 'u_r'
            
        Returns:
            Default value for new lookup table entries.
        """
        defaults = {
            'q_r': self.lookup_default_q_r,
            'v_r': self.lookup_default_v_r,
            'v_h_e': self.lookup_default_v_h_e,
            'x_h': self.lookup_default_x_h,
            'u_r': self.lookup_default_y,  # For U_r, this is default y value
        }
        return defaults.get(network_name, 0.0)
    
    def save_yaml(self, path: str) -> str:
        """
        Save all configuration parameters to a YAML file with structured sections.
        
        Args:
            path: Path to save the YAML file.
            
        Returns:
            Actual path where the file was saved (may differ if fallback used).
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for save_yaml(). Install with: pip install pyyaml"
            )
        
        # Organize fields into logical sections
        config_dict = {
            'theory_parameters': {
                'discount_factors': {
                    'gamma_r': self.gamma_r,
                    'gamma_h': self.gamma_h,
                },
                'power_metric': {
                    'zeta': self.zeta,
                    'xi': self.xi,
                    'eta': self.eta,
                },
                'robot_policy': {
                    'beta_r': self.beta_r,
                },
            },
            'exploration': {
                'robot': {
                    'epsilon_r_start': self.epsilon_r_start,
                    'epsilon_r_end': self.epsilon_r_end,
                    'epsilon_r_decay_steps': self.epsilon_r_decay_steps,
                },
                'human': {
                    'epsilon_h_start': self.epsilon_h_start,
                    'epsilon_h_end': self.epsilon_h_end,
                    'epsilon_h_decay_steps': self.epsilon_h_decay_steps,
                },
            },
            'curiosity': {
                'rnd': {
                    'use_rnd': self.use_rnd,
                    'use_human_action_rnd': self.use_human_action_rnd,
                    'rnd_feature_dim': self.rnd_feature_dim,
                    'rnd_hidden_dim': self.rnd_hidden_dim,
                    'rnd_bonus_coef_r': self.rnd_bonus_coef_r,
                    'rnd_bonus_coef_h': self.rnd_bonus_coef_h,
                    'lr_rnd': self.lr_rnd,
                    'rnd_weight_decay': self.rnd_weight_decay,
                    'rnd_grad_clip': self.rnd_grad_clip,
                    'normalize_rnd': self.normalize_rnd,
                    'rnd_normalization_decay': self.rnd_normalization_decay,
                },
                'count_based': {
                    'use_count_based_curiosity': self.use_count_based_curiosity,
                    'count_curiosity_scale': self.count_curiosity_scale,
                    'count_curiosity_use_ucb': self.count_curiosity_use_ucb,
                    'count_curiosity_bonus_coef_r': self.count_curiosity_bonus_coef_r,
                    'count_curiosity_bonus_coef_h': self.count_curiosity_bonus_coef_h,
                },
            },
            'learning_rates': {
                'base_rates': {
                    'lr_q_r': self.lr_q_r,
                    'lr_v_r': self.lr_v_r,
                    'lr_v_h_e': self.lr_v_h_e,
                    'lr_x_h': self.lr_x_h,
                    'lr_u_r': self.lr_u_r,
                },
                'schedule': {
                    'use_sqrt_lr_decay': self.use_sqrt_lr_decay,
                    'lr_constant_fraction': self.lr_constant_fraction,
                    'constant_lr_then_1_over_t': self.constant_lr_then_1_over_t,
                },
                'adaptive_lookup': {
                    'lookup_use_adaptive_lr': self.lookup_use_adaptive_lr,
                    'lookup_adaptive_lr_min': self.lookup_adaptive_lr_min,
                },
                'adaptive_rnd': {
                    'rnd_use_adaptive_lr': self.rnd_use_adaptive_lr,
                    'rnd_adaptive_lr_scale': self.rnd_adaptive_lr_scale,
                    'rnd_adaptive_lr_min': self.rnd_adaptive_lr_min,
                    'rnd_adaptive_lr_max': self.rnd_adaptive_lr_max,
                },
            },
            'warmup': {
                'stage_durations': {
                    'warmup_v_h_e_steps': self.warmup_v_h_e_steps,
                    'warmup_x_h_steps': self.warmup_x_h_steps,
                    'warmup_u_r_steps': self.warmup_u_r_steps,
                    'warmup_q_r_steps': self.warmup_q_r_steps,
                    'warmup_v_r_steps': self.warmup_v_r_steps,
                    'beta_r_rampup_steps': self.beta_r_rampup_steps,
                },
                'computed_thresholds': {
                    'warmup_v_h_e_end': self._warmup_v_h_e_end,
                    'warmup_x_h_end': self._warmup_x_h_end,
                    'warmup_u_r_end': self._warmup_u_r_end,
                    'warmup_q_r_end': self._warmup_q_r_end,
                    'warmup_v_r_end': self._warmup_v_r_end,
                    'total_warmup_steps': self.get_total_warmup_steps(),
                },
            },
            'target_networks': {
                'q_r_target_update_interval': self.q_r_target_update_interval,
                'v_r_target_update_interval': self.v_r_target_update_interval,
                'v_h_e_target_update_interval': self.v_h_e_target_update_interval,
                'x_h_target_update_interval': self.x_h_target_update_interval,
                'u_r_target_update_interval': self.u_r_target_update_interval,
            },
            'training': {
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'x_h_batch_size': self.x_h_batch_size,
                'num_training_steps': self.num_training_steps,
                'steps_per_episode': self.steps_per_episode,
                'training_steps_per_env_step': self.training_steps_per_env_step,
                'env_steps_per_training_step': self.env_steps_per_training_step,
                'goal_resample_prob': self.goal_resample_prob,
                'use_model_based_targets': self.use_model_based_targets,
            },
            'sampling': {
                'u_r_sample_humans': self.u_r_sample_humans,
                'x_h_sample_humans': self.x_h_sample_humans,
            },
            'regularization': {
                'weight_decay': {
                    'q_r_weight_decay': self.q_r_weight_decay,
                    'v_r_weight_decay': self.v_r_weight_decay,
                    'v_h_e_weight_decay': self.v_h_e_weight_decay,
                    'x_h_weight_decay': self.x_h_weight_decay,
                    'u_r_weight_decay': self.u_r_weight_decay,
                },
                'gradient_clipping': {
                    'q_r_grad_clip': self.q_r_grad_clip,
                    'v_r_grad_clip': self.v_r_grad_clip,
                    'v_h_e_grad_clip': self.v_h_e_grad_clip,
                    'x_h_grad_clip': self.x_h_grad_clip,
                    'u_r_grad_clip': self.u_r_grad_clip,
                    'auto_grad_clip': self.auto_grad_clip,
                    'auto_grad_clip_reference_lr': self.auto_grad_clip_reference_lr,
                },
                'dropout': {
                    'q_r_dropout': self.q_r_dropout,
                    'v_r_dropout': self.v_r_dropout,
                    'v_h_e_dropout': self.v_h_e_dropout,
                    'x_h_dropout': self.x_h_dropout,
                    'u_r_dropout': self.u_r_dropout,
                },
            },
            'network_modes': {
                'v_r_use_network': self.v_r_use_network,
                'u_r_use_network': self.u_r_use_network,
                'x_h_use_network': self.x_h_use_network,
                'use_encoders': self.use_encoders,
                'include_step_count': self.include_step_count,
                'use_z_space_transform': self.use_z_space_transform,
            },
            'lookup_tables': {
                'use_lookup_tables': self.use_lookup_tables,
                'network_flags': {
                    'use_lookup_q_r': self.use_lookup_q_r,
                    'use_lookup_v_r': self.use_lookup_v_r,
                    'use_lookup_v_h_e': self.use_lookup_v_h_e,
                    'use_lookup_x_h': self.use_lookup_x_h,
                    'use_lookup_u_r': self.use_lookup_u_r,
                },
                'default_values': {
                    'lookup_default_q_r': self.lookup_default_q_r,
                    'lookup_default_v_r': self.lookup_default_v_r,
                    'lookup_default_v_h_e': self.lookup_default_v_h_e,
                    'lookup_default_x_h': self.lookup_default_x_h,
                    'lookup_default_y': self.lookup_default_y,
                },
            },
            'async_training': {
                'async_training': self.async_training,
                'num_actors': self.num_actors,
                'actor_sync_freq': self.actor_sync_freq,
                'rnd_sync_freq': self.rnd_sync_freq,
                'async_min_buffer_size': self.async_min_buffer_size,
                'max_env_steps_per_training_step': self.max_env_steps_per_training_step,
                'async_queue_size': self.async_queue_size,
            },
            'network_architecture': {
                'hidden_dim': self.hidden_dim,
                'state_feature_dim': self.state_feature_dim,
                'goal_feature_dim': self.goal_feature_dim,
                'agent_embedding_dim': self.agent_embedding_dim,
                'agent_position_feature_dim': self.agent_position_feature_dim,
                'agent_feature_dim': self.agent_feature_dim,
            },
            'profiling': {
                'profile_batching': self.profile_batching,
                'profile_batching_interval': self.profile_batching_interval,
            },
            'checkpointing_and_memory': {
                'checkpoint_interval': self.checkpoint_interval,
                'min_free_memory_fraction': self.min_free_memory_fraction,
                'memory_check_interval': self.memory_check_interval,
                'memory_pause_duration': self.memory_pause_duration,
            },
        }
        
        # Try to save to the specified path
        actual_path = path
        try:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        except (IOError, OSError) as e:
            # Fallback to /tmp if original path fails
            fallback_path = os.path.join('/tmp', os.path.basename(path))
            warnings.warn(
                f"Could not write to {path}: {e}. Falling back to {fallback_path}",
                UserWarning,
                stacklevel=2
            )
            with open(fallback_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            actual_path = fallback_path
        
        return actual_path
