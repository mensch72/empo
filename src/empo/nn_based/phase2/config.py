"""
Configuration for Phase 2 robot policy learning.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Set
import math


@dataclass
class Phase2Config:
    """
    Configuration for Phase 2 training.
    
    Phase 2 learns the robot policy to softly maximize aggregate human power
    as defined in equations (4)-(9) of the EMPO paper.
    
    Warm-up Phase:
        Training proceeds in stages to break mutual network dependencies:
        1. V_h^e only (steps 0 to warmup_v_h_e_steps)
        2. V_h^e + X_h (steps warmup_v_h_e_steps to warmup_x_h_steps)
        3. V_h^e + X_h + U_r (steps warmup_x_h_steps to warmup_u_r_steps)
        4. V_h^e + X_h + U_r + Q_r (steps warmup_u_r_steps to warmup_q_r_steps)
        5. All networks including V_r if enabled (after warmup_q_r_steps)
        
        During warm-up, beta_r=0 (uniform random robot policy).
        After warm-up, beta_r ramps up to nominal value over beta_r_rampup_steps.
        Learning rates are constant during warm-up, then decay as 1/sqrt(t).
    
    Attributes:
        gamma_r: Robot discount factor.
        gamma_h: Human discount factor (for V_h^e computation).
        zeta: Risk/reliability preference parameter (ζ >= 1, 1 = neutral).
        xi: Inter-human inequality aversion (ξ >= 1 to protect last bit of power).
        eta: Additional intertemporal inequality aversion (η >= 1, 1 = neutral).
        beta_r: Robot power-law policy exponent (< ∞ to prevent overoptimization risks).
        epsilon_r_start: Initial exploration rate.
        epsilon_r_end: Final exploration rate.
        epsilon_r_decay_steps: Steps over which epsilon decays.
        lr_q_r: Learning rate for Q_r network.
        lr_v_r: Learning rate for V_r network.
        lr_v_h_e: Learning rate for V_h^e network.
        lr_x_h: Learning rate for X_h network.
        lr_u_r: Learning rate for U_r network.
        v_r_target_update_freq: Steps between V_r target network updates.
        v_h_target_update_freq: Steps between V_h^e target network updates.
        buffer_size: Replay buffer capacity.
        batch_size: Training batch size.
        num_episodes: Total training episodes.
        steps_per_episode: Steps per episode.
        updates_per_step: Gradient updates per environment step.
        goal_resample_prob: Probability of resampling goals each step.
        hidden_dim: Hidden layer dimension for networks.
        state_feature_dim: State encoder output dimension.
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
    
    # Exploration (in addition to power-law policy randomization)
    epsilon_r_start: float = 1.0
    epsilon_r_end: float = 0.01
    epsilon_r_decay_steps: int = 10000
    
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
    # Stage 2: V_h^e + X_h (warmup_x_h_steps duration)
    # Stage 3: V_h^e + X_h + U_r (warmup_u_r_steps duration) - SKIPPED if u_r_use_network=False
    # Stage 4: V_h^e + X_h + (U_r) + Q_r (warmup_q_r_steps duration)
    # Stage 5: All networks with beta_r ramp-up (beta_r_rampup_steps duration)
    # Stage 6: Full training with LR decay (remainder)
    # 
    # NOTE: warmup_u_r_steps is set to 0 if u_r_use_network=False (in __post_init__)
    warmup_v_h_e_steps: int = 1000   # Duration of V_h^e-only stage
    warmup_x_h_steps: int = 1000     # Duration of V_h^e + X_h stage  
    warmup_u_r_steps: int = 1000     # Duration of V_h^e + X_h + U_r stage (0 if u_r_use_network=False)
    warmup_q_r_steps: int = 1000     # Duration of V_h^e + X_h + (U_r) + Q_r stage
    
    # Beta_r schedule: ramps from 0 to beta_r over this many steps after warm-up ends
    beta_r_rampup_steps: int = 2000
    
    # Learning rate schedule after warm-up
    # After warm-up, use 1/sqrt(t) decay: lr(t) = lr_base * sqrt(warmup) / sqrt(t)
    # This is a compromise between 1/t (for expectations) and constant (for Q-learning)
    use_sqrt_lr_decay: bool = True
    
    # Legacy 1/t decay settings (DEPRECATED - use use_sqrt_lr_decay instead)
    # These flags take precedence over use_sqrt_lr_decay if enabled.
    lr_x_h_warmup_steps: int = 1000  # Steps before 1/t decay starts (0 = always 1/t)
    lr_u_r_warmup_steps: int = 1000  # Steps before 1/t decay starts (0 = always 1/t)
    lr_x_h_use_1_over_t: bool = False  # DEPRECATED: Whether to use legacy 1/t decay for X_h
    lr_u_r_use_1_over_t: bool = False  # DEPRECATED: Whether to use legacy 1/t decay for U_r
    
    # Target network updates
    v_r_target_update_freq: int = 100
    v_h_target_update_freq: int = 100
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    x_h_batch_size: Optional[int] = None  # Larger batch for X_h (None = use batch_size)
    
    # Training
    num_episodes: int = 10000
    steps_per_episode: int = 50
    updates_per_step: int = 1
    
    # Goal resampling
    goal_resample_prob: float = 0.01
    
    # U_r loss computation: number of humans to sample (None = all humans)
    u_r_sample_humans: Optional[int] = None
    
    # X_h loss computation: number of human-goal pairs to sample (None = all from transition)
    x_h_sample_humans: Optional[int] = None
    
    # Regularization options for all networks (weight decay, gradient clipping, dropout)
    # These help with training stability, especially for high-variance learning.
    # Weight decay (L2 regularization) for each network's optimizer
    q_r_weight_decay: float = 1e-3 #1e-4
    v_r_weight_decay: float = 1e-3 #1e-4
    v_h_e_weight_decay: float = 1e-3 #1e-4
    x_h_weight_decay: float = 1e-3 #1e-4
    u_r_weight_decay: float = 1e-3 #1e-4
    
    # Max gradient norm for each network (0 or None to disable clipping)
    # If auto_grad_clip is True, these values are scaled by the learning rate.
    q_r_grad_clip: Optional[float] = 1.0
    v_r_grad_clip: Optional[float] = 1.0
    v_h_e_grad_clip: Optional[float] = 1.0
    x_h_grad_clip: Optional[float] = 1.0
    u_r_grad_clip: Optional[float] = 1.0
    
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
    
    # Whether to include step count (remaining time) in state encoding.
    # Set to False to verify that identical grid states get identical values.
    include_step_count: bool = True
    
    # Profiling: if True, collect timing statistics for batched computation stages.
    # This adds minimal overhead but provides detailed breakdown of where time is spent.
    # Results are printed periodically and at the end of training.
    profile_batching: bool = False
    profile_batching_interval: int = 100  # Print stats every N training steps
    
    def __post_init__(self):
        """Compute cumulative warmup thresholds and apply network flags."""
        # Warn about deprecated legacy LR decay flags
        if self.lr_x_h_use_1_over_t or self.lr_u_r_use_1_over_t:
            warnings.warn(
                "lr_x_h_use_1_over_t and lr_u_r_use_1_over_t are deprecated. "
                "Use use_sqrt_lr_decay=True instead for 1/√t decay. "
                "Legacy 1/t flags take precedence if enabled.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Override U_r warmup duration to 0 if not using U_r network
        if not self.u_r_use_network:
            self.warmup_u_r_steps = 0
        
        # Compute cumulative thresholds from per-stage durations
        # These are used internally for all step-based comparisons
        self._warmup_v_h_e_end = self.warmup_v_h_e_steps
        self._warmup_x_h_end = self._warmup_v_h_e_end + self.warmup_x_h_steps
        self._warmup_u_r_end = self._warmup_x_h_end + self.warmup_u_r_steps
        self._warmup_q_r_end = self._warmup_u_r_end + self.warmup_q_r_steps
    
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
    # More actors = more diverse data, but diminishing returns after ~4-8.
    num_actors: int = 4
    
    # Steps between syncing frozen policy from learner to actors.
    # Lower = more on-policy but more sync overhead.
    # Higher = more off-policy but more efficient.
    actor_sync_freq: int = 100
    
    # Minimum transitions in buffer before training starts (for async mode).
    # Ensures actors have collected enough initial data.
    async_min_buffer_size: int = 1000
    
    # Queue size for actor-to-learner transition queue.
    # Should be large enough to buffer actor output during learner training.
    async_queue_size: int = 10000
    
    # Network architecture
    hidden_dim: int = 256
    state_feature_dim: int = 256
    
    def get_epsilon(self, step: int) -> float:
        """Get epsilon value for given training step."""
        if step >= self.epsilon_r_decay_steps:
            return self.epsilon_r_end
        
        # Linear decay
        decay_rate = (self.epsilon_r_start - self.epsilon_r_end) / self.epsilon_r_decay_steps
        return self.epsilon_r_start - decay_rate * step
    
    def get_lr_x_h(self, update_count: int) -> float:
        """Get X_h learning rate with optional 1/t decay.
        
        After warmup, decays as lr_base * warmup / t to satisfy Robbins-Monro
        conditions for converging to true expectation.
        """
        if not self.lr_x_h_use_1_over_t:
            return self.lr_x_h
        
        warmup = max(1, self.lr_x_h_warmup_steps)
        if update_count <= warmup:
            return self.lr_x_h
        
        # 1/t decay: lr = lr_base * warmup / t
        return self.lr_x_h * warmup / update_count
    
    def get_lr_u_r(self, update_count: int) -> float:
        """Get U_r learning rate with optional 1/t decay.
        
        After warmup, decays as lr_base * warmup / t to satisfy Robbins-Monro
        conditions for converging to true expectation.
        """
        if not self.lr_u_r_use_1_over_t:
            return self.lr_u_r
        
        warmup = max(1, self.lr_u_r_warmup_steps)
        if update_count <= warmup:
            return self.lr_u_r
        
        # 1/t decay: lr = lr_base * warmup / t
        return self.lr_u_r * warmup / update_count
    
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
        return self._warmup_q_r_end + self.beta_r_rampup_steps
    
    def is_in_warmup(self, step: int) -> bool:
        """Check if we're still in the warm-up phase (before all networks active)."""
        return step < self._warmup_q_r_end
    
    def is_in_rampup(self, step: int) -> bool:
        """Check if we're in the beta_r ramp-up phase."""
        return self._warmup_q_r_end <= step < self._warmup_q_r_end + self.beta_r_rampup_steps
    
    def is_fully_trained(self, step: int) -> bool:
        """Check if we're past all warmup/rampup phases (LR decay starts here)."""
        return step >= self._warmup_q_r_end + self.beta_r_rampup_steps
    
    def get_active_networks(self, step: int) -> Set[str]:
        """
        Get the set of networks that should be trained at the given step.
        
        Args:
            step: Current training step.
            
        Returns:
            Set of network names to train: subset of {'v_h_e', 'x_h', 'u_r', 'q_r', 'v_r'}
        """
        active = set()
        
        # V_h^e is always active (it's the foundation)
        active.add('v_h_e')
        
        # X_h starts after V_h^e warmup
        if step >= self._warmup_v_h_e_end:
            active.add('x_h')
        
        # U_r starts after X_h warmup (only if using network mode)
        if step >= self._warmup_x_h_end and self.u_r_use_network:
            active.add('u_r')
        
        # Q_r starts after U_r warmup (U_r duration is 0 if not using U_r network)
        if step >= self._warmup_u_r_end:
            active.add('q_r')
        
        # V_r starts after Q_r warmup (only if using network mode)
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
        warmup_end = self._warmup_q_r_end
        
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
        
        During warm-up: constant learning rate
        After warm-up: 1/sqrt(t) decay if use_sqrt_lr_decay is True
        
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
        
        # Check legacy 1/t decay for X_h and U_r
        if network_name == 'x_h' and self.lr_x_h_use_1_over_t:
            return self.get_lr_x_h(update_count)
        if network_name == 'u_r' and self.lr_u_r_use_1_over_t:
            return self.get_lr_u_r(update_count)
        
        # During warm-up and beta_r ramp-up: constant learning rate
        # LR decay only starts after beta_r ramp-up is complete
        full_warmup_end = self._warmup_q_r_end + self.beta_r_rampup_steps
        if step < full_warmup_end or not self.use_sqrt_lr_decay:
            return base_lr
        
        # After full warmup (including ramp-up): 1/sqrt(t) decay
        # We count steps since full warmup ended
        t = max(1, step - full_warmup_end + 1)  # +1 to avoid division issues
        
        # 1/sqrt(t) decay: lr = lr_base / sqrt(t)
        return base_lr / math.sqrt(t)
    
    def get_warmup_stage(self, step: int) -> int:
        """
        Get numeric warm-up stage (0-5).
        
        When u_r_use_network=True:
            0: Stage 1 - V_h^e only
            1: Stage 2 - V_h^e + X_h
            2: Stage 3 - V_h^e + X_h + U_r
            3: Stage 4 - V_h^e + X_h + U_r + Q_r
            4: Post-warmup (beta_r ramping)
            5: Post-warmup (beta_r at nominal)
        
        When u_r_use_network=False (U_r stage skipped, warmup_u_r_steps=0):
            0: Stage 1 - V_h^e only
            1: Stage 2 - V_h^e + X_h
            3: Stage 3 - V_h^e + X_h + Q_r
            4: Post-warmup (beta_r ramping)
            5: Post-warmup (beta_r at nominal)
        """
        if step < self._warmup_v_h_e_end:
            return 0  # V_h^e only
        elif step < self._warmup_x_h_end:
            return 1  # + X_h
        elif step < self._warmup_u_r_end:
            # This branch only reached if u_r_use_network=True (else warmup_u_r_steps=0)
            return 2  # + U_r (training U_r before Q_r)
        elif step < self._warmup_q_r_end:
            return 3  # + Q_r
        elif step < self._warmup_q_r_end + self.beta_r_rampup_steps:
            return 4  # beta_r ramping
        else:
            return 5  # full training with LR decay
    
    def get_warmup_stage_name(self, step: int) -> str:
        """Get human-readable name of current warm-up stage."""
        stage = self.get_warmup_stage(step)
        
        if self.u_r_use_network:
            names = {
                0: "Stage 1: V_h^e only",
                1: "Stage 2: V_h^e + X_h",
                2: "Stage 3: V_h^e + X_h + U_r",
                3: "Stage 4: V_h^e + X_h + U_r + Q_r",
                4: "β_r ramping (constant LR)",
                5: "Full training (LR decay)",
            }
        else:
            # U_r computed directly from X_h, not trained (stage 2 skipped)
            names = {
                0: "Stage 1: V_h^e only",
                1: "Stage 2: V_h^e + X_h",
                3: "Stage 3: V_h^e + X_h + Q_r",  # Stage 3 (was 4 with U_r)
                4: "β_r ramping (constant LR)",
                5: "Full training (LR decay)",
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
            transitions.append((self._warmup_v_h_e_end, "X_h starts"))
        
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
            transitions.append((self._warmup_q_r_end, "Warmup ends"))
        if self.beta_r_rampup_steps > 0:
            transitions.append((self._warmup_q_r_end + self.beta_r_rampup_steps, "β_r ramp complete"))
        return transitions
