"""
Configuration for Phase 2 robot policy learning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Phase2Config:
    """
    Configuration for Phase 2 training.
    
    Phase 2 learns the robot policy to softly maximize aggregate human power
    as defined in equations (4)-(9) of the EMPO paper.
    
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
    beta_r: float = 10.0  # Power-law policy exponent
    
    # Exploration (in addition to power-law policy randomization)
    epsilon_r_start: float = 1.0
    epsilon_r_end: float = 0.01
    epsilon_r_decay_steps: int = 10000
    
    # Learning rates (may need adjustment for time-scale separation)
    lr_q_r: float = 1e-3
    lr_v_r: float = 1e-3
    lr_v_h_e: float = 1e-3
    lr_x_h: float = 1e-3
    lr_u_r: float = 1e-3
    
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
    
    # X_h regularization options (to help with high-variance learning)
    x_h_weight_decay: float = 0.0  # L2 regularization for X_h optimizer (e.g., 1e-4)
    x_h_grad_clip: Optional[float] = None  # Max gradient norm for X_h (e.g., 1.0)
    x_h_dropout: float = 0.0  # Dropout rate for X_h network (e.g., 0.1)
    
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
