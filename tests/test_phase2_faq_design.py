#!/usr/bin/env python3
"""
Test suite for Phase 2 FAQ design choices.

These tests verify that all design choices documented in docs/FAQ.md
are properly implemented in the training code.

See FAQ.md for detailed justifications of each design choice.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from empo.learning_based.phase2.config import Phase2Config


# =============================================================================
# 1. WARMUP STAGES & SEQUENTIAL NETWORK ACTIVATION
# =============================================================================

class TestWarmupStages:
    """Test multi-stage warmup with sequential network activation."""
    
    def test_warmup_stage_progression_with_u_r_network(self):
        """Test warmup stages when using U_r network."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=1000,
            warmup_q_r_steps=1000,
            beta_r_rampup_steps=2000,
            u_r_use_network=True,
        )
        
        # Stage 0: V_h^e only
        assert config.get_warmup_stage(0) == 0
        assert config.get_active_networks(0) == {'v_h_e'}
        
        # Stage 1: V_h^e + X_h
        assert config.get_warmup_stage(1500) == 1
        assert config.get_active_networks(1500) == {'v_h_e', 'x_h'}
        
        # Stage 2: V_h^e + X_h + U_r
        assert config.get_warmup_stage(2500) == 2
        assert config.get_active_networks(2500) == {'v_h_e', 'x_h', 'u_r'}
        
        # Stage 3: V_h^e + X_h + U_r + Q_r
        assert config.get_warmup_stage(3500) == 3
        assert config.get_active_networks(3500) == {'v_h_e', 'x_h', 'u_r', 'q_r'}
        
        # Stage 5: Beta_r ramping (stage 4 is V_r warmup, skipped when v_r_use_network=False)
        assert config.get_warmup_stage(4500) == 5
        
        # Stage 6: Full training
        assert config.get_warmup_stage(7000) == 6
    
    def test_warmup_stage_progression_without_u_r_network(self):
        """Test warmup stages when U_r is computed directly (no network)."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=1000,  # Should be overridden to 0
            warmup_q_r_steps=1000,
            u_r_use_network=False,
        )
        
        # U_r warmup should be automatically set to 0
        assert config.warmup_u_r_steps == 0
        
        # Stage progression skips U_r stage
        assert config.get_active_networks(0) == {'v_h_e'}
        assert config.get_active_networks(1500) == {'v_h_e', 'x_h'}
        # Stage 2 is skipped, goes directly to Q_r
        assert config.get_active_networks(2500) == {'v_h_e', 'x_h', 'q_r'}
    
    def test_v_r_activation_when_using_network(self):
        """Test V_r is only active when v_r_use_network=True."""
        config_no_v_r = Phase2Config(v_r_use_network=False, warmup_q_r_steps=100)
        config_with_v_r = Phase2Config(v_r_use_network=True, warmup_q_r_steps=100)
        
        step = config_no_v_r._warmup_q_r_end + 100
        
        assert 'v_r' not in config_no_v_r.get_active_networks(step)
        assert 'v_r' in config_with_v_r.get_active_networks(step)


# =============================================================================
# 2. BETA_R = 0 DURING WARMUP + SIGMOID RAMP-UP
# =============================================================================

class TestBetaRSchedule:
    """Test beta_r = 0 during warmup and sigmoid ramp-up."""
    
    def test_beta_r_zero_during_warmup(self):
        """Beta_r should be exactly 0 during warmup."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            beta_r=10.0,
        )
        
        warmup_end = config._warmup_v_r_end
        
        # All warmup steps should have beta_r = 0
        for step in [0, 500, 1000, 2000, warmup_end - 1]:
            assert config.get_effective_beta_r(step) == 0.0, f"beta_r should be 0 at step {step}"
    
    def test_beta_r_sigmoid_rampup(self):
        """Beta_r should ramp up using sigmoid curve after warmup."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            beta_r_rampup_steps=2000,
            beta_r=10.0,
        )
        
        warmup_end = config._warmup_v_r_end  # = 3000 (v_r_use_network=False by default)
        rampup_end = warmup_end + config.beta_r_rampup_steps  # = 5000
        
        # Just after warmup: small but non-zero
        beta_start = config.get_effective_beta_r(warmup_end + 1)
        assert 0 < beta_start < 1.0
        
        # Midpoint: approximately half of nominal
        beta_mid = config.get_effective_beta_r(warmup_end + 1000)
        assert 4.0 < beta_mid < 6.0, f"Expected ~5.0 at midpoint, got {beta_mid}"
        
        # After ramp: nominal value
        beta_full = config.get_effective_beta_r(rampup_end + 1000)
        assert beta_full == config.beta_r
    
    def test_beta_r_immediate_when_no_rampup(self):
        """Beta_r should be immediate when rampup_steps = 0."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
            beta_r_rampup_steps=0,
            beta_r=10.0,
        )
        
        warmup_end = config._warmup_v_r_end
        
        # Last warmup step should still be 0
        assert config.get_effective_beta_r(warmup_end - 1) == 0.0
        # At warmup_end with no rampup, beta_r should immediately be nominal
        assert config.get_effective_beta_r(warmup_end) == 10.0


# =============================================================================
# 3. 1/√t LEARNING RATE DECAY
# =============================================================================

class TestLearningRateDecay:
    """Test 1/sqrt(t) learning rate decay after warmup."""
    
    def test_constant_lr_during_warmup(self):
        """Learning rate should be constant during warmup and ramp-up."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            beta_r_rampup_steps=2000,
            use_sqrt_lr_decay=True,
            lr_q_r=1e-4,
        )
        
        full_warmup_end = config._warmup_v_r_end + config.beta_r_rampup_steps
        
        # All steps during warmup and ramp-up should have base LR
        for step in [0, 1000, 2000, 3000, 4000, full_warmup_end - 1]:
            lr = config.get_learning_rate('q_r', step, step)
            assert lr == config.lr_q_r, f"LR should be constant at step {step}"
    
    def test_sqrt_decay_after_warmup(self):
        """Learning rate should decay as 1/sqrt(step) after full warmup.
        
        The decay uses lr = base_lr * sqrt(decay_start_step / step) to ensure
        continuity at decay_start_step while decaying proportionally to 1/sqrt(step).
        """
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            beta_r_rampup_steps=2000,
            use_sqrt_lr_decay=True,
            lr_constant_fraction=0.0,  # Start decay immediately after warmup
            constant_lr_then_1_over_t=False,  # Use 1/sqrt(t) not 1/t
            lr_q_r=1e-4,
        )
        
        full_warmup_end = config._warmup_v_r_end + config.beta_r_rampup_steps
        
        # Get LRs at different steps after warmup
        step1 = full_warmup_end + 1000
        step2 = full_warmup_end + 5000
        lr1 = config.get_learning_rate('q_r', step1, 0)
        lr2 = config.get_learning_rate('q_r', step2, 0)
        
        # Verify ratio follows 1/sqrt(step): lr2/lr1 = sqrt(step1/step2)
        expected_ratio = math.sqrt(step1 / step2)
        actual_ratio = lr2 / lr1
        
        assert abs(expected_ratio - actual_ratio) < 0.001
    
    def test_no_decay_when_disabled(self):
        """Learning rate should be constant when sqrt decay is disabled."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
            beta_r_rampup_steps=100,
            use_sqrt_lr_decay=False,
            lr_q_r=1e-4,
        )
        
        full_warmup_end = config._warmup_v_r_end + config.beta_r_rampup_steps
        
        lr_before = config.get_learning_rate('q_r', full_warmup_end + 100, 0)
        lr_after = config.get_learning_rate('q_r', full_warmup_end + 10000, 0)
        
        assert lr_before == lr_after == config.lr_q_r

    def test_lr_constant_fraction(self):
        """Learning rate should stay constant until lr_constant_fraction of total steps."""
        config = Phase2Config(
            num_training_steps=100000,
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
            beta_r_rampup_steps=100,
            use_sqrt_lr_decay=True,
            lr_constant_fraction=0.8,  # Decay starts at step 80000
            lr_q_r=1e-4,
        )
        
        full_warmup_end = config._warmup_v_r_end + config.beta_r_rampup_steps
        decay_start = int(0.8 * 100000)  # 80000
        
        # Before decay_start: should be constant at base LR
        lr_early = config.get_learning_rate('q_r', full_warmup_end + 1000, 0)
        lr_mid = config.get_learning_rate('q_r', 50000, 0)
        lr_just_before = config.get_learning_rate('q_r', decay_start - 1, 0)
        
        assert lr_early == config.lr_q_r, f"Expected constant LR early, got {lr_early}"
        assert lr_mid == config.lr_q_r, f"Expected constant LR mid, got {lr_mid}"
        assert lr_just_before == config.lr_q_r, f"Expected constant LR just before decay, got {lr_just_before}"
        
        # At and after decay_start: should start decaying
        lr_at_start = config.get_learning_rate('q_r', decay_start, 0)
        lr_late = config.get_learning_rate('q_r', 90000, 0)
        
        assert lr_at_start == config.lr_q_r, f"Expected base LR at decay start (t=1), got {lr_at_start}"
        assert lr_late < config.lr_q_r, f"Expected decayed LR late, got {lr_late}"

    def test_constant_lr_then_1_over_t(self):
        """Learning rate should decay as 1/step when constant_lr_then_1_over_t is True.
        
        The decay uses lr = base_lr * decay_start_step / step to ensure
        continuity at decay_start_step while decaying proportionally to 1/step.
        """
        config = Phase2Config(
            num_training_steps=100000,
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=100,
            warmup_u_r_steps=0,
            warmup_q_r_steps=100,
            beta_r_rampup_steps=100,
            use_sqrt_lr_decay=True,
            lr_constant_fraction=0.5,  # Decay starts at step 50000
            constant_lr_then_1_over_t=True,  # Use 1/step instead of 1/sqrt(step)
            lr_q_r=1e-3,
        )
        
        decay_start = 50000
        
        # At decay_start: lr = base_lr (continuous with constant phase)
        lr_at_start = config.get_learning_rate('q_r', decay_start, 0)
        assert abs(lr_at_start - config.lr_q_r) < 1e-10, f"At decay_start, expected {config.lr_q_r}, got {lr_at_start}"
        
        # At step 100000: lr = base_lr * 50000 / 100000 = base_lr / 2
        lr_at_100k = config.get_learning_rate('q_r', 100000, 0)
        expected_at_100k = config.lr_q_r / 2
        assert abs(lr_at_100k - expected_at_100k) < 1e-10, f"At step 100000, expected {expected_at_100k}, got {lr_at_100k}"
        
        # Verify 1/step ratio (not 1/sqrt(step))
        # For 1/step: lr(step2)/lr(step1) = step1/step2
        # For 1/sqrt(step): lr(step2)/lr(step1) = sqrt(step1)/sqrt(step2)
        step1, step2 = 60000, 80000
        lr1 = config.get_learning_rate('q_r', step1, 0)
        lr2 = config.get_learning_rate('q_r', step2, 0)
        
        actual_ratio = lr2 / lr1
        expected_1_over_step_ratio = step1 / step2  # = 0.75
        expected_sqrt_ratio = math.sqrt(step1 / step2)  # ≈ 0.866
        
        assert abs(actual_ratio - expected_1_over_step_ratio) < 0.001, \
            f"Expected 1/step ratio {expected_1_over_step_ratio}, got {actual_ratio}"


# =============================================================================
# 4. EPSILON-GREEDY EXPLORATION
# =============================================================================

class TestEpsilonSchedule:
    """Test epsilon-greedy exploration schedule."""
    
    def test_epsilon_r_linear_decay(self):
        """Robot epsilon should decay linearly."""
        config = Phase2Config(
            epsilon_r_start=1.0,
            epsilon_r_end=0.01,
            epsilon_r_decay_steps=10000,
        )
        
        assert config.get_epsilon_r(0) == 1.0
        assert abs(config.get_epsilon_r(5000) - 0.505) < 0.01
        assert config.get_epsilon_r(10000) == 0.01
    
    def test_epsilon_r_stays_at_end_value(self):
        """Robot epsilon should stay at end value after decay completes."""
        config = Phase2Config(
            epsilon_r_start=1.0,
            epsilon_r_end=0.01,
            epsilon_r_decay_steps=10000,
        )
        
        # All steps after decay should be at end value
        for step in [10000, 15000, 100000]:
            assert config.get_epsilon_r(step) == 0.01
    
    def test_epsilon_r_non_zero_end(self):
        """Robot epsilon end value should be non-zero (prevents deterministic policy)."""
        config = Phase2Config()
        assert config.epsilon_r_end > 0
    
    def test_epsilon_h_linear_decay(self):
        """Human epsilon should decay linearly."""
        config = Phase2Config(
            epsilon_h_start=1.0,
            epsilon_h_end=0.02,
            epsilon_h_decay_steps=5000,
        )
        
        assert config.get_epsilon_h(0) == 1.0
        assert abs(config.get_epsilon_h(2500) - 0.51) < 0.01
        assert config.get_epsilon_h(5000) == 0.02
    
    def test_epsilon_h_default_matches_epsilon_r(self):
        """Human epsilon defaults should match robot epsilon defaults."""
        config = Phase2Config()
        assert config.epsilon_h_start == config.epsilon_r_start
        assert config.epsilon_h_end == config.epsilon_r_end
        assert config.epsilon_h_decay_steps == config.epsilon_r_decay_steps


# =============================================================================
# 5. AUTO-SCALED GRADIENT CLIPPING
# =============================================================================

class TestGradientClipping:
    """Test auto-scaled gradient clipping proportional to learning rate."""
    
    def test_clip_at_reference_lr(self):
        """At reference LR, effective clip should equal base clip."""
        config = Phase2Config(
            q_r_grad_clip=1.0,
            auto_grad_clip=True,
            auto_grad_clip_reference_lr=1e-4,
        )
        
        clip = config.get_effective_grad_clip('q_r', 1e-4)
        assert clip == 1.0
    
    def test_clip_scales_with_lr(self):
        """Effective clip should scale proportionally to LR."""
        config = Phase2Config(
            q_r_grad_clip=1.0,
            auto_grad_clip=True,
            auto_grad_clip_reference_lr=1e-4,
        )
        
        # 10x higher LR -> 10x higher clip
        assert config.get_effective_grad_clip('q_r', 1e-3) == 10.0
        
        # 10x lower LR -> 10x lower clip
        assert config.get_effective_grad_clip('q_r', 1e-5) == 0.1
    
    def test_clip_disabled_when_none(self):
        """Clipping should be disabled when set to None."""
        config = Phase2Config(q_r_grad_clip=None)
        assert config.get_effective_grad_clip('q_r', 1e-4) is None
    
    def test_no_scaling_when_auto_disabled(self):
        """When auto_grad_clip=False, use base clip directly."""
        config = Phase2Config(
            q_r_grad_clip=1.0,
            auto_grad_clip=False,
        )
        
        # Should always return base clip regardless of LR
        assert config.get_effective_grad_clip('q_r', 1e-3) == 1.0
        assert config.get_effective_grad_clip('q_r', 1e-5) == 1.0


# =============================================================================
# 6. POWER-LAW SOFTMAX POLICY (-Q)^{-β}
# =============================================================================

class TestPowerLawPolicy:
    """Test power-law softmax policy: π ∝ (-Q)^{-β}."""
    
    def _get_policy(self, q_values, beta_r):
        """Compute power-law policy: π ∝ (-Q)^{-β}."""
        if beta_r == 0.0:
            uniform = torch.ones_like(q_values)
            return uniform / uniform.sum(dim=-1, keepdim=True)
        
        neg_q = -q_values
        neg_q = torch.clamp(neg_q, min=1e-10)
        
        log_unnormalized = -beta_r * torch.log(neg_q)
        log_unnormalized = log_unnormalized - log_unnormalized.max(dim=-1, keepdim=True)[0]
        unnormalized = torch.exp(log_unnormalized)
        policy = unnormalized / unnormalized.sum(dim=-1, keepdim=True)
        return policy
    
    def test_uniform_at_beta_zero(self):
        """With β=0, policy should be uniform."""
        q_values = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        policy = self._get_policy(q_values, 0.0)
        
        expected = torch.ones_like(policy) / 4
        assert torch.allclose(policy, expected, atol=0.001)
    
    def test_concentrated_at_high_beta(self):
        """With high β, policy should concentrate on best (least negative) action."""
        q_values = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        policy = self._get_policy(q_values, 100.0)
        
        # First action (Q=-1) is best
        assert policy[0, 0] > 0.99
    
    def test_monotonic_concentration(self):
        """Higher β should give more concentrated policy."""
        q_values = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        
        policy_low = self._get_policy(q_values, 1.0)
        policy_mid = self._get_policy(q_values, 5.0)
        policy_high = self._get_policy(q_values, 10.0)
        
        # Best action probability should increase with beta
        assert policy_low[0, 0] < policy_mid[0, 0] < policy_high[0, 0]


# =============================================================================
# 7. Q_r USES -SOFTPLUS FOR NEGATIVE Q VALUES
# =============================================================================

class TestNegativeSoftplus:
    """Test that -softplus ensures Q values are always negative."""
    
    def test_all_outputs_negative(self):
        """All outputs of -softplus should be negative."""
        raw_values = torch.tensor([[-10.0, -1.0, 0.0, 1.0, 10.0, 100.0]])
        q_values = -F.softplus(raw_values)
        
        assert (q_values < 0).all()
    
    def test_maps_real_to_negative(self):
        """-softplus should map ℝ → (-∞, 0)."""
        # Test reasonable range values (extreme values may underflow)
        raw_values = torch.tensor([[-100.0, 100.0]])
        q_values = -F.softplus(raw_values)
        
        # Both should be negative
        assert q_values[0, 0] < 0
        assert q_values[0, 1] < 0
        
        # Large positive raw value -> large negative Q
        assert q_values[0, 1] < -99


# =============================================================================
# 8. TARGET NETWORK MANAGEMENT (tested in trainer integration tests)
# =============================================================================

# Note: Target network tests require trainer instantiation
# See test_phase2_trainer_integration.py for full tests


# =============================================================================
# 9. REPLAY BUFFER CLEARING (tested in trainer integration tests)
# =============================================================================

# Note: Buffer clearing tests require trainer instantiation
# See test_phase2_trainer_integration.py for full tests


# =============================================================================
# 10. SEPARATE X_H BATCH SIZE
# =============================================================================

class TestXhBatchSize:
    """Test separate larger batch size for X_h network."""
    
    def test_x_h_batch_size_configurable(self):
        """X_h batch size should be separately configurable."""
        config = Phase2Config(
            batch_size=64,
            x_h_batch_size=128,
        )
        
        assert config.batch_size == 64
        assert config.x_h_batch_size == 128
    
    def test_x_h_batch_defaults_to_none(self):
        """When None, trainer should use regular batch_size."""
        config = Phase2Config(batch_size=64, x_h_batch_size=None)
        assert config.x_h_batch_size is None


# =============================================================================
# 11-12. WEIGHT DECAY AND GRADIENT CLIPPING
# =============================================================================

class TestRegularization:
    """Test weight decay and gradient clipping configuration."""
    
    def test_weight_decay_all_networks(self):
        """All networks should have configurable weight decay."""
        config = Phase2Config()
        
        for net in ['q_r', 'v_r', 'v_h_e', 'x_h', 'u_r']:
            wd = getattr(config, f'{net}_weight_decay')
            assert wd > 0, f"{net} should have non-zero weight decay"
    
    def test_grad_clip_all_networks(self):
        """All networks should have configurable gradient clipping."""
        config = Phase2Config()
        
        for net in ['q_r', 'v_r', 'v_h_e', 'x_h', 'u_r']:
            clip = getattr(config, f'{net}_grad_clip')
            assert clip is not None and clip > 0


# =============================================================================
# 13. MODEL-BASED TARGETS
# =============================================================================

class TestModelBasedTargets:
    """Test model-based targets configuration."""
    
    def test_model_based_targets_default_true(self):
        """Model-based targets should be enabled by default."""
        config = Phase2Config()
        assert config.use_model_based_targets is True
    
    def test_model_based_targets_configurable(self):
        """Model-based targets should be toggleable."""
        config_on = Phase2Config(use_model_based_targets=True)
        config_off = Phase2Config(use_model_based_targets=False)
        
        assert config_on.use_model_based_targets is True
        assert config_off.use_model_based_targets is False


# =============================================================================
# 14. ASYNC ACTOR-LEARNER ARCHITECTURE
# =============================================================================

class TestAsyncConfig:
    """Test async actor-learner configuration."""
    
    def test_async_config_defaults(self):
        """Async config should have sensible defaults."""
        config = Phase2Config()
        
        assert config.async_training is False  # Off by default
        assert config.num_actors >= 1
        assert config.actor_sync_freq > 0
        assert config.async_min_buffer_size > 0
        assert config.async_queue_size > 0
    
    def test_async_config_customizable(self):
        """All async parameters should be customizable."""
        config = Phase2Config(
            async_training=True,
            num_actors=8,
            actor_sync_freq=50,
            async_min_buffer_size=500,
            async_queue_size=5000,
        )
        
        assert config.async_training is True
        assert config.num_actors == 8
        assert config.actor_sync_freq == 50
        assert config.async_min_buffer_size == 500
        assert config.async_queue_size == 5000


# =============================================================================
# 15. GOAL RESAMPLING
# =============================================================================

class TestGoalResampling:
    """Test goal resampling probability configuration."""
    
    def test_goal_resample_prob_default(self):
        """Goal resampling should have a small default probability."""
        config = Phase2Config()
        assert 0 < config.goal_resample_prob < 0.1
    
    def test_goal_resample_prob_configurable(self):
        """Goal resampling probability should be configurable."""
        config = Phase2Config(goal_resample_prob=0.05)
        assert config.goal_resample_prob == 0.05


# =============================================================================
# 16. DROPOUT CONFIGURATION
# =============================================================================

class TestDropoutConfig:
    """Test dropout configuration for all networks."""
    
    def test_dropout_all_networks(self):
        """All networks should have configurable dropout."""
        config = Phase2Config()
        
        for net in ['q_r', 'v_r', 'v_h_e', 'x_h', 'u_r']:
            dropout = getattr(config, f'{net}_dropout')
            assert 0 <= dropout <= 1
    
    def test_default_dropout_high(self):
        """Default dropout should be high (0.5) for regularization."""
        config = Phase2Config()
        assert config.q_r_dropout == 0.5
        assert config.v_h_e_dropout == 0.5


# =============================================================================
# 17. DIRECT U_R / V_R / X_H COMPUTATION MODES
# =============================================================================

class TestDirectComputationModes:
    """Test direct U_r, V_r, and X_h computation modes."""
    
    def test_u_r_use_network_default_false(self):
        """U_r should use direct computation by default."""
        config = Phase2Config()
        assert config.u_r_use_network is False
    
    def test_v_r_use_network_default_false(self):
        """V_r should use direct computation by default."""
        config = Phase2Config()
        assert config.v_r_use_network is False
    
    def test_x_h_use_network_default_true(self):
        """X_h should use network by default (unlike U_r and V_r)."""
        config = Phase2Config()
        assert config.x_h_use_network is True
    
    def test_x_h_use_network_configurable(self):
        """X_h computation mode should be configurable."""
        config_with_network = Phase2Config(x_h_use_network=True)
        config_no_network = Phase2Config(x_h_use_network=False)
        
        assert config_with_network.x_h_use_network is True
        assert config_no_network.x_h_use_network is False
    
    def test_x_h_use_network_false_sets_warmup_to_zero(self):
        """When x_h_use_network=False, warmup_x_h_steps should be set to 0."""
        config = Phase2Config(
            warmup_x_h_steps=1000,  # Should be overridden to 0
            x_h_use_network=False,
        )
        assert config.warmup_x_h_steps == 0
    
    def test_network_modes_configurable(self):
        """Network computation modes should be configurable."""
        config = Phase2Config(u_r_use_network=True, v_r_use_network=True)
        assert config.u_r_use_network is True
        assert config.v_r_use_network is True
    
    def test_warmup_stages_with_x_h_use_network_false(self):
        """Test warmup stages when X_h is computed directly (no network)."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=1000,  # Should be overridden to 0
            warmup_u_r_steps=0,     # U_r not using network
            warmup_q_r_steps=1000,
            x_h_use_network=False,
            u_r_use_network=False,
        )
        
        # X_h warmup should be automatically set to 0
        assert config.warmup_x_h_steps == 0
        
        # Stage progression skips X_h stage
        assert config.get_active_networks(0) == {'v_h_e'}          # Stage 0: V_h^e only
        assert config.get_active_networks(1500) == {'v_h_e', 'q_r'}  # Stage goes directly to Q_r
        # X_h should NOT be in active networks when x_h_use_network=False
        assert 'x_h' not in config.get_active_networks(1500)
    
    def test_warmup_stage_name_without_x_h_network(self):
        """Test stage names reflect skipped X_h stage."""
        config = Phase2Config(
            warmup_v_h_e_steps=100,
            x_h_use_network=False,
            u_r_use_network=False,
        )
        
        # After V_h^e stage, should go directly to Q_r
        stage_name = config.get_warmup_stage_name(150)
        assert 'Q_r' in stage_name or 'V_h^e' in stage_name
    
    def test_combined_x_h_and_u_r_false(self):
        """Test both X_h and U_r computed directly."""
        config = Phase2Config(
            warmup_v_h_e_steps=1000,
            warmup_q_r_steps=1000,
            x_h_use_network=False,
            u_r_use_network=False,
        )
        
        # Both warmup stages should be 0
        assert config.warmup_x_h_steps == 0
        assert config.warmup_u_r_steps == 0
        
        # Active networks at various stages
        assert config.get_active_networks(0) == {'v_h_e'}
        # After v_h_e warmup, Q_r starts immediately
        assert config.get_active_networks(1500) == {'v_h_e', 'q_r'}
        # Neither X_h nor U_r should be active
        assert 'x_h' not in config.get_active_networks(1500)
        assert 'u_r' not in config.get_active_networks(1500)


# =============================================================================
# 18. TENSORBOARD LOGGING (tested in trainer integration tests)
# =============================================================================

# Note: TensorBoard logging tests require trainer instantiation
# See test_phase2_trainer_integration.py for full tests


# =============================================================================
# 19. INCLUDE STEP COUNT IN STATE
# =============================================================================

class TestIncludeStepCount:
    """Test include_step_count configuration."""
    
    def test_include_step_count_default_true(self):
        """Step count should be included by default."""
        config = Phase2Config()
        assert config.include_step_count is True
    
    def test_include_step_count_configurable(self):
        """Step count inclusion should be toggleable."""
        config_on = Phase2Config(include_step_count=True)
        config_off = Phase2Config(include_step_count=False)
        
        assert config_on.include_step_count is True
        assert config_off.include_step_count is False


# =============================================================================
# 20. TRAINING TERMINOLOGY
# =============================================================================

class TestTrainingTerminology:
    """Test that training terminology is consistent with FAQ glossary."""
    
    def test_num_training_steps_exists(self):
        """num_training_steps should be the main duration parameter."""
        config = Phase2Config(num_training_steps=100000)
        assert config.num_training_steps == 100000
    
    def test_steps_per_episode_exists(self):
        """steps_per_episode should control env reset interval."""
        config = Phase2Config(steps_per_episode=50)
        assert config.steps_per_episode == 50
    
    def test_training_steps_per_env_step_exists(self):
        """training_steps_per_env_step should control update ratio."""
        config = Phase2Config(training_steps_per_env_step=2.0)
        assert config.training_steps_per_env_step == 2.0
    
    def test_target_update_intervals_in_training_steps(self):
        """Target update intervals should be in training steps."""
        config = Phase2Config(
            v_r_target_update_interval=100,
            v_h_e_target_update_interval=200,
            x_h_target_update_interval=300,
            u_r_target_update_interval=400,
        )
        
        assert config.v_r_target_update_interval == 100
        assert config.v_h_e_target_update_interval == 200
        assert config.x_h_target_update_interval == 300
        assert config.u_r_target_update_interval == 400


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
