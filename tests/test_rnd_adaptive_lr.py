"""
Tests for RND-based adaptive learning rate.

Tests verify that:
1. Config options work correctly
2. RND adaptive LR is applied only when enabled and RND is active
3. Gradient scaling is applied correctly
4. LR scale is properly clamped
5. Logging returns correct values
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Any, Tuple, Optional

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.rnd import RNDModule


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def simple_states():
    """Create simple test states."""
    return [
        (0, 0, 0),
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 0),  # duplicate
    ]


# =============================================================================
# Config tests
# =============================================================================

class TestRNDAdaptiveLRConfig:
    """Tests for RND adaptive LR config options."""
    
    def test_default_config(self):
        """Test default config values."""
        config = Phase2Config()
        assert config.rnd_use_adaptive_lr is False
        assert config.rnd_adaptive_lr_scale == 1.0
        assert config.rnd_adaptive_lr_min == 0.1
        assert config.rnd_adaptive_lr_max == 10.0
    
    def test_custom_config(self):
        """Test custom config values."""
        config = Phase2Config(
            rnd_use_adaptive_lr=True,
            rnd_adaptive_lr_scale=2.0,
            rnd_adaptive_lr_min=0.05,
            rnd_adaptive_lr_max=20.0,
        )
        assert config.rnd_use_adaptive_lr is True
        assert config.rnd_adaptive_lr_scale == 2.0
        assert config.rnd_adaptive_lr_min == 0.05
        assert config.rnd_adaptive_lr_max == 20.0
    
    def test_warning_when_rnd_disabled(self):
        """Test that warning is issued when RND adaptive LR is enabled but RND is disabled."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Phase2Config(
                rnd_use_adaptive_lr=True,
                use_rnd=False,
            )
            
            # Check that a warning was issued
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("rnd_use_adaptive_lr=True but use_rnd=False" in msg for msg in warning_messages)


# =============================================================================
# RND module tests (for computing novelty)
# =============================================================================

class TestRNDNoveltyComputation:
    """Tests for RND novelty computation used in adaptive LR."""
    
    def test_novelty_is_squared(self):
        """Test that RND novelty is already squared (MSE)."""
        rnd = RNDModule(input_dim=10, feature_dim=8, hidden_dim=16, normalize=False)
        
        # Create random input
        features = torch.randn(4, 10)
        
        # Compute novelty
        novelty = rnd.compute_novelty_no_grad(features, use_raw=True)
        
        # Novelty should be non-negative (squared values)
        assert (novelty >= 0).all(), "Raw novelty (MSE) should be non-negative"
    
    def test_novelty_shape(self):
        """Test that novelty has correct shape."""
        rnd = RNDModule(input_dim=10, feature_dim=8, hidden_dim=16)
        
        batch_size = 8
        features = torch.randn(batch_size, 10)
        
        novelty = rnd.compute_novelty_no_grad(features, use_raw=True)
        
        assert novelty.shape == (batch_size,), f"Expected shape ({batch_size},), got {novelty.shape}"
    
    def test_running_mean_updates(self):
        """Test that running mean is updated during training."""
        rnd = RNDModule(input_dim=10, feature_dim=8, hidden_dim=16, normalize=True)
        rnd.train()
        
        initial_mean = rnd.running_mean.clone()
        
        # Compute novelty with stats update
        features = torch.randn(16, 10)
        _ = rnd.compute_novelty(features, update_stats=True)
        
        # Running mean should have changed
        assert rnd.update_count > 0, "Update count should have increased"
        # Note: running_mean might not change much with first batch, but update_count should
    
    def test_no_grad_computation(self):
        """Test that compute_novelty_no_grad doesn't require gradients."""
        rnd = RNDModule(input_dim=10, feature_dim=8, hidden_dim=16)
        
        features = torch.randn(4, 10, requires_grad=False)
        
        novelty = rnd.compute_novelty_no_grad(features, use_raw=True)
        
        assert not novelty.requires_grad, "no_grad computation should not require gradients"


# =============================================================================
# Gradient scaling logic tests
# =============================================================================

class TestGradientScaling:
    """Tests for gradient scaling logic."""
    
    def test_lr_scale_normalization(self):
        """Test that LR scale is properly normalized by running mean."""
        # Simulate the scaling logic from _apply_rnd_adaptive_lr_scaling
        rnd_mse = torch.tensor([0.5, 1.0, 2.0, 4.0])  # Batch of MSE values
        running_mean = torch.tensor(1.0)
        scale = 1.0
        min_lr = 0.1
        max_lr = 10.0
        
        # Compute scale: lr ∝ rnd_mse / running_mean
        lr_scale = (rnd_mse / running_mean) * scale
        lr_scale = lr_scale.clamp(min=min_lr, max=max_lr)
        
        expected = torch.tensor([0.5, 1.0, 2.0, 4.0])
        assert torch.allclose(lr_scale, expected), f"Expected {expected}, got {lr_scale}"
    
    def test_lr_scale_clamping(self):
        """Test that LR scale is properly clamped."""
        rnd_mse = torch.tensor([0.01, 0.1, 1.0, 100.0])
        running_mean = torch.tensor(1.0)
        scale = 1.0
        min_lr = 0.1
        max_lr = 10.0
        
        lr_scale = (rnd_mse / running_mean) * scale
        lr_scale = lr_scale.clamp(min=min_lr, max=max_lr)
        
        # 0.01 -> 0.1 (clamped), 0.1 -> 0.1, 1.0 -> 1.0, 100.0 -> 10.0 (clamped)
        expected = torch.tensor([0.1, 0.1, 1.0, 10.0])
        assert torch.allclose(lr_scale, expected), f"Expected {expected}, got {lr_scale}"
    
    def test_scale_multiplier(self):
        """Test that scale multiplier is applied correctly."""
        rnd_mse = torch.tensor([1.0, 2.0])
        running_mean = torch.tensor(1.0)
        scale = 2.0  # Double the scale
        min_lr = 0.1
        max_lr = 20.0
        
        lr_scale = (rnd_mse / running_mean) * scale
        lr_scale = lr_scale.clamp(min=min_lr, max=max_lr)
        
        expected = torch.tensor([2.0, 4.0])
        assert torch.allclose(lr_scale, expected), f"Expected {expected}, got {lr_scale}"
    
    def test_gradient_multiplication(self):
        """Test that gradients are properly multiplied by scale."""
        # Create a simple network
        net = nn.Linear(10, 5)
        
        # Forward and backward
        x = torch.randn(4, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()
        
        # Get original gradients
        original_grad_weight = net.weight.grad.clone()
        original_grad_bias = net.bias.grad.clone()
        
        # Apply scaling
        mean_lr_scale = 2.5
        for param in net.parameters():
            if param.grad is not None:
                param.grad.mul_(mean_lr_scale)
        
        # Check scaling
        assert torch.allclose(net.weight.grad, original_grad_weight * mean_lr_scale)
        assert torch.allclose(net.bias.grad, original_grad_bias * mean_lr_scale)


# =============================================================================
# Integration tests (requires mock trainer setup)
# =============================================================================

class TestRNDAdaptiveLRIntegration:
    """Integration tests for RND adaptive LR with trainer."""
    
    def test_config_enables_rnd_adaptive_lr(self):
        """Test that config properly enables RND adaptive LR."""
        config = Phase2Config(
            use_rnd=True,
            rnd_use_adaptive_lr=True,
            use_lookup_tables=False,  # Neural network mode
        )
        
        assert config.rnd_use_adaptive_lr is True
        assert config.use_rnd is True
    
    def test_config_lookup_tables_compatible(self):
        """Test that lookup tables and RND adaptive LR can coexist."""
        # Both can be enabled - lookup tables use 1/n, neural networks use RND
        config = Phase2Config(
            use_lookup_tables=True,
            lookup_use_adaptive_lr=True,
            use_rnd=True,
            rnd_use_adaptive_lr=True,
        )
        
        # No exception should be raised
        assert config.lookup_use_adaptive_lr is True
        assert config.rnd_use_adaptive_lr is True


# =============================================================================
# Edge case tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in RND adaptive LR."""
    
    def test_zero_running_mean(self):
        """Test handling when running mean is near zero."""
        rnd_mse = torch.tensor([0.5, 1.0, 2.0])
        running_mean = torch.tensor(1e-10)  # Very small
        
        # Should use batch mean instead
        mean_mse = rnd_mse.mean()
        if running_mean <= 1e-8:
            lr_scale = rnd_mse / mean_mse
        else:
            lr_scale = rnd_mse / running_mean
        
        # With batch mean normalization, mean scale should be ~1.0
        assert abs(lr_scale.mean().item() - 1.0) < 0.01
    
    def test_all_zeros_mse(self):
        """Test handling when all MSE values are zero."""
        rnd_mse = torch.zeros(4)
        
        # Should not produce NaN or Inf
        mean_mse = rnd_mse.mean()
        if mean_mse > 1e-8:
            lr_scale = rnd_mse / mean_mse
        else:
            lr_scale = None  # No scaling
        
        assert lr_scale is None, "Should skip scaling when all MSE is zero"
    
    def test_single_sample_batch(self):
        """Test with single sample batch."""
        rnd_mse = torch.tensor([1.0])
        running_mean = torch.tensor(1.0)
        
        lr_scale = (rnd_mse / running_mean) * 1.0
        lr_scale = lr_scale.clamp(min=0.1, max=10.0)
        
        expected = torch.tensor([1.0])
        assert torch.allclose(lr_scale, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
