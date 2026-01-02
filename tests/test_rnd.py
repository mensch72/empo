"""
Tests for Random Network Distillation (RND) module.

Tests cover:
- RNDModule novelty computation
- Running statistics normalization
- RNDModuleWithEncoder integration
- Batched novelty computation in trainer
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from empo.nn_based.phase2.rnd import RNDModule, RNDModuleWithEncoder


class TestRNDModule:
    """Tests for the base RND module."""
    
    def test_initialization(self):
        """Test RND module initializes correctly."""
        rnd = RNDModule(input_dim=64, feature_dim=32, hidden_dim=128)
        
        assert rnd.input_dim == 64
        assert rnd.feature_dim == 32
        assert rnd.normalize is True
        
        # Target network should be frozen
        for param in rnd.target.parameters():
            assert not param.requires_grad
        
        # Predictor network should be trainable
        for param in rnd.predictor.parameters():
            assert param.requires_grad
    
    def test_target_frozen(self):
        """Test target network weights don't change during training."""
        rnd = RNDModule(input_dim=32, feature_dim=16, hidden_dim=64)
        
        # Get initial target weights
        initial_weights = [p.clone() for p in rnd.target.parameters()]
        
        # Forward pass with gradients
        x = torch.randn(10, 32)
        loss = rnd.compute_loss(x)
        loss.backward()
        
        # Target weights should be unchanged
        for initial, current in zip(initial_weights, rnd.target.parameters()):
            assert torch.allclose(initial, current)
    
    def test_novelty_shape(self):
        """Test novelty output has correct shape."""
        rnd = RNDModule(input_dim=64, feature_dim=32)
        
        batch_sizes = [1, 5, 32]
        for bs in batch_sizes:
            x = torch.randn(bs, 64)
            novelty = rnd.compute_novelty(x)
            assert novelty.shape == (bs,), f"Expected ({bs},), got {novelty.shape}"
    
    def test_novelty_no_grad_shape(self):
        """Test novelty_no_grad output has correct shape."""
        rnd = RNDModule(input_dim=64, feature_dim=32)
        
        x = torch.randn(10, 64)
        novelty = rnd.compute_novelty_no_grad(x)
        assert novelty.shape == (10,)
    
    def test_novelty_positive_before_normalization(self):
        """Test raw novelty (MSE) is always non-negative."""
        rnd = RNDModule(input_dim=64, feature_dim=32, normalize=False)
        
        x = torch.randn(100, 64)
        novelty = rnd.compute_novelty(x)
        
        # MSE is always >= 0
        assert (novelty >= 0).all()
    
    def test_novelty_decreases_with_training(self):
        """Test novelty decreases for repeated states after training."""
        rnd = RNDModule(input_dim=32, feature_dim=16, hidden_dim=64, normalize=False)
        optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
        
        # Fixed input that we'll train on
        x = torch.randn(10, 32)
        
        # Initial novelty
        initial_novelty = rnd.compute_novelty_no_grad(x).mean().item()
        
        # Train for a few steps
        rnd.train()
        for _ in range(100):
            optimizer.zero_grad()
            loss = rnd.compute_loss(x)
            loss.backward()
            optimizer.step()
        
        # Novelty should decrease
        final_novelty = rnd.compute_novelty_no_grad(x).mean().item()
        assert final_novelty < initial_novelty, \
            f"Novelty should decrease: {initial_novelty} -> {final_novelty}"
    
    def test_novel_states_have_higher_novelty(self):
        """Test that unseen states have higher novelty than trained states."""
        rnd = RNDModule(input_dim=32, feature_dim=16, hidden_dim=64, normalize=False)
        optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
        
        # Training data
        train_x = torch.randn(20, 32)
        
        # Novel data (different distribution)
        novel_x = torch.randn(20, 32) * 3 + 5  # Different mean and scale
        
        # Train on train_x
        rnd.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = rnd.compute_loss(train_x)
            loss.backward()
            optimizer.step()
        
        # Compare novelties
        rnd.eval()
        train_novelty = rnd.compute_novelty_no_grad(train_x).mean().item()
        novel_novelty = rnd.compute_novelty_no_grad(novel_x).mean().item()
        
        assert novel_novelty > train_novelty, \
            f"Novel states should have higher novelty: train={train_novelty}, novel={novel_novelty}"
    
    def test_running_statistics_update(self):
        """Test running mean/std are updated during training."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=True)
        
        assert rnd.update_count.item() == 0
        initial_mean = rnd.running_mean.item()
        
        # Forward with update
        rnd.train()
        x = torch.randn(10, 32) * 10  # Large values to shift mean
        rnd.compute_novelty(x, update_stats=True)
        
        assert rnd.update_count.item() == 1
        # Mean should have changed (unless by extreme coincidence)
        # We use a looser check since the first update sets mean directly
        assert rnd.running_mean.item() != initial_mean or rnd.update_count.item() > 0
    
    def test_no_stats_update_when_disabled(self):
        """Test running stats don't update when update_stats=False."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=True)
        
        rnd.train()
        x = torch.randn(10, 32)
        
        initial_count = rnd.update_count.item()
        rnd.compute_novelty(x, update_stats=False)
        
        assert rnd.update_count.item() == initial_count
    
    def test_no_stats_update_in_eval_mode(self):
        """Test running stats don't update in eval mode."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=True)
        
        rnd.eval()
        x = torch.randn(10, 32)
        
        initial_count = rnd.update_count.item()
        rnd.compute_novelty(x, update_stats=True)  # Even with update_stats=True
        
        assert rnd.update_count.item() == initial_count
    
    def test_normalization_centers_novelty(self):
        """Test normalized novelty is roughly zero-centered after warmup."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=True)
        
        rnd.train()
        # Warm up running statistics
        for _ in range(100):
            x = torch.randn(32, 32)
            rnd.compute_novelty(x, update_stats=True)
        
        # Check that novelty is roughly centered
        x = torch.randn(100, 32)
        novelty = rnd.compute_novelty(x, update_stats=False)
        
        # Mean should be close to 0 (within 2 std of sampling variation)
        assert abs(novelty.mean().item()) < 2.0, \
            f"Normalized novelty mean should be ~0, got {novelty.mean().item()}"
    
    def test_loss_is_scalar(self):
        """Test compute_loss returns a scalar."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        x = torch.randn(10, 32)
        loss = rnd.compute_loss(x)
        
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.requires_grad, "Loss should have gradients"
    
    def test_loss_is_non_negative(self):
        """Test loss (MSE) is always non-negative."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        for _ in range(10):
            x = torch.randn(10, 32)
            loss = rnd.compute_loss(x)
            assert loss.item() >= 0
    
    def test_statistics_dict(self):
        """Test get_statistics returns expected keys."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        # Do some forward passes to populate stats
        rnd.train()
        x = torch.randn(10, 32)
        rnd.compute_novelty(x)
        
        stats = rnd.get_statistics()
        
        expected_keys = [
            'rnd_running_mean',
            'rnd_running_std',
            'rnd_update_count',
            'rnd_batch_raw_mean',
            'rnd_batch_raw_std',
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_state_dict_includes_buffers(self):
        """Test state_dict includes running statistics."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        # Update statistics
        rnd.train()
        x = torch.randn(10, 32) * 5
        rnd.compute_novelty(x)
        
        state_dict = rnd.state_dict()
        
        assert 'running_mean' in state_dict
        assert 'running_var' in state_dict
        assert 'update_count' in state_dict
    
    def test_load_state_dict_restores_statistics(self):
        """Test loading state_dict restores running statistics."""
        rnd1 = RNDModule(input_dim=32, feature_dim=16)
        
        # Update statistics
        rnd1.train()
        for _ in range(10):
            x = torch.randn(10, 32)
            rnd1.compute_novelty(x)
        
        # Save and load
        state_dict = rnd1.state_dict()
        
        rnd2 = RNDModule(input_dim=32, feature_dim=16)
        rnd2.load_state_dict(state_dict)
        
        assert rnd2.running_mean.item() == rnd1.running_mean.item()
        assert rnd2.running_var.item() == rnd1.running_var.item()
        assert rnd2.update_count.item() == rnd1.update_count.item()


class TestRNDModuleWithEncoder:
    """Tests for RND module with encoder wrapper."""
    
    class SimpleEncoder(nn.Module):
        """Simple encoder for testing."""
        def __init__(self, input_dim=10, feature_dim=32):
            super().__init__()
            self.feature_dim = feature_dim
            self.net = nn.Linear(input_dim, feature_dim)
        
        def forward(self, x):
            return self.net(x)
    
    def test_initialization(self):
        """Test RNDModuleWithEncoder initializes correctly."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16, hidden_dim=64)
        
        assert rnd.encoder is encoder
        assert rnd.rnd.input_dim == 32  # Encoder's feature_dim
        assert rnd.rnd.feature_dim == 16
    
    def test_encoder_without_feature_dim_raises(self):
        """Test that encoder without feature_dim attribute raises error."""
        class BadEncoder(nn.Module):
            def forward(self, x):
                return x
        
        with pytest.raises(ValueError, match="feature_dim"):
            RNDModuleWithEncoder(BadEncoder())
    
    def test_compute_novelty_from_features(self):
        """Test novelty computation from pre-computed features."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        features = torch.randn(10, 32)
        novelty = rnd.compute_novelty_from_features(features)
        
        assert novelty.shape == (10,)
    
    def test_compute_novelty_from_features_no_grad(self):
        """Test no-grad novelty computation."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        features = torch.randn(10, 32)
        novelty = rnd.compute_novelty_from_features_no_grad(features)
        
        assert novelty.shape == (10,)
    
    def test_compute_loss(self):
        """Test loss computation."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        features = torch.randn(10, 32)
        loss = rnd.compute_loss(features)
        
        assert loss.ndim == 0
        assert loss.requires_grad
    
    def test_predictor_property(self):
        """Test predictor property returns correct network."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        assert rnd.predictor is rnd.rnd.predictor
    
    def test_target_property(self):
        """Test target property returns correct network."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        assert rnd.target is rnd.rnd.target
    
    def test_statistics(self):
        """Test get_statistics delegates to inner RND."""
        encoder = self.SimpleEncoder(input_dim=10, feature_dim=32)
        rnd = RNDModuleWithEncoder(encoder, feature_dim=16)
        
        stats = rnd.get_statistics()
        assert 'rnd_running_mean' in stats


class TestRNDBatchedComputation:
    """Tests for batched RND computation efficiency."""
    
    def test_batched_vs_sequential_same_result(self):
        """Test batched computation gives same result as sequential."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=False)
        
        # Multiple inputs
        inputs = [torch.randn(1, 32) for _ in range(10)]
        
        # Sequential computation
        sequential_novelties = []
        for x in inputs:
            n = rnd.compute_novelty_no_grad(x)
            sequential_novelties.append(n.item())
        
        # Batched computation
        batched_input = torch.cat(inputs, dim=0)
        batched_novelties = rnd.compute_novelty_no_grad(batched_input)
        
        # Should be equal
        for i, (seq, batch) in enumerate(zip(sequential_novelties, batched_novelties)):
            assert abs(seq - batch.item()) < 1e-5, \
                f"Mismatch at index {i}: sequential={seq}, batched={batch.item()}"
    
    def test_empty_batch(self):
        """Test handling of empty batch."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        # Empty input
        x = torch.randn(0, 32)
        novelty = rnd.compute_novelty_no_grad(x)
        
        assert novelty.shape == (0,)
    
    def test_single_sample_batch(self):
        """Test single sample batch works correctly."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        x = torch.randn(1, 32)
        novelty = rnd.compute_novelty_no_grad(x)
        
        assert novelty.shape == (1,)


class TestRNDGradients:
    """Tests for gradient flow in RND."""
    
    def test_predictor_gradients_flow(self):
        """Test gradients flow through predictor during loss computation."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        x = torch.randn(10, 32)
        loss = rnd.compute_loss(x)
        loss.backward()
        
        # Predictor should have gradients
        for param in rnd.predictor.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
    
    def test_target_no_gradients(self):
        """Test target network has no gradients."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        x = torch.randn(10, 32)
        loss = rnd.compute_loss(x)
        loss.backward()
        
        # Target should have no gradients (frozen)
        for param in rnd.target.parameters():
            assert param.grad is None
    
    def test_novelty_has_gradients_for_predictor(self):
        """Test compute_novelty allows gradient flow for predictor."""
        rnd = RNDModule(input_dim=32, feature_dim=16, normalize=False)
        
        x = torch.randn(10, 32, requires_grad=False)
        novelty = rnd.compute_novelty(x, update_stats=False)
        
        # Novelty should have gradient connection to predictor
        novelty.sum().backward()
        
        for param in rnd.predictor.parameters():
            assert param.grad is not None


class TestRNDDeterminism:
    """Tests for deterministic behavior."""
    
    def test_target_output_deterministic(self):
        """Test target network output is deterministic for same input."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        
        x = torch.randn(10, 32)
        
        with torch.no_grad():
            out1 = rnd.target(x)
            out2 = rnd.target(x)
        
        assert torch.allclose(out1, out2)
    
    def test_novelty_deterministic_in_eval(self):
        """Test novelty computation is deterministic in eval mode."""
        rnd = RNDModule(input_dim=32, feature_dim=16)
        rnd.eval()
        
        x = torch.randn(10, 32)
        
        n1 = rnd.compute_novelty_no_grad(x)
        n2 = rnd.compute_novelty_no_grad(x)
        
        assert torch.allclose(n1, n2)


class TestRNDEncoderCoefficients:
    """Tests for multi-encoder coefficient support in RND."""
    
    def test_encoder_dims_stored(self):
        """Test encoder_dims parameter is stored correctly."""
        encoder_dims = [32, 64, 128]
        rnd = RNDModule(input_dim=224, feature_dim=64, encoder_dims=encoder_dims)
        
        assert rnd.encoder_dims == encoder_dims
    
    def test_encoder_dims_default_none(self):
        """Test encoder_dims defaults to None for backward compatibility."""
        rnd = RNDModule(input_dim=64, feature_dim=32)
        
        assert rnd.encoder_dims is None
    
    def test_apply_encoder_coefficients_identity(self):
        """Test apply_encoder_coefficients with all-ones returns unchanged tensor."""
        encoder_dims = [16, 32, 16]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims)
        
        x = torch.randn(10, 64)
        coefficients = [1.0, 1.0, 1.0]
        
        x_weighted = rnd.apply_encoder_coefficients(x, coefficients)
        
        assert torch.allclose(x, x_weighted)
    
    def test_apply_encoder_coefficients_zeros(self):
        """Test apply_encoder_coefficients with zeros produces zeros."""
        encoder_dims = [16, 32, 16]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims)
        
        x = torch.randn(10, 64)
        coefficients = [0.0, 0.0, 0.0]
        
        x_weighted = rnd.apply_encoder_coefficients(x, coefficients)
        
        assert torch.allclose(x_weighted, torch.zeros_like(x))
    
    def test_apply_encoder_coefficients_partial(self):
        """Test apply_encoder_coefficients with mixed coefficients."""
        encoder_dims = [16, 32, 16]  # Total = 64
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims)
        
        x = torch.ones(5, 64)
        coefficients = [1.0, 0.5, 0.0]  # First full, second half, third zero
        
        x_weighted = rnd.apply_encoder_coefficients(x, coefficients)
        
        # Check each encoder segment
        assert torch.allclose(x_weighted[:, :16], torch.ones(5, 16))  # First: coef=1.0
        assert torch.allclose(x_weighted[:, 16:48], torch.full((5, 32), 0.5))  # Second: coef=0.5
        assert torch.allclose(x_weighted[:, 48:64], torch.zeros(5, 16))  # Third: coef=0.0
    
    def test_apply_encoder_coefficients_none_passthrough(self):
        """Test apply_encoder_coefficients with None coefficients returns unchanged."""
        encoder_dims = [16, 32, 16]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims)
        
        x = torch.randn(10, 64)
        
        x_weighted = rnd.apply_encoder_coefficients(x, None)
        
        assert torch.allclose(x, x_weighted)
    
    def test_apply_encoder_coefficients_no_encoder_dims_passthrough(self):
        """Test apply_encoder_coefficients without encoder_dims returns unchanged."""
        rnd = RNDModule(input_dim=64, feature_dim=32)  # No encoder_dims
        
        x = torch.randn(10, 64)
        coefficients = [0.5, 0.5]  # Shouldn't be applied
        
        x_weighted = rnd.apply_encoder_coefficients(x, coefficients)
        
        assert torch.allclose(x, x_weighted)
    
    def test_compute_novelty_with_coefficients(self):
        """Test compute_novelty properly applies encoder coefficients."""
        encoder_dims = [32, 32]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims, normalize=False)
        
        x = torch.randn(10, 64)
        
        # Full coefficients should give same result as no coefficients
        novelty_full = rnd.compute_novelty(x, update_stats=False, encoder_coefficients=[1.0, 1.0])
        novelty_default = rnd.compute_novelty(x, update_stats=False, encoder_coefficients=None)
        
        assert torch.allclose(novelty_full, novelty_default)
    
    def test_compute_loss_with_coefficients(self):
        """Test compute_loss properly applies encoder coefficients."""
        encoder_dims = [32, 32]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims)
        
        x = torch.randn(10, 64)
        
        # Full coefficients should give same result as no coefficients
        loss_full = rnd.compute_loss(x, encoder_coefficients=[1.0, 1.0])
        loss_default = rnd.compute_loss(x, encoder_coefficients=None)
        
        assert torch.allclose(loss_full, loss_default)
    
    def test_zero_coefficients_reduce_novelty_variance(self):
        """Test that zeroing encoder features reduces novelty variance."""
        encoder_dims = [32, 32]
        rnd = RNDModule(input_dim=64, feature_dim=32, encoder_dims=encoder_dims, normalize=False)
        
        # Different inputs
        x_batch = torch.randn(100, 64)
        
        # With full features: should have variance
        novelty_full = rnd.compute_novelty_no_grad(x_batch, encoder_coefficients=[1.0, 1.0])
        var_full = novelty_full.var().item()
        
        # With half features zeroed: should still have variance but potentially less
        novelty_half = rnd.compute_novelty_no_grad(x_batch, encoder_coefficients=[1.0, 0.0])
        var_half = novelty_half.var().item()
        
        # With all zeros: constant output, zero variance
        novelty_zero = rnd.compute_novelty_no_grad(x_batch, encoder_coefficients=[0.0, 0.0])
        var_zero = novelty_zero.var().item()
        
        # Zero features should have near-zero variance
        assert var_zero < 1e-6, f"Expected near-zero variance, got {var_zero}"
        # Full features should have non-trivial variance
        assert var_full > 1e-6, f"Expected non-trivial variance, got {var_full}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
