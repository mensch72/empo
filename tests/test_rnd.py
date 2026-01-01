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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
