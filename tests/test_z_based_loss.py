#!/usr/bin/env python3
"""
Tests for the use_z_based_loss configuration option.

This tests:
1. Config validation (use_z_based_loss requires use_z_space_transform)
2. should_use_z_loss() method behavior
3. Correct loss function selection in trainer

See docs/VALUE_TRANSFORMATIONS.md for full rationale.
"""

import pytest
import warnings
import torch
import numpy as np

from empo.nn_based.phase2.config import Phase2Config


class TestZBasedLossConfig:
    """Test Phase2Config options for z-based loss."""
    
    def test_default_values(self):
        """Test that defaults are use_z_space_transform=False, use_z_based_loss=False."""
        config = Phase2Config()
        assert config.use_z_space_transform is False
        assert config.use_z_based_loss is False
    
    def test_z_based_loss_without_transform_warns_and_disables(self):
        """Test that use_z_based_loss=True without use_z_space_transform=True warns and disables."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Phase2Config(
                use_z_space_transform=False,
                use_z_based_loss=True
            )
            # Should have emitted a warning
            assert len(w) >= 1
            assert any("use_z_based_loss=True but use_z_space_transform=False" in str(warning.message) 
                      for warning in w)
            # Should have been disabled
            assert config.use_z_based_loss is False
    
    def test_z_based_loss_with_transform_allowed(self):
        """Test that use_z_based_loss=True with use_z_space_transform=True is allowed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = Phase2Config(
                use_z_space_transform=True,
                use_z_based_loss=True
            )
            # Should NOT have emitted a warning about this combination
            z_loss_warnings = [warning for warning in w 
                              if "use_z_based_loss" in str(warning.message)]
            assert len(z_loss_warnings) == 0
            # Both should be True
            assert config.use_z_space_transform is True
            assert config.use_z_based_loss is True


class TestShouldUseZLoss:
    """Test the should_use_z_loss() method."""
    
    def test_should_use_z_loss_disabled_without_transform(self):
        """Test should_use_z_loss returns False when use_z_space_transform=False."""
        config = Phase2Config(
            use_z_space_transform=False,
            use_z_based_loss=False,
            num_training_steps=10000,
        )
        # Should always return False
        for step in [0, 1000, 5000, 9999]:
            assert config.should_use_z_loss(step) is False
    
    def test_should_use_z_loss_disabled_when_z_based_loss_false(self):
        """Test should_use_z_loss returns False when use_z_based_loss=False (default)."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=False,  # Default: Q-space loss throughout
            num_training_steps=10000,
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            warmup_v_r_steps=0,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.7,
            constant_lr_then_1_over_t=True,
        )
        # Should always return False regardless of phase
        for step in [0, 1000, 3000, 5000, 7000, 9999]:
            assert config.should_use_z_loss(step) is False
    
    def test_should_use_z_loss_enabled_in_constant_lr_phase(self):
        """Test should_use_z_loss returns True during constant LR phase when enabled."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=True,  # Legacy mode
            num_training_steps=10000,
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            warmup_v_r_steps=0,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.7,  # Decay starts at step 7000
            constant_lr_then_1_over_t=True,
        )
        # Total warmup = 1000 + 0 + 0 + 1000 + 0 + 1000 = 3000
        # Decay starts at max(3000, 0.7 * 10000) = 7000
        
        # During warmup and constant LR phase: should use z-loss
        assert config.should_use_z_loss(0) is True
        assert config.should_use_z_loss(1000) is True
        assert config.should_use_z_loss(3000) is True
        assert config.should_use_z_loss(5000) is True
        assert config.should_use_z_loss(6999) is True
    
    def test_should_use_z_loss_disabled_in_decay_phase(self):
        """Test should_use_z_loss returns False during decay phase even when enabled."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=True,  # Legacy mode
            num_training_steps=10000,
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            warmup_v_r_steps=0,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.7,  # Decay starts at step 7000
            constant_lr_then_1_over_t=True,
        )
        # In decay phase (step >= 7000): should NOT use z-loss
        assert config.should_use_z_loss(7000) is False
        assert config.should_use_z_loss(8000) is False
        assert config.should_use_z_loss(9999) is False
    
    def test_should_use_z_loss_when_no_decay_phase(self):
        """Test should_use_z_loss when constant_lr_then_1_over_t=False."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=True,
            num_training_steps=10000,
            warmup_v_h_e_steps=1000,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
            warmup_q_r_steps=1000,
            warmup_v_r_steps=0,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.7,
            constant_lr_then_1_over_t=False,  # No decay phase
        )
        # Without decay phase, is_in_decay_phase always returns False
        # So should_use_z_loss should always return True when use_z_based_loss=True
        for step in [0, 1000, 5000, 7000, 9999]:
            assert config.should_use_z_loss(step) is True


class TestDecayPhaseConsistency:
    """Test consistency between is_in_decay_phase and should_use_z_loss."""
    
    def test_consistency_z_based_loss_true(self):
        """Test that should_use_z_loss = True iff NOT in_decay_phase (when use_z_based_loss=True)."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=True,
            num_training_steps=10000,
            warmup_v_h_e_steps=500,
            warmup_x_h_steps=500,
            warmup_u_r_steps=500,
            warmup_q_r_steps=500,
            warmup_v_r_steps=500,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.5,
            constant_lr_then_1_over_t=True,
            x_h_use_network=True,
            u_r_use_network=True,
            v_r_use_network=True,
        )
        
        for step in range(0, 10000, 100):
            in_decay = config.is_in_decay_phase(step)
            use_z = config.should_use_z_loss(step)
            # They should be logical opposites when use_z_based_loss=True
            assert use_z == (not in_decay), f"At step {step}: in_decay={in_decay}, use_z={use_z}"
    
    def test_consistency_z_based_loss_false(self):
        """Test that should_use_z_loss = False always (when use_z_based_loss=False)."""
        config = Phase2Config(
            use_z_space_transform=True,
            use_z_based_loss=False,  # Default: Q-space loss
            num_training_steps=10000,
            warmup_v_h_e_steps=500,
            warmup_x_h_steps=500,
            warmup_u_r_steps=500,
            warmup_q_r_steps=500,
            warmup_v_r_steps=500,
            beta_r_rampup_steps=1000,
            lr_constant_fraction=0.5,
            constant_lr_then_1_over_t=True,
            x_h_use_network=True,
            u_r_use_network=True,
            v_r_use_network=True,
        )
        
        for step in range(0, 10000, 100):
            use_z = config.should_use_z_loss(step)
            assert use_z is False, f"At step {step}: use_z should be False but got {use_z}"


class TestValueTransformImports:
    """Test that value_transforms module is importable and works correctly."""
    
    def test_import_transforms(self):
        """Test that transform functions are importable."""
        from empo.nn_based.phase2.value_transforms import (
            to_z_space, from_z_space, y_to_z_space, z_to_y_space
        )
        
        # Basic sanity checks
        eta, xi = 1.1, 1.0
        
        # Q -> z -> Q roundtrip
        q = torch.tensor([-10.0, -100.0, -1000.0])
        z = to_z_space(q, eta, xi)
        q_back = from_z_space(z, eta, xi)
        assert torch.allclose(q, q_back, rtol=1e-5)
        
        # z should be in (0, 1]
        assert (z > 0).all() and (z <= 1).all()
        
        # y -> z -> y roundtrip
        y = torch.tensor([1.0, 10.0, 100.0])
        z_y = y_to_z_space(y, xi)
        y_back = z_to_y_space(z_y, xi)
        assert torch.allclose(y, y_back, rtol=1e-5)
        
        # z should be in (0, 1]
        assert (z_y > 0).all() and (z_y <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
