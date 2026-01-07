"""
Tests for HumanActionRNDModule and its integration with sample_human_actions.

These tests verify that:
1. HumanActionRNDModule produces different novelty for different (state, agent, action) tuples
2. The human RND is actually used during action sampling
3. Curiosity bonus changes the action distribution as expected
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from empo.learning_based.phase2.rnd import HumanActionRNDModule


class TestHumanActionRNDModule:
    """Unit tests for HumanActionRNDModule."""
    
    def test_module_creation(self):
        """Test that module can be created with expected dimensions."""
        module = HumanActionRNDModule(
            state_feature_dim=64,
            agent_feature_dim=32,
            num_actions=6,
            feature_dim=32,
            hidden_dim=128,
        )
        
        assert module.state_feature_dim == 64
        assert module.agent_feature_dim == 32
        assert module.num_actions == 6
        
    def test_output_shape(self):
        """Test that output has correct shape (batch_size, num_actions)."""
        module = HumanActionRNDModule(
            state_feature_dim=64,
            agent_feature_dim=32,
            num_actions=6,
        )
        
        batch_size = 4
        state_features = torch.randn(batch_size, 64)
        agent_features = torch.randn(batch_size, 32)
        
        novelty = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        
        assert novelty.shape == (batch_size, 6)
        
    def test_novelty_is_positive(self):
        """Test that raw novelty (MSE) is always non-negative."""
        module = HumanActionRNDModule(
            state_feature_dim=64,
            agent_feature_dim=32,
            num_actions=6,
            normalize=False,  # Raw MSE should be positive
        )
        
        state_features = torch.randn(10, 64)
        agent_features = torch.randn(10, 32)
        
        novelty = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        
        assert (novelty >= 0).all(), "Raw novelty should be non-negative"
        
    def test_different_agents_get_different_novelty(self):
        """Test that different agent features produce different novelty scores."""
        module = HumanActionRNDModule(
            state_feature_dim=64,
            agent_feature_dim=32,
            num_actions=6,
        )
        
        # Same state, different agents
        state_features = torch.randn(1, 64).expand(2, -1)  # Same state for both
        agent_features = torch.randn(2, 32)  # Different agents
        
        novelty = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        
        # Novelty should be different for different agents
        assert not torch.allclose(novelty[0], novelty[1]), \
            "Different agents should have different novelty scores"
            
    def test_same_input_same_output(self):
        """Test that same (state, agent) always produces same novelty."""
        module = HumanActionRNDModule(
            state_feature_dim=64,
            agent_feature_dim=32,
            num_actions=6,
        )
        module.eval()  # Ensure deterministic behavior
        
        state_features = torch.randn(1, 64)
        agent_features = torch.randn(1, 32)
        
        novelty1 = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        novelty2 = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        
        assert torch.allclose(novelty1, novelty2), \
            "Same input should produce same novelty"
            
    def test_training_reduces_novelty(self):
        """Test that training the predictor reduces novelty for trained inputs."""
        torch.manual_seed(42)
        
        module = HumanActionRNDModule(
            state_feature_dim=32,
            agent_feature_dim=16,
            num_actions=4,
            normalize=False,
        )
        module.train()
        
        # Fixed input
        state_features = torch.randn(8, 32)
        agent_features = torch.randn(8, 16)
        actions = torch.randint(0, 4, (8,))
        
        # Initial novelty
        initial_novelty = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        initial_mean = initial_novelty.mean().item()
        
        # Train on this data
        optimizer = torch.optim.Adam(module.predictor.parameters(), lr=0.01)
        for _ in range(100):
            optimizer.zero_grad()
            loss = module.compute_loss(state_features, agent_features, actions)
            loss.backward()
            optimizer.step()
        
        # Final novelty
        final_novelty = module.compute_action_novelties_no_grad(
            state_features, agent_features, use_raw=True
        )
        final_mean = final_novelty.mean().item()
        
        assert final_mean < initial_mean * 0.8, \
            f"Training should reduce novelty: {initial_mean:.4f} -> {final_mean:.4f}"
            
    def test_loss_only_trains_taken_actions(self):
        """Test that loss computation focuses on taken actions."""
        module = HumanActionRNDModule(
            state_feature_dim=32,
            agent_feature_dim=16,
            num_actions=4,
            normalize=False,
        )
        
        state_features = torch.randn(2, 32)
        agent_features = torch.randn(2, 16)
        actions = torch.tensor([0, 2])  # Only actions 0 and 2
        
        # Should compute loss without error
        loss = module.compute_loss(state_features, agent_features, actions)
        
        assert loss.shape == (), "Loss should be scalar"
        assert loss.item() > 0, "Initial loss should be positive"
        

class TestHumanRNDIntegration:
    """Integration tests verifying human RND is used in sample_human_actions."""
    
    def test_curiosity_changes_action_distribution(self):
        """Test that curiosity bonus actually changes action probabilities."""
        # Create a mock human RND that returns high novelty for action 0
        mock_human_rnd = MagicMock()
        # Return high novelty for action 0, low for others
        novelty_scores = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1, 0.1]])
        mock_human_rnd.compute_action_novelties_no_grad.return_value = novelty_scores
        
        # Uniform prior probabilities
        prior_probs = np.array([1/6] * 6)
        
        # Apply the same logic as sample_human_actions
        bonus_coef = 0.5
        novelty = novelty_scores[0].cpu().numpy()
        novelty = np.maximum(0.0, novelty)
        scale_factors = np.exp(bonus_coef * novelty)
        modified_probs = prior_probs * scale_factors
        modified_probs = modified_probs / modified_probs.sum()
        
        # Action 0 should now have much higher probability
        assert modified_probs[0] > 0.5, \
            f"High novelty action should have high probability: {modified_probs}"
        assert modified_probs[0] > modified_probs[1] * 5, \
            "High novelty action should be much more likely than low novelty"
            
    def test_zero_bonus_coef_preserves_prior(self):
        """Test that zero bonus coefficient preserves original distribution."""
        novelty_scores = torch.tensor([[10.0, 0.1, 0.1, 0.1, 0.1, 0.1]])
        prior_probs = np.array([1/6] * 6)
        
        bonus_coef = 0.0  # No curiosity
        novelty = novelty_scores[0].cpu().numpy()
        novelty = np.maximum(0.0, novelty)
        scale_factors = np.exp(bonus_coef * novelty)
        modified_probs = prior_probs * scale_factors
        modified_probs = modified_probs / modified_probs.sum()
        
        np.testing.assert_array_almost_equal(
            modified_probs, prior_probs,
            err_msg="Zero bonus should preserve original distribution"
        )
        
    def test_human_rnd_called_when_enabled(self):
        """Test that human RND's compute method is called when enabled."""
        from empo.learning_based.phase2.config import Phase2Config
        
        # Create minimal mock trainer
        mock_trainer = MagicMock()
        # Use Phase2Config with no epsilon (set start=end=0) 
        mock_trainer.config = Phase2Config(
            use_rnd=True,
            use_human_action_rnd=True,
            rnd_bonus_coef_h=0.5,
            epsilon_h_start=0.0,  # No epsilon exploration
            epsilon_h_end=0.0,
        )
        mock_trainer.training_step_count = 1000
        mock_trainer.human_agent_indices = [0]
        mock_trainer.env.action_space.n = 6
        mock_trainer.human_exploration_policy = None
        
        # Mock the human RND
        mock_human_rnd = MagicMock()
        mock_human_rnd.compute_action_novelties_no_grad.return_value = torch.zeros(1, 6)
        mock_trainer.networks.human_rnd = mock_human_rnd
        
        # Mock get_human_features_for_rnd
        mock_trainer.get_human_features_for_rnd.return_value = (
            torch.zeros(1, 64),  # state_features
            torch.zeros(1, 32),  # agent_features
        )
        
        # Mock the _get_curiosity_bonus_coef_h method
        mock_trainer._get_curiosity_bonus_coef_h.return_value = 0.5
        
        # Mock human policy prior
        mock_trainer.human_policy_prior.return_value = np.array([1/6] * 6)
        mock_trainer.human_policy_prior.sample.return_value = 0
        
        # Import and call the actual method with the mock
        from empo.learning_based.phase2.trainer import BasePhase2Trainer
        
        # Call sample_human_actions using the mock as self
        state = "mock_state"
        goals = {0: "mock_goal"}
        
        # We need to call the unbound method with our mock
        BasePhase2Trainer.sample_human_actions(mock_trainer, state, goals)
        
        # Verify human RND was called
        mock_trainer.get_human_features_for_rnd.assert_called_once()
        mock_human_rnd.compute_action_novelties_no_grad.assert_called_once()
        
    def test_human_rnd_not_called_when_disabled(self):
        """Test that human RND is not called when disabled."""
        from empo.learning_based.phase2.config import Phase2Config
        
        mock_trainer = MagicMock()
        mock_trainer.config = Phase2Config(
            use_rnd=False,  # Disabled
            use_human_action_rnd=False,  # Human RND also disabled
        )
        mock_trainer.training_step_count = 1000
        mock_trainer.human_agent_indices = [0]
        mock_trainer.env.action_space.n = 6
        mock_trainer.human_exploration_policy = None
        mock_trainer.networks.human_rnd = None  # No human RND
        
        # No need to mock config methods - real Phase2Config has them
        mock_trainer._get_curiosity_bonus_coef_h.return_value = 0.0
        
        mock_trainer.human_policy_prior.sample.return_value = 0
        
        from empo.learning_based.phase2.trainer import BasePhase2Trainer
        
        state = "mock_state"
        goals = {0: "mock_goal"}
        
        BasePhase2Trainer.sample_human_actions(mock_trainer, state, goals)
        
        # Verify human RND was NOT called
        mock_trainer.get_human_features_for_rnd.assert_not_called()


class TestHumanRNDStatisticalBehavior:
    """Statistical tests to verify human RND affects action selection."""
    
    def test_high_novelty_action_selected_more_often(self):
        """Test that actions with high novelty are selected more frequently."""
        np.random.seed(42)
        
        # Simulate the action selection logic from sample_human_actions
        num_samples = 10000
        num_actions = 6
        
        # High novelty for action 0, low for others
        novelty = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1])
        prior_probs = np.array([1/6] * 6)
        bonus_coef = 0.5
        
        scale_factors = np.exp(bonus_coef * novelty)
        modified_probs = prior_probs * scale_factors
        modified_probs = modified_probs / modified_probs.sum()
        
        # Sample many times
        actions = np.random.choice(num_actions, size=num_samples, p=modified_probs)
        
        # Count action 0
        action_0_count = (actions == 0).sum()
        action_0_fraction = action_0_count / num_samples
        
        # Action 0 should be selected much more than 1/6 of the time
        assert action_0_fraction > 0.5, \
            f"High novelty action should be selected >50% of time, got {action_0_fraction:.2%}"
        
        # And more than any other individual action
        for a in range(1, num_actions):
            action_a_fraction = (actions == a).sum() / num_samples
            assert action_0_fraction > action_a_fraction * 2, \
                f"Action 0 should be selected much more than action {a}"
                
    def test_uniform_novelty_preserves_prior(self):
        """Test that uniform novelty doesn't change the distribution much."""
        np.random.seed(42)
        
        num_actions = 6
        
        # Same novelty for all actions
        novelty = np.array([1.0] * num_actions)
        prior_probs = np.array([1 / num_actions] * num_actions)
        bonus_coef = 0.5
        
        scale_factors = np.exp(bonus_coef * novelty)
        modified_probs = prior_probs * scale_factors
        modified_probs = modified_probs / modified_probs.sum()
        
        # Should still be uniform
        np.testing.assert_array_almost_equal(
            modified_probs, prior_probs, decimal=5,
            err_msg="Uniform novelty should preserve uniform prior"
        )
        

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
