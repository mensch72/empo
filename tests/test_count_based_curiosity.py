"""
Tests for count-based curiosity module.

Tests cover:
- Basic visit counting
- Bonus computation (simple and UCB)
- State dict save/load
- Statistics
- Integration with trainer
"""

import pytest
import math

from empo.learning_based.phase2.count_based_curiosity import CountBasedCuriosity


class TestCountBasedCuriosity:
    """Tests for the CountBasedCuriosity class."""
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        curiosity = CountBasedCuriosity()
        
        assert curiosity.scale == 1.0
        assert curiosity.use_ucb is False
        assert curiosity.num_unique_states == 0
        assert curiosity.total_visits == 0
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        curiosity = CountBasedCuriosity(scale=2.0, use_ucb=True, min_bonus=1e-8)
        
        assert curiosity.scale == 2.0
        assert curiosity.use_ucb is True
        assert curiosity.min_bonus == 1e-8
    
    def test_record_visit(self):
        """Test recording visits to states."""
        curiosity = CountBasedCuriosity()
        
        state1 = (1, 2, 3)
        state2 = (4, 5, 6)
        
        curiosity.record_visit(state1)
        assert curiosity.get_visit_count(state1) == 1
        assert curiosity.get_visit_count(state2) == 0
        assert curiosity.num_unique_states == 1
        assert curiosity.total_visits == 1
        
        curiosity.record_visit(state1)
        assert curiosity.get_visit_count(state1) == 2
        assert curiosity.num_unique_states == 1
        assert curiosity.total_visits == 2
        
        curiosity.record_visit(state2)
        assert curiosity.get_visit_count(state2) == 1
        assert curiosity.num_unique_states == 2
        assert curiosity.total_visits == 3
    
    def test_record_visits_batch(self):
        """Test recording multiple visits at once."""
        curiosity = CountBasedCuriosity()
        
        states = [(1, 2), (3, 4), (1, 2), (5, 6)]
        curiosity.record_visits(states)
        
        assert curiosity.get_visit_count((1, 2)) == 2
        assert curiosity.get_visit_count((3, 4)) == 1
        assert curiosity.get_visit_count((5, 6)) == 1
        assert curiosity.num_unique_states == 3
        assert curiosity.total_visits == 4
    
    def test_simple_bonus_novel_state(self):
        """Test bonus for unvisited states with simple formula."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=False)
        
        state = (1, 2, 3)
        bonus = curiosity.get_bonus(state)
        
        # For count=0: bonus = scale / sqrt(0 + 1) = 1.0
        assert bonus == pytest.approx(1.0)
    
    def test_simple_bonus_decreases_with_visits(self):
        """Test bonus decreases as visits increase."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=False)
        
        state = (1, 2, 3)
        
        bonus_0 = curiosity.get_bonus(state)
        curiosity.record_visit(state)
        bonus_1 = curiosity.get_bonus(state)
        curiosity.record_visit(state)
        bonus_2 = curiosity.get_bonus(state)
        
        # Bonus should decrease: 1/sqrt(1) > 1/sqrt(2) > 1/sqrt(3)
        assert bonus_0 > bonus_1 > bonus_2
        
        # Check specific values
        assert bonus_0 == pytest.approx(1.0)  # 1/sqrt(1)
        assert bonus_1 == pytest.approx(1.0 / math.sqrt(2))  # 1/sqrt(2)
        assert bonus_2 == pytest.approx(1.0 / math.sqrt(3))  # 1/sqrt(3)
    
    def test_ucb_bonus(self):
        """Test UCB-style bonus formula."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=True)
        
        # Record some visits
        states = [(1,), (2,), (3,)]
        for _ in range(10):
            for s in states:
                curiosity.record_visit(s)
        
        # UCB bonus: scale * sqrt(log(total) / (count + 1))
        state = (1,)
        bonus = curiosity.get_bonus(state)
        
        # count=10, total=30
        expected = 1.0 * math.sqrt(math.log(30) / (10 + 1))
        assert bonus == pytest.approx(expected)
    
    def test_ucb_bonus_novel_state(self):
        """Test UCB bonus for novel state with some history."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=True)
        
        # Build up some visit history
        for _ in range(100):
            curiosity.record_visit((1,))
        
        # Novel state should have high bonus
        novel_bonus = curiosity.get_bonus((2,))  # Never visited
        visited_bonus = curiosity.get_bonus((1,))  # Visited 100 times
        
        assert novel_bonus > visited_bonus
    
    def test_normalized_bonus(self):
        """Test normalized bonus is in approximately [0, 1] range."""
        curiosity = CountBasedCuriosity(scale=2.0, use_ucb=False)
        
        state = (1, 2, 3)
        
        # For novel state, normalized bonus = 1.0
        normalized = curiosity.get_normalized_bonus(state)
        assert normalized == pytest.approx(1.0)
        
        # After many visits, normalized bonus approaches 0
        for _ in range(100):
            curiosity.record_visit(state)
        
        normalized = curiosity.get_normalized_bonus(state)
        assert 0 < normalized < 0.2  # Much smaller than 1.0
    
    def test_get_bonuses_batch(self):
        """Test getting bonuses for multiple states at once."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=False)
        
        # Visit some states
        curiosity.record_visit((1,))
        curiosity.record_visit((1,))
        curiosity.record_visit((2,))
        
        states = [(1,), (2,), (3,)]  # 2 visits, 1 visit, 0 visits
        bonuses = curiosity.get_bonuses(states)
        
        assert len(bonuses) == 3
        assert bonuses[0] == pytest.approx(1.0 / math.sqrt(3))  # count=2
        assert bonuses[1] == pytest.approx(1.0 / math.sqrt(2))  # count=1
        assert bonuses[2] == pytest.approx(1.0)  # count=0
    
    def test_statistics(self):
        """Test get_statistics returns expected values."""
        curiosity = CountBasedCuriosity()
        
        # Record some visits
        curiosity.record_visit((1,))
        curiosity.record_visit((1,))
        curiosity.record_visit((2,))
        curiosity.record_visit((3,))
        
        stats = curiosity.get_statistics()
        
        assert stats['count_curiosity_unique_states'] == 3
        assert stats['count_curiosity_total_visits'] == 4
        assert stats['count_curiosity_mean_visits'] == pytest.approx(4 / 3)
        assert stats['count_curiosity_max_visits'] == 2
        assert stats['count_curiosity_min_visits'] == 1
    
    def test_statistics_empty(self):
        """Test statistics with no visits."""
        curiosity = CountBasedCuriosity()
        
        stats = curiosity.get_statistics()
        
        assert stats['count_curiosity_unique_states'] == 0
        assert stats['count_curiosity_total_visits'] == 0
        assert stats['count_curiosity_mean_visits'] == 0.0
        assert stats['count_curiosity_max_visits'] == 0
        assert stats['count_curiosity_min_visits'] == 0
    
    def test_reset(self):
        """Test reset clears all state."""
        curiosity = CountBasedCuriosity()
        
        # Add some data
        curiosity.record_visit((1,))
        curiosity.record_visit((2,))
        
        assert curiosity.num_unique_states == 2
        
        # Reset
        curiosity.reset()
        
        assert curiosity.num_unique_states == 0
        assert curiosity.total_visits == 0
        assert curiosity.get_visit_count((1,)) == 0
    
    def test_state_dict_save_load(self):
        """Test saving and loading state dict."""
        curiosity1 = CountBasedCuriosity(scale=2.0, use_ucb=True)
        
        # Record some visits
        curiosity1.record_visit((1, 2))
        curiosity1.record_visit((1, 2))
        curiosity1.record_visit((3, 4))
        
        # Save state
        state_dict = curiosity1.state_dict()
        
        # Create new instance and load
        curiosity2 = CountBasedCuriosity()
        curiosity2.load_state_dict(state_dict)
        
        # Verify state was restored
        assert curiosity2.scale == 2.0
        assert curiosity2.use_ucb is True
        assert curiosity2.get_visit_count((1, 2)) == 2
        assert curiosity2.get_visit_count((3, 4)) == 1
        assert curiosity2.total_visits == 3
        assert curiosity2.num_unique_states == 2
    
    def test_min_bonus(self):
        """Test minimum bonus prevents zero bonus."""
        curiosity = CountBasedCuriosity(scale=1.0, min_bonus=0.01)
        
        state = (1,)
        
        # After many visits, bonus should not go below min_bonus
        for _ in range(10000):
            curiosity.record_visit(state)
        
        bonus = curiosity.get_bonus(state)
        assert bonus >= 0.01
    
    def test_hashable_states(self):
        """Test various hashable state types work correctly."""
        curiosity = CountBasedCuriosity()
        
        # Tuple state
        curiosity.record_visit((1, 2, 3))
        assert curiosity.get_visit_count((1, 2, 3)) == 1
        
        # String state
        curiosity.record_visit("state_1")
        assert curiosity.get_visit_count("state_1") == 1
        
        # Integer state
        curiosity.record_visit(42)
        assert curiosity.get_visit_count(42) == 1
        
        # Nested tuple state
        curiosity.record_visit((1, (2, 3), 4))
        assert curiosity.get_visit_count((1, (2, 3), 4)) == 1
    
    def test_repr(self):
        """Test string representation."""
        curiosity = CountBasedCuriosity(scale=2.0, use_ucb=True)
        curiosity.record_visit((1,))
        
        repr_str = repr(curiosity)
        
        assert "CountBasedCuriosity" in repr_str
        assert "scale=2.0" in repr_str
        assert "use_ucb=True" in repr_str
        assert "unique_states=1" in repr_str
        assert "total_visits=1" in repr_str


class TestCountBasedCuriosityScaling:
    """Tests for bonus scaling behavior."""
    
    def test_scale_multiplies_bonus(self):
        """Test scale parameter multiplies the bonus."""
        curiosity1 = CountBasedCuriosity(scale=1.0)
        curiosity2 = CountBasedCuriosity(scale=2.0)
        
        state = (1, 2, 3)
        
        bonus1 = curiosity1.get_bonus(state)
        bonus2 = curiosity2.get_bonus(state)
        
        assert bonus2 == pytest.approx(2.0 * bonus1)
    
    def test_bonus_converges_to_zero(self):
        """Test bonus converges toward zero with many visits."""
        curiosity = CountBasedCuriosity(scale=1.0, use_ucb=False, min_bonus=0.0)
        
        state = (1,)
        
        bonuses = []
        for i in range(100):
            bonuses.append(curiosity.get_bonus(state))
            curiosity.record_visit(state)
        
        # Bonus should decrease monotonically
        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1]
        
        # Final bonus should be very small (1/sqrt(100) = 0.1)
        assert bonuses[-1] <= 0.1


class TestCountBasedCuriosityEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_batch_bonuses(self):
        """Test getting bonuses for empty list."""
        curiosity = CountBasedCuriosity()
        bonuses = curiosity.get_bonuses([])
        assert bonuses == []
    
    def test_empty_batch_visits(self):
        """Test recording empty list of visits."""
        curiosity = CountBasedCuriosity()
        curiosity.record_visits([])
        assert curiosity.num_unique_states == 0
    
    def test_zero_scale(self):
        """Test behavior with zero scale."""
        curiosity = CountBasedCuriosity(scale=0.0, min_bonus=0.0)
        
        bonus = curiosity.get_bonus((1, 2, 3))
        # Should return min_bonus since computed bonus is 0
        assert bonus == 0.0
    
    def test_very_large_visit_count(self):
        """Test with very large visit counts."""
        curiosity = CountBasedCuriosity(scale=1.0, min_bonus=1e-10)
        
        state = (1,)
        # Simulate many visits by directly setting the count
        curiosity._visit_counts[state] = 1000000
        curiosity._total_visits = 1000000
        
        bonus = curiosity.get_bonus(state)
        # Should be small but not zero
        assert bonus > 0
        assert bonus < 0.01


class TestCountBasedCuriosityConfig:
    """Test count-based curiosity configuration in Phase2Config."""
    
    def test_config_default_values(self):
        """Test default config values for count-based curiosity."""
        from empo.learning_based.phase2.config import Phase2Config
        
        config = Phase2Config()
        
        assert config.use_count_based_curiosity is False
        assert config.count_curiosity_scale == 1.0
        assert config.count_curiosity_use_ucb is False
        # Default bonus coefficients are 0.1 (same as RND defaults)
        assert config.count_curiosity_bonus_coef_r == 0.1
        assert config.count_curiosity_bonus_coef_h == 0.1
    
    def test_config_with_curiosity_enabled(self):
        """Test config with count-based curiosity enabled."""
        from empo.learning_based.phase2.config import Phase2Config
        
        config = Phase2Config(
            use_count_based_curiosity=True,
            count_curiosity_scale=2.0,
            count_curiosity_use_ucb=True,
            count_curiosity_bonus_coef_r=0.1,
            count_curiosity_bonus_coef_h=0.05,
        )
        
        assert config.use_count_based_curiosity is True
        assert config.count_curiosity_scale == 2.0
        assert config.count_curiosity_use_ucb is True
        assert config.count_curiosity_bonus_coef_r == 0.1
        assert config.count_curiosity_bonus_coef_h == 0.05


class TestCountBasedCuriosityFactory:
    """Test the factory function for creating count-based curiosity."""
    
    def test_create_count_based_curiosity(self):
        """Test creating curiosity module from config."""
        from empo.learning_based.phase2.config import Phase2Config
        from empo.learning_based.phase2.network_factory import create_count_based_curiosity
        
        config = Phase2Config(
            use_count_based_curiosity=True,
            count_curiosity_scale=2.5,
            count_curiosity_use_ucb=True,
        )
        
        curiosity = create_count_based_curiosity(config)
        
        assert curiosity is not None
        assert curiosity.scale == 2.5
        assert curiosity.use_ucb is True
    
    def test_create_returns_none_when_disabled(self):
        """Test factory returns None when curiosity is disabled."""
        from empo.learning_based.phase2.config import Phase2Config
        from empo.learning_based.phase2.network_factory import create_count_based_curiosity
        
        config = Phase2Config(use_count_based_curiosity=False)
        curiosity = create_count_based_curiosity(config)
        
        assert curiosity is None


class TestCountBasedCuriosityIntegration:
    """Integration tests with Phase2Networks and trainer."""
    
    def test_phase2_networks_with_count_curiosity(self):
        """Test Phase2Networks can hold count_curiosity field."""
        from empo.learning_based.phase2.trainer import Phase2Networks
        
        curiosity = CountBasedCuriosity(scale=1.0)
        
        # Create minimal Phase2Networks with count_curiosity
        networks = Phase2Networks(
            q_r=None,
            v_h_e=None,
            x_h=None,
            u_r=None,
            v_r=None,
            count_curiosity=curiosity,
        )
        
        assert networks.count_curiosity is curiosity
    
    def test_curiosity_module_export(self):
        """Test CountBasedCuriosity is exported from phase2 module."""
        from empo.learning_based.phase2 import CountBasedCuriosity as ExportedClass
        
        curiosity = ExportedClass(scale=1.0)
        assert curiosity.scale == 1.0
    
    def test_factory_export(self):
        """Test create_count_based_curiosity is exported."""
        from empo.learning_based.phase2 import create_count_based_curiosity
        from empo.learning_based.phase2.config import Phase2Config
        
        config = Phase2Config(use_count_based_curiosity=True)
        curiosity = create_count_based_curiosity(config)
        assert curiosity is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
