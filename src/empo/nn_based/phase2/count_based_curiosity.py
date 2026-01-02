"""
Count-based curiosity module for tabular (lookup table) Phase 2 training.

Count-based curiosity maintains a dictionary mapping states to visit counts,
and provides exploration bonuses based on inverse visit frequency. This is
simple and effective for tabular settings where states are hashable.

Unlike RND (which generalizes novelty across similar states via neural networks),
count-based curiosity provides exact per-state novelty tracking. This makes it
ideal for lookup table training where there's no function approximation.

Bonus formulas:
- Simple: bonus(s) = scale / sqrt(visits[s] + 1)
- UCB-style: bonus(s) = scale * sqrt(log(total_visits) / (visits[s] + 1))

Reference:
    Bellemare, M., Srinivasan, S., Ostrovski, G., Schaul, T., Saxton, D., & Munos, R.
    (2016). Unifying Count-Based Exploration and Intrinsic Motivation. NeurIPS 2016.
"""

from typing import Dict, Hashable, Tuple, Any, Optional
import math


class CountBasedCuriosity:
    """
    Count-based curiosity module for exploration in tabular settings.
    
    Maintains visit counts for states and computes exploration bonuses
    based on inverse visit frequency. States must be hashable.
    
    Args:
        scale: Scale factor for curiosity bonus.
        use_ucb: If True, use UCB-style bonus with log(total)/count.
                 If False, use simple 1/sqrt(count) bonus.
        min_bonus: Minimum bonus value (prevents zero bonus for highly visited states).
    
    Example:
        >>> curiosity = CountBasedCuriosity(scale=1.0)
        >>> state = (1, 2, 3)  # Hashable state
        >>> bonus = curiosity.get_bonus(state)  # High bonus (novel)
        >>> curiosity.record_visit(state)
        >>> bonus = curiosity.get_bonus(state)  # Lower bonus (visited)
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        use_ucb: bool = False,
        min_bonus: float = 1e-6,
    ):
        self.scale = scale
        self.use_ucb = use_ucb
        self.min_bonus = min_bonus
        
        # State visit counts: Dict[Hashable, int]
        self._visit_counts: Dict[Hashable, int] = {}
        self._total_visits: int = 0
    
    def get_visit_count(self, state: Hashable) -> int:
        """
        Get the visit count for a state.
        
        Args:
            state: Hashable state.
            
        Returns:
            Number of times this state has been visited.
        """
        return self._visit_counts.get(state, 0)
    
    def record_visit(self, state: Hashable) -> None:
        """
        Record a visit to a state.
        
        Args:
            state: Hashable state that was visited.
        """
        self._visit_counts[state] = self._visit_counts.get(state, 0) + 1
        self._total_visits += 1
    
    def record_visits(self, states: list) -> None:
        """
        Record visits to multiple states.
        
        Args:
            states: List of hashable states that were visited.
        """
        for state in states:
            self.record_visit(state)
    
    def get_bonus(self, state: Hashable) -> float:
        """
        Compute curiosity bonus for a state.
        
        For novel states (count=0), returns maximum bonus.
        For visited states, bonus decreases with visit count.
        
        Args:
            state: Hashable state.
            
        Returns:
            Curiosity bonus value (always positive).
        """
        count = self._visit_counts.get(state, 0)
        
        if self.use_ucb:
            # UCB-style bonus: scale * sqrt(log(total) / (count + 1))
            # For count=0, this is scale * sqrt(log(total))
            # As count grows, bonus shrinks
            if self._total_visits <= 1:
                log_total = 1.0  # Avoid log(0)
            else:
                log_total = math.log(self._total_visits)
            bonus = self.scale * math.sqrt(log_total / (count + 1))
        else:
            # Simple bonus: scale / sqrt(count + 1)
            # For count=0, this is scale
            # For count=1, this is scale/sqrt(2) ≈ 0.707*scale
            bonus = self.scale / math.sqrt(count + 1)
        
        return max(bonus, self.min_bonus)
    
    def get_bonuses(self, states: list) -> list:
        """
        Compute curiosity bonuses for multiple states.
        
        Args:
            states: List of hashable states.
            
        Returns:
            List of curiosity bonus values.
        """
        return [self.get_bonus(state) for state in states]
    
    def get_normalized_bonus(self, state: Hashable) -> float:
        """
        Compute normalized curiosity bonus (approximately in [0, 1] range).
        
        Normalizes the bonus by dividing by scale, so:
        - Novel states have bonus close to 1.0
        - Highly visited states have bonus close to 0.0
        
        Args:
            state: Hashable state.
            
        Returns:
            Normalized curiosity bonus in approximately [0, 1].
        """
        bonus = self.get_bonus(state)
        if self.scale > 0:
            return bonus / self.scale
        return bonus
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the curiosity module.
        
        Returns:
            Dict with statistics for logging/debugging.
        """
        counts = list(self._visit_counts.values()) if self._visit_counts else [0]
        
        return {
            'count_curiosity_unique_states': len(self._visit_counts),
            'count_curiosity_total_visits': self._total_visits,
            'count_curiosity_mean_visits': sum(counts) / len(counts) if counts else 0.0,
            'count_curiosity_max_visits': max(counts) if counts else 0,
            'count_curiosity_min_visits': min(counts) if counts else 0,
        }
    
    def reset(self) -> None:
        """Reset all visit counts."""
        self._visit_counts.clear()
        self._total_visits = 0
    
    @property
    def num_unique_states(self) -> int:
        """Number of unique states visited."""
        return len(self._visit_counts)
    
    @property
    def total_visits(self) -> int:
        """Total number of visits recorded."""
        return self._total_visits
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dict for saving/loading.
        
        Returns:
            Dict containing all state needed to restore the module.
        """
        return {
            'visit_counts': dict(self._visit_counts),
            'total_visits': self._total_visits,
            'scale': self.scale,
            'use_ucb': self.use_ucb,
            'min_bonus': self.min_bonus,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state from a state dict.
        
        Args:
            state_dict: State dict from state_dict().
        """
        self._visit_counts = dict(state_dict.get('visit_counts', {}))
        self._total_visits = state_dict.get('total_visits', 0)
        # Optionally update config (only if present and different)
        if 'scale' in state_dict:
            self.scale = state_dict['scale']
        if 'use_ucb' in state_dict:
            self.use_ucb = state_dict['use_ucb']
        if 'min_bonus' in state_dict:
            self.min_bonus = state_dict['min_bonus']
    
    def __repr__(self) -> str:
        return (
            f"CountBasedCuriosity(scale={self.scale}, use_ucb={self.use_ucb}, "
            f"unique_states={self.num_unique_states}, total_visits={self.total_visits})"
        )
