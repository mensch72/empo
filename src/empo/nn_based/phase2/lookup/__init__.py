"""
Lookup table implementations for Phase 2 networks.

This module provides tabular (dictionary-based) implementations of all Phase 2 networks:
- LookupTableRobotQNetwork: Q_r(s, a_r) lookup table
- LookupTableRobotValueNetwork: V_r(s) lookup table  
- LookupTableHumanGoalAbilityNetwork: V_h^e(s, g_h) lookup table
- LookupTableAggregateGoalAbilityNetwork: X_h(s) lookup table
- LookupTableIntrinsicRewardNetwork: U_r(s) lookup table

These provide exact value storage without function approximation error, useful for:
- Small state spaces (< 100K states)
- Debugging and interpretability
- Baseline comparisons with neural approaches

Key design principles:
1. Same API as neural versions (forward(), encode_and_forward(), get_config())
2. Uses torch.nn.Parameter for gradient tracking and optimizer compatibility
3. Lazy parameter creation (entries created on first access)
4. States must be hashable (enforced by WorldModel.get_state() interface)
"""

from .robot_q_network import LookupTableRobotQNetwork
from .robot_value_network import LookupTableRobotValueNetwork
from .human_goal_ability import LookupTableHumanGoalAbilityNetwork
from .aggregate_goal_ability import LookupTableAggregateGoalAbilityNetwork
from .intrinsic_reward_network import LookupTableIntrinsicRewardNetwork

__all__ = [
    'LookupTableRobotQNetwork',
    'LookupTableRobotValueNetwork',
    'LookupTableHumanGoalAbilityNetwork',
    'LookupTableAggregateGoalAbilityNetwork',
    'LookupTableIntrinsicRewardNetwork',
]
