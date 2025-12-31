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
from .null_encoders import NullStateEncoder, NullGoalEncoder, NullAgentEncoder


def is_lookup_table_network(network) -> bool:
    """
    Check if a network is a lookup table implementation.
    
    Args:
        network: A network instance (neural or lookup table).
    
    Returns:
        True if the network is a lookup table implementation.
    """
    return isinstance(network, (
        LookupTableRobotQNetwork,
        LookupTableRobotValueNetwork,
        LookupTableHumanGoalAbilityNetwork,
        LookupTableAggregateGoalAbilityNetwork,
        LookupTableIntrinsicRewardNetwork,
    ))


def get_all_lookup_tables(networks) -> dict:
    """
    Get all lookup table networks from a Phase2Networks container.
    
    Args:
        networks: Phase2Networks instance.
    
    Returns:
        Dict mapping network name to network instance for all lookup table networks.
    """
    lookup_tables = {}
    network_map = {
        'q_r': networks.q_r,
        'v_h_e': networks.v_h_e,
        'x_h': networks.x_h,
    }
    if networks.u_r is not None:
        network_map['u_r'] = networks.u_r
    if networks.v_r is not None:
        network_map['v_r'] = networks.v_r
    
    for name, net in network_map.items():
        if is_lookup_table_network(net):
            lookup_tables[name] = net
    
    return lookup_tables


def get_total_table_size(networks) -> int:
    """
    Get total number of entries across all lookup table networks.
    
    Args:
        networks: Phase2Networks instance.
    
    Returns:
        Total number of table entries (for memory monitoring).
    """
    total = 0
    for net in get_all_lookup_tables(networks).values():
        total += len(net.table)
    return total


__all__ = [
    'LookupTableRobotQNetwork',
    'LookupTableRobotValueNetwork',
    'LookupTableHumanGoalAbilityNetwork',
    'LookupTableAggregateGoalAbilityNetwork',
    'LookupTableIntrinsicRewardNetwork',
    'NullStateEncoder',
    'NullGoalEncoder',
    'NullAgentEncoder',
    'is_lookup_table_network',
    'get_all_lookup_tables',
    'get_total_table_size',
]
