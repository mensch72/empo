"""
Network Factory for Phase 2 Networks.

Provides factory functions to create Phase 2 networks based on configuration,
automatically choosing between lookup table and neural implementations.
"""

from typing import Any, Optional, Tuple

from .config import Phase2Config
from .robot_q_network import BaseRobotQNetwork
from .robot_value_network import BaseRobotValueNetwork
from .human_goal_ability import BaseHumanGoalAchievementNetwork
from .aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from .intrinsic_reward_network import BaseIntrinsicRewardNetwork

from .lookup import (
    LookupTableRobotQNetwork,
    LookupTableRobotValueNetwork,
    LookupTableHumanGoalAbilityNetwork,
    LookupTableAggregateGoalAbilityNetwork,
    LookupTableIntrinsicRewardNetwork,
)


def create_robot_q_network(
    config: Phase2Config,
    num_actions: int,
    num_robots: int = 1,
    neural_network_factory: Optional[callable] = None,
    **kwargs
) -> BaseRobotQNetwork:
    """
    Create a robot Q-network based on configuration.
    
    Args:
        config: Phase2Config instance.
        num_actions: Number of actions per robot.
        num_robots: Number of robot agents.
        neural_network_factory: Optional callable to create neural network.
            Required if not using lookup tables. Called with (num_actions, num_robots, **kwargs).
        **kwargs: Additional arguments passed to neural network factory.
    
    Returns:
        BaseRobotQNetwork instance (lookup table or neural).
    
    Raises:
        ValueError: If lookup tables not enabled and no neural factory provided.
    """
    if config.should_use_lookup_table('q_r'):
        return LookupTableRobotQNetwork(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=config.beta_r,
            default_q_r=config.get_lookup_default('q_r'),
        )
    
    if neural_network_factory is None:
        raise ValueError(
            "neural_network_factory required when not using lookup tables for Q_r"
        )
    
    return neural_network_factory(num_actions, num_robots, **kwargs)


def create_robot_value_network(
    config: Phase2Config,
    neural_network_factory: Optional[callable] = None,
    **kwargs
) -> Optional[BaseRobotValueNetwork]:
    """
    Create a robot value network based on configuration.
    
    Args:
        config: Phase2Config instance.
        neural_network_factory: Optional callable to create neural network.
            Required if using V_r network and not using lookup tables.
        **kwargs: Additional arguments passed to neural network factory.
    
    Returns:
        BaseRobotValueNetwork instance if v_r_use_network is True, else None.
    """
    if not config.v_r_use_network:
        return None
    
    if config.should_use_lookup_table('v_r'):
        return LookupTableRobotValueNetwork(
            gamma_r=config.gamma_r,
            default_v_r=config.get_lookup_default('v_r'),
        )
    
    if neural_network_factory is None:
        raise ValueError(
            "neural_network_factory required when using V_r network but not lookup tables"
        )
    
    return neural_network_factory(**kwargs)


def create_human_goal_ability_network(
    config: Phase2Config,
    neural_network_factory: Optional[callable] = None,
    **kwargs
) -> BaseHumanGoalAchievementNetwork:
    """
    Create a human goal ability network (V_h^e) based on configuration.
    
    Args:
        config: Phase2Config instance.
        neural_network_factory: Optional callable to create neural network.
            Required if not using lookup tables.
        **kwargs: Additional arguments passed to neural network factory.
    
    Returns:
        BaseHumanGoalAchievementNetwork instance.
    """
    if config.should_use_lookup_table('v_h_e'):
        return LookupTableHumanGoalAbilityNetwork(
            gamma_h=config.gamma_h,
            default_v_h_e=config.get_lookup_default('v_h_e'),
        )
    
    if neural_network_factory is None:
        raise ValueError(
            "neural_network_factory required when not using lookup tables for V_h^e"
        )
    
    return neural_network_factory(**kwargs)


def create_aggregate_goal_ability_network(
    config: Phase2Config,
    neural_network_factory: Optional[callable] = None,
    **kwargs
) -> BaseAggregateGoalAbilityNetwork:
    """
    Create an aggregate goal ability network (X_h) based on configuration.
    
    Args:
        config: Phase2Config instance.
        neural_network_factory: Optional callable to create neural network.
            Required if not using lookup tables.
        **kwargs: Additional arguments passed to neural network factory.
    
    Returns:
        BaseAggregateGoalAbilityNetwork instance.
    """
    if config.should_use_lookup_table('x_h'):
        return LookupTableAggregateGoalAbilityNetwork(
            default_x_h=config.get_lookup_default('x_h'),
        )
    
    if neural_network_factory is None:
        raise ValueError(
            "neural_network_factory required when not using lookup tables for X_h"
        )
    
    return neural_network_factory(**kwargs)


def create_intrinsic_reward_network(
    config: Phase2Config,
    neural_network_factory: Optional[callable] = None,
    **kwargs
) -> Optional[BaseIntrinsicRewardNetwork]:
    """
    Create an intrinsic reward network (U_r) based on configuration.
    
    Args:
        config: Phase2Config instance.
        neural_network_factory: Optional callable to create neural network.
            Required if using U_r network and not using lookup tables.
        **kwargs: Additional arguments passed to neural network factory.
    
    Returns:
        BaseIntrinsicRewardNetwork instance if u_r_use_network is True, else None.
    """
    if not config.u_r_use_network:
        return None
    
    if config.should_use_lookup_table('u_r'):
        return LookupTableIntrinsicRewardNetwork(
            eta=config.eta,
            default_y=config.get_lookup_default('u_r'),
        )
    
    if neural_network_factory is None:
        raise ValueError(
            "neural_network_factory required when using U_r network but not lookup tables"
        )
    
    return neural_network_factory(**kwargs)


def create_all_phase2_lookup_networks(
    config: Phase2Config,
    num_actions: int,
    num_robots: int = 1,
) -> Tuple[
    BaseRobotQNetwork,
    BaseHumanGoalAchievementNetwork,
    BaseAggregateGoalAbilityNetwork,
    Optional[BaseIntrinsicRewardNetwork],
    Optional[BaseRobotValueNetwork],
]:
    """
    Create all Phase 2 networks as lookup tables.
    
    Convenience function to create a complete set of lookup table networks.
    Requires config.use_lookup_tables to be True.
    
    Args:
        config: Phase2Config instance with use_lookup_tables=True.
        num_actions: Number of actions per robot.
        num_robots: Number of robot agents.
    
    Returns:
        Tuple of (q_r, v_h_e, x_h, u_r, v_r) networks.
        u_r and v_r may be None depending on config.
    
    Raises:
        ValueError: If config.use_lookup_tables is False.
    """
    if not config.use_lookup_tables:
        raise ValueError(
            "config.use_lookup_tables must be True for create_all_phase2_lookup_networks"
        )
    
    q_r = LookupTableRobotQNetwork(
        num_actions=num_actions,
        num_robots=num_robots,
        beta_r=config.beta_r,
        default_q_r=config.get_lookup_default('q_r'),
    )
    
    v_h_e = LookupTableHumanGoalAbilityNetwork(
        gamma_h=config.gamma_h,
        default_v_h_e=config.get_lookup_default('v_h_e'),
    )
    
    x_h = LookupTableAggregateGoalAbilityNetwork(
        default_x_h=config.get_lookup_default('x_h'),
    )
    
    u_r = None
    if config.u_r_use_network:
        u_r = LookupTableIntrinsicRewardNetwork(
            eta=config.eta,
            default_y=config.get_lookup_default('u_r'),
        )
    
    v_r = None
    if config.v_r_use_network:
        v_r = LookupTableRobotValueNetwork(
            gamma_r=config.gamma_r,
            default_v_r=config.get_lookup_default('v_r'),
        )
    
    return q_r, v_h_e, x_h, u_r, v_r
