"""
Phase 2: Robot Policy Learning for Empowerment-Based AI

This module implements the neural network-based learning approach for Phase 2
of the EMPO framework (equations 4-9 from the paper).

Phase 2 computes:
- Q_r(s, a_r): Robot state-action value (eq. 4)
- π_r(s): Robot policy (eq. 5) 
- V_h^e(s, g_h): Effective human goal achievement ability (eq. 6)
- X_h(s): Aggregate goal achievement ability (eq. 7)
- U_r(s): Intrinsic robot reward (eq. 8)
- V_r(s): Robot state value (eq. 9)
"""

from .config import Phase2Config
from .robot_q_network import BaseRobotQNetwork
from .replay_buffer import Phase2Transition, Phase2ReplayBuffer
from .human_goal_ability import BaseHumanGoalAchievementNetwork
from .robot_value_network import BaseRobotValueNetwork
from .intrinsic_reward_network import BaseIntrinsicRewardNetwork
from .aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from .trainer import Phase2Networks, BasePhase2Trainer
from .robot_policy import BaseRobotPolicy

__all__ = [
    'Phase2Config',
    'BaseRobotQNetwork',
    'Phase2Transition',
    'Phase2ReplayBuffer',
    'BaseHumanGoalAchievementNetwork',
    'BaseRobotValueNetwork',
    'BaseIntrinsicRewardNetwork',
    'BaseAggregateGoalAbilityNetwork',
    'Phase2Networks',
    'BasePhase2Trainer',
    'BaseRobotPolicy',
]
