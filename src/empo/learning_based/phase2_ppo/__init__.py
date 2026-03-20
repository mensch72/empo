"""
Phase 2 PPO: Robot Policy Approximation via Proximal Policy Optimization.

This module implements a PPO-based approach for Phase 2 of the EMPO framework,
computing an explicit (π_r, V_r) actor-critic that approximates the solution
to equations (4)-(9) of the EMPO paper.

This is a **parallel** implementation to the existing DQN-style Phase 2 trainer.
It does NOT modify or extend any existing code in ``learning_based/phase2/``.
The existing DQN trainer, its networks, config, replay buffer, and all supporting
code remain untouched.

Key differences from the DQN path:
- Explicit policy network π_r (actor) instead of Q_r-derived power-law softmax
- PPO on-policy rollout buffer instead of off-policy replay buffer
- Intrinsic reward U_r(s) fed as the environment reward signal to PPO
- Auxiliary networks (V_h^e, X_h, U_r) still trained separately via replay buffer

Shared read-only imports from the DQN path:
- ``Phase2Transition``, ``Phase2ReplayBuffer`` — for auxiliary network training
- ``BaseHumanGoalAchievementNetwork``, ``BaseAggregateGoalAbilityNetwork``,
  ``BaseIntrinsicRewardNetwork`` — base classes for auxiliary networks
- ``WorldModelFactory`` — for creating environment instances
"""

from .config import PPOPhase2Config
from .actor_critic import EMPOActorCritic
from .env_wrapper import EMPOMultiGridEnv
from .trainer import PPOPhase2Trainer

__all__ = [
    "PPOPhase2Config",
    "EMPOActorCritic",
    "EMPOMultiGridEnv",
    "PPOPhase2Trainer",
]
