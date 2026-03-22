"""
Phase 1 PPO: Goal-Conditioned Human Policy Prior via PufferLib PPO.

This module implements a PPO-based approach for Phase 1 of the EMPO framework,
computing a goal-conditioned policy π_h(a|s,g) that approximates the human
policy prior.  The reward signal is binary goal achievement plus optional
reward shaping.

**PufferLib integration**: The training loop uses ``pufferlib.pufferl.PuffeRL``
for rollout collection, advantage computation (GAE), and PPO clipped surrogate
updates.  ``pufferlib.vector.make()`` creates vectorised environments.
``GoalConditionedActorCritic`` follows PufferLib's
``forward(observations, state) → (logits, value)`` policy convention.

This is a **parallel** implementation to the existing DQN-style Phase 1 trainer.
It does NOT modify or extend any existing code in ``learning_based/phase1/``.
The existing DQN trainer, its networks, config, replay buffer, and all supporting
code remain untouched.

Key differences from the DQN path:
- Explicit policy network π_h (actor) instead of Q-derived Boltzmann softmax
- PufferLib PPO on-policy training instead of off-policy replay buffer
- Goal included in observation (goal-conditioned observation vector)
- No separate Q-network; value function V_h(s,g) via PPO critic
"""

from .config import PPOPhase1Config
from .actor_critic import GoalConditionedActorCritic
from .env_wrapper import Phase1PPOEnv
from .trainer import PPOPhase1Trainer

__all__ = [
    "PPOPhase1Config",
    "GoalConditionedActorCritic",
    "Phase1PPOEnv",
    "PPOPhase1Trainer",
]