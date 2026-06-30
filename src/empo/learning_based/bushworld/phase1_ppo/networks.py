"""
Factory helpers for creating BushWorld PPO Phase 1 networks.

Provides :func:`create_bushworld_phase1_ppo_networks`, which builds a
:class:`BushWorldStateEncoder` and :class:`BushWorldGoalEncoder` together with a
shared :class:`~empo.learning_based.phase1_ppo.actor_critic.GoalConditionedActorCritic`,
wired so the observation dimensionality matches the concatenated encoder outputs.

This module does NOT modify any code in ``learning_based/bushworld/phase1/``.
"""

from __future__ import annotations

from typing import Any, Tuple

from empo.learning_based.phase1_ppo.actor_critic import GoalConditionedActorCritic
from empo.learning_based.phase1_ppo.config import PPOPhase1Config

from empo.learning_based.bushworld.goal_encoder import BushWorldGoalEncoder
from empo.learning_based.bushworld.state_encoder import BushWorldStateEncoder


def create_bushworld_phase1_ppo_networks(
    env: Any,
    config: PPOPhase1Config,
    *,
    feature_dim: int = 128,
    goal_feature_dim: int = 32,
    use_encoders: bool = True,
    device: str = "cpu",
) -> Tuple[GoalConditionedActorCritic, BushWorldStateEncoder, BushWorldGoalEncoder]:
    """Create all networks for BushWorld PPO Phase 1 training.

    Parameters
    ----------
    env : BushWorld
        A BushWorld environment instance used to infer grid size, ``B``,
        ``num_robots`` and ``max_steps``.
    config : PPOPhase1Config
        PPO Phase 1 configuration.
    feature_dim : int
        Output dimensionality of the state encoder.
    goal_feature_dim : int
        Output dimensionality of the goal encoder.
    use_encoders : bool
        Whether the encoders use neural network layers (True) or
        identity/flattening mode (False).
    device : str
        Torch device string.

    Returns
    -------
    actor_critic : GoalConditionedActorCritic
        The PPO actor-critic network.
    state_encoder : BushWorldStateEncoder
        The state encoder (also needed by :class:`BushWorldPhase1PPOEnv`).
    goal_encoder : BushWorldGoalEncoder
        The goal encoder (also needed by :class:`BushWorldPhase1PPOEnv`).
    """
    num_robots = len(env.robot_agent_indices)

    state_encoder = BushWorldStateEncoder(
        grid_height=env.height,
        grid_width=env.width,
        B=env.B,
        num_robots=num_robots,
        max_steps=env.max_steps,
        feature_dim=feature_dim,
        hidden_dim=feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    goal_encoder = BushWorldGoalEncoder(
        grid_height=env.height,
        grid_width=env.width,
        feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    obs_dim = state_encoder.feature_dim + goal_encoder.feature_dim

    actor_critic = GoalConditionedActorCritic(
        obs_dim=obs_dim,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
    ).to(device)

    return actor_critic, state_encoder, goal_encoder
