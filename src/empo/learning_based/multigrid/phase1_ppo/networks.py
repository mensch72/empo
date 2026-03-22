"""
Factory helpers for creating MultiGrid PPO Phase 1 networks.

Provides :func:`create_multigrid_phase1_ppo_networks` which builds a shared
:class:`MultiGridStateEncoder` and :class:`MultiGridGoalEncoder` together
with a :class:`GoalConditionedActorCritic` — all wired so that the
observation dimensionality matches the concatenated encoder outputs.

This module does NOT modify any code in ``learning_based/multigrid/phase1/``.
"""

from __future__ import annotations

from typing import Any, Tuple

from empo.learning_based.phase1_ppo.actor_critic import GoalConditionedActorCritic
from empo.learning_based.phase1_ppo.config import PPOPhase1Config

from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.goal_encoder import MultiGridGoalEncoder
from empo.learning_based.multigrid.constants import NUM_STANDARD_COLORS
from empo.learning_based.multigrid.feature_extraction import get_num_agents_per_color


def create_multigrid_phase1_ppo_networks(
    env: Any,
    config: PPOPhase1Config,
    *,
    feature_dim: int = 256,
    goal_feature_dim: int = 32,
    use_encoders: bool = True,
    include_step_count: bool = True,
    device: str = "cpu",
) -> Tuple[GoalConditionedActorCritic, MultiGridStateEncoder, MultiGridGoalEncoder]:
    """Create all networks for MultiGrid PPO Phase 1 training.

    Builds a :class:`MultiGridStateEncoder` and :class:`MultiGridGoalEncoder`,
    then creates a :class:`GoalConditionedActorCritic` whose observation
    dimensionality equals ``state_encoder.feature_dim + goal_encoder.feature_dim``.

    Parameters
    ----------
    env : MultiGridEnv
        A MultiGrid environment instance used to infer grid size, agent
        colours, etc.
    config : PPOPhase1Config
        PPO Phase 1 configuration.
    feature_dim : int
        Output dimensionality of the shared state encoder.
    goal_feature_dim : int
        Output dimensionality of the goal encoder.
    use_encoders : bool
        Whether the encoders use neural network layers (True) or
        identity/flattening mode (False).
    include_step_count : bool
        Whether to include the environment step count in global features.
    device : str
        Torch device string.

    Returns
    -------
    actor_critic : GoalConditionedActorCritic
        The PPO actor-critic network.
    state_encoder : MultiGridStateEncoder
        The shared state encoder (also needed by ``MultiGridPhase1PPOEnv``).
    goal_encoder : MultiGridGoalEncoder
        The goal encoder (also needed by ``MultiGridPhase1PPOEnv``).
    """
    num_agents_per_color = get_num_agents_per_color(env)
    grid_height = env.height
    grid_width = env.width

    # ── Shared state encoder ────────────────────────────────────────────
    state_encoder = MultiGridStateEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=NUM_STANDARD_COLORS,
        feature_dim=feature_dim,
        include_step_count=include_step_count,
        use_encoders=use_encoders,
    ).to(device)

    # ── Goal encoder ────────────────────────────────────────────────────
    goal_encoder = MultiGridGoalEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    # ── Actor-critic ────────────────────────────────────────────────────
    # The observation is the concatenation of state features and goal features.
    # Use the actual feature_dim from the encoder (may differ in identity mode).
    obs_dim = state_encoder.feature_dim + goal_encoder.feature_dim

    actor_critic = GoalConditionedActorCritic(
        obs_dim=obs_dim,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
    ).to(device)

    return actor_critic, state_encoder, goal_encoder
