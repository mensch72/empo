"""
Factory helpers for creating MultiGrid PPO Phase 2 networks.

Provides :func:`create_multigrid_ppo_networks` which builds a shared
:class:`MultiGridStateEncoder` together with an :class:`EMPOActorCritic`
and a :class:`PPOAuxiliaryNetworks` container — all wired to use the same
encoder for consistent state-feature extraction.

This module does NOT modify any code in ``learning_based/multigrid/phase2/``.
"""

from __future__ import annotations

from typing import Any, Tuple

from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.trainer import PPOAuxiliaryNetworks

from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.goal_encoder import MultiGridGoalEncoder
from empo.learning_based.multigrid.agent_encoder import AgentIdentityEncoder
from empo.learning_based.multigrid.constants import NUM_STANDARD_COLORS
from empo.learning_based.multigrid.feature_extraction import get_num_agents_per_color

from empo.learning_based.multigrid.phase2.human_goal_ability import (
    MultiGridHumanGoalAchievementNetwork,
)
from empo.learning_based.multigrid.phase2.aggregate_goal_ability import (
    MultiGridAggregateGoalAbilityNetwork,
)
from empo.learning_based.multigrid.phase2.intrinsic_reward_network import (
    MultiGridIntrinsicRewardNetwork,
)


def create_multigrid_ppo_networks(
    env: Any,
    config: PPOPhase2Config,
    *,
    feature_dim: int = 256,
    goal_feature_dim: int = 32,
    max_agents: int = 10,
    agent_embedding_dim: int = 16,
    agent_position_feature_dim: int = 32,
    agent_feature_dim: int = 32,
    use_x_h: bool = True,
    use_u_r: bool = True,
    use_encoders: bool = True,
    include_step_count: bool = True,
    device: str = "cpu",
) -> Tuple[EMPOActorCritic, PPOAuxiliaryNetworks, MultiGridStateEncoder]:
    """Create all networks for MultiGrid PPO Phase 2 training.

    Builds a :class:`MultiGridStateEncoder`, an :class:`EMPOActorCritic`
    whose encoder input dimension matches the state encoder output, and
    a :class:`PPOAuxiliaryNetworks` containing V_h^e (and optionally X_h
    and U_r) that share the same state encoder for consistent feature
    extraction.

    Parameters
    ----------
    env : MultiGridEnv
        A MultiGrid environment instance used to infer grid size, agent
        colours, etc.
    config : PPOPhase2Config
        PPO Phase 2 configuration.
    feature_dim : int
        Output dimensionality of the shared state encoder.
    goal_feature_dim : int
        Output dimensionality of the goal encoder.
    max_agents : int
        Maximum number of agents for the identity encoder.
    agent_embedding_dim : int
        Dimensionality of the agent identity embedding.
    agent_position_feature_dim : int
        Agent position encoding output dimension.
    agent_feature_dim : int
        Agent feature encoding output dimension.
    use_x_h : bool
        Whether to create the X_h aggregate goal ability network.
    use_u_r : bool
        Whether to create the U_r intrinsic reward network.
    use_encoders : bool
        Whether the state encoder uses neural network layers (True) or
        identity/flattening mode (False).
    include_step_count : bool
        Whether to include the environment step count in global features.
    device : str
        Torch device string.

    Returns
    -------
    actor_critic : EMPOActorCritic
        The PPO actor-critic network.
    auxiliary_networks : PPOAuxiliaryNetworks
        Container with V_h^e (and optionally X_h, U_r).
    state_encoder : MultiGridStateEncoder
        The shared state encoder (also needed by ``MultiGridWorldModelEnv``).
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

    # ── Actor-critic ────────────────────────────────────────────────────
    # The actor-critic receives a flat observation vector of dimension
    # ``state_encoder.feature_dim`` (the encoder is applied inside the
    # env wrapper, not inside the actor-critic).
    actor_critic = EMPOActorCritic(
        state_encoder=None,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
        num_robots=config.num_robots,
        obs_dim=state_encoder.feature_dim,
    ).to(device)

    # ── Goal encoder (for V_h^e) ────────────────────────────────────────
    goal_encoder = MultiGridGoalEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    # ── Agent identity encoder (for V_h^e, X_h) ────────────────────────
    agent_encoder = AgentIdentityEncoder(
        num_agents=max_agents,
        embedding_dim=agent_embedding_dim,
        position_feature_dim=agent_position_feature_dim,
        agent_feature_dim=agent_feature_dim,
        grid_height=grid_height,
        grid_width=grid_width,
        use_encoders=use_encoders,
    ).to(device)

    # Use the encoder's actual feature_dim (post-construction) for the
    # auxiliary networks' ``state_feature_dim``.  When ``use_encoders=False``
    # (identity mode), ``MultiGridStateEncoder`` overwrites ``feature_dim``
    # to match the true flattened output size, which may differ from the
    # ``feature_dim`` parameter passed to this factory.
    actual_state_feature_dim = state_encoder.feature_dim

    # ── V_h^e ───────────────────────────────────────────────────────────
    v_h_e = MultiGridHumanGoalAchievementNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=actual_state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=config.hidden_dim,
        gamma_h=config.gamma_h,
        max_agents=max_agents,
        agent_embedding_dim=agent_embedding_dim,
        state_encoder=state_encoder,
        goal_encoder=goal_encoder,
        agent_encoder=agent_encoder,
    ).to(device)

    # ── X_h (optional) ──────────────────────────────────────────────────
    x_h = None
    if use_x_h:
        x_h = MultiGridAggregateGoalAbilityNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            state_feature_dim=actual_state_feature_dim,
            hidden_dim=config.hidden_dim,
            zeta=config.zeta,
            gamma_h=config.gamma_h,
            feasible_range=(1.0, float('inf')) if config.use_simplified_x_h else (0.0, 1.0),
            max_agents=max_agents,
            agent_embedding_dim=agent_embedding_dim,
            state_encoder=state_encoder,
            agent_encoder=agent_encoder,
        ).to(device)

    # ── U_r (optional) ──────────────────────────────────────────────────
    u_r = None
    if use_u_r:
        u_r = MultiGridIntrinsicRewardNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            state_feature_dim=actual_state_feature_dim,
            hidden_dim=config.hidden_dim,
            eta=config.eta,
            state_encoder=state_encoder,
        ).to(device)

    auxiliary_networks = PPOAuxiliaryNetworks(
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
    )

    return actor_critic, auxiliary_networks, state_encoder
