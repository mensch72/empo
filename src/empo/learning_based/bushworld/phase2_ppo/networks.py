"""
Factory helpers for creating BushWorld PPO Phase 2 networks.

Provides :func:`create_bushworld_ppo_networks`, which builds a shared
:class:`BushWorldStateEncoder` together with an
:class:`~empo.learning_based.phase2_ppo.actor_critic.EMPOActorCritic` and a
:class:`~empo.learning_based.phase2_ppo.trainer.PPOAuxiliaryNetworks` container
(V_h^e, optionally X_h and U_r) — all wired to use the same state encoder for
consistent feature extraction.

This module does NOT modify any code in ``learning_based/bushworld/phase2/``.
"""

from __future__ import annotations

from typing import Any, Tuple

from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.trainer import PPOAuxiliaryNetworks

from empo.learning_based.bushworld.agent_encoder import BushWorldAgentEncoder
from empo.learning_based.bushworld.goal_encoder import BushWorldGoalEncoder
from empo.learning_based.bushworld.state_encoder import BushWorldStateEncoder
from empo.learning_based.bushworld.phase2.aggregate_goal_ability import (
    BushWorldAggregateGoalAbilityNetwork,
)
from empo.learning_based.bushworld.phase2.human_goal_ability import (
    BushWorldHumanGoalAchievementNetwork,
)
from empo.learning_based.bushworld.phase2.intrinsic_reward_network import (
    BushWorldIntrinsicRewardNetwork,
)


def create_bushworld_ppo_networks(
    env: Any,
    config: PPOPhase2Config,
    *,
    feature_dim: int = 128,
    goal_feature_dim: int = 32,
    agent_feature_dim: int = 16,
    use_x_h: bool = True,
    use_u_r: bool = True,
    use_encoders: bool = True,
    device: str = "cpu",
) -> Tuple[EMPOActorCritic, PPOAuxiliaryNetworks, BushWorldStateEncoder]:
    """Create all networks for BushWorld PPO Phase 2 training.

    Parameters
    ----------
    env : BushWorld
        A BushWorld environment instance used to infer grid size, ``B``,
        ``num_robots``, ``num_players`` and ``max_steps``.
    config : PPOPhase2Config
        PPO Phase 2 configuration.
    feature_dim : int
        Output dimensionality of the shared state encoder.
    goal_feature_dim : int
        Output dimensionality of the goal encoder (for V_h^e).
    agent_feature_dim : int
        Output dimensionality of the agent encoder (for V_h^e and X_h).
    use_x_h : bool
        Whether to create the X_h aggregate goal ability network.
    use_u_r : bool
        Whether to create the U_r intrinsic reward network.
    use_encoders : bool
        Whether the encoders use neural network layers (True) or
        identity/flattening mode (False).
    device : str
        Torch device string.

    Returns
    -------
    actor_critic : EMPOActorCritic
        The PPO actor-critic network.
    auxiliary_networks : PPOAuxiliaryNetworks
        Container with V_h^e (and optionally X_h, U_r).
    state_encoder : BushWorldStateEncoder
        The shared state encoder (also needed by :class:`BushWorldWorldModelEnv`).
    """
    num_robots = len(env.robot_agent_indices)
    num_agents = env.num_players
    grid_height = env.height
    grid_width = env.width
    B = env.B
    max_steps = env.max_steps

    # ── Shared state encoder ────────────────────────────────────────────
    state_encoder = BushWorldStateEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        B=B,
        num_robots=num_robots,
        max_steps=max_steps,
        feature_dim=feature_dim,
        hidden_dim=feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    actual_state_feature_dim = state_encoder.feature_dim

    # ── Actor-critic (encoder applied in the env wrapper) ───────────────
    actor_critic = EMPOActorCritic(
        state_encoder=None,
        hidden_dim=config.hidden_dim,
        num_actions=config.num_actions,
        num_robots=config.num_robots,
        obs_dim=actual_state_feature_dim,
    ).to(device)

    # ── Goal + agent encoders (for V_h^e / X_h) ─────────────────────────
    goal_encoder = BushWorldGoalEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    agent_encoder = BushWorldAgentEncoder(
        num_agents=num_agents,
        grid_height=grid_height,
        grid_width=grid_width,
        output_dim=agent_feature_dim,
        hidden_dim=agent_feature_dim,
        use_encoders=use_encoders,
    ).to(device)

    # ── V_h^e ───────────────────────────────────────────────────────────
    v_h_e = BushWorldHumanGoalAchievementNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        B=B,
        num_robots=num_robots,
        num_agents=num_agents,
        max_steps=max_steps,
        state_feature_dim=actual_state_feature_dim,
        goal_feature_dim=goal_encoder.feature_dim,
        agent_feature_dim=agent_encoder.output_dim,
        hidden_dim=config.hidden_dim,
        gamma_h=config.gamma_h,
        use_encoders=use_encoders,
        state_encoder=state_encoder,
        goal_encoder=goal_encoder,
        agent_encoder=agent_encoder,
    ).to(device)

    # ── X_h (optional) ──────────────────────────────────────────────────
    x_h = None
    if use_x_h:
        x_h = BushWorldAggregateGoalAbilityNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            num_agents=num_agents,
            max_steps=max_steps,
            state_feature_dim=actual_state_feature_dim,
            agent_feature_dim=agent_encoder.output_dim,
            hidden_dim=config.hidden_dim,
            zeta=config.zeta,
            use_encoders=use_encoders,
            state_encoder=state_encoder,
            agent_encoder=agent_encoder,
        ).to(device)

    # ── U_r (optional) ──────────────────────────────────────────────────
    u_r = None
    if use_u_r:
        u_r = BushWorldIntrinsicRewardNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            max_steps=max_steps,
            state_feature_dim=actual_state_feature_dim,
            hidden_dim=config.hidden_dim,
            xi=config.xi,
            eta=config.eta,
            use_encoders=use_encoders,
            state_encoder=state_encoder,
        ).to(device)

    auxiliary_networks = PPOAuxiliaryNetworks(
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
    )

    return actor_critic, auxiliary_networks, state_encoder
