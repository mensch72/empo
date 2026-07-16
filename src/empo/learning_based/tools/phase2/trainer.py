"""
Tools-specific Phase 2 (DQN) trainer.

Provides :class:`ToolsPhase2Trainer` and the convenience function
:func:`train_tools_phase2` that mirrors
``empo.learning_based.multigrid.phase2.train_multigrid_phase2``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.phase2.trainer import BasePhase2Trainer, Phase2Networks
from empo.learning_based.phase2.network_factory import create_count_based_curiosity
from empo.learning_based.tools.state_encoder import ToolsStateEncoder
from empo.world_specific_helpers.tools import ToolsWorldModel

from .networks import (
    ToolsRobotQNetwork,
    ToolsRobotValueNetwork,
    ToolsHumanGoalAchievementNetwork,
    ToolsAggregateGoalAbilityNetwork,
    ToolsIntrinsicRewardNetwork,
)

# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------


class ToolsPhase2Trainer(BasePhase2Trainer):
    """Phase 2 (DQN) trainer for the tools environment.

    Implements the single abstract method ``get_state_features_for_rnd``
    required by :class:`BasePhase2Trainer`.  All other training logic
    (replay buffer, warm-up schedule, gradient steps) is inherited.
    """

    def get_state_features_for_rnd(
        self,
        states: List[Any],
        encoder_coefficients: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Encode a batch of tools states for RND novelty computation.

        Uses the shared state encoder (detached) to produce feature vectors.
        Since tools environments are small, RND is unlikely to be used, but
        this method is required by the base class.
        """
        coefficients = encoder_coefficients or [1.0]
        encoder: ToolsStateEncoder = self.networks.v_h_e.state_encoder
        features_list: List[torch.Tensor] = []
        for state in states:
            x = encoder.tensorize_state(state, self.env, device=self.device)
            feat = encoder(x).detach()
            features_list.append(feat.squeeze(0))
        return torch.stack(features_list), coefficients

    # ------------------------------------------------------------------
    # Override to respect per-agent action counts
    # ------------------------------------------------------------------

    def sample_human_actions(
        self,
        state: Any,
        goals: Dict[int, Any],
    ) -> List[int]:
        """Sample human actions using per-agent action counts.

        In the tools environment every agent can have a different number
        of valid actions (depending on its ``give_targets``).  The base
        implementation uses the global ``env.action_space.n`` which is
        the *maximum* across agents and can produce invalid indices for
        agents with fewer actions.
        """
        epsilon_h = self.config.get_epsilon_h(self.training_step_count)
        actions: List[int] = []
        for h in self.human_agent_indices:
            n = self.env.n_actions_per_agent[h]
            goal = goals.get(h)
            if torch.rand(1).item() < epsilon_h:
                actions.append(np.random.randint(0, n))
            else:
                probs = self.human_policy_prior(state, h, goal)
                probs = np.asarray(probs[:n], dtype=np.float64)
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(n) / n
                actions.append(int(np.random.choice(n, p=probs)))
        return actions


# -------------------------------------------------------------------
# Network factory
# -------------------------------------------------------------------


def create_tools_phase2_networks(
    env: ToolsWorldModel,
    config: Phase2Config,
    *,
    hidden_dim: int = 128,
    feature_dim: int = 64,
    device: str = "cpu",
) -> Phase2Networks:
    """Create all Phase 2 (DQN) networks for a tools environment.

    Parameters
    ----------
    env : ToolsWorldModel
        A tools world model instance (must already be ``reset()``).
    config : Phase2Config
        Phase 2 DQN configuration.
    hidden_dim : int
        MLP hidden width.
    feature_dim : int
        State encoder output dimension.
    device : str
        Torch device string.

    Returns
    -------
    Phase2Networks
        Container with q_r, v_h_e, x_h, u_r, v_r (and count curiosity
        if configured).
    """
    n_agents = env.n_agents
    n_tools = env.n_tools
    max_steps = env.max_steps
    num_robots = len(env.robot_agent_indices)
    num_actions = max(env.n_actions_per_agent[r] for r in env.robot_agent_indices)

    # Shared state encoder
    shared_encoder = ToolsStateEncoder(
        n_agents=n_agents,
        n_tools=n_tools,
        max_steps=max_steps,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    # Q_r
    q_r = ToolsRobotQNetwork(
        state_encoder=shared_encoder,
        num_robot_actions=num_actions,
        num_robots=num_robots,
        hidden_dim=hidden_dim,
        beta_r=config.beta_r,
        use_z_space=getattr(config, "use_z_space_transform", False),
        eta=config.eta,
        xi=config.xi,
    ).to(device)

    # V_h^e
    v_h_e = ToolsHumanGoalAchievementNetwork(
        state_encoder=shared_encoder,
        n_agents=n_agents,
        n_tools=n_tools,
        hidden_dim=hidden_dim,
        gamma_h=config.gamma_h,
    ).to(device)

    # X_h
    x_h: Optional[ToolsAggregateGoalAbilityNetwork] = None
    if getattr(config, "x_h_use_network", True):
        x_h = ToolsAggregateGoalAbilityNetwork(
            state_encoder=shared_encoder,
            n_agents=n_agents,
            hidden_dim=hidden_dim,
            zeta=config.zeta,
        ).to(device)

    # U_r (always use network for tools)
    u_r = ToolsIntrinsicRewardNetwork(
        state_encoder=shared_encoder,
        hidden_dim=hidden_dim,
        xi=config.xi,
        eta=config.eta,
    ).to(device)

    # V_r
    v_r = ToolsRobotValueNetwork(
        state_encoder=shared_encoder,
        hidden_dim=hidden_dim,
        gamma_r=config.gamma_r,
        use_z_space=getattr(config, "use_z_space_transform", False),
        eta=config.eta,
        xi=config.xi,
    ).to(device)

    count_curiosity = create_count_based_curiosity(config)

    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r,
        count_curiosity=count_curiosity,
    )


# -------------------------------------------------------------------
# Convenience entry-point
# -------------------------------------------------------------------


def train_tools_phase2(
    world_model: ToolsWorldModel,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_policy_prior: Callable,
    goal_sampler: Callable,
    config: Optional[Phase2Config] = None,
    num_training_steps: Optional[int] = None,
    hidden_dim: int = 128,
    feature_dim: int = 64,
    device: str = "cpu",
    verbose: bool = True,
    debug: bool = False,
    tensorboard_dir: Optional[str] = None,
    restore_networks_path: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[
    ToolsRobotQNetwork, Phase2Networks, List[Dict[str, float]], ToolsPhase2Trainer
]:
    """Train Phase 2 robot policy for a tools environment.

    This is the tools counterpart of
    ``empo.learning_based.multigrid.phase2.train_multigrid_phase2``.

    Returns
    -------
    q_r : ToolsRobotQNetwork
    networks : Phase2Networks
    history : list[dict[str, float]]
    trainer : ToolsPhase2Trainer
    """
    if config is None:
        config = Phase2Config()

    if num_training_steps is not None:
        config.num_training_steps = num_training_steps

    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    if feature_dim is None:
        feature_dim = getattr(config, "state_feature_dim", 64)

    networks = create_tools_phase2_networks(
        env=world_model,
        config=config,
        hidden_dim=hidden_dim,
        feature_dim=feature_dim,
        device=device,
    )

    trainer = ToolsPhase2Trainer(
        env=world_model,
        networks=networks,
        config=config,
        human_agent_indices=human_agent_indices,
        robot_agent_indices=robot_agent_indices,
        human_policy_prior=human_policy_prior,
        goal_sampler=goal_sampler,
        device=device,
        verbose=verbose,
        debug=debug,
        tensorboard_dir=tensorboard_dir,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
    )

    if restore_networks_path is not None:
        if verbose:
            print(f"\nRestoring networks from: {restore_networks_path}")
        trainer.load_all_networks(restore_networks_path)

    if verbose:
        print(f"\nTraining for {config.num_training_steps} training steps...")
        print(f"  Steps per episode: {config.steps_per_episode}")
        print(f"  Training steps per env step: {config.training_steps_per_env_step}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size}")
        if tensorboard_dir:
            print(f"  TensorBoard: {tensorboard_dir}")
        print(f"\n{config.format_stages_table()}")

    history = trainer.train(config.num_training_steps)

    if verbose:
        print("\nTraining completed!")
        if history:
            final = history[-1]
            print(f"  Final losses: {final}")

    return networks.q_r, networks, history, trainer
