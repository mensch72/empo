"""
BushWorld-specific Phase 2 Trainer.

Provides the training entry point for Phase 2 of the EMPO framework
(equations 4-9) specialised for BushWorld, reusing the shared
:class:`BasePhase2Trainer` exactly like the multigrid implementation.

Unlike multigrid, BushWorld networks each own their encoders (they are not
shared between networks). This keeps the wiring simple and avoids a single
encoder's parameters being updated by multiple optimisers. The RND-specific
hooks therefore use the V_h^e network's encoders as the canonical state/agent
encoders for novelty features.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.phase2.network_factory import create_count_based_curiosity
from empo.learning_based.phase2.trainer import BasePhase2Trainer, Phase2Networks

from .aggregate_goal_ability import BushWorldAggregateGoalAbilityNetwork
from .human_goal_ability import BushWorldHumanGoalAchievementNetwork
from .intrinsic_reward_network import BushWorldIntrinsicRewardNetwork
from .robot_q_network import BushWorldRobotQNetwork
from .robot_value_network import BushWorldRobotValueNetwork


class BushWorldPhase2Trainer(BasePhase2Trainer):
    """Phase 2 trainer for BushWorld environments."""

    def __init__(
        self,
        env: Any,
        networks: Phase2Networks,
        config: Phase2Config,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        human_policy_prior: Callable,
        goal_sampler: Callable,
        device: str = "cpu",
        verbose: bool = False,
        debug: bool = False,
        tensorboard_dir: Optional[str] = None,
        profiler: Optional[Any] = None,
        world_model_factory: Optional[Any] = None,
        robot_exploration_policy: Optional[Any] = None,
        human_exploration_policy: Optional[Any] = None,
        checkpoint_interval: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(
            env=env,
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
            profiler=profiler,
            world_model_factory=world_model_factory,
            robot_exploration_policy=robot_exploration_policy,
            human_exploration_policy=human_exploration_policy,
            checkpoint_interval=checkpoint_interval,
            checkpoint_path=checkpoint_path,
        )

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #
    def _iter_unique_encoders(self):
        """Yield ``(name, encoder)`` for all unique encoders used by Phase 2 nets."""
        seen_encoder_ids = set()
        for network_name in ("q_r", "v_h_e", "x_h", "u_r", "v_r"):
            network = getattr(self.networks, network_name, None)
            if network is None:
                continue
            for encoder_name in ("state_encoder", "goal_encoder", "agent_encoder"):
                encoder = getattr(network, encoder_name, None)
                if encoder is None:
                    continue
                encoder_id = id(encoder)
                if encoder_id in seen_encoder_ids:
                    continue
                seen_encoder_ids.add(encoder_id)
                yield f"{network_name}.{encoder_name}", encoder

    def clear_caches(self):
        """Clear encoder caches across all instantiated BushWorld Phase 2 networks."""
        for _name, encoder in self._iter_unique_encoders():
            if hasattr(encoder, "clear_cache"):
                encoder.clear_cache()

    def get_cache_stats(self) -> Dict[str, Tuple[int, int]]:
        stats: Dict[str, Tuple[int, int]] = {}
        for name, encoder in self._iter_unique_encoders():
            if hasattr(encoder, "get_cache_stats"):
                stats[name] = encoder.get_cache_stats()
        return stats

    def reset_cache_stats(self):
        for _name, encoder in self._iter_unique_encoders():
            if hasattr(encoder, "reset_cache_stats"):
                encoder.reset_cache_stats()

    # ------------------------------------------------------------------ #
    # RND feature hooks
    # ------------------------------------------------------------------ #
    def get_rnd_encoder_coefficients(self, step: int) -> List[float]:
        """BushWorld uses a single state encoder for RND (or none)."""
        if self.networks.rnd_encoder_dims is None:
            return []
        return [1.0 for _ in self.networks.rnd_encoder_dims]

    def get_state_features_for_rnd(
        self,
        states: List[Any],
        encoder_coefficients: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, List[float]]:
        if encoder_coefficients is None:
            encoder_coefficients = self.get_rnd_encoder_coefficients(self.training_step_count)

        state_encoder = self.networks.v_h_e.state_encoder
        if self.networks.rnd is not None:
            feature_dim = self.networks.rnd.input_dim
        else:
            feature_dim = state_encoder.feature_dim

        if not states:
            return torch.zeros(0, feature_dim, device=self.device), encoder_coefficients

        if hasattr(state_encoder, "tensorize_state"):
            inputs = torch.cat(
                [state_encoder.tensorize_state(s, self.env, self.device) for s in states],
                dim=0,
            )
            with torch.no_grad():
                features = state_encoder.forward(inputs)
            return features, encoder_coefficients

        # Lookup-table fallback (RND should be disabled in this mode).
        features = []
        for state in states:
            state_hash = hash(state) % (2 ** 31)
            gen = torch.Generator(device=self.device).manual_seed(state_hash)
            features.append(torch.randn(feature_dim, device=self.device, generator=gen))
        return torch.stack(features), encoder_coefficients

    def get_human_features_for_rnd(
        self, state: Any, human_agent_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_humans = len(human_agent_indices)
        state_encoder = self.networks.v_h_e.state_encoder
        agent_encoder = self.networks.v_h_e.agent_encoder
        state_dim = state_encoder.feature_dim
        agent_dim = agent_encoder.output_dim

        if num_humans == 0:
            return (
                torch.zeros(0, state_dim, device=self.device),
                torch.zeros(0, agent_dim, device=self.device),
            )

        if hasattr(state_encoder, "tensorize_state"):
            state_input = state_encoder.tensorize_state(state, self.env, self.device)
            with torch.no_grad():
                state_features_single = state_encoder.forward(state_input)
            state_features = state_features_single.expand(num_humans, -1)

            agent_features_list = []
            for h in human_agent_indices:
                with torch.no_grad():
                    agent_input = agent_encoder.tensorize_single(h, state, self.env, self.device)
                    agent_features_list.append(agent_encoder.forward(agent_input))
            agent_features = torch.cat(agent_features_list, dim=0)
            return state_features, agent_features

        # Lookup-table fallback.
        state_hash = hash(state) % (2 ** 31)
        gen = torch.Generator(device=self.device).manual_seed(state_hash)
        state_features = torch.randn(num_humans, state_dim, device=self.device, generator=gen)
        agent_features_list = []
        for h in human_agent_indices:
            agent_hash = hash((state, h)) % (2 ** 31)
            gen = torch.Generator(device=self.device).manual_seed(agent_hash)
            agent_features_list.append(torch.randn(1, agent_dim, device=self.device, generator=gen))
        return state_features, torch.cat(agent_features_list, dim=0)


def create_phase2_networks(
    env: Any,
    config: Phase2Config,
    num_robots: int,
    num_actions: int,
    hidden_dim: Optional[int] = None,
    device: str = "cpu",
    goal_feature_dim: Optional[int] = None,
) -> Phase2Networks:
    """Create all Phase 2 networks for a BushWorld environment.

    Supports the same lookup-table / neural / mixed modes as the multigrid
    factory. In neural mode each network owns its encoders.
    """
    use_neural_q_r = not config.should_use_lookup_table("q_r")
    use_neural_v_h_e = not config.should_use_lookup_table("v_h_e")
    use_neural_x_h = config.x_h_use_network and not config.should_use_lookup_table("x_h")
    use_neural_u_r = config.u_r_use_network and not config.should_use_lookup_table("u_r")
    use_neural_v_r = config.v_r_use_network and not config.should_use_lookup_table("v_r")

    any_neural = (
        use_neural_q_r or use_neural_v_h_e or use_neural_x_h or use_neural_u_r or use_neural_v_r
    )

    if not any_neural:
        from empo.learning_based.phase2.lookup import (
            LookupTableAggregateGoalAbilityNetwork,
            LookupTableHumanGoalAbilityNetwork,
            LookupTableIntrinsicRewardNetwork,
            LookupTableRobotQNetwork,
            LookupTableRobotValueNetwork,
        )

        q_r = LookupTableRobotQNetwork(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=config.beta_r,
            default_q_r=config.get_lookup_default("q_r"),
            include_step_count=config.include_step_count,
        )
        v_h_e = LookupTableHumanGoalAbilityNetwork(
            gamma_h=config.gamma_h,
            default_v_h_e=config.get_lookup_default("v_h_e"),
            include_step_count=config.include_step_count,
        )
        x_h = None
        if config.x_h_use_network:
            x_h = LookupTableAggregateGoalAbilityNetwork(
                default_x_h=config.get_lookup_default("x_h"),
                include_step_count=config.include_step_count,
            )
        u_r = None
        if config.u_r_use_network:
            u_r = LookupTableIntrinsicRewardNetwork(
                eta=config.eta,
                default_y=config.get_lookup_default("u_r"),
                include_step_count=config.include_step_count,
            )
        v_r = None
        if config.v_r_use_network:
            v_r = LookupTableRobotValueNetwork(
                gamma_r=config.gamma_r,
                default_v_r=config.get_lookup_default("v_r"),
                include_step_count=config.include_step_count,
            )
        if config.use_rnd:
            import warnings

            warnings.warn(
                "RND curiosity exploration is disabled in full tabular mode. "
                "RND requires learned state representations, but tabular mode uses exact "
                "state hashes. For small state spaces, epsilon-greedy exploration is sufficient.",
                UserWarning,
                stacklevel=2,
            )
        count_curiosity = create_count_based_curiosity(config)
        return Phase2Networks(
            q_r=q_r,
            v_h_e=v_h_e,
            x_h=x_h,
            u_r=u_r,
            v_r=v_r,
            rnd=None,
            count_curiosity=count_curiosity,
        )

    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    if goal_feature_dim is None:
        goal_feature_dim = config.goal_feature_dim

    grid_height = env.height
    grid_width = env.width
    B = env.B
    max_steps = env.max_steps
    num_agents = env.num_players
    use_encoders = config.use_encoders

    # --- V_h^e ---
    if use_neural_v_h_e:
        v_h_e = BushWorldHumanGoalAchievementNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            num_agents=num_agents,
            max_steps=max_steps,
            state_feature_dim=hidden_dim,
            goal_feature_dim=goal_feature_dim,
            hidden_dim=hidden_dim,
            gamma_h=config.gamma_h,
            use_encoders=use_encoders,
        ).to(device)
    else:
        from empo.learning_based.phase2.lookup import LookupTableHumanGoalAbilityNetwork

        v_h_e = LookupTableHumanGoalAbilityNetwork(
            gamma_h=config.gamma_h,
            default_v_h_e=config.get_lookup_default("v_h_e"),
            include_step_count=config.include_step_count,
        )

    # --- Q_r ---
    if use_neural_q_r:
        q_r = BushWorldRobotQNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robot_actions=num_actions,
            num_robots=num_robots,
            num_humans=env.num_humans,
            max_steps=max_steps,
            state_feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            beta_r=config.beta_r,
            use_encoders=use_encoders,
        ).to(device)
    else:
        from empo.learning_based.phase2.lookup import LookupTableRobotQNetwork

        q_r = LookupTableRobotQNetwork(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=config.beta_r,
            default_q_r=config.get_lookup_default("q_r"),
            include_step_count=config.include_step_count,
        )

    # --- X_h ---
    x_h = None
    if config.x_h_use_network:
        if use_neural_x_h:
            x_h = BushWorldAggregateGoalAbilityNetwork(
                grid_height=grid_height,
                grid_width=grid_width,
                B=B,
                num_robots=num_robots,
                num_agents=num_agents,
                max_steps=max_steps,
                state_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                zeta=config.zeta,
                use_encoders=use_encoders,
            ).to(device)
        else:
            from empo.learning_based.phase2.lookup import LookupTableAggregateGoalAbilityNetwork

            x_h = LookupTableAggregateGoalAbilityNetwork(
                default_x_h=config.get_lookup_default("x_h"),
                include_step_count=config.include_step_count,
            )

    # --- U_r ---
    u_r = None
    if config.u_r_use_network:
        if use_neural_u_r:
            u_r = BushWorldIntrinsicRewardNetwork(
                grid_height=grid_height,
                grid_width=grid_width,
                B=B,
                num_robots=num_robots,
                max_steps=max_steps,
                state_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                xi=config.xi,
                eta=config.eta,
                use_encoders=use_encoders,
            ).to(device)
        else:
            from empo.learning_based.phase2.lookup import LookupTableIntrinsicRewardNetwork

            u_r = LookupTableIntrinsicRewardNetwork(
                eta=config.eta,
                default_y=config.get_lookup_default("u_r"),
                include_step_count=config.include_step_count,
            )

    # --- V_r ---
    v_r = None
    if config.v_r_use_network:
        if use_neural_v_r:
            v_r = BushWorldRobotValueNetwork(
                grid_height=grid_height,
                grid_width=grid_width,
                B=B,
                num_robots=num_robots,
                max_steps=max_steps,
                state_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                gamma_r=config.gamma_r,
                use_encoders=use_encoders,
            ).to(device)
        else:
            from empo.learning_based.phase2.lookup import LookupTableRobotValueNetwork

            v_r = LookupTableRobotValueNetwork(
                gamma_r=config.gamma_r,
                default_v_r=config.get_lookup_default("v_r"),
                include_step_count=config.include_step_count,
            )

    rnd = None
    rnd_encoder_dims = None
    if config.use_rnd and use_neural_v_h_e:
        from empo.learning_based.phase2.rnd import RNDModule

        rnd_encoder_dims = [v_h_e.state_encoder.feature_dim]
        rnd = RNDModule(
            input_dim=sum(rnd_encoder_dims),
            encoder_dims=rnd_encoder_dims,
            feature_dim=config.rnd_feature_dim,
            hidden_dim=config.rnd_hidden_dim,
            normalize=config.normalize_rnd,
            normalization_decay=config.rnd_normalization_decay,
        ).to(device)
    elif config.use_rnd:
        import warnings

        warnings.warn(
            "RND requires a neural V_h^e; disabling RND for this BushWorld configuration.",
            UserWarning,
            stacklevel=2,
        )

    count_curiosity = create_count_based_curiosity(config)

    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r,
        rnd=rnd,
        rnd_encoder_dims=rnd_encoder_dims,
        count_curiosity=count_curiosity,
    )


def train_bushworld_phase2(
    world_model: Any,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_policy_prior: Callable,
    goal_sampler: Callable,
    config: Optional[Phase2Config] = None,
    num_training_steps: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    goal_feature_dim: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
    debug: bool = False,
    tensorboard_dir: Optional[str] = None,
    profiler: Optional[Any] = None,
    restore_networks_path: Optional[str] = None,
    world_model_factory: Optional[Any] = None,
    robot_exploration_policy: Optional[Any] = None,
    human_exploration_policy: Optional[Any] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Any, Phase2Networks, List[Dict[str, float]], "BushWorldPhase2Trainer"]:
    """Train a Phase 2 robot policy for a BushWorld environment.

    Returns ``(q_r_network, all_networks, training_history, trainer)``.
    """
    if config is None:
        config = Phase2Config()
    if num_training_steps is not None:
        config.num_training_steps = num_training_steps
    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    if goal_feature_dim is None:
        goal_feature_dim = config.goal_feature_dim

    if hasattr(world_model.action_space, "n"):
        num_actions = world_model.action_space.n
    else:
        num_actions = len(world_model.action_space)
    num_robots = len(robot_agent_indices)

    if verbose:
        print("Creating Phase 2 networks:")
        print(f"  Grid: {world_model.width}x{world_model.height}")
        print(f"  Humans: {len(human_agent_indices)}")
        print(f"  Robots: {num_robots}")
        print(f"  Actions per robot: {num_actions}")
        print(f"  Joint action space: {num_actions ** num_robots} actions")

    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=num_robots,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        device=device,
        goal_feature_dim=goal_feature_dim,
    )

    trainer = BushWorldPhase2Trainer(
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
        profiler=profiler,
        world_model_factory=world_model_factory,
        robot_exploration_policy=robot_exploration_policy,
        human_exploration_policy=human_exploration_policy,
        checkpoint_interval=checkpoint_interval,
        checkpoint_path=checkpoint_path,
    )

    if restore_networks_path is not None:
        if verbose:
            print(f"\nRestoring networks from: {restore_networks_path}")
        trainer.load_all_networks(restore_networks_path)

    if verbose:
        print(f"\nTraining for {config.num_training_steps} training steps...")

    history = trainer.train(config.num_training_steps)

    if verbose:
        print("\nTraining completed!")

    return networks.q_r, networks, history, trainer
