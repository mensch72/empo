"""
Multigrid-specific Phase 2 Trainer.

This module provides the training function for Phase 2 of the EMPO framework
(equations 4-9) specialized for multigrid environments.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from gym_multigrid.multigrid import MultiGridEnv

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.trainer import BasePhase2Trainer, Phase2Networks
from empo.nn_based.phase2.replay_buffer import Phase2Transition
from empo.nn_based.multigrid import MultiGridStateEncoder
from empo.nn_based.multigrid.goal_encoder import MultiGridGoalEncoder
from empo.nn_based.multigrid.agent_encoder import AgentIdentityEncoder

from .robot_q_network import MultiGridRobotQNetwork
from .human_goal_ability import MultiGridHumanGoalAchievementNetwork
from .aggregate_goal_ability import MultiGridAggregateGoalAbilityNetwork
from .intrinsic_reward_network import MultiGridIntrinsicRewardNetwork
from .robot_value_network import MultiGridRobotValueNetwork

# Lookup table network imports
from empo.nn_based.phase2.lookup import is_lookup_table_network


class MultiGridPhase2Trainer(BasePhase2Trainer):
    """
    Phase 2 trainer for multigrid environments.
    
    Implements environment-specific methods for:
    - State encoding using MultiGridStateEncoder
    - Goal achievement checking
    - Environment stepping
    
    All networks share the same encoders (state, goal, agent identity), which
    have internal caching for raw tensor extraction. This ensures:
    1. Consistent representations across all networks
    2. Efficient caching without redundant computation
    3. Proper gradient flow (caches raw tensors, not NN outputs)
    
    .. warning:: ASYNC TRAINING / PICKLE COMPATIBILITY
    
        The **entire trainer object** is pickled and sent to spawned actor processes
        during async training (via bound method ``_actor_process_entry``). This means
        all attributes must be picklable.
        
        To avoid breaking async functionality:
        
        1. **Pickle size matters.** Large pickles (>64MB) can exceed Docker's default
           shared memory and cause SIGBUS errors. When use_encoders=False, we avoid
           creating unused NN layers to keep pickle size small (~10MB vs ~130MB).
        
        2. **All attributes must be picklable.** Avoid:
           - Open file handles, database connections
           - Lambda functions or local functions as attributes
           - Thread locks (use multiprocessing locks from context instead)
           - Non-picklable third-party objects
        
        3. **The env attribute may not be picklable.** Use ``world_model_factory``
           to create fresh environment instances in worker processes.
        
        4. **Caches are NOT preserved.** The encoder caches (_raw_cache) will be
           empty in worker processes. This is expected and fine.
        
        5. **Test with async mode after changes:** Always verify changes work with
           ``--async`` flag in the phase2 demo, both inside and outside Docker.
    
    Args:
        env: MultiGridEnv instance.
        networks: Phase2Networks container.
        config: Phase2Config.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: HumanPolicyPrior instance with .sample(state, human_idx, goal) method.
        goal_sampler: PossibleGoalSampler instance with .sample(state, human_idx) -> (goal, weight) method.
        device: Torch device.
        verbose: Enable progress output (tqdm progress bar).
        debug: Enable verbose debug output.
        tensorboard_dir: Directory for TensorBoard logs (optional).
        robot_exploration_policy: Optional policy for robot epsilon exploration. Can be:
            - None: Use uniform random policy for exploration (default).
            - List[float]: Fixed action probabilities (length = num_action_combinations).
            - Callable[[state, world_model], List[float]]: Function returning action probabilities.
            - RobotPolicy: A proper policy object with sample(state) method returning action tuple.
        human_exploration_policy: Optional policy for human epsilon exploration. Can be:
            - None: Use uniform random policy for exploration (default).
            - HumanExplorationPolicy: A policy object with sample(state, human_idx, goal) method.
        world_model_factory: Optional factory for creating world models. Required for
            async training where the environment cannot be pickled.
    """
    
    def __init__(
        self,
        env: MultiGridEnv,
        networks: Phase2Networks,
        config: Phase2Config,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        human_policy_prior: Callable,
        goal_sampler: Callable,
        device: str = 'cpu',
        verbose: bool = False,
        debug: bool = False,
        tensorboard_dir: Optional[str] = None,
        profiler: Optional[Any] = None,
        world_model_factory: Optional[Any] = None,
        robot_exploration_policy: Optional[Any] = None,
        human_exploration_policy: Optional[Any] = None,
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
        )
        
        # Caching is now handled internally by the shared encoders.
        # The shared state/goal/agent encoders cache raw tensor extraction
        # (but not NN output, to preserve gradient flow).

    def clear_caches(self):
        """
        Clear encoder caches. Call after each training step to prevent memory growth.
        
        This clears the caches in the shared encoders that all networks use.
        For lookup table networks with null encoders, this is a no-op.
        """
        # All networks share encoders from V_h^e, so we clear through V_h^e
        # (NullEncoders have no-op clear_cache methods)
        self.networks.v_h_e.state_encoder.clear_cache()
        self.networks.v_h_e.goal_encoder.clear_cache()
        self.networks.v_h_e.agent_encoder.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Tuple[int, int]]:
        """
        Get cache hit/miss statistics from all encoders.
        
        For null encoders (when V_h^e is lookup table), returns (0, 0).
        
        Returns:
            Dict mapping encoder name to (hits, misses) tuple.
        """
        return {
            'state': self.networks.v_h_e.state_encoder.get_cache_stats(),
            'goal': self.networks.v_h_e.goal_encoder.get_cache_stats(),
            'agent': self.networks.v_h_e.agent_encoder.get_cache_stats(),
        }
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters for all encoders.
        
        For null encoders, this is a no-op.
        """
        self.networks.v_h_e.state_encoder.reset_cache_stats()
        self.networks.v_h_e.goal_encoder.reset_cache_stats()
        self.networks.v_h_e.agent_encoder.reset_cache_stats()

    def get_rnd_encoder_coefficients(self, step: int) -> List[float]:
        """
        Compute warmup coefficients for each encoder used by RND.
        
        Each encoder's coefficient ramps from 0 to 1 during the warmup stage
        when that encoder is introduced. This provides smooth transitions
        as new encoders come online.
        
        Encoder introduction order (matching rnd_encoder_dims):
        - shared_state_encoder: Stage 0 (V_h^e warmup)
        - x_h_own_state_encoder: Stage 1 (X_h warmup)
        - u_r_own_state_encoder: Stage 2 (U_r warmup, if u_r_use_network)
        - q_r_own_state_encoder: Stage 3 (Q_r warmup)
        
        Args:
            step: Current training step.
            
        Returns:
            List of coefficients [0, 1] for each encoder, in same order as
            rnd_encoder_dims.
        """
        if self.networks.rnd_encoder_dims is None:
            return []
        
        # Get warmup stage boundaries from config
        v_h_e_end = self.config._warmup_v_h_e_end
        x_h_end = self.config._warmup_x_h_end
        u_r_end = self.config._warmup_u_r_end
        q_r_end = self.config._warmup_q_r_end
        
        coefficients = []
        encoder_idx = 0
        
        # Shared encoder (introduced at step 0, ramps during V_h^e warmup)
        if encoder_idx < len(self.networks.rnd_encoder_dims):
            if v_h_e_end <= 0:
                coef = 1.0
            elif step >= v_h_e_end:
                coef = 1.0
            else:
                coef = step / v_h_e_end
            coefficients.append(coef)
            encoder_idx += 1
        
        # X_h own encoder (introduced at v_h_e_end, ramps during X_h warmup)
        if encoder_idx < len(self.networks.rnd_encoder_dims):
            stage_start = v_h_e_end
            stage_end = x_h_end
            stage_duration = stage_end - stage_start
            if stage_duration <= 0 or step >= stage_end:
                coef = 1.0
            elif step < stage_start:
                coef = 0.0
            else:
                coef = (step - stage_start) / stage_duration
            coefficients.append(coef)
            encoder_idx += 1
        
        # U_r own encoder (if present, introduced at x_h_end, ramps during U_r warmup)
        if encoder_idx < len(self.networks.rnd_encoder_dims) and self.config.u_r_use_network:
            stage_start = x_h_end
            stage_end = u_r_end
            stage_duration = stage_end - stage_start
            if stage_duration <= 0 or step >= stage_end:
                coef = 1.0
            elif step < stage_start:
                coef = 0.0
            else:
                coef = (step - stage_start) / stage_duration
            coefficients.append(coef)
            encoder_idx += 1
        
        # Q_r own encoder (introduced at u_r_end, ramps during Q_r warmup)
        if encoder_idx < len(self.networks.rnd_encoder_dims):
            stage_start = u_r_end  # u_r_end == x_h_end if u_r_use_network=False
            stage_end = q_r_end
            stage_duration = stage_end - stage_start
            if stage_duration <= 0 or step >= stage_end:
                coef = 1.0
            elif step < stage_start:
                coef = 0.0
            else:
                coef = (step - stage_start) / stage_duration
            coefficients.append(coef)
            encoder_idx += 1
        
        return coefficients

    def get_state_features_for_rnd(
        self,
        states: List[Any],
        encoder_coefficients: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Get concatenated state features from all encoders for RND.
        
        Uses all available state encoders (shared + own encoders) to produce
        concatenated features for RND novelty computation. Each encoder's
        features can be weighted by coefficients for smooth warmup transitions.
        
        Args:
            states: List of states (tuples from WorldModel.get_state()).
            encoder_coefficients: Optional pre-computed coefficients. If None,
                                 will compute from current training_step_count.
            
        Returns:
            Tuple of:
            - Feature tensor of shape (batch_size, total_feature_dim)
            - Encoder coefficients used (for passing to RND)
        """
        # Compute coefficients if not provided
        if encoder_coefficients is None:
            encoder_coefficients = self.get_rnd_encoder_coefficients(
                self.training_step_count
            )
        
        # Get feature dim from RND network
        if self.networks.rnd is not None:
            feature_dim = self.networks.rnd.input_dim
        else:
            feature_dim = self.config.hidden_dim
            
        if not states:
            return torch.zeros(0, feature_dim, device=self.device), encoder_coefficients
        
        # Collect all encoder outputs
        all_features = []
        
        # Shared state encoder from V_h^e
        shared_encoder = self.networks.v_h_e.state_encoder
        
        # Check if we have a neural encoder (not a NullEncoder)
        if hasattr(shared_encoder, 'tensorize_state'):
            # Tensorize all states once (shared by all encoders via cache)
            batch_tensors = []
            for state in states:
                tensors = shared_encoder.tensorize_state(state, self.env)
                batch_tensors.append(tensors)
            
            grid_tensors = torch.cat([t[0] for t in batch_tensors], dim=0).to(self.device)
            global_features = torch.cat([t[1] for t in batch_tensors], dim=0).to(self.device)
            agent_features = torch.cat([t[2] for t in batch_tensors], dim=0).to(self.device)
            interactive_features = torch.cat([t[3] for t in batch_tensors], dim=0).to(self.device)
            
            # Forward through shared encoder (detached - RND trains its own networks)
            with torch.no_grad():
                shared_features = shared_encoder.forward(
                    grid_tensors, global_features, agent_features, interactive_features
                )
            all_features.append(shared_features)
            
            # X_h own state encoder (if neural X_h)
            if hasattr(self.networks.x_h, 'own_state_encoder') and self.networks.x_h.own_state_encoder is not None:
                x_h_encoder = self.networks.x_h.own_state_encoder
                with torch.no_grad():
                    x_h_features = x_h_encoder.forward(
                        grid_tensors, global_features, agent_features, interactive_features
                    )
                all_features.append(x_h_features)
            
            # U_r own state encoder (if neural U_r with network enabled)
            if (self.config.u_r_use_network and 
                self.networks.u_r is not None and 
                hasattr(self.networks.u_r, 'own_state_encoder') and 
                self.networks.u_r.own_state_encoder is not None):
                u_r_encoder = self.networks.u_r.own_state_encoder
                with torch.no_grad():
                    u_r_features = u_r_encoder.forward(
                        grid_tensors, global_features, agent_features, interactive_features
                    )
                all_features.append(u_r_features)
            
            # Q_r own state encoder (if neural Q_r)
            if hasattr(self.networks.q_r, 'own_state_encoder') and self.networks.q_r.own_state_encoder is not None:
                q_r_encoder = self.networks.q_r.own_state_encoder
                with torch.no_grad():
                    q_r_features = q_r_encoder.forward(
                        grid_tensors, global_features, agent_features, interactive_features
                    )
                all_features.append(q_r_features)
            
            # Concatenate all features
            concatenated = torch.cat(all_features, dim=-1)
            return concatenated, encoder_coefficients
        else:
            # Lookup table mode with NullEncoder - RND should be disabled
            # Fall back to hash-based features as before
            if self.networks.rnd is not None:
                feature_dim = self.networks.rnd.input_dim
            else:
                feature_dim = self.config.hidden_dim
            
            features = []
            for state in states:
                state_hash = hash(state) % (2**31)
                gen = torch.Generator(device=self.device).manual_seed(state_hash)
                feat = torch.randn(feature_dim, device=self.device, generator=gen)
                features.append(feat)
            
            return torch.stack(features), encoder_coefficients
    
    def get_human_features_for_rnd(
        self,
        state: Any,
        human_agent_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get state and agent features for human action RND module.
        
        Uses the shared state encoder and agent encoder to produce features
        for each human agent in the given state.
        
        Args:
            state: Current state (tuple from get_state()).
            human_agent_indices: List of human agent indices to get features for.
            
        Returns:
            Tuple of:
            - state_features: (num_humans, state_feature_dim) - same for all humans
            - agent_features: (num_humans, agent_feature_dim) - different for each human
        """
        import torch
        
        num_humans = len(human_agent_indices)
        if num_humans == 0:
            # No humans - return empty tensors
            state_dim = self.networks.v_h_e.state_encoder.feature_dim
            agent_dim = self.networks.v_h_e.agent_encoder.output_dim
            return (
                torch.zeros(0, state_dim, device=self.device),
                torch.zeros(0, agent_dim, device=self.device),
            )
        
        # Get shared encoders from V_h^e
        state_encoder = self.networks.v_h_e.state_encoder
        agent_encoder = self.networks.v_h_e.agent_encoder
        
        # Check if we have neural encoders
        if hasattr(state_encoder, 'tensorize_state'):
            # Tensorize the state once
            tensors = state_encoder.tensorize_state(state, self.env)
            grid_tensor = tensors[0].to(self.device)  # (1, C, H, W)
            global_features = tensors[1].to(self.device)  # (1, global_dim)
            agent_features_raw = tensors[2].to(self.device)  # (1, agent_dim)
            interactive_features = tensors[3].to(self.device)  # (1, interactive_dim)
            
            # Forward through state encoder (detached - human RND has its own networks)
            with torch.no_grad():
                state_features_single = state_encoder.forward(
                    grid_tensor, global_features, agent_features_raw, interactive_features
                )  # (1, state_feature_dim)
            
            # Expand to num_humans (same state features for all humans)
            state_features = state_features_single.expand(num_humans, -1)  # (num_humans, state_feature_dim)
            
            # Get agent features for each human
            agent_features_list = []
            for h in human_agent_indices:
                with torch.no_grad():
                    # Use tensorize_single to extract tensors, then forward through agent encoder
                    # tensorize_single returns: idx (1,), query_grid (1, 1, H, W), query_features (1, feat_dim)
                    # forward() expects: idx (batch,), grid (batch, 1, H, W), features (batch, feat_dim)
                    # So shapes already match with batch=1
                    idx_tensor, query_grid, query_features = agent_encoder.tensorize_single(
                        h, state, self.env, self.device
                    )
                    agent_feat = agent_encoder.forward(
                        idx_tensor, query_grid, query_features
                    )  # (1, agent_feature_dim)
                agent_features_list.append(agent_feat)
            
            agent_features = torch.cat(agent_features_list, dim=0)  # (num_humans, agent_feature_dim)
            
            return state_features, agent_features
        else:
            # Lookup table mode - human action RND should not be enabled
            # But provide a fallback with random features
            state_dim = getattr(state_encoder, 'feature_dim', self.config.hidden_dim)
            agent_dim = getattr(agent_encoder, 'output_dim', 80)  # Default agent feature dim
            
            state_hash = hash(state) % (2**31)
            gen = torch.Generator(device=self.device).manual_seed(state_hash)
            state_features = torch.randn(num_humans, state_dim, device=self.device, generator=gen)
            
            agent_features_list = []
            for h in human_agent_indices:
                agent_hash = hash((state, h)) % (2**31)
                gen = torch.Generator(device=self.device).manual_seed(agent_hash)
                agent_feat = torch.randn(1, agent_dim, device=self.device, generator=gen)
                agent_features_list.append(agent_feat)
            agent_features = torch.cat(agent_features_list, dim=0)
            
            return state_features, agent_features


def create_phase2_networks(
    env: MultiGridEnv,
    config: Phase2Config,
    num_robots: int,
    num_actions: int,
    hidden_dim: Optional[int] = None,
    device: str = 'cpu',
    goal_feature_dim: Optional[int] = None,
    max_agents: int = 10,
    agent_embedding_dim: Optional[int] = None,
    agent_position_feature_dim: Optional[int] = None,
    agent_feature_dim: Optional[int] = None
) -> Phase2Networks:
    """
    Create all Phase 2 networks for a multigrid environment.
    
    Supports three modes based on config:
    
    1. All lookup tables (config.use_lookup_tables=True and all use_lookup_* flags True):
       Creates all networks as lookup tables (no encoders needed).
    
    2. All neural networks (config.use_lookup_tables=False):
       Creates neural networks with SHARED encoders:
       - One state encoder shared by Q_r, V_h^e, X_h, U_r, V_r
       - One goal encoder shared by V_h^e
       - One agent encoder shared by V_h^e, X_h
    
    3. Mixed mode (config.use_lookup_tables=True with some use_lookup_* flags False):
       Creates lookup tables for specified networks and neural networks for others.
       Important: If V_h^e is a lookup table (no encoders), other neural networks
       cannot share encoders from it. Instead, standalone encoders are created
       and passed to each neural network that needs them.
    
    This ensures consistent encoding across networks and enables
    efficient caching of raw tensor extraction.
    
    Args:
        env: MultiGridEnv instance.
        config: Phase2Config.
        num_robots: Number of robot agents.
        num_actions: Number of actions per robot.
        hidden_dim: Hidden dimension for networks (default: config.hidden_dim).
        device: Torch device.
        goal_feature_dim: Goal encoder output dimension (default: config.goal_feature_dim).
        max_agents: Max number of agents for identity encoding.
        agent_embedding_dim: Dimension of agent identity embedding (default: config.agent_embedding_dim).
        agent_position_feature_dim: Agent encoder position encoding output dimension (default: config.agent_position_feature_dim).
        agent_feature_dim: Agent encoder feature encoding output dimension (default: config.agent_feature_dim).
    
    Returns:
        Phase2Networks container with all networks using shared encoders.
    """
    # Determine which networks need neural implementations
    # (i.e., lookup tables are NOT enabled for them)
    use_neural_q_r = not config.should_use_lookup_table('q_r')
    use_neural_v_h_e = not config.should_use_lookup_table('v_h_e')
    use_neural_x_h = not config.should_use_lookup_table('x_h')
    use_neural_u_r = config.u_r_use_network and not config.should_use_lookup_table('u_r')
    use_neural_v_r = config.v_r_use_network and not config.should_use_lookup_table('v_r')
    
    any_neural = use_neural_q_r or use_neural_v_h_e or use_neural_x_h or use_neural_u_r or use_neural_v_r
    
    # If no neural networks needed, use all lookup tables
    if not any_neural:
        from empo.nn_based.phase2.lookup import (
            LookupTableRobotQNetwork,
            LookupTableRobotValueNetwork,
            LookupTableHumanGoalAbilityNetwork,
            LookupTableAggregateGoalAbilityNetwork,
            LookupTableIntrinsicRewardNetwork,
        )
        
        q_r = LookupTableRobotQNetwork(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=config.beta_r,
            default_q_r=config.get_lookup_default('q_r'),
            include_step_count=config.include_step_count,
        )
        
        v_h_e = LookupTableHumanGoalAbilityNetwork(
            gamma_h=config.gamma_h,
            default_v_h_e=config.get_lookup_default('v_h_e'),
            include_step_count=config.include_step_count,
        )
        
        x_h = LookupTableAggregateGoalAbilityNetwork(
            default_x_h=config.get_lookup_default('x_h'),
            include_step_count=config.include_step_count,
        )
        
        u_r = None
        if config.u_r_use_network:
            u_r = LookupTableIntrinsicRewardNetwork(
                eta=config.eta,
                default_y=config.get_lookup_default('u_r'),
                include_step_count=config.include_step_count,
            )
        
        v_r = None
        if config.v_r_use_network:
            v_r = LookupTableRobotValueNetwork(
                gamma_r=config.gamma_r,
                default_v_r=config.get_lookup_default('v_r'),
                include_step_count=config.include_step_count,
            )
        
        # RND doesn't make sense with full lookup table mode - warn and disable
        # RND needs learned state representations; tabular mode uses exact state hashes.
        # For small state spaces, epsilon-greedy exploration is sufficient.
        rnd = None
        if config.use_rnd:
            import warnings
            warnings.warn(
                "RND curiosity exploration is disabled in full tabular mode. "
                "RND requires learned state representations, but tabular mode uses exact state hashes. "
                "For small state spaces, epsilon-greedy exploration (already enabled) is sufficient.",
                UserWarning,
                stacklevel=2
            )
        
        return Phase2Networks(
            q_r=q_r,
            v_h_e=v_h_e,
            x_h=x_h,
            u_r=u_r,
            v_r=v_r,
            rnd=rnd,
        )
    
    # At least one network needs neural implementation
    # 
    # Strategy: Create V_h^e first (neural or lookup), then get shared encoders from it.
    # - If V_h^e is neural: it provides trained shared encoders
    # - If V_h^e is lookup: it provides NullEncoders that output zeros
    # Either way, other networks get encoders from v_h_e and can use them uniformly.
    
    # Use config values as defaults
    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    if goal_feature_dim is None:
        goal_feature_dim = config.goal_feature_dim
    if agent_embedding_dim is None:
        agent_embedding_dim = config.agent_embedding_dim
    if agent_position_feature_dim is None:
        agent_position_feature_dim = config.agent_position_feature_dim
    if agent_feature_dim is None:
        agent_feature_dim = config.agent_feature_dim
    
    # Count agents per color
    num_agents_per_color = {}
    for agent in env.agents:
        color = agent.color
        num_agents_per_color[color] = num_agents_per_color.get(color, 0) + 1
    
    # Common parameters
    grid_height = env.height
    grid_width = env.width
    
    # =========================================================================
    # Create V_h^e first (it provides shared encoders to other networks)
    # =========================================================================
    if use_neural_v_h_e:
        # Neural V_h^e: create real encoders that will be trained
        shared_state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=7,
            feature_dim=hidden_dim,
            include_step_count=config.include_step_count,
            use_encoders=config.use_encoders,
        ).to(device)
        
        shared_goal_encoder = MultiGridGoalEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=goal_feature_dim,
            use_encoders=config.use_encoders,
        ).to(device)
        
        shared_agent_encoder = AgentIdentityEncoder(
            num_agents=max_agents,
            embedding_dim=agent_embedding_dim,
            position_feature_dim=agent_position_feature_dim,
            agent_feature_dim=agent_feature_dim,
            grid_height=grid_height,
            grid_width=grid_width,
            use_encoders=config.use_encoders,
        ).to(device)
        
        v_h_e = MultiGridHumanGoalAchievementNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            state_feature_dim=hidden_dim,
            goal_feature_dim=goal_feature_dim,
            hidden_dim=hidden_dim,
            gamma_h=config.gamma_h,
            dropout=config.v_h_e_dropout,
            max_agents=max_agents,
            agent_embedding_dim=agent_embedding_dim,
            state_encoder=shared_state_encoder,
            goal_encoder=shared_goal_encoder,
            agent_encoder=shared_agent_encoder,
        ).to(device)
    else:
        # Lookup V_h^e: provides NullEncoders that output zeros
        from empo.nn_based.phase2.lookup import LookupTableHumanGoalAbilityNetwork
        v_h_e = LookupTableHumanGoalAbilityNetwork(
            gamma_h=config.gamma_h,
            default_v_h_e=config.get_lookup_default('v_h_e'),
            include_step_count=config.include_step_count,
            state_feature_dim=hidden_dim,
            goal_feature_dim=goal_feature_dim,
            agent_feature_dim=agent_embedding_dim + agent_position_feature_dim + agent_feature_dim,
        )
        # Get null encoders from lookup V_h^e
        shared_state_encoder = v_h_e.state_encoder
        shared_agent_encoder = v_h_e.agent_encoder
    
    # =========================================================================
    # Create OWN encoders for Q_r and X_h (trained with their respective losses)
    # =========================================================================
    # These allow Q_r and X_h to learn additional features beyond those learned by V_h^e.
    # When V_h^e is a lookup table (null encoders), these own encoders do ALL the encoding.
    # Note: own encoders share cache with shared encoders to avoid redundant tensorization
    # (but when shared encoders are null, there's nothing to share - that's fine).
    
    q_r_own_state_encoder = None
    if use_neural_q_r:
        share_cache_with = shared_state_encoder if use_neural_v_h_e else None
        q_r_own_state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=7,
            feature_dim=hidden_dim,
            include_step_count=config.include_step_count,
            use_encoders=config.use_encoders,
            share_cache_with=share_cache_with,
        ).to(device)
    
    x_h_own_state_encoder = None
    x_h_own_agent_encoder = None
    if use_neural_x_h:
        state_share_cache_with = shared_state_encoder if use_neural_v_h_e else None
        x_h_own_state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=7,
            feature_dim=hidden_dim,
            include_step_count=config.include_step_count,
            use_encoders=config.use_encoders,
            share_cache_with=state_share_cache_with,
        ).to(device)
        agent_share_cache_with = shared_agent_encoder if use_neural_v_h_e else None
        x_h_own_agent_encoder = AgentIdentityEncoder(
            num_agents=max_agents,
            embedding_dim=agent_embedding_dim,
            position_feature_dim=agent_position_feature_dim,
            agent_feature_dim=agent_feature_dim,
            grid_height=grid_height,
            grid_width=grid_width,
            use_encoders=config.use_encoders,
            share_cache_with=agent_share_cache_with,
        ).to(device)
    
    u_r_own_state_encoder = None
    if use_neural_u_r and config.u_r_use_network:
        share_cache_with = shared_state_encoder if use_neural_v_h_e else None
        u_r_own_state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=7,
            feature_dim=hidden_dim,
            include_step_count=config.include_step_count,
            use_encoders=config.use_encoders,
            share_cache_with=share_cache_with,
        ).to(device)
    
    v_r_own_state_encoder = None
    if use_neural_v_r and config.v_r_use_network:
        share_cache_with = shared_state_encoder if use_neural_v_h_e else None
        v_r_own_state_encoder = MultiGridStateEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=7,
            feature_dim=hidden_dim,
            include_step_count=config.include_step_count,
            use_encoders=config.use_encoders,
            share_cache_with=share_cache_with,
        ).to(device)
    
    # =========================================================================
    # Create remaining networks - use shared encoders from V_h^e
    # =========================================================================
    
    # Q_r network
    if use_neural_q_r:
        q_r = MultiGridRobotQNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_robot_actions=num_actions,
            num_robots=num_robots,
            num_agents_per_color=num_agents_per_color,
            state_feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            beta_r=config.beta_r,
            dropout=config.q_r_dropout,
            state_encoder=shared_state_encoder,       # From V_h^e (real or null)
            own_state_encoder=q_r_own_state_encoder,  # OWN (trained with Q_r)
        ).to(device)
    else:
        from empo.nn_based.phase2.lookup import LookupTableRobotQNetwork
        q_r = LookupTableRobotQNetwork(
            num_actions=num_actions,
            num_robots=num_robots,
            beta_r=config.beta_r,
            default_q_r=config.get_lookup_default('q_r'),
            include_step_count=config.include_step_count,
        )
    
    # X_h network
    if use_neural_x_h:
        x_h = MultiGridAggregateGoalAbilityNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_agents_per_color=num_agents_per_color,
            state_feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            zeta=config.zeta,
            dropout=config.x_h_dropout,
            max_agents=max_agents,
            agent_embedding_dim=agent_embedding_dim,
            state_encoder=shared_state_encoder,       # From V_h^e (SHARED, used detached)
            agent_encoder=shared_agent_encoder,       # From V_h^e (SHARED, used detached)
            own_state_encoder=x_h_own_state_encoder,  # OWN (trained with X_h)
            own_agent_encoder=x_h_own_agent_encoder,  # OWN (trained with X_h)
        ).to(device)
    else:
        from empo.nn_based.phase2.lookup import LookupTableAggregateGoalAbilityNetwork
        x_h = LookupTableAggregateGoalAbilityNetwork(
            default_x_h=config.get_lookup_default('x_h'),
            include_step_count=config.include_step_count,
        )
    
    # U_r network (optional)
    u_r = None
    if config.u_r_use_network:
        if use_neural_u_r:
            u_r = MultiGridIntrinsicRewardNetwork(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                state_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                xi=config.xi,
                eta=config.eta,
                dropout=config.u_r_dropout,
                state_encoder=shared_state_encoder,       # SHARED (used detached)
                own_state_encoder=u_r_own_state_encoder,  # OWN (trained with U_r)
            ).to(device)
        else:
            from empo.nn_based.phase2.lookup import LookupTableIntrinsicRewardNetwork
            u_r = LookupTableIntrinsicRewardNetwork(
                eta=config.eta,
                default_y=config.get_lookup_default('u_r'),
                include_step_count=config.include_step_count,
            )
    
    # V_r network (optional)
    v_r = None
    if config.v_r_use_network:
        if use_neural_v_r:
            v_r = MultiGridRobotValueNetwork(
                grid_height=grid_height,
                grid_width=grid_width,
                num_agents_per_color=num_agents_per_color,
                state_feature_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=config.v_r_dropout,
                state_encoder=shared_state_encoder,       # SHARED (used detached)
                own_state_encoder=v_r_own_state_encoder,  # OWN (trained with V_r)
            ).to(device)
        else:
            from empo.nn_based.phase2.lookup import LookupTableRobotValueNetwork
            v_r = LookupTableRobotValueNetwork(
                gamma_r=config.gamma_r,
                default_v_r=config.get_lookup_default('v_r'),
                include_step_count=config.include_step_count,
            )
    
    # =========================================================================
    # Create RND module for curiosity-driven exploration (optional)
    # =========================================================================
    # RND uses concatenated features from ALL state encoders (shared + own).
    # Each encoder's features are weighted by a coefficient that ramps 0→1
    # during the warmup stage when that encoder is introduced.
    # 
    # Encoder introduction order:
    # - Stage 0 (V_h^e): shared_state_encoder
    # - Stage 1 (X_h): x_h_own_state_encoder
    # - Stage 2 (U_r): u_r_own_state_encoder (if u_r_use_network)
    # - Stage 3 (Q_r): q_r_own_state_encoder
    # 
    # This smooth weighting avoids RND novelty spikes when new encoders are added.
    rnd = None
    rnd_encoder_dims = None  # Track individual encoder dims for coefficient weighting
    if config.use_rnd:
        from empo.nn_based.phase2.rnd import RNDModule
        
        # Collect all encoder dimensions in introduction order
        # Note: Some encoders may be None if network is lookup table
        rnd_encoder_dims = []
        
        # Shared state encoder (from V_h^e) - always present
        rnd_encoder_dims.append(shared_state_encoder.feature_dim)
        
        # X_h own state encoder (if neural X_h)
        if x_h_own_state_encoder is not None:
            rnd_encoder_dims.append(x_h_own_state_encoder.feature_dim)
        
        # U_r own state encoder (if neural U_r with network enabled)
        if u_r_own_state_encoder is not None:
            rnd_encoder_dims.append(u_r_own_state_encoder.feature_dim)
        
        # Q_r own state encoder (if neural Q_r)
        if q_r_own_state_encoder is not None:
            rnd_encoder_dims.append(q_r_own_state_encoder.feature_dim)
        
        rnd_input_dim = sum(rnd_encoder_dims)
        
        rnd = RNDModule(
            input_dim=rnd_input_dim,
            encoder_dims=rnd_encoder_dims,
            feature_dim=config.rnd_feature_dim,
            hidden_dim=config.rnd_hidden_dim,
            normalize=config.normalize_rnd,
            normalization_decay=config.rnd_normalization_decay,
        ).to(device)
    
    # =========================================================================
    # Create Human Action RND (if enabled)
    # =========================================================================
    # Human Action RND outputs per-action novelty for (state, human, action) tuples.
    # Input: state features (from shared encoder) + agent features (from agent encoder)
    # Output: novelty scores for each action
    human_rnd = None
    if config.use_rnd and config.use_human_action_rnd:
        from empo.nn_based.phase2.rnd import HumanActionRNDModule
        
        # State features from shared state encoder
        state_feature_dim = shared_state_encoder.feature_dim
        # Agent features from shared agent encoder
        agent_feature_dim = shared_agent_encoder.output_dim
        
        human_rnd = HumanActionRNDModule(
            state_feature_dim=state_feature_dim,
            agent_feature_dim=agent_feature_dim,
            num_actions=num_actions,
            feature_dim=config.rnd_feature_dim,
            hidden_dim=config.rnd_hidden_dim,
            normalize=config.normalize_rnd,
            normalization_decay=config.rnd_normalization_decay,
        ).to(device)
    
    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r,
        rnd=rnd,
        rnd_encoder_dims=rnd_encoder_dims,  # Store for coefficient computation
        human_rnd=human_rnd,
    )


def train_multigrid_phase2(
    world_model: MultiGridEnv,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_policy_prior: Callable,
    goal_sampler: Callable,
    config: Optional[Phase2Config] = None,
    num_training_steps: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    goal_feature_dim: Optional[int] = None,
    agent_embedding_dim: Optional[int] = None,
    agent_position_feature_dim: Optional[int] = None,
    agent_feature_dim: Optional[int] = None,
    device: str = 'cpu',
    verbose: bool = True,
    debug: bool = False,
    tensorboard_dir: Optional[str] = None,
    profiler: Optional[Any] = None,
    restore_networks_path: Optional[str] = None,
    world_model_factory: Optional[Any] = None,
    robot_exploration_policy: Optional[Any] = None,
    human_exploration_policy: Optional[Any] = None,
) -> Tuple[MultiGridRobotQNetwork, Phase2Networks, List[Dict[str, float]], "MultiGridPhase2Trainer"]:
    """
    Train Phase 2 robot policy for a multigrid environment.
    
    This function trains neural networks to learn the robot policy that maximizes
    aggregate human power, as defined in equations (4)-(9) of the EMPO paper.
    
    Args:
        world_model: MultiGridEnv instance.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: HumanPolicyPrior instance with .sample(state, human_idx, goal) method.
        goal_sampler: PossibleGoalSampler instance with .sample(state, human_idx) -> (goal, weight) method.
        config: Phase2Config (uses defaults if None).
        num_training_steps: Override config.num_training_steps if provided.
        hidden_dim: Hidden dimension for networks (default: config.hidden_dim).
        goal_feature_dim: Goal encoder output dimension (default: config.goal_feature_dim).
        agent_embedding_dim: Dimension of agent identity embedding (default: config.agent_embedding_dim).
        agent_position_feature_dim: Agent encoder position encoding output dimension (default: config.agent_position_feature_dim).
        agent_feature_dim: Agent encoder feature encoding output dimension (default: config.agent_feature_dim).
        device: Torch device.
        verbose: Enable progress bar (tqdm).
        debug: Enable verbose debug output.
        tensorboard_dir: Directory for TensorBoard logs (optional).
        profiler: Torch profiler instance (optional).
        restore_networks_path: Path to checkpoint for restoring networks.
                               Skips warmup/rampup stages since they were already done.
        world_model_factory: Optional factory for creating world models. Required for
            async training where the environment cannot be pickled.
        robot_exploration_policy: Optional policy for robot epsilon exploration. Can be:
            - None: Use uniform random policy for exploration (default).
            - List[float]: Fixed action probabilities (length = num_action_combinations).
            - Callable[[state, world_model], List[float]]: Function returning action probabilities.
            - RobotPolicy: A proper policy object with sample(state) method returning action tuple.
        human_exploration_policy: Optional policy for human epsilon exploration. Can be:
            - None: Use uniform random policy for exploration (default).
            - HumanExplorationPolicy: A policy object with sample(state, human_idx, goal) method.
    
    Returns:
        Tuple of (robot_q_network, all_networks, training_history, trainer).
    """
    if config is None:
        config = Phase2Config()
    
    if num_training_steps is not None:
        config.num_training_steps = num_training_steps
    
    # Use config values as defaults for encoder dimensions
    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    if goal_feature_dim is None:
        goal_feature_dim = config.goal_feature_dim
    if agent_embedding_dim is None:
        agent_embedding_dim = config.agent_embedding_dim
    if agent_position_feature_dim is None:
        agent_position_feature_dim = config.agent_position_feature_dim
    if agent_feature_dim is None:
        agent_feature_dim = config.agent_feature_dim
    
    # Determine number of actions from environment
    if hasattr(world_model.actions, 'n'):
        num_actions = world_model.actions.n
    elif hasattr(world_model.actions, 'available'):
        num_actions = len(world_model.actions.available)
    else:
        num_actions = len(world_model.actions)
    num_robots = len(robot_agent_indices)
    
    if verbose:
        print(f"Creating Phase 2 networks:")
        print(f"  Grid: {world_model.width}x{world_model.height}")
        print(f"  Humans: {len(human_agent_indices)}")
        print(f"  Robots: {num_robots}")
        print(f"  Actions per robot: {num_actions}")
        print(f"  Joint action space: {num_actions ** num_robots} actions")
    
    # Create networks
    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=num_robots,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        device=device,
        goal_feature_dim=goal_feature_dim,
        agent_embedding_dim=agent_embedding_dim,
        agent_position_feature_dim=agent_position_feature_dim,
        agent_feature_dim=agent_feature_dim
    )
    
    if debug:
        print("[DEBUG] Creating trainer...")
    
    # Create trainer
    trainer = MultiGridPhase2Trainer(
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
    )
    
    # Restore networks if checkpoint provided (skips warmup/rampup since already done)
    if restore_networks_path is not None:
        if verbose:
            print(f"\nRestoring networks from: {restore_networks_path}")
        trainer.load_all_networks(restore_networks_path)
        if verbose:
            print(f"  Restored networks at env step {trainer.total_env_steps}")
            print(f"  Warmup/rampup stages will be skipped")
    
    if verbose:
        print(f"\nTraining for {config.num_training_steps} training steps...")
        print(f"  Steps per episode: {config.steps_per_episode}")
        print(f"  Training steps per env step: {config.training_steps_per_env_step}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size}")
        if config.use_rnd:
            print(f"  Curiosity (RND): enabled (bonus_coef_r={config.rnd_bonus_coef_r})")
        if tensorboard_dir:
            print(f"  TensorBoard: {tensorboard_dir}")
    
    # Train
    history = trainer.train(config.num_training_steps)
    
    if verbose:
        print(f"\nTraining completed!")
        if history:
            final_losses = history[-1]
            print(f"  Final losses: {final_losses}")
    
    return networks.q_r, networks, history, trainer
