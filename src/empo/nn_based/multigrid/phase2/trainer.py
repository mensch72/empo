"""
Multigrid-specific Phase 2 Trainer.

This module provides the training function for Phase 2 of the EMPO framework
(equations 4-9) specialized for multigrid environments.
"""

import random
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
    
    Args:
        env: MultiGridEnv instance.
        networks: Phase2Networks container.
        config: Phase2Config.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: Callable returning human action given state, index, goal.
        goal_sampler: Callable returning a goal for a human.
        device: Torch device.
        verbose: Enable progress output (tqdm progress bar).
        debug: Enable verbose debug output.
        tensorboard_dir: Directory for TensorBoard logs (optional).
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
    ):
        self.env = env
        super().__init__(
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
        )
        
        # Caching is now handled internally by the shared encoders.
        # The shared state/goal/agent encoders cache raw tensor extraction
        # (but not NN output, to preserve gradient flow).
    
    def clear_caches(self):
        """
        Clear encoder caches. Call after each training step to prevent memory growth.
        
        This clears the caches in the shared encoders that all networks use.
        """
        # All networks share the same encoders, so we only need to clear once
        self.networks.q_r.state_encoder.clear_cache()
        self.networks.v_h_e.goal_encoder.clear_cache()
        self.networks.v_h_e.agent_encoder.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Tuple[int, int]]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dict with stats from each encoder type.
        """
        return {
            'state': self.networks.q_r.state_encoder.get_cache_stats(),
            'goal': self.networks.v_h_e.goal_encoder.get_cache_stats(),
            'agent': self.networks.v_h_e.agent_encoder.get_cache_stats(),
        }
    
    def _tensorize_state_cached(
        self,
        state: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode state using the shared encoder (which handles caching internally).
        
        The shared state encoder caches raw tensor extraction (before NN forward)
        to avoid redundant computation when the same state is encoded multiple times.
        The encoding is agent-agnostic.
        
        Args:
            state: Raw environment state.
        
        Returns:
            Tuple of (grid_tensor, global_features, agent_features, interactive_features).
        """
        # The state encoder now handles caching internally
        state_encoder = self.networks.q_r.state_encoder
        return state_encoder.tensorize_state(state, None, self.device)
    
    def tensorize_state(self, state: Any) -> Dict[str, torch.Tensor]:
        """Encode multigrid state to tensors."""
        # The networks handle their own encoding
        return {'state': state}
    
    def sample_robot_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action using policy with epsilon-greedy exploration.
        
        Uses cached state encoding for efficiency.
        During warm-up, uses effective beta_r = 0 (uniform random policy).
        
        Args:
            state: Current state.
        
        Returns:
            Tuple of robot actions.
        """
        epsilon = self.config.get_epsilon(self.total_steps)
        effective_beta_r = self.config.get_effective_beta_r(self.total_steps)
        
        with torch.no_grad():
            grid, glob, agent, interactive = self._tensorize_state_cached(state)
            q_values = self.networks.q_r.forward(grid, glob, agent, interactive)
            return self.networks.q_r.sample_action(q_values, epsilon, beta_r=effective_beta_r)
    
    def check_goal_achieved(self, state: Any, human_idx: int, goal: Any) -> bool:
        """Check if human's goal is achieved."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[human_idx]
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        
        # Goals should have a target_pos attribute
        if hasattr(goal, 'target_pos'):
            goal_pos = goal.target_pos
            return agent_x == goal_pos[0] and agent_y == goal_pos[1]
        elif hasattr(goal, 'is_achieved'):
            return goal.is_achieved(state)
        else:
            return False
    
    def step_environment(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        human_actions: List[int]
    ) -> Any:
        """Execute actions and return next state."""
        # Build action list for all agents
        actions = []
        human_idx = 0
        robot_idx = 0
        
        for agent_idx in range(len(self.env.agents)):
            if agent_idx in self.human_agent_indices:
                actions.append(human_actions[human_idx])
                human_idx += 1
            elif agent_idx in self.robot_agent_indices:
                actions.append(robot_action[robot_idx])
                robot_idx += 1
            else:
                actions.append(0)  # Idle for unknown agents
        
        # Step environment
        self.env.step(actions)
        return self.env.get_state()
    
    def reset_environment(self) -> Any:
        """Reset environment and return initial state."""
        self.env.reset()
        return self.env.get_state()
    
    def collect_transition(
        self,
        state: Any,
        goals: Dict[int, Any]
    ) -> Tuple[Phase2Transition, Any]:
        """
        Collect one transition, including pre-computed compact features.
        
        Overrides base class to add compact_features and next_compact_features
        to the transition. These include:
        - Expensive-to-compute but small-to-store tensors (global, agent, interactive)
        - Compressed grid tensor that captures ALL grid information from world_model
        
        This allows the trainer to reconstruct the full tensorized state WITHOUT
        access to the world_model, which is essential for off-policy learning
        where transitions may come from different episodes/world layouts.
        
        Args:
            state: Current state.
            goals: Current goal assignments.
        
        Returns:
            Tuple of (transition, next_state).
        """
        # Get base transition using parent implementation
        transition, next_state = super().collect_transition(state, goals)
        
        # Compute compact features for both state and next_state
        state_encoder = self.networks.q_r.state_encoder
        
        # Get global, agent, interactive features (expensive but small)
        global_feats, agent_feats, interactive_feats = state_encoder.tensorize_state_compact(
            state, self.env, self.device
        )
        next_global_feats, next_agent_feats, next_interactive_feats = state_encoder.tensorize_state_compact(
            next_state, self.env, self.device
        )
        
        # Get agent colors for grid compression
        agent_colors = [getattr(agent, 'color', 'grey') for agent in self.env.agents]
        
        # Compress the grid (captures ALL grid info including static objects)
        step_count, agent_states, mobile_objects, mutable_objects = state
        compressed_grid = state_encoder.compress_grid(
            self.env, agent_states, mobile_objects, mutable_objects, agent_colors
        )
        
        next_step_count, next_agent_states, next_mobile_objects, next_mutable_objects = next_state
        next_compressed_grid = state_encoder.compress_grid(
            self.env, next_agent_states, next_mobile_objects, next_mutable_objects, agent_colors
        )
        
        # Pack all compact features including compressed grid
        compact_features = (global_feats, agent_feats, interactive_feats, compressed_grid)
        next_compact_features = (next_global_feats, next_agent_feats, next_interactive_feats, next_compressed_grid)
        
        # Update transition with compact features
        transition.compact_features = compact_features
        transition.next_compact_features = next_compact_features
        
        return transition, next_state
    
    def tensorize_states_batch(
        self,
        states: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch of states into tensor format for batched forward passes.
        The encoding is agent-agnostic.
        
        Args:
            states: List of raw environment states.
        
        Returns:
            Tuple of (grid_tensors, global_features, agent_features, interactive_features)
            all with batch dimension.
        """
        # Get the state encoder from any of the networks (they share architecture)
        state_encoder = self.networks.q_r.state_encoder
        
        # Encode each state
        grid_list = []
        global_list = []
        agent_list = []
        interactive_list = []
        
        for state in states:
            grid, glob, agent, interactive = state_encoder.tensorize_state(
                state, None, self.device
            )
            grid_list.append(grid)
            global_list.append(glob)
            agent_list.append(agent)
            interactive_list.append(interactive)
        
        # Stack into batched tensors
        return (
            torch.cat(grid_list, dim=0),
            torch.cat(global_list, dim=0),
            torch.cat(agent_list, dim=0),
            torch.cat(interactive_list, dim=0)
        )
    
    def _compute_model_based_targets_for_transition(
        self,
        state: Any,
        human_actions: List[int],
        goals: Dict[int, Any],
        effective_beta_r: float,
        human_indices_for_v_h_e: Optional[List[int]] = None,
        goals_for_v_h_e: Optional[List[Any]] = None,
        cached_trans_probs: Optional[Dict[int, List[Tuple[float, Any]]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[float, float]]]]:
        """
        Compute model-based targets for Q_r and optionally V_h^e for a single transition.
        
        Uses the ACTUALLY TAKEN human actions from the transition, combined with
        ALL possible robot actions. Uses cached transition probabilities if provided,
        otherwise computes them on the fly.
        
        For Q_r: Returns E[V_r(s')] for all robot actions (uniform weighting).
        For V_h^e: Returns E[goal_achieved], E[V_h^e(s')] averaged over robot actions
                   weighted by current robot policy.
        
        Args:
            state: Current state s.
            human_actions: Actually taken human actions from the transition.
            goals: Human goals dict.
            effective_beta_r: Beta for robot policy.
            human_indices_for_v_h_e: Optional list of human indices to compute V_h^e for.
            goals_for_v_h_e: Optional list of corresponding goals.
            cached_trans_probs: Optional pre-computed transition probabilities from
                transition.transition_probs_by_action. If provided, avoids re-computing.
        
        Returns:
            Tuple of:
            - expected_v_r: Tensor of shape (num_actions,) with E[V_r(s')] per action
            - v_h_e_targets: List of (exp_achieved, exp_v_h_e) per human, or None
        """
        num_actions = self.networks.q_r.num_action_combinations
        expected_v_r = torch.zeros(num_actions, device=self.device)
        
        # If computing V_h^e, we need robot policy weights
        compute_v_h_e = human_indices_for_v_h_e is not None and goals_for_v_h_e is not None
        
        # Get current robot policy for weighting V_h^e (computed once)
        if compute_v_h_e:
            s_encoded = self._batch_tensorize_states([state])
            own_s_encoded = self._batch_tensorize_states_with_encoder(
                [state], self.networks.q_r.own_state_encoder
            )
            with torch.no_grad():
                q_r_current = self.networks.q_r.forward(*s_encoded, *own_s_encoded)
                robot_policy = self.networks.q_r.get_policy(q_r_current, beta_r=effective_beta_r)
                robot_policy = robot_policy.squeeze()  # Shape: (num_actions,)
            
            # Accumulators for V_h^e: one per human
            n_humans = len(human_indices_for_v_h_e)
            v_h_e_achieved_accum = [0.0] * n_humans
            v_h_e_value_accum = [0.0] * n_humans
        
        # Iterate over all robot actions
        for action_idx in range(num_actions):
            # Get transition probabilities - use cache if available
            if cached_trans_probs is not None:
                trans_probs = cached_trans_probs.get(action_idx, [])
                if not trans_probs:
                    # Terminal state or no entry
                    expected_v_r[action_idx] = 0.0
                    continue
            else:
                # Compute on the fly (fallback)
                robot_action = self.networks.q_r.action_index_to_tuple(action_idx)
                
                # Build full action vector using ACTUAL human actions
                actions = []
                human_idx_iter = 0
                robot_idx = 0
                for agent_idx in range(len(self.env.agents)):
                    if agent_idx in self.human_agent_indices:
                        actions.append(human_actions[human_idx_iter])
                        human_idx_iter += 1
                    elif agent_idx in self.robot_agent_indices:
                        actions.append(robot_action[robot_idx])
                        robot_idx += 1
                    else:
                        actions.append(0)
                
                trans_probs = self.env.transition_probabilities(state, actions)
                
                if trans_probs is None:
                    # Terminal state
                    expected_v_r[action_idx] = 0.0
                    continue
            
            # Compute expected V_r over successor states
            action_expected_v_r = 0.0
            
            # For V_h^e: accumulate per-action contributions
            if compute_v_h_e:
                action_achieved = [0.0] * n_humans
                action_v_h_e = [0.0] * n_humans
            
            for prob, next_state in trans_probs:
                if prob > 0:
                    # Encode next state once
                    s_prime_encoded = self._batch_tensorize_states([next_state])
                    
                    # Compute V_r(s')
                    with torch.no_grad():
                        if self.config.v_r_use_network:
                            v_r_next = self.networks.v_r_target.forward(*s_prime_encoded)
                        else:
                            _, u_r_next = self.networks.u_r_target.forward(*s_prime_encoded)
                            own_s_prime = self._batch_tensorize_states_with_encoder(
                                [next_state], self.networks.q_r.own_state_encoder
                            )
                            q_r_next = self.networks.q_r.forward(*s_prime_encoded, *own_s_prime)
                            pi_r_next = self.networks.q_r.get_policy(q_r_next, beta_r=effective_beta_r)
                            v_r_next = self.networks.v_r.compute_from_components(
                                u_r_next, q_r_next, pi_r_next
                            )
                        action_expected_v_r += prob * v_r_next.item()
                    
                    # Compute V_h^e contributions (reuse s_prime_encoded)
                    if compute_v_h_e:
                        for h_i, (h_idx, goal) in enumerate(zip(human_indices_for_v_h_e, goals_for_v_h_e)):
                            # Check goal achievement
                            achieved = self.check_goal_achieved(next_state, h_idx, goal)
                            action_achieved[h_i] += prob * (1.0 if achieved else 0.0)
                            
                            # Compute V_h^e(s')
                            goal_features = self._batch_tensorize_goals([goal])
                            v_h_e_idx, v_h_e_grid, v_h_e_feat = self._batch_tensorize_agent_identities(
                                [h_idx], [next_state]
                            )
                            with torch.no_grad():
                                v_h_e_next = self.networks.v_h_e_target.forward(
                                    s_prime_encoded[0], s_prime_encoded[1],
                                    s_prime_encoded[2], s_prime_encoded[3],
                                    goal_features, v_h_e_idx, v_h_e_grid, v_h_e_feat
                                )
                                action_v_h_e[h_i] += prob * v_h_e_next.item()
            
            expected_v_r[action_idx] = action_expected_v_r
            
            # Weight V_h^e by robot policy
            if compute_v_h_e:
                weight = robot_policy[action_idx].item()
                for h_i in range(n_humans):
                    v_h_e_achieved_accum[h_i] += weight * action_achieved[h_i]
                    v_h_e_value_accum[h_i] += weight * action_v_h_e[h_i]
        
        # Build V_h^e targets
        v_h_e_targets = None
        if compute_v_h_e:
            v_h_e_targets = [
                (v_h_e_achieved_accum[i], v_h_e_value_accum[i])
                for i in range(n_humans)
            ]
        
        return expected_v_r, v_h_e_targets

    def compute_losses(
        self,
        batch: List[Phase2Transition],
        x_h_batch: Optional[List[Phase2Transition]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        """
        Compute losses for all networks using batched operations for efficiency.
        
        This implementation encodes each unique state ONCE and passes agent identity
        separately to human-specific networks (V_h^e, X_h). This dramatically reduces
        computation compared to encoding the same state multiple times for different
        agent perspectives.
        
        Args:
            batch: List of transitions for most networks.
            x_h_batch: Optional larger batch for X_h (defaults to batch).
        
        Returns:
            Tuple of (losses dict, prediction_stats dict).
            prediction_stats maps network names to {'mean': float, 'std': float}.
        """
        if x_h_batch is None:
            x_h_batch = batch
        
        if self.debug:
            print(f"[DEBUG] compute_losses: processing batch of {len(batch)} transitions (agent-identity mode)")
        
        n = len(batch)
        
        # Track prediction statistics
        prediction_stats = {}
        
        # Get encoders
        state_encoder = self.networks.q_r.state_encoder
        goal_encoder = self.networks.v_h_e.goal_encoder
        
        # ========================================================================
        # Stage 1: Tensorize all unique states ONCE (agent-agnostic)
        # ========================================================================
        # State tensorization is agent-agnostic - no query agent needed
        # Agent identity is handled separately by AgentIdentityEncoder
        states = [t.state for t in batch]
        next_states = [t.next_state for t in batch]
        
        # Extract pre-computed compact features if available
        compact_features = [t.compact_features for t in batch]
        next_compact_features = [t.next_compact_features for t in batch]
        
        # Tensorize states - uses compact features when available (OPTIMIZATION)
        # Grid tensor is always computed fresh (cheap), while global/agent/interactive
        # are retrieved from pre-computed compact features (avoids expensive grid scan)
        s_encoded = self._batch_tensorize_from_compact(states, compact_features)
        s_prime_encoded = self._batch_tensorize_from_compact(next_states, next_compact_features)
        
        # ========================================================================
        # Phase 2: Collect human-agent pairs for V_h^e
        # Each transition may have multiple humans with goals
        # ========================================================================
        v_h_e_indices = []  # transition index for each entry
        v_h_e_goals = []
        v_h_e_achieved = []
        v_h_e_human_indices = []
        v_h_e_states = []  # states for agent identity encoding
        v_h_e_next_states = []
        v_h_e_goals_dicts = []  # full goals dict for each entry (for model-based targets)
        v_h_e_human_actions = []  # actual human actions taken (for model-based targets)
        
        for i, t in enumerate(batch):
            for h, g_h in t.goals.items():
                v_h_e_indices.append(i)
                v_h_e_goals.append(g_h)
                v_h_e_achieved.append(self.check_goal_achieved(t.next_state, h, g_h))
                v_h_e_human_indices.append(h)
                v_h_e_states.append(t.state)
                v_h_e_next_states.append(t.next_state)
                v_h_e_goals_dicts.append(t.goals)  # Store full goals dict
                v_h_e_human_actions.append(t.human_actions)  # Store actual human actions
        
        if v_h_e_goals:
            goal_features = self._batch_tensorize_goals(v_h_e_goals)
            
            # Encode agent identities (index + grid + features)
            v_h_e_idx, v_h_e_grid, v_h_e_feat = self._batch_tensorize_agent_identities(
                v_h_e_human_indices, v_h_e_states
            )
            v_h_e_prime_idx, v_h_e_prime_grid, v_h_e_prime_feat = self._batch_tensorize_agent_identities(
                v_h_e_human_indices, v_h_e_next_states
            )
            
            # Get encoded states for these transitions (index into batched encodings)
            v_h_e_trans_indices = torch.tensor(v_h_e_indices, device=self.device)
            s_h_grid = s_encoded[0][v_h_e_trans_indices]
            s_h_glob = s_encoded[1][v_h_e_trans_indices]
            s_h_agent = s_encoded[2][v_h_e_trans_indices]
            s_h_interactive = s_encoded[3][v_h_e_trans_indices]
            
            s_prime_h_grid = s_prime_encoded[0][v_h_e_trans_indices]
            s_prime_h_glob = s_prime_encoded[1][v_h_e_trans_indices]
            s_prime_h_agent = s_prime_encoded[2][v_h_e_trans_indices]
            s_prime_h_interactive = s_prime_encoded[3][v_h_e_trans_indices]
        
        # ========================================================================
        # Phase 3: Collect for X_h loss (potentially larger batch)
        # ========================================================================
        x_h_transition_indices = []
        x_h_goals = []
        x_h_human_indices = []
        x_h_states_for_identity = []
        
        # For X_h, we need to encode states from x_h_batch (which may differ from batch)
        x_h_states = [t.state for t in x_h_batch]
        x_h_s_all_encoded = self._batch_tensorize_states(x_h_states)
        
        for i, t in enumerate(x_h_batch):
            if self.config.x_h_sample_humans is None:
                humans_for_x_h = list(t.goals.keys())
            else:
                n_sample = min(self.config.x_h_sample_humans, len(t.goals))
                humans_for_x_h = random.sample(list(t.goals.keys()), n_sample)
            
            for h_x in humans_for_x_h:
                x_h_transition_indices.append(i)
                x_h_goals.append(t.goals[h_x])
                x_h_human_indices.append(h_x)
                x_h_states_for_identity.append(t.state)
        
        if x_h_goals:
            x_h_goal_features = self._batch_tensorize_goals(x_h_goals)
            
            # Encode agent identities
            x_h_idx, x_h_agent_grid, x_h_agent_feat = self._batch_tensorize_agent_identities(
                x_h_human_indices, x_h_states_for_identity
            )
            
            # Index into X_h state encodings
            x_h_trans_indices = torch.tensor(x_h_transition_indices, device=self.device)
            x_h_grid = x_h_s_all_encoded[0][x_h_trans_indices]
            x_h_glob = x_h_s_all_encoded[1][x_h_trans_indices]
            x_h_agent = x_h_s_all_encoded[2][x_h_trans_indices]
            x_h_interactive = x_h_s_all_encoded[3][x_h_trans_indices]
        
        # ========================================================================
        # Phase 4: Compute all forward passes in batches
        # ========================================================================
        # Detach shared encoder outputs for all networks except V_h^e
        # V_h^e is the only network that trains the shared encoders
        s_encoded_detached = tuple(t.detach() for t in s_encoded)
        
        # ----- Q_r: Robot Q-values -----
        # Q_r uses: shared encoder (detached) + own encoder (trained)
        # Encode with Q_r's own state encoder (also agent-agnostic)
        own_s_encoded = self._batch_tensorize_states_with_encoder(
            [t.state for t in batch],
            self.networks.q_r.own_state_encoder
        )
        q_r_all = self.networks.q_r.forward(
            *s_encoded_detached,  # Shared encoder (frozen for Q_r)
            *own_s_encoded        # Own encoder (trained with Q_r)
        )
        robot_actions = [t.robot_action for t in batch]
        action_indices = torch.tensor(
            [self.networks.q_r.action_tuple_to_index(a) for a in robot_actions],
            device=self.device
        )
        q_r_pred = q_r_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)  # (n,)
        
        # ----- Q_r target: γ_r * E[V_r(s')] -----
        # Use effective beta_r for policy (0 during warm-up for independence)
        effective_beta_r = self.config.get_effective_beta_r(self.total_steps)
        
        # Pre-compute model-based targets for both Q_r and V_h^e in a single pass
        # This avoids calling transition_probabilities twice for the same transitions
        model_based_q_r_targets = None
        model_based_v_h_e_targets = None
        
        if self.config.use_model_based_targets and hasattr(self.env, 'transition_probabilities'):
            # Compute both Q_r and V_h^e targets in one pass through transitions
            with torch.no_grad():
                all_target_q_r = []
                
                # Pre-allocate V_h^e targets (will fill in based on transition index)
                v_h_e_targets_by_entry = [0.0] * len(v_h_e_goals) if v_h_e_goals else []
                
                # Group V_h^e entries by transition index for efficient lookup
                from collections import defaultdict
                transition_to_v_h_e_entries = defaultdict(list)  # trans_idx -> [(entry_idx, h_idx, g), ...]
                if v_h_e_goals:
                    for entry_idx, trans_idx in enumerate(v_h_e_indices):
                        h_idx = v_h_e_human_indices[entry_idx]
                        g = v_h_e_goals[entry_idx]
                        transition_to_v_h_e_entries[trans_idx].append((entry_idx, h_idx, g))
                
                # Process each transition ONCE for both Q_r and V_h^e
                for trans_idx, t in enumerate(batch):
                    # Get V_h^e humans/goals for this transition (if any)
                    v_h_e_entries = transition_to_v_h_e_entries.get(trans_idx, [])
                    
                    # Use cached transition probabilities if available
                    cached_trans_probs = t.transition_probs_by_action
                    
                    if v_h_e_entries:
                        human_indices = [e[1] for e in v_h_e_entries]
                        goals_list = [e[2] for e in v_h_e_entries]
                        
                        # Single call computes BOTH Q_r and V_h^e targets
                        expected_v_r, v_h_e_results = self._compute_model_based_targets_for_transition(
                            t.state, t.human_actions, t.goals, effective_beta_r,
                            human_indices_for_v_h_e=human_indices,
                            goals_for_v_h_e=goals_list,
                            cached_trans_probs=cached_trans_probs
                        )
                        
                        # Store V_h^e targets
                        if v_h_e_results:
                            for i, (entry_idx, _, _) in enumerate(v_h_e_entries):
                                exp_achieved, exp_v_h_e_next = v_h_e_results[i]
                                target_v = exp_achieved + (1.0 - exp_achieved) * exp_v_h_e_next
                                v_h_e_targets_by_entry[entry_idx] = target_v
                    else:
                        # No V_h^e for this transition, just compute Q_r
                        expected_v_r, _ = self._compute_model_based_targets_for_transition(
                            t.state, t.human_actions, t.goals, effective_beta_r,
                            cached_trans_probs=cached_trans_probs
                        )
                    
                    all_target_q_r.append(self.config.gamma_r * expected_v_r)
                
                # Store results for use later
                model_based_q_r_targets = torch.stack(all_target_q_r)
                if v_h_e_goals:
                    model_based_v_h_e_targets = torch.tensor(v_h_e_targets_by_entry, device=self.device)
            
            # Q_r loss over ALL actions
            loss_q_r = ((q_r_all - model_based_q_r_targets) ** 2).mean()
            target_q_r = model_based_q_r_targets.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        else:
            # Sample-based targets: use observed next state only
            with torch.no_grad():
                if self.config.v_r_use_network:
                    v_r_next = self.networks.v_r_target.forward(*s_prime_encoded)
                else:
                    # Use frozen u_r_target for stable V_r computation
                    _, u_r_next = self.networks.u_r_target.forward(*s_prime_encoded)
                    # Q_r needs both encoders for proper inference
                    own_s_prime_encoded = self._batch_tensorize_states_with_encoder(
                        [t.next_state for t in batch],
                        self.networks.q_r.own_state_encoder
                    )
                    q_r_next = self.networks.q_r.forward(*s_prime_encoded, *own_s_prime_encoded)
                    pi_r_next = self.networks.q_r.get_policy(q_r_next, beta_r=effective_beta_r)
                    v_r_next = self.networks.v_r.compute_from_components(
                        u_r_next, q_r_next, pi_r_next
                    )
            target_q_r = self.config.gamma_r * v_r_next.squeeze()
            loss_q_r = ((q_r_pred - target_q_r) ** 2).mean()
        
        # Track Q_r prediction stats
        with torch.no_grad():
            prediction_stats['q_r'] = {
                'mean': q_r_pred.mean().item(),
                'std': q_r_pred.std().item() if q_r_pred.numel() > 1 else 0.0,
                'target_mean': target_q_r.mean().item()
            }
        
        # ----- U_r: Intrinsic reward (only when using network mode) -----
        loss_u_r = torch.tensor(0.0, device=self.device)
        if self.config.u_r_use_network:
            # U_r uses only shared encoder (detached) - no own encoder
            y_pred, u_r_pred = self.networks.u_r.forward(*s_encoded_detached)
            
            # Compute X_h for U_r target using agent identity encoding
            u_r_x_h_transition_indices = []
            u_r_x_h_human_indices = []
            u_r_x_h_states = []  # states for agent identity encoding
            u_r_transition_boundaries = [0]
            
            for i, t in enumerate(batch):
                if self.config.u_r_sample_humans is None:
                    humans_for_u_r = self.human_agent_indices
                else:
                    n_sample = min(self.config.u_r_sample_humans, len(self.human_agent_indices))
                    humans_for_u_r = random.sample(self.human_agent_indices, n_sample)
                
                for h_u in humans_for_u_r:
                    u_r_x_h_transition_indices.append(i)
                    u_r_x_h_human_indices.append(h_u)
                    u_r_x_h_states.append(t.state)
                u_r_transition_boundaries.append(len(u_r_x_h_transition_indices))
            
            with torch.no_grad():
                u_r_trans_idx = torch.tensor(u_r_x_h_transition_indices, device=self.device)
                
                # Encode agent identities for X_h computation
                u_r_idx, u_r_agent_grid, u_r_agent_feat = self._batch_tensorize_agent_identities(
                    u_r_x_h_human_indices, u_r_x_h_states
                )
                
                u_r_x_h_grid = s_encoded[0][u_r_trans_idx]
                u_r_x_h_glob = s_encoded[1][u_r_trans_idx]
                u_r_x_h_agent = s_encoded[2][u_r_trans_idx]
                u_r_x_h_interactive = s_encoded[3][u_r_trans_idx]
                
                x_h_for_u_r = self.networks.x_h_target.forward(
                    u_r_x_h_grid, u_r_x_h_glob, u_r_x_h_agent, u_r_x_h_interactive,
                    u_r_idx, u_r_agent_grid, u_r_agent_feat
                )
                # Hard clamp X_h for inference (soft clamp is only for training X_h)
                x_h_for_u_r = self.networks.x_h_target.apply_hard_clamp(x_h_for_u_r)
            
            # Compute U_r targets by aggregating X_h values for each transition
            u_r_targets = []
            for i in range(n):
                start_idx = u_r_transition_boundaries[i]
                end_idx = u_r_transition_boundaries[i + 1]
                x_h_values = x_h_for_u_r[start_idx:end_idx].squeeze()
                
                if x_h_values.dim() == 0:
                    x_h_values = x_h_values.unsqueeze(0)
                # Clamp X_h to (0, 1] to prevent X_h^{-ξ} explosion when X_h is near 0
                x_h_clamped = torch.clamp(x_h_values, min=1e-3, max=1.0)
                x_h_sum = (x_h_clamped ** (-self.config.xi)).sum()
                x_h_avg = x_h_sum / x_h_clamped.numel()
                # target_y = E[X_h^{-ξ}] directly (y is defined as E[X_h^{-ξ}])
                target_y = x_h_avg
                u_r_targets.append(target_y)
            
            u_r_targets = torch.stack(u_r_targets)
            loss_u_r = ((y_pred.squeeze() - u_r_targets) ** 2).mean()
            
            # Track U_r prediction stats (the actual U_r values, not y)
            # U_r = -y^η where η = -1/ξ, so U_r is negative
            with torch.no_grad():
                # Compute target U_r from target y: U_r = -y^η where η = -1/ξ
                target_u_r = -torch.pow(u_r_targets, -1.0 / self.config.xi)
                prediction_stats['u_r'] = {
                    'mean': u_r_pred.mean().item(),
                    'std': u_r_pred.std().item() if u_r_pred.numel() > 1 else 0.0,
                    'target_mean': target_u_r.mean().item()
                }
        
        # ----- V_h^e: Human goal achievement -----
        loss_v_h_e = torch.tensor(0.0, device=self.device)
        if v_h_e_goals:
            # V_h^e trains the shared encoders - do NOT detach
            v_h_e_pred = self.networks.v_h_e.forward(
                s_h_grid, s_h_glob, s_h_agent, s_h_interactive, 
                goal_features,
                v_h_e_idx, v_h_e_grid, v_h_e_feat
            )
            
            if model_based_v_h_e_targets is not None:
                # Use pre-computed model-based targets (computed jointly with Q_r above)
                target_v_h_e = model_based_v_h_e_targets
            else:
                # Sample-based targets: use observed transition
                goal_achieved_t = torch.tensor(
                    [1.0 if a else 0.0 for a in v_h_e_achieved],
                    device=self.device
                )
                
                with torch.no_grad():
                    v_h_e_next = self.networks.v_h_e_target.forward(
                        s_prime_h_grid, s_prime_h_glob, s_prime_h_agent, s_prime_h_interactive,
                        goal_features, v_h_e_prime_idx, v_h_e_prime_grid, v_h_e_prime_feat
                    )
                
                target_v_h_e = self.networks.v_h_e.compute_td_target(
                    goal_achieved_t, v_h_e_next.squeeze()
                )
            
            loss_v_h_e = ((v_h_e_pred.squeeze() - target_v_h_e) ** 2).mean()
            
            # Track V_h^e prediction stats
            with torch.no_grad():
                prediction_stats['v_h_e'] = {
                    'mean': v_h_e_pred.mean().item(),
                    'std': v_h_e_pred.std().item() if v_h_e_pred.numel() > 1 else 0.0,
                    'target_mean': target_v_h_e.mean().item()
                }
        
        # ----- X_h: Aggregate goal ability -----
        loss_x_h = torch.tensor(0.0, device=self.device)
        if x_h_goals:
            # X_h uses: shared agent encoder (detached) + own agent encoder (trained)
            # Encode with X_h's own agent encoder
            own_x_h_idx, own_x_h_agent_grid, own_x_h_agent_feat = self._batch_tensorize_agent_identities_with_encoder(
                x_h_human_indices, x_h_states_for_identity,
                self.networks.x_h.own_agent_encoder
            )
            
            # Detach shared state encoder outputs, pass own agent encoder outputs
            x_h_pred = self.networks.x_h.forward(
                x_h_grid.detach(), x_h_glob.detach(), x_h_agent.detach(), x_h_interactive.detach(),
                x_h_idx.detach(), x_h_agent_grid.detach(), x_h_agent_feat.detach(),  # Shared (frozen)
                own_x_h_idx, own_x_h_agent_grid, own_x_h_agent_feat  # Own (trained)
            )
            
            with torch.no_grad():
                v_h_e_for_x = self.networks.v_h_e_target.forward(
                    x_h_grid, x_h_glob, x_h_agent, x_h_interactive, x_h_goal_features,
                    x_h_idx, x_h_agent_grid, x_h_agent_feat
                )
                # Hard clamp V_h^e for inference (soft clamp is only for training V_h^e)
                v_h_e_for_x = self.networks.v_h_e_target.apply_hard_clamp(v_h_e_for_x)
            
            target_x_h = self.networks.x_h.compute_target(v_h_e_for_x.squeeze())
            loss_x_h = ((x_h_pred.squeeze() - target_x_h) ** 2).mean()
            
            # Track X_h prediction stats
            with torch.no_grad():
                prediction_stats['x_h'] = {
                    'mean': x_h_pred.mean().item(),
                    'std': x_h_pred.std().item() if x_h_pred.numel() > 1 else 0.0,
                    'target_mean': target_x_h.mean().item()
                }
        
        # ----- V_r: Robot value (if using network mode) -----
        losses = {
            'v_h_e': loss_v_h_e,
            'x_h': loss_x_h,
            'u_r': loss_u_r,
            'q_r': loss_q_r,
        }
        
        if self.config.v_r_use_network:
            # V_r uses only shared encoder (detached) - no own encoder
            v_r_pred = self.networks.v_r.forward(*s_encoded_detached)
            
            with torch.no_grad():
                # Use frozen u_r_target for stable V_r target computation
                _, u_r = self.networks.u_r_target.forward(*s_encoded_detached)
                # Q_r needs both encoders for proper inference
                own_s_for_v = self._batch_tensorize_states_with_encoder(
                    [t.state for t in batch],
                    self.networks.q_r.own_state_encoder
                )
                q_r_for_v = self.networks.q_r.forward(*s_encoded_detached, *own_s_for_v)
                # Use effective beta_r for policy (0 during warm-up for independence)
                pi_r = self.networks.q_r.get_policy(q_r_for_v, beta_r=effective_beta_r)
            
            target_v_r = self.networks.v_r.compute_from_components(
                u_r.squeeze(), q_r_for_v, pi_r
            )
            losses['v_r'] = ((v_r_pred.squeeze() - target_v_r) ** 2).mean()
            
            # Track V_r prediction stats
            with torch.no_grad():
                prediction_stats['v_r'] = {
                    'mean': v_r_pred.mean().item(),
                    'std': v_r_pred.std().item() if v_r_pred.numel() > 1 else 0.0,
                    'target_mean': target_v_r.mean().item()
                }
        
        # Clear caches after each compute_losses call to prevent memory growth
        # The caches are useful within a single batch computation but stale afterwards
        self.clear_caches()
        
        return losses, prediction_stats
    
    def _batch_tensorize_from_compact(
        self,
        states: List[Any],
        compact_features_list: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch tensorize states using pre-computed compact features.
        
        This is the optimized path that avoids ALL access to world_model:
        - Global, agent, interactive features are retrieved from compact_features_list
        - Grid tensor is decompressed from the compressed_grid in compact_features
        
        The compressed grid captures ALL grid information (including static objects
        from world_model), so training can proceed without any world_model access.
        
        If compact_features is None for a state, falls back to full tensorization
        (requires world_model access, only works for on-policy training).
        
        Args:
            states: List of states to tensorize (only used for fallback path).
            compact_features_list: Pre-computed (global, agent, interactive, compressed_grid)
                tuples, or None for states without pre-computed features.
        
        Returns:
            Tuple of batched tensors (grid, global, agent, interactive).
        """
        state_encoder = self.networks.q_r.state_encoder
        
        # Check if we can use the fully vectorized path (all compact_features available)
        all_have_compact = all(c is not None for c in compact_features_list)
        
        if all_have_compact:
            # FAST PATH: Fully vectorized batch decompression
            # Extract compressed grids and stack
            compressed_grids = torch.stack([c[3] for c in compact_features_list], dim=0)
            
            grid_batch = state_encoder.decompress_grid_batch_to_tensor(
                compressed_grids, self.device
            )
            
            # Stack the other features
            global_batch = torch.cat([c[0].to(self.device) for c in compact_features_list], dim=0)
            agent_batch = torch.cat([c[1].to(self.device) for c in compact_features_list], dim=0)
            interactive_batch = torch.cat([c[2].to(self.device) for c in compact_features_list], dim=0)
            
            return (grid_batch, global_batch, agent_batch, interactive_batch)
        
        # SLOW PATH: Mixed batch with some missing compact_features
        grid_list = []
        global_list = []
        agent_list = []
        interactive_list = []
        
        for state, compact in zip(states, compact_features_list):
            if compact is not None:
                # Use pre-computed compact features
                glob, agent, interactive, compressed_grid = compact
                # Decompress the grid (vectorized for single state)
                grid = state_encoder.decompress_grid_to_tensor(compressed_grid, self.device)
            else:
                # Fall back to full tensorization (requires world_model - LEGACY)
                # This path only works if we have access to world_model via self.env
                # In off-policy training with different world layouts, this will fail!
                grid, glob, agent, interactive = state_encoder.tensorize_state(
                    state, None, self.device
                )
            
            grid_list.append(grid)
            global_list.append(glob.to(self.device))
            agent_list.append(agent.to(self.device))
            interactive_list.append(interactive.to(self.device))
        
        return (
            torch.cat(grid_list, dim=0),
            torch.cat(global_list, dim=0),
            torch.cat(agent_list, dim=0),
            torch.cat(interactive_list, dim=0)
        )
    
    def _batch_tensorize_states(
        self,
        states: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch tensorize multiple states (agent-agnostic).
        
        The state encoder produces agent-agnostic representations.
        Agent identity is handled separately by AgentIdentityEncoder.
        
        Args:
            states: List of states to tensorize.
        
        Returns:
            Tuple of batched tensors (grid, global, agent, interactive).
        """
        state_encoder = self.networks.q_r.state_encoder
        
        grid_list = []
        global_list = []
        agent_list = []
        interactive_list = []
        
        for state in states:
            # The encoder handles caching internally
            # Encoding is agent-agnostic
            grid, glob, agent, interactive = state_encoder.tensorize_state(
                state, None, self.device
            )
            
            grid_list.append(grid)
            global_list.append(glob)
            agent_list.append(agent)
            interactive_list.append(interactive)
        
        return (
            torch.cat(grid_list, dim=0),
            torch.cat(global_list, dim=0),
            torch.cat(agent_list, dim=0),
            torch.cat(interactive_list, dim=0)
        )
    
    def _batch_tensorize_states_with_encoder(
        self,
        states: List[Any],
        state_encoder: 'MultiGridStateEncoder'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch encode multiple states using a specific encoder (agent-agnostic).
        
        This is used for encoding with network-specific encoders (e.g., Q_r's own encoder).
        The encoding is agent-agnostic - agent identity is handled separately.
        
        Args:
            states: List of states to encode.
            state_encoder: The specific state encoder to use.
        
        Returns:
            Tuple of batched tensors (grid, global, agent, interactive).
        """
        grid_list = []
        global_list = []
        agent_list = []
        interactive_list = []
        
        for state in states:
            # Encoding is agent-agnostic
            grid, glob, agent, interactive = state_encoder.tensorize_state(
                state, None, self.device
            )
            
            grid_list.append(grid)
            global_list.append(glob)
            agent_list.append(agent)
            interactive_list.append(interactive)
        
        return (
            torch.cat(grid_list, dim=0),
            torch.cat(global_list, dim=0),
            torch.cat(agent_list, dim=0),
            torch.cat(interactive_list, dim=0)
        )
    
    def _batch_tensorize_agent_identities_with_encoder(
        self,
        human_indices: List[int],
        states: List[Any],
        agent_encoder: 'AgentIdentityEncoder'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch encode agent identities using a specific encoder.
        
        This is used for encoding with network-specific encoders (e.g., X_h's own encoder).
        
        Args:
            human_indices: List of human agent indices.
            states: List of states (for position info).
            agent_encoder: The specific agent encoder to use.
        
        Returns:
            Tuple of (indices, grids, features) tensors.
        """
        idx_list = []
        grid_list = []
        feat_list = []
        
        for h_idx, state in zip(human_indices, states):
            idx, grid, feat = agent_encoder.encode_single(
                h_idx, state, None, self.device
            )
            idx_list.append(idx)
            grid_list.append(grid)
            feat_list.append(feat)
        
        return (
            torch.cat(idx_list, dim=0),
            torch.cat(grid_list, dim=0),
            torch.cat(feat_list, dim=0)
        )
    
    def _batch_tensorize_goals(self, goals: List[Any]) -> torch.Tensor:
        """
        Batch encode multiple goals efficiently using a single forward pass.
        
        Note: We intentionally do NOT cache goal encoder outputs because that
        would break gradient flow. The goal encoder has trainable nn.Linear
        parameters that need gradients to update during backprop.
        
        Args:
            goals: List of goals to encode.
        
        Returns:
            Batched goal features tensor of shape (len(goals), feature_dim).
        """
        goal_encoder = self.networks.v_h_e.goal_encoder
        
        # Extract coordinates for all goals (cheap, no neural network)
        coords_list = []
        for goal in goals:
            goal_coords = goal_encoder.tensorize_goal(goal, self.device)
            coords_list.append(goal_coords)
        
        # Stack coordinates: (num_goals, 4)
        coords_batch = torch.cat(coords_list, dim=0)
        
        # Single batched forward pass - preserves gradient flow
        return goal_encoder(coords_batch)
    
    def _batch_tensorize_agent_identities(
        self,
        agent_indices: List[int],
        states: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch encode agent identities (index + position grid + features).
        
        Uses the shared agent encoder which handles caching internally.
        
        Args:
            agent_indices: List of agent indices to encode.
            states: List of states (to extract agent positions and features).
        
        Returns:
            Tuple of (agent_idx_tensor, query_agent_grids, query_agent_features).
        """
        agent_encoder = self.networks.v_h_e.agent_encoder
        
        # Use the encoder's batch method which handles caching internally
        return agent_encoder.encode_batch(
            agent_indices, states, self.env, self.device
        )


def create_phase2_networks(
    env: MultiGridEnv,
    config: Phase2Config,
    num_robots: int,
    num_actions: int,
    hidden_dim: int = 256,
    device: str = 'cpu',
    goal_feature_dim: int = 64,
    max_agents: int = 10,
    agent_embedding_dim: int = 16
) -> Phase2Networks:
    """
    Create all Phase 2 networks for a multigrid environment.
    
    Creates SHARED encoders that are used by all networks:
    - One state encoder shared by Q_r, V_h^e, X_h, U_r, V_r
    - One goal encoder shared by V_h^e
    - One agent encoder shared by V_h^e, X_h
    
    This ensures consistent encoding across networks and enables
    efficient caching of raw tensor extraction.
    
    Args:
        env: MultiGridEnv instance.
        config: Phase2Config.
        num_robots: Number of robot agents.
        num_actions: Number of actions per robot.
        hidden_dim: Hidden dimension for networks (also state_feature_dim).
        device: Torch device.
        goal_feature_dim: Goal encoder output dimension.
        max_agents: Max number of agents for identity encoding.
        agent_embedding_dim: Dimension of agent identity embedding.
    
    Returns:
        Phase2Networks container with all networks using shared encoders.
    """
    # Count agents per color
    num_agents_per_color = {}
    for agent in env.agents:
        color = agent.color
        num_agents_per_color[color] = num_agents_per_color.get(color, 0) + 1
    
    # Common parameters
    grid_height = env.height
    grid_width = env.width
    
    # Create SHARED encoders (one instance used by all networks)
    shared_state_encoder = MultiGridStateEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=7,
        feature_dim=hidden_dim,
        include_step_count=config.include_step_count,
    ).to(device)
    
    shared_goal_encoder = MultiGridGoalEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        feature_dim=goal_feature_dim
    ).to(device)
    
    shared_agent_encoder = AgentIdentityEncoder(
        num_agents=max_agents,
        embedding_dim=agent_embedding_dim,
        grid_height=grid_height,
        grid_width=grid_width
    ).to(device)
    
    # Create OWN encoders for Q_r and X_h (trained with their respective losses)
    # These allow Q_r and X_h to learn additional features beyond those learned by V_h^e
    q_r_own_state_encoder = MultiGridStateEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=7,
        feature_dim=hidden_dim,
        include_step_count=config.include_step_count,
    ).to(device)
    
    x_h_own_agent_encoder = AgentIdentityEncoder(
        num_agents=max_agents,
        embedding_dim=agent_embedding_dim,
        grid_height=grid_height,
        grid_width=grid_width
    ).to(device)
    
    # Create networks with SHARED encoders and OWN encoders where applicable
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
        state_encoder=shared_state_encoder,       # SHARED (frozen for Q_r)
        own_state_encoder=q_r_own_state_encoder,  # OWN (trained with Q_r)
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
        state_encoder=shared_state_encoder,  # SHARED
        goal_encoder=shared_goal_encoder,    # SHARED
        agent_encoder=shared_agent_encoder,  # SHARED
    ).to(device)
    
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
        state_encoder=shared_state_encoder,       # SHARED (frozen for X_h)
        agent_encoder=shared_agent_encoder,       # SHARED (frozen for X_h)
        own_agent_encoder=x_h_own_agent_encoder,  # OWN (trained with X_h)
    ).to(device)
    
    u_r = MultiGridIntrinsicRewardNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        xi=config.xi,
        eta=config.eta,
        dropout=config.u_r_dropout,
        state_encoder=shared_state_encoder,  # SHARED
    ).to(device)
    
    v_r = MultiGridRobotValueNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        dropout=config.v_r_dropout,
        state_encoder=shared_state_encoder,  # SHARED
    ).to(device)
    
    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r
    )


def train_multigrid_phase2(
    world_model: MultiGridEnv,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_policy_prior: Callable,
    goal_sampler: Callable,
    config: Optional[Phase2Config] = None,
    num_episodes: Optional[int] = None,
    hidden_dim: int = 256,
    goal_feature_dim: int = 64,
    agent_embedding_dim: int = 16,
    device: str = 'cpu',
    verbose: bool = True,
    debug: bool = False,
    tensorboard_dir: Optional[str] = None,
    profiler: Optional[Any] = None,
) -> Tuple[MultiGridRobotQNetwork, Phase2Networks, List[Dict[str, float]]]:
    """
    Train Phase 2 robot policy for a multigrid environment.
    
    This function trains neural networks to learn the robot policy that maximizes
    aggregate human power, as defined in equations (4)-(9) of the EMPO paper.
    
    Args:
        world_model: MultiGridEnv instance.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: Callable(state, human_idx, goal) -> action.
        goal_sampler: Callable(state, human_idx) -> goal.
        config: Phase2Config (uses defaults if None).
        num_episodes: Override config.num_episodes if provided.
        hidden_dim: Hidden dimension for networks.
        goal_feature_dim: Goal encoder output dimension.
        agent_embedding_dim: Dimension of agent identity embedding.
        device: Torch device.
        verbose: Enable progress bar (tqdm).
        debug: Enable verbose debug output.
        tensorboard_dir: Directory for TensorBoard logs (optional).
    
    Returns:
        Tuple of (robot_q_network, all_networks, training_history).
    """
    if config is None:
        config = Phase2Config()
    
    if num_episodes is not None:
        config.num_episodes = num_episodes
    
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
        agent_embedding_dim=agent_embedding_dim
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
    )
    
    if verbose:
        print(f"\nTraining for {config.num_episodes} episodes...")
        print(f"  Steps per episode: {config.steps_per_episode}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size}")
        if tensorboard_dir:
            print(f"  TensorBoard: {tensorboard_dir}")
    
    # Train
    history = trainer.train(config.num_episodes)
    
    if verbose:
        print(f"\nTraining completed!")
        if history:
            final_losses = history[-1]
            print(f"  Final losses: {final_losses}")
    
    return networks.q_r, networks, history
