"""
Multigrid-specific Phase 2 Trainer.

This module provides the training function for Phase 2 of the EMPO framework
(equations 4-9) specialized for multigrid environments.
"""

import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from gym_multigrid.multigrid import MultiGridEnv

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.trainer import BasePhase2Trainer, Phase2Networks
from empo.nn_based.phase2.replay_buffer import Phase2Transition
from empo.nn_based.phase2.robot_value_network import compute_v_r_from_components
from empo.nn_based.multigrid import MultiGridStateEncoder
from empo.nn_based.multigrid.goal_encoder import MultiGridGoalEncoder
from empo.nn_based.multigrid.agent_encoder import AgentIdentityEncoder

from .robot_q_network import MultiGridRobotQNetwork
from .human_goal_ability import MultiGridHumanGoalAchievementNetwork
from .aggregate_goal_ability import MultiGridAggregateGoalAbilityNetwork
from .intrinsic_reward_network import MultiGridIntrinsicRewardNetwork
from .robot_value_network import MultiGridRobotValueNetwork

# Lookup table network imports
from empo.nn_based.phase2.lookup import (
    LookupTableRobotQNetwork,
    LookupTableRobotValueNetwork,
    LookupTableHumanGoalAbilityNetwork,
    LookupTableAggregateGoalAbilityNetwork,
    LookupTableIntrinsicRewardNetwork,
    is_lookup_table_network,
)


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
    ):
        self.env = env
        self.world_model_factory = world_model_factory
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
    
    def __getstate__(self):
        """Exclude env from pickling (for async training).
        
        Calls parent __getstate__ then also excludes env.
        """
        # Get parent state (excludes writer, profiler, replay_buffer)
        state = super().__getstate__()
        # Also exclude env - it contains thread locks
        state['env'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling; env will be recreated from factory.
        
        Calls parent __setstate__ then allows env to be lazily created.
        """
        super().__setstate__(state)
        # Note: env stays None until _ensure_world_model() is called
    
    def _ensure_world_model(self):
        """
        Ensure world model is available, creating from factory if needed.
        
        Called by reset_environment() when env is None (in async actor processes).
        After creating the env, updates goal_sampler and human_policy_prior.
        """
        if self.env is None:
            if self.world_model_factory is None:
                raise RuntimeError(
                    "No world_model_factory provided. For async training, "
                    "you must pass a world_model_factory to the trainer."
                )
            self._create_env_from_factory()
    
    def _create_env_from_factory(self):
        """
        Create environment from factory and update dependent components.
        
        Called by _ensure_world_model() for initial creation, and by
        reset_environment() for ensemble mode (new env each episode).
        """
        # Create env from factory
        self.env = self.world_model_factory.create()
        # Update goal_sampler and human_policy_prior with new env
        if hasattr(self.goal_sampler, 'set_world_model'):
            self.goal_sampler.set_world_model(self.env)
        if hasattr(self.human_policy_prior, 'set_world_model'):
            self.human_policy_prior.set_world_model(self.env)
    

    def clear_caches(self):
        """
        Clear encoder caches. Call after each training step to prevent memory growth.
        
        This clears the caches in the shared encoders that all networks use.
        For lookup table networks, this is a no-op (they don't have encoder caches).
        """
        # Check if using lookup tables (no encoder caches)
        if is_lookup_table_network(self.networks.q_r):
            return
        # All networks share the same encoders, so we only need to clear once
        self.networks.q_r.state_encoder.clear_cache()
        self.networks.v_h_e.goal_encoder.clear_cache()
        self.networks.v_h_e.agent_encoder.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Tuple[int, int]]:
        """
        Get cache hit/miss statistics from all encoders.
        
        For lookup table networks, returns empty dict (no encoder caches).
        
        Returns:
            Dict mapping encoder name to (hits, misses) tuple.
        """
        if is_lookup_table_network(self.networks.q_r):
            return {}
        return {
            'state': self.networks.q_r.state_encoder.get_cache_stats(),
            'goal': self.networks.v_h_e.goal_encoder.get_cache_stats(),
            'agent': self.networks.v_h_e.agent_encoder.get_cache_stats(),
        }
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters for all encoders.
        
        For lookup table networks, this is a no-op (they don't have encoder caches).
        """
        if is_lookup_table_network(self.networks.q_r):
            return
        self.networks.q_r.state_encoder.reset_cache_stats()
        self.networks.v_h_e.goal_encoder.reset_cache_stats()
        self.networks.v_h_e.agent_encoder.reset_cache_stats()
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
    
    def compute_losses(
        self,
        batch: List[Phase2Transition],
        x_h_batch: Optional[List[Phase2Transition]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        """
        Compute losses for all networks using batched forward passes.
        
        This override uses the forward_batch methods on all networks for efficient
        batched computation. Works for both neural networks and lookup tables.
        
        Args:
            batch: List of transitions for most networks.
            x_h_batch: Optional larger batch for X_h (defaults to batch).
        
        Returns:
            Tuple of (losses dict, prediction_stats dict).
        """
        if x_h_batch is None:
            x_h_batch = batch
        
        n = len(batch)
        
        # Check which networks are active in current warmup stage
        active_networks = self.config.get_active_networks(self.training_step_count)
        v_h_e_active = 'v_h_e' in active_networks
        x_h_active = 'x_h' in active_networks
        u_r_active = 'u_r' in active_networks
        q_r_active = 'q_r' in active_networks
        v_r_active = 'v_r' in active_networks
        
        # Track prediction statistics
        prediction_stats = {}
        
        # Get effective beta_r (needed for V_r target computation)
        effective_beta_r = self.config.get_effective_beta_r(self.training_step_count)
        
        # ===================================================================
        # Stage 1: Collect data from transitions for batched processing
        # ===================================================================
        
        # Extract states and next_states
        states = [t.state for t in batch]
        next_states = [t.next_state for t in batch]
        
        # Collect V_h^e data: (state_idx, state, next_state, human_idx, goal)
        v_h_e_indices = []  # which transition this came from
        v_h_e_states = []
        v_h_e_next_states = []
        v_h_e_human_indices = []
        v_h_e_goals = []
        v_h_e_achieved = []
        
        for i, t in enumerate(batch):
            for h, g_h in t.goals.items():
                v_h_e_indices.append(i)
                v_h_e_states.append(t.state)
                v_h_e_next_states.append(t.next_state)
                v_h_e_human_indices.append(h)
                v_h_e_goals.append(g_h)
                v_h_e_achieved.append(self.check_goal_achieved(t.next_state, h, g_h))
        
        # Collect X_h data
        x_h_states = []
        x_h_human_indices = []
        x_h_goals = []
        x_h_goal_weights = []
        
        if x_h_active:
            for t in x_h_batch:
                if self.config.x_h_sample_humans is None:
                    humans_for_x_h = list(t.goals.keys())
                else:
                    n_sample = min(self.config.x_h_sample_humans, len(t.goals))
                    humans_for_x_h = random.sample(list(t.goals.keys()), n_sample)
                
                for h_x in humans_for_x_h:
                    x_h_states.append(t.state)
                    x_h_human_indices.append(h_x)
                    x_h_goals.append(t.goals[h_x])
                    x_h_goal_weights.append(t.goal_weights[h_x])
        
        # Collect U_r data - flatten all (state, human) pairs for batched X_h computation
        u_r_flat_states = []      # states repeated for each human
        u_r_flat_humans = []      # human indices
        u_r_humans_per_state = [] # how many humans sampled for each state
        
        if self.config.u_r_use_network and u_r_active:
            for t in batch:
                if self.config.u_r_sample_humans is None:
                    humans_for_u_r = list(self.human_agent_indices)
                else:
                    n_sample = min(self.config.u_r_sample_humans, len(self.human_agent_indices))
                    humans_for_u_r = random.sample(list(self.human_agent_indices), n_sample)
                
                u_r_humans_per_state.append(len(humans_for_u_r))
                for h in humans_for_u_r:
                    u_r_flat_states.append(t.state)
                    u_r_flat_humans.append(h)
        
        # ===================================================================
        # Stage 2: Batched forward passes
        # ===================================================================
        
        losses = {
            'v_h_e': torch.tensor(0.0, device=self.device),
            'x_h': torch.tensor(0.0, device=self.device),
            'q_r': torch.tensor(0.0, device=self.device),
        }
        if self.config.u_r_use_network:
            losses['u_r'] = torch.tensor(0.0, device=self.device)
        if self.config.v_r_use_network:
            losses['v_r'] = torch.tensor(0.0, device=self.device)
        
        # ----- V_h^e loss (batched) -----
        if v_h_e_states:
            # Forward pass for current states
            v_h_e_pred = self.networks.v_h_e.forward_batch(
                v_h_e_states, v_h_e_goals, v_h_e_human_indices,
                self.env, self.device
            )
            
            # Target: check goal achieved + V_h^e(s', g_h) from target network
            goal_achieved_t = torch.tensor(
                [1.0 if a else 0.0 for a in v_h_e_achieved],
                device=self.device
            )
            
            with torch.no_grad():
                v_h_e_next = self.networks.v_h_e_target.forward_batch(
                    v_h_e_next_states, v_h_e_goals, v_h_e_human_indices,
                    self.env, self.device
                )
            
            # TD target: achieved + (1 - achieved) * gamma_h * V_h^e_next
            target_v_h_e = self.networks.v_h_e.compute_td_target(
                goal_achieved_t, v_h_e_next.squeeze()
            )
            
            losses['v_h_e'] = ((v_h_e_pred.squeeze() - target_v_h_e) ** 2).mean()
            
            with torch.no_grad():
                prediction_stats['v_h_e'] = {
                    'mean': v_h_e_pred.mean().item(),
                    'std': v_h_e_pred.std().item() if v_h_e_pred.numel() > 1 else 0.0,
                    'target_mean': target_v_h_e.mean().item()
                }
        
        # ----- X_h loss (batched, potentially larger batch) -----
        if x_h_active and x_h_states:
            # Forward pass
            x_h_pred = self.networks.x_h.forward_batch(
                x_h_states, x_h_human_indices,
                self.env, self.device
            )
            
            # Target from V_h^e target network
            with torch.no_grad():
                v_h_e_for_x = self.networks.v_h_e_target.forward_batch(
                    x_h_states, x_h_goals, x_h_human_indices,
                    self.env, self.device
                )
                # Hard clamp for inference
                v_h_e_for_x = self.networks.v_h_e_target.apply_hard_clamp(v_h_e_for_x)
            
            # Compute targets: w_h * V_h^e(s, g_h)^zeta
            x_h_weights_tensor = torch.tensor(x_h_goal_weights, device=self.device, dtype=torch.float32)
            target_x_h = self.networks.x_h.compute_target(v_h_e_for_x.squeeze(), x_h_weights_tensor)
            
            losses['x_h'] = ((x_h_pred.squeeze() - target_x_h) ** 2).mean()
            
            with torch.no_grad():
                prediction_stats['x_h'] = {
                    'mean': x_h_pred.mean().item(),
                    'std': x_h_pred.std().item() if x_h_pred.numel() > 1 else 0.0,
                    'target_mean': target_x_h.mean().item()
                }
        
        # ----- U_r loss (fully batched, only if using network mode) -----
        if self.config.u_r_use_network and u_r_active and u_r_flat_states:
            # Forward pass on unique states for U_r predictions
            y_pred, _ = self.networks.u_r.forward_batch(
                states, self.env, self.device
            )
            
            # Single batched X_h computation for all (state, human) pairs
            with torch.no_grad():
                x_h_all = self.networks.x_h_target.forward_batch(
                    u_r_flat_states, u_r_flat_humans,
                    self.env, self.device
                ).squeeze()
                
                # Clamp X_h values and compute X_h^{-xi}
                x_h_clamped = torch.clamp(x_h_all, min=1e-3, max=1.0)
                x_h_power = x_h_clamped ** (-self.config.xi)
                
                # Aggregate by state using scatter_add: sum X_h^{-xi} for each state
                # Build state indices: [0,0,0, 1,1,1, 2,2,2, ...] based on humans_per_state
                state_indices = []
                for state_idx, n_humans in enumerate(u_r_humans_per_state):
                    state_indices.extend([state_idx] * n_humans)
                state_indices_t = torch.tensor(state_indices, device=self.device)
                
                n_states = len(batch)
                x_h_sums = torch.zeros(n_states, device=self.device)
                x_h_sums.scatter_add_(0, state_indices_t, x_h_power)
                
                # Average: y = E[X_h^{-xi}]
                humans_per_state_t = torch.tensor(u_r_humans_per_state, device=self.device, dtype=torch.float32)
                u_r_targets_tensor = x_h_sums / humans_per_state_t
            
            losses['u_r'] = ((y_pred.squeeze() - u_r_targets_tensor) ** 2).mean()
            
            with torch.no_grad():
                # U_r = -y^eta
                target_u_r = -(u_r_targets_tensor ** self.config.eta)
                prediction_stats['u_r'] = {
                    'mean': y_pred.mean().item(),
                    'std': y_pred.std().item() if y_pred.numel() > 1 else 0.0,
                    'target_mean': target_u_r.mean().item()
                }
        
        # ----- Q_r loss (batched) -----
        if q_r_active:
            q_r_all = self.networks.q_r.forward_batch(states, self.env, self.device)
            
            # Gather Q-values for taken actions
            robot_actions = [t.robot_action for t in batch]
            action_indices = torch.tensor(
                [self.networks.q_r.action_tuple_to_index(a) for a in robot_actions],
                device=self.device
            )
            q_r_pred = q_r_all.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            
            # Compute targets
            with torch.no_grad():
                if self.config.v_r_use_network:
                    v_r_next = self.networks.v_r_target.forward_batch(
                        next_states, self.env, self.device
                    )
                else:
                    # Compute V_r from components
                    u_r_next = self._compute_u_r_batch_target(next_states)
                    q_r_next = self.networks.q_r_target.forward_batch(
                        next_states, self.env, self.device
                    )
                    pi_r_next = self.networks.q_r_target.get_policy(q_r_next, beta_r=effective_beta_r)
                    v_r_next = compute_v_r_from_components(
                        u_r_next.squeeze(), q_r_next, pi_r_next
                    )
            
            target_q_r = self.config.gamma_r * v_r_next.squeeze()
            losses['q_r'] = ((q_r_pred - target_q_r) ** 2).mean()
            
            with torch.no_grad():
                prediction_stats['q_r'] = {
                    'mean': q_r_pred.mean().item(),
                    'std': q_r_pred.std().item() if q_r_pred.numel() > 1 else 0.0,
                    'target_mean': target_q_r.mean().item()
                }
        
        # ----- V_r loss (batched, only if using network mode) -----
        if self.config.v_r_use_network and v_r_active:
            v_r_pred = self.networks.v_r.forward_batch(states, self.env, self.device)
            
            with torch.no_grad():
                u_r_for_v = self._compute_u_r_batch_target(states)
                q_r_for_v = self.networks.q_r_target.forward_batch(
                    states, self.env, self.device
                )
                pi_r = self.networks.q_r_target.get_policy(q_r_for_v, beta_r=effective_beta_r)
            
            target_v_r = compute_v_r_from_components(
                u_r_for_v.squeeze(), q_r_for_v, pi_r
            )
            losses['v_r'] = ((v_r_pred.squeeze() - target_v_r) ** 2).mean()
            
            with torch.no_grad():
                prediction_stats['v_r'] = {
                    'mean': v_r_pred.mean().item(),
                    'std': v_r_pred.std().item() if v_r_pred.numel() > 1 else 0.0,
                    'target_mean': target_v_r.mean().item()
                }
        
        # Clear caches after each compute_losses call
        self.clear_caches()
        
        return losses, prediction_stats
    
    def _compute_u_r_batch_target(self, states: List[Any]) -> torch.Tensor:
        """
        Compute U_r for a batch of states using target networks (fully batched).
        
        If u_r_use_network, uses U_r target network.
        Otherwise, computes from X_h target values.
        
        Args:
            states: List of states.
        
        Returns:
            U_r values tensor of shape (batch_size,).
        """
        if self.config.u_r_use_network:
            _, u_r = self.networks.u_r_target.forward_batch(states, self.env, self.device)
            return u_r
        else:
            # Compute from X_h: U_r = -E[X_h^{-xi}]^eta
            # Flatten all (state, human) pairs for single batched computation
            n_states = len(states)
            n_humans = len(self.human_agent_indices)
            
            flat_states = []
            flat_humans = []
            for state in states:
                for h in self.human_agent_indices:
                    flat_states.append(state)
                    flat_humans.append(h)
            
            # Single batched forward pass for all (state, human) pairs
            x_h_all = self.networks.x_h_target.forward_batch(
                flat_states, flat_humans, self.env, self.device
            ).squeeze()
            
            # Reshape to (n_states, n_humans) and compute mean over humans
            x_h_reshaped = x_h_all.view(n_states, n_humans)
            x_h_clamped = torch.clamp(x_h_reshaped, min=1e-3, max=1.0)
            
            # y = E[X_h^{-xi}] = mean over humans
            y = (x_h_clamped ** (-self.config.xi)).mean(dim=1)
            
            # U_r = -y^eta
            u_r = -(y ** self.config.eta)
            return u_r
    
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
        """Reset environment and return initial state.
        
        Always uses the world_model_factory if available. The factory determines
        whether to create a new environment or reuse an existing one:
        - CachedWorldModelFactory: returns same env (reset by env.reset() below)
        - EnsembleWorldModelFactory: returns new env each episode
        """
        if self.world_model_factory is not None:
            self._create_env_from_factory()
        elif self.env is None:
            raise RuntimeError(
                "No world_model_factory provided and env is None. "
                "For async training, you must pass a world_model_factory."
            )
        
        self.env.reset()
        return self.env.get_state()
    

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
    # In identity mode (use_encoders=False), encoders now compute their own
    # output dimensions internally. We don't need to set them here.
    # The hidden_dim for downstream MLP heads stays at default (small).
    
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
    
    # Create SHARED encoders (one instance used by all networks)
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
    
    # Agent encoder (identity mode computes its own output dim internally)
    shared_agent_encoder = AgentIdentityEncoder(
        num_agents=max_agents,
        embedding_dim=agent_embedding_dim,
        position_feature_dim=agent_position_feature_dim,
        agent_feature_dim=agent_feature_dim,
        grid_height=grid_height,
        grid_width=grid_width,
        use_encoders=config.use_encoders,
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
        use_encoders=config.use_encoders,
    ).to(device)
    
    x_h_own_agent_encoder = AgentIdentityEncoder(
        num_agents=max_agents,
        embedding_dim=agent_embedding_dim,
        position_feature_dim=agent_position_feature_dim,
        agent_feature_dim=agent_feature_dim,
        grid_height=grid_height,
        grid_width=grid_width,
        use_encoders=config.use_encoders,
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
    
    # Only create U_r network if u_r_use_network=True
    u_r = None
    if config.u_r_use_network:
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
    
    # Only create V_r network if v_r_use_network=True
    v_r = None
    if config.v_r_use_network:
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
