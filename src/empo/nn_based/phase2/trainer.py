"""
Base Trainer for Phase 2 Robot Policy Learning.

This module provides the training loop and loss computation for Phase 2
of the EMPO framework (equations 4-9).
"""

import glob
import os
import random
import copy
import multiprocessing as mp
import shutil
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import Phase2Config
from .replay_buffer import Phase2Transition, Phase2ReplayBuffer
from .robot_q_network import BaseRobotQNetwork
from .human_goal_ability import BaseHumanGoalAchievementNetwork
from .aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from .intrinsic_reward_network import BaseIntrinsicRewardNetwork
from .robot_value_network import BaseRobotValueNetwork, compute_v_r_from_components
from .lookup import get_all_lookup_tables, get_total_table_size
from .rnd import RNDModule

# Try to import tensorboard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@dataclass
class Phase2Networks:
    """Container for all Phase 2 networks."""
    q_r: BaseRobotQNetwork
    v_h_e: BaseHumanGoalAchievementNetwork
    x_h: BaseAggregateGoalAbilityNetwork
    # U_r and V_r are optional - only created when use_network=True
    u_r: Optional[BaseIntrinsicRewardNetwork] = None
    v_r: Optional[BaseRobotValueNetwork] = None
    
    # RND module for curiosity-driven exploration (optional)
    rnd: Optional[RNDModule] = None
    
    # Target networks (frozen copies for stable training)
    q_r_target: Optional[BaseRobotQNetwork] = None
    v_r_target: Optional[BaseRobotValueNetwork] = None
    v_h_e_target: Optional[BaseHumanGoalAchievementNetwork] = None
    x_h_target: Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r_target: Optional[BaseIntrinsicRewardNetwork] = None


class BasePhase2Trainer(ABC):
    """
    Base class for Phase 2 training.
    
    Implements the training loop for learning the robot policy to maximize
    aggregate human power as defined in equations (4)-(9).
    
    Provides generic implementations for environment interaction using the
    standard WorldModel API (get_state, set_state, step, reset) and the
    PossibleGoal API (is_achieved). Subclasses may override these methods
    for environment-specific optimizations.
    
    Args:
        env: Environment instance (WorldModel).
        networks: Phase2Networks container with all networks.
        config: Phase2Config with hyperparameters.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: HumanPolicyPrior instance with .sample(state, human_idx, goal) method.
        goal_sampler: PossibleGoalSampler instance with .sample(state, human_idx) -> (goal, weight) method.
        device: Torch device for computation.
        verbose: Enable progress output (tqdm progress bar).
        debug: Enable verbose debug output (very detailed, for debugging).
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
        world_model_factory: Optional factory for creating world models. Required for
            async training where the environment cannot be pickled. In ensemble mode,
            agent indices are automatically updated by calling the env's
            get_human_agent_indices() and get_robot_agent_indices() methods.
    """
    
    def __init__(
        self,
        env: Any,
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
        self.env = env
        self.world_model_factory = world_model_factory
        self.networks = networks
        self.config = config
        self.human_agent_indices = list(human_agent_indices)
        self.robot_agent_indices = list(robot_agent_indices)
        self.human_policy_prior = human_policy_prior
        self.goal_sampler = goal_sampler
        self.device = device
        self.verbose = verbose
        self.debug = debug
        self.profiler = profiler
        self.robot_exploration_policy = robot_exploration_policy
        self.human_exploration_policy = human_exploration_policy
        
        # Compute the total number of agents from the indices
        self._update_num_agents()
        
        # Initialize TensorBoard writer if requested
        self.writer = None
        if tensorboard_dir is not None and HAS_TENSORBOARD:
            # Archive old TensorBoard data to prevent mixing with new run
            self._archive_tensorboard_data(tensorboard_dir)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
            # Add static remarks to TensorBoard
            self.writer.add_text(
                'Loss/caution',
                "Losses are NOT expected to drop to zero as they are MSE losses "
                "that are lower bounded by the variance of the stochastic variable they track!",
                global_step=0
            )
            self.writer.add_text('GradNorm/remark', "Gradients are clipped.", global_step=0)
            if config.async_training:
                max_steps = config.max_env_steps_per_training_step
                if max_steps is not None:
                    self.writer.add_text(
                        'Progress/remark',
                        f"Async mode: at most {max_steps} env steps per training step. Buffer cleared before and after beta_r ramp-up.",
                        global_step=0
                    )
                else:
                    self.writer.add_text(
                        'Progress/remark',
                        "Async mode: unlimited env steps per training step. Buffer cleared before and after beta_r ramp-up.",
                        global_step=0
                    )
            else:
                self.writer.add_text(
                    'Progress/remark',
                    "Sync mode: one env step per training step. Buffer cleared before and after beta_r ramp-up.",
                    global_step=0
                )
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Initializing target networks...")
        
        # Initialize target networks
        self._init_target_networks()
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Initializing optimizers...")
        
        # Initialize optimizers
        self.optimizers = self._init_optimizers()
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Creating replay buffer...")
        
        # Replay buffer
        self.replay_buffer = Phase2ReplayBuffer(capacity=config.buffer_size)
        
        # Training step counters
        self.total_env_steps = 0  # environment interaction steps
        self.training_step_count = 0  # gradient update steps (learning steps)
        
        # Shared env_steps counter for async mode (set by _learner_loop)
        # When buffer is cleared, this gets reset to allow actors to resume production
        self._shared_env_steps = None
        
        # Per-network update counters for 1/t learning rate schedules
        self.update_counts = {
            'q_r': 0,
            'v_r': 0,
            'v_h_e': 0,
            'x_h': 0,
            'u_r': 0,
            'rnd': 0,
        }
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Initialization complete.")
    
    def __getstate__(self):
        """Exclude unpicklable objects for async training (multiprocessing).
        
        Excludes:
        - TensorBoard writer (contains thread locks)
        - Profiler (contains thread locks)
        - Replay buffer (not needed in actor processes)
        - Environment (may contain thread locks; recreated from factory)
        """
        state = self.__dict__.copy()
        # Don't pickle TensorBoard writer - contains thread locks
        state['writer'] = None
        # Don't pickle profiler - may contain locks
        state['profiler'] = None
        # Don't pickle replay buffer - not needed in actor processes
        state['replay_buffer'] = None
        # Don't pickle env - it may contain thread locks
        state['env'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling for async training."""
        self.__dict__.update(state)
        # Recreate empty replay buffer if needed
        if self.replay_buffer is None:
            self.replay_buffer = Phase2ReplayBuffer(capacity=self.config.buffer_size)
        # Note: env stays None until _ensure_world_model() is called
    
    def _archive_tensorboard_data(self, tensorboard_dir: str) -> None:
        """Archive existing TensorBoard event files to a zip before starting a new run.
        
        This prevents old data from mixing with new runs in TensorBoard visualization.
        Old files are moved to a timestamped zip archive in the same directory.
        
        Args:
            tensorboard_dir: Path to the TensorBoard log directory.
        """
        if not os.path.exists(tensorboard_dir):
            return
        
        # Find all event files
        event_files = glob.glob(os.path.join(tensorboard_dir, 'events.out.tfevents.*'))
        if not event_files:
            return
        
        # Create archive filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f'archived_runs_{timestamp}.zip'
        archive_path = os.path.join(tensorboard_dir, archive_name)
        
        # Create zip archive
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for event_file in event_files:
                    # Add file to archive with just the filename (not full path)
                    zf.write(event_file, os.path.basename(event_file))
            
            # Remove original files after successful archiving
            for event_file in event_files:
                os.remove(event_file)
            
            if self.verbose:
                print(f"[TensorBoard] Archived {len(event_files)} old event files to {archive_name}")
        except Exception as e:
            # Don't fail training if archiving fails, just warn
            print(f"[TensorBoard] Warning: Failed to archive old data: {e}")
    
    def _init_target_networks(self):
        """Initialize target networks as copies of main networks."""
        self.networks.q_r_target = copy.deepcopy(self.networks.q_r)
        self.networks.v_h_e_target = copy.deepcopy(self.networks.v_h_e)
        self.networks.x_h_target = copy.deepcopy(self.networks.x_h)
        
        # Only create U_r/V_r targets if the networks exist
        if self.networks.u_r is not None:
            self.networks.u_r_target = copy.deepcopy(self.networks.u_r)
        if self.networks.v_r is not None:
            self.networks.v_r_target = copy.deepcopy(self.networks.v_r)
        
        # Freeze target networks (no gradients)
        for param in self.networks.q_r_target.parameters():
            param.requires_grad = False
        for param in self.networks.v_h_e_target.parameters():
            param.requires_grad = False
        for param in self.networks.x_h_target.parameters():
            param.requires_grad = False
        if self.networks.u_r_target is not None:
            for param in self.networks.u_r_target.parameters():
                param.requires_grad = False
        if self.networks.v_r_target is not None:
            for param in self.networks.v_r_target.parameters():
                param.requires_grad = False
        
        # Set target networks to eval mode (disables dropout during inference)
        self.networks.q_r_target.eval()
        self.networks.v_h_e_target.eval()
        self.networks.x_h_target.eval()
        if self.networks.u_r_target is not None:
            self.networks.u_r_target.eval()
        if self.networks.v_r_target is not None:
            self.networks.v_r_target.eval()
    
    def _init_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for each network with weight decay.
        
        For lookup table networks that may have empty parameter lists,
        creates a placeholder parameter to satisfy the optimizer constructor.
        """
        def make_optimizer(network, lr, weight_decay):
            """Create optimizer, handling empty parameter case for lookup tables."""
            params = list(network.parameters())
            if len(params) == 0:
                # Lookup tables start empty - create placeholder parameter
                # This will be replaced when optimizer is recreated after tables grow
                placeholder = nn.Parameter(torch.zeros(1, requires_grad=True))
                return optim.Adam([placeholder], lr=lr, weight_decay=weight_decay)
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        optimizers = {
            'q_r': make_optimizer(
                self.networks.q_r,
                lr=self.config.lr_q_r,
                weight_decay=self.config.q_r_weight_decay
            ),
            'v_h_e': make_optimizer(
                self.networks.v_h_e,
                lr=self.config.lr_v_h_e,
                weight_decay=self.config.v_h_e_weight_decay
            ),
            'x_h': make_optimizer(
                self.networks.x_h,
                lr=self.config.lr_x_h,
                weight_decay=self.config.x_h_weight_decay
            ),
        }
        # Only create U_r optimizer if using network (not direct computation)
        if self.config.u_r_use_network:
            optimizers['u_r'] = make_optimizer(
                self.networks.u_r,
                lr=self.config.lr_u_r,
                weight_decay=self.config.u_r_weight_decay
            )
        # Only create V_r optimizer if using network (not direct computation)
        if self.config.v_r_use_network:
            optimizers['v_r'] = make_optimizer(
                self.networks.v_r,
                lr=self.config.lr_v_r,
                weight_decay=self.config.v_r_weight_decay
            )
        # Create RND optimizer if using curiosity exploration
        if self.config.use_rnd and self.networks.rnd is not None:
            # Only train the predictor network (target is frozen)
            optimizers['rnd'] = optim.Adam(
                self.networks.rnd.predictor.parameters(),
                lr=self.config.lr_rnd,
                weight_decay=self.config.rnd_weight_decay
            )
        return optimizers
    
    def _add_new_lookup_params_to_optimizers(self):
        """
        Add new lookup table parameters to optimizers incrementally.
        
        Lookup tables create new parameters dynamically as new states are encountered.
        This method adds any newly created parameters to existing optimizers without
        recreating them, preserving momentum/adaptive state for existing parameters.
        
        This is very cheap (~0.002ms) so it's called every training step.
        """
        # Collect new parameters from all lookup table networks
        # Each network tracks new params since last call to get_new_params()
        new_params_by_optimizer = {
            'q_r': [],
            'v_h_e': [],
            'x_h': [],
        }
        
        # Check each network for new params
        if hasattr(self.networks.q_r, 'get_new_params'):
            new_params_by_optimizer['q_r'].extend(self.networks.q_r.get_new_params())
        if hasattr(self.networks.v_h_e, 'get_new_params'):
            new_params_by_optimizer['v_h_e'].extend(self.networks.v_h_e.get_new_params())
        if hasattr(self.networks.x_h, 'get_new_params'):
            new_params_by_optimizer['x_h'].extend(self.networks.x_h.get_new_params())
        if self.config.u_r_use_network and hasattr(self.networks.u_r, 'get_new_params'):
            # U_r shares optimizer with x_h
            new_params_by_optimizer['x_h'].extend(self.networks.u_r.get_new_params())
        if self.config.v_r_use_network and hasattr(self.networks.v_r, 'get_new_params'):
            # V_r shares optimizer with q_r
            new_params_by_optimizer['q_r'].extend(self.networks.v_r.get_new_params())
        
        # Add new parameters incrementally to each optimizer
        total_new = 0
        for name, new_params in new_params_by_optimizer.items():
            if new_params and name in self.optimizers:
                optimizer = self.optimizers[name]
                # Direct append to first param group is 300x faster than add_param_group
                for param in new_params:
                    optimizer.param_groups[0]['params'].append(param)
                total_new += len(new_params)
        
        if self.debug and total_new > 0:
            total_entries = get_total_table_size(self.networks)
            print(f"[DEBUG] Added {total_new} new params to optimizers "
                  f"(step {self.training_step_count}, {total_entries} total entries)")
    
    def _compute_param_norms(self) -> Dict[str, float]:
        """Compute L2 norms of network parameters for monitoring."""
        norms = {}
        networks = {
            'q_r': self.networks.q_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
        }
        # Only include U_r if using network mode
        if self.config.u_r_use_network:
            networks['u_r'] = self.networks.u_r
        # Only include V_r if using network mode
        if self.config.v_r_use_network:
            networks['v_r'] = self.networks.v_r
        for name, net in networks.items():
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm += p.data.norm(2).item() ** 2
            norms[name] = total_norm ** 0.5
        return norms
    
    def _compute_grad_norms(self) -> Dict[str, float]:
        """Compute L2 norms of gradients for monitoring."""
        norms = {}
        networks = {
            'q_r': self.networks.q_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
        }
        # Only include U_r if using network mode
        if self.config.u_r_use_network:
            networks['u_r'] = self.networks.u_r
        # Only include V_r if using network mode
        if self.config.v_r_use_network:
            networks['v_r'] = self.networks.v_r
        for name, net in networks.items():
            total_norm = 0.0
            for p in net.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            norms[name] = total_norm ** 0.5
        return norms
    
    def update_target_networks(self):
        """Update target networks (hard copy) at their individual intervals."""
        step = self.training_step_count
        
        # Update each target network at its own interval
        if step % self.config.q_r_target_update_interval == 0:
            self.networks.q_r_target.load_state_dict(self.networks.q_r.state_dict())
            self.networks.q_r_target.eval()
        
        if step % self.config.v_h_e_target_update_interval == 0:
            self.networks.v_h_e_target.load_state_dict(self.networks.v_h_e.state_dict())
            self.networks.v_h_e_target.eval()
        
        if step % self.config.x_h_target_update_interval == 0:
            self.networks.x_h_target.load_state_dict(self.networks.x_h.state_dict())
            self.networks.x_h_target.eval()
        
        if self.config.u_r_use_network and self.networks.u_r_target is not None:
            if step % self.config.u_r_target_update_interval == 0:
                self.networks.u_r_target.load_state_dict(self.networks.u_r.state_dict())
                self.networks.u_r_target.eval()
        
        if self.config.v_r_use_network and self.networks.v_r_target is not None:
            if step % self.config.v_r_target_update_interval == 0:
                self.networks.v_r_target.load_state_dict(self.networks.v_r.state_dict())
                self.networks.v_r_target.eval()
    
    # NOTE: tensorize_state commented out - never called anywhere in codebase
    # @abstractmethod
    # def tensorize_state(self, state: Any) -> Dict[str, torch.Tensor]:
    #     """
    #     Convert raw state to tensors (preprocessing, NOT neural network encoding).
    #     
    #     Args:
    #         state: Raw environment state.
    #     
    #     Returns:
    #         Dict of input tensors.
    #     """
    #     pass
    
    def _update_num_agents(self):
        """Update num_agents from current human_agent_indices and robot_agent_indices."""
        all_indices = self.human_agent_indices + self.robot_agent_indices
        self.num_agents = max(all_indices) + 1 if all_indices else 0
    
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
        
        Automatically updates human_agent_indices and robot_agent_indices
        by calling the env's get_human_agent_indices() and get_robot_agent_indices()
        methods (important for ensemble mode where agent positions/indices can vary).
        """
        # Create env from factory
        self.env = self.world_model_factory.create()
        
        # Update agent indices from the new environment using WorldModel API
        if hasattr(self.env, 'get_human_agent_indices') and hasattr(self.env, 'get_robot_agent_indices'):
            self.human_agent_indices = self.env.get_human_agent_indices()
            self.robot_agent_indices = self.env.get_robot_agent_indices()
            self._update_num_agents()
        
        # Update goal_sampler and human_policy_prior with new env
        if hasattr(self.goal_sampler, 'set_world_model'):
            self.goal_sampler.set_world_model(self.env)
        if hasattr(self.human_policy_prior, 'set_world_model'):
            self.human_policy_prior.set_world_model(self.env)
    
    def check_goal_achieved(self, state: Any, human_idx: int, goal: Any) -> bool:
        """
        Check if a human's goal is achieved in the given state.
        
        Uses the standard PossibleGoal API: goal.is_achieved(state).
        
        Args:
            state: Current environment state.
            human_idx: Index of the human agent (unused, goals check state directly).
            goal: The goal to check (must have is_achieved method).
        
        Returns:
            True if goal is achieved, False otherwise.
        """
        return goal.is_achieved(state)
    
    def step_environment(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        human_actions: List[int]
    ) -> Any:
        """
        Execute actions in the environment.
        
        Builds a full action list for all agents based on human_agent_indices
        and robot_agent_indices, then steps the environment.
        
        Args:
            state: Current state.
            robot_action: Tuple of robot actions.
            human_actions: List of human actions.
        
        Returns:
            Next state.
        """
        # Build action list for all agents
        actions = [0] * self.num_agents  # Default to idle for any gaps
        
        for i, human_idx in enumerate(self.human_agent_indices):
            actions[human_idx] = human_actions[i]
        
        for i, robot_idx in enumerate(self.robot_agent_indices):
            actions[robot_idx] = robot_action[i]
        
        # Step environment and get new state (standard WorldModel API)
        self.env.step(actions)
        return self.env.get_state()
    
    def reset_environment(self) -> Any:
        """
        Reset the environment to initial state.
        
        If a world_model_factory is provided, uses it to create/reset the environment.
        This supports both cached mode (same env reused) and ensemble mode (new env
        each episode).
        
        Also calls reset() on the robot_exploration_policy and human_exploration_policy
        if they are policy objects with reset methods.
        
        Returns:
            Initial state.
        """
        from .robot_policy import RobotPolicy
        from empo.human_policy_prior import HumanPolicyPrior
        
        if self.world_model_factory is not None:
            self._create_env_from_factory()
        elif self.env is None:
            raise RuntimeError(
                "No world_model_factory provided and env is None. "
                "For async training, you must pass a world_model_factory."
            )
        
        self.env.reset()
        
        # Reset exploration policies if they need world model context
        if isinstance(self.robot_exploration_policy, RobotPolicy):
            self.robot_exploration_policy.reset(self.env)
        if isinstance(self.human_exploration_policy, HumanPolicyPrior):
            self.human_exploration_policy.set_world_model(self.env)
        
        return self.env.get_state()
    
    def sample_robot_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action using policy with epsilon-greedy exploration.
        
        Uses q_r_target (frozen copy) for stable action sampling, consistent
        with async mode where actors use a periodically-synced copy.
        
        During warm-up, uses effective beta_r = 0 (uniform random policy).
        After warm-up, beta_r ramps up to nominal value.
        
        With probability epsilon, samples from the exploration policy
        (uniform random by default, or custom if robot_exploration_policy is set).
        
        When RND is enabled, curiosity bonus can be added to exploration:
        - During epsilon exploration: bias toward actions leading to novel states
        - During policy-based selection: add curiosity bonus to Q-values
        
        Args:
            state: Current state.
        
        Returns:
            Tuple of robot actions.
        """
        from .robot_policy import RobotPolicy
        
        epsilon = self.config.get_epsilon_r(self.training_step_count)
        effective_beta_r = self.config.get_effective_beta_r(self.training_step_count)
        
        # Epsilon exploration: sample from exploration policy (with optional curiosity bias)
        if torch.rand(1).item() < epsilon:
            return self._sample_robot_exploration_action(state)
        
        # Otherwise: sample from learned policy (with optional curiosity bonus)
        with torch.no_grad():
            q_values = self.networks.q_r_target.forward(
                state, None, self.device
            )
            
            # Add curiosity bonus to Q-values if RND is enabled
            if self.config.use_rnd and self.networks.rnd is not None and self.config.rnd_bonus_coef_r > 0:
                q_values = self._add_curiosity_bonus_to_q_values(state, q_values)
            
            return self.networks.q_r_target.sample_action(
                q_values, beta_r=effective_beta_r
            )
    
    def _sample_robot_exploration_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action during epsilon exploration.
        
        When RND is enabled, uses curiosity-weighted sampling instead of uniform random.
        Otherwise falls back to configured exploration policy.
        
        Args:
            state: Current state.
            
        Returns:
            Tuple of robot actions.
        """
        from .robot_policy import RobotPolicy
        
        num_action_combinations = self.networks.q_r_target.num_action_combinations
        
        # If RND is enabled, use curiosity-weighted exploration
        if self.config.use_rnd and self.networks.rnd is not None and self.config.rnd_bonus_coef_r > 0:
            return self._sample_curiosity_exploration_action(state)
        
        # Otherwise use configured exploration policy
        if self.robot_exploration_policy is None:
            # Uniform random exploration
            flat_idx = torch.randint(0, num_action_combinations, (1,)).item()
            return self.networks.q_r_target.action_index_to_tuple(flat_idx)
        elif isinstance(self.robot_exploration_policy, RobotPolicy):
            # RobotPolicy object: use its sample() method directly
            action = self.robot_exploration_policy.sample(state)
            # Ensure return type is tuple
            if isinstance(action, (list, tuple)):
                return tuple(action)
            else:
                return (action,)
        elif callable(self.robot_exploration_policy):
            # Callable exploration policy: get probabilities from function
            probs = self.robot_exploration_policy(state, self.env)
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            probs_tensor = probs_tensor / probs_tensor.sum()  # Normalize
            flat_idx = torch.multinomial(probs_tensor, 1).item()
            return self.networks.q_r_target.action_index_to_tuple(flat_idx)
        else:
            # List/array exploration policy: use fixed probabilities
            probs_tensor = torch.tensor(self.robot_exploration_policy, dtype=torch.float32)
            probs_tensor = probs_tensor / probs_tensor.sum()  # Normalize
            flat_idx = torch.multinomial(probs_tensor, 1).item()
            return self.networks.q_r_target.action_index_to_tuple(flat_idx)
    
    def _sample_curiosity_exploration_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action weighted by curiosity bonus for successor states.
        
        Uses transition_probabilities to compute expected novelty for each action,
        then samples proportionally to novelty.
        
        Args:
            state: Current state.
            
        Returns:
            Tuple of robot actions.
        """
        num_action_combinations = self.networks.q_r_target.num_action_combinations
        
        # Get current human actions (sample from exploration or prior)
        # For simplicity, use uniform random human actions during curiosity exploration
        num_human_actions = self.env.action_space.n
        human_actions = tuple(
            torch.randint(0, num_human_actions, (1,)).item()
            for _ in self.human_agent_indices
        )
        
        # Compute expected novelty for each robot action
        novelty_scores = []
        
        for flat_idx in range(num_action_combinations):
            robot_action = self.networks.q_r_target.action_index_to_tuple(flat_idx)
            
            # Build full action tuple: robots first, then humans
            actions = robot_action + human_actions
            
            # Get transition probabilities
            try:
                transitions = self.env.transition_probabilities(state, actions)
            except Exception:
                # If transition_probabilities fails, use uniform
                novelty_scores.append(1.0)
                continue
            
            # Handle None or empty transitions
            if transitions is None or len(transitions) == 0:
                novelty_scores.append(1.0)  # Unknown = maximally novel
                continue
            
            # Compute expected novelty over successor states
            expected_novelty = 0.0
            next_states = []
            probs = []
            for prob, next_state in transitions:
                next_states.append(next_state)
                probs.append(prob)
            
            if next_states:
                # Batch compute novelty for all successor states
                novelties = self.compute_novelty_for_states(next_states)
                for prob, novelty in zip(probs, novelties):
                    expected_novelty += prob * max(0.0, novelty.item())  # Clamp negative novelty
            
            novelty_scores.append(max(expected_novelty, 1e-6))  # Avoid zero
        
        # Convert to probabilities (softmax-like with temperature)
        novelty_tensor = torch.tensor(novelty_scores, dtype=torch.float32)
        # Add small uniform component to ensure exploration
        probs = novelty_tensor / novelty_tensor.sum()
        probs = 0.9 * probs + 0.1 / num_action_combinations  # 10% uniform
        probs = probs / probs.sum()
        
        flat_idx = torch.multinomial(probs, 1).item()
        return self.networks.q_r_target.action_index_to_tuple(flat_idx)
    
    def _add_curiosity_bonus_to_q_values(
        self,
        state: Any,
        q_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply curiosity bonus to Q-values for each action using multiplicative scaling.
        
        The bonus is based on the expected novelty of successor states.
        
        Since the power-law policy π_r(a) ∝ (-Q_r)^{-β_r} requires Q < 0,
        we cannot use additive bonuses that might make Q >= 0.
        Instead, we use multiplicative scaling:
        
            Q_effective = Q * exp(-bonus_coef * novelty)
        
        Since Q < 0 and exp(...) > 0, Q_effective remains negative.
        High novelty → smaller exp factor → Q_effective closer to 0 (better).
        This encourages exploration of novel states while preserving Q < 0.
        
        Args:
            state: Current state.
            q_values: Q-values tensor of shape (num_action_combinations,), all negative.
            
        Returns:
            Scaled Q-values with curiosity bonus applied, still all negative.
        """
        num_action_combinations = self.networks.q_r_target.num_action_combinations
        
        # For efficiency during training, sample human actions
        num_human_actions = self.env.action_space.n
        human_actions = tuple(
            torch.randint(0, num_human_actions, (1,)).item()
            for _ in self.human_agent_indices
        )
        
        # Compute expected novelty for each action
        novelties = torch.zeros(num_action_combinations, device=self.device)
        
        for flat_idx in range(num_action_combinations):
            robot_action = self.networks.q_r_target.action_index_to_tuple(flat_idx)
            actions = robot_action + human_actions
            
            try:
                transitions = self.env.transition_probabilities(state, actions)
            except Exception:
                continue
            
            if transitions is None or len(transitions) == 0:
                continue
            
            # Expected novelty over successor states
            expected_novelty = 0.0
            next_states = [ns for _, ns in transitions]
            probs = [p for p, _ in transitions]
            
            if next_states:
                state_novelties = self.compute_novelty_for_states(next_states)
                for prob, novelty in zip(probs, state_novelties):
                    expected_novelty += prob * max(0.0, novelty.item())  # Clamp negative
            
            novelties[flat_idx] = expected_novelty
        
        # Multiplicative scaling: Q_eff = Q * exp(-bonus_coef * novelty)
        # High novelty → smaller scale factor → Q closer to 0 (better for power-law)
        scale_factors = torch.exp(-self.config.rnd_bonus_coef_r * novelties)
        return q_values * scale_factors
    
    def sample_human_actions(
        self,
        state: Any,
        goals: Dict[int, Any]
    ) -> List[int]:
        """
        Sample human actions from the human policy prior with epsilon-greedy exploration.
        
        With probability epsilon_h, samples from the exploration policy
        (uniform random by default, or custom if human_exploration_policy is set).
        
        Args:
            state: Current state.
            goals: Dict mapping human index to their goal.
        
        Returns:
            List of human actions.
        """
        from empo.human_policy_prior import HumanPolicyPrior
        
        epsilon_h = self.config.get_epsilon_h(self.training_step_count)
        
        actions = []
        for h in self.human_agent_indices:
            goal = goals.get(h)
            
            # Epsilon exploration: sample from exploration policy
            if torch.rand(1).item() < epsilon_h:
                if self.human_exploration_policy is None:
                    # Uniform random exploration
                    num_actions = self.env.action_space.n
                    action = torch.randint(0, num_actions, (1,)).item()
                elif isinstance(self.human_exploration_policy, HumanPolicyPrior):
                    # HumanPolicyPrior object: use its sample() method
                    action = self.human_exploration_policy.sample(state, h, goal)
                else:
                    # Fallback: use uniform random
                    num_actions = self.env.action_space.n
                    action = torch.randint(0, num_actions, (1,)).item()
            else:
                # Use the human policy prior
                action = self.human_policy_prior.sample(state, h, goal)
            
            actions.append(action)
        return actions
    
    def collect_transition(
        self,
        state: Any,
        goals: Dict[int, Any],
        goal_weights: Dict[int, float],
        terminal: bool = False
    ) -> Tuple[Phase2Transition, Any]:
        """
        Collect one transition from the environment.
        
        Args:
            state: Current state.
            goals: Current goal assignments.
            goal_weights: Weights for each goal (from goal sampler).
            terminal: Whether this transition ends the episode (after this step,
                the environment will be reset). When True, the V_h^e TD target
                should not bootstrap from next_state.
        
        Returns:
            Tuple of (transition, next_state).
        """
        if self.debug:
            print(f"[DEBUG] collect_transition: sampling robot action...")
        
        # Sample actions
        robot_action = self.sample_robot_action(state)
        
        if self.debug:
            print(f"[DEBUG] collect_transition: robot_action={robot_action}, sampling human actions...")
        
        human_actions = self.sample_human_actions(state, goals)
        
        if self.debug:
            print(f"[DEBUG] collect_transition: human_actions={human_actions}, stepping environment...")
        
        # Step environment
        next_state = self.step_environment(state, robot_action, human_actions)
        
        if self.debug:
            print(f"[DEBUG] collect_transition: environment stepped, creating transition...")
        
        # Pre-compute transition probabilities for all robot actions (for model-based targets)
        # OPTIMIZATION: Only cache when Q_r is active - during early warmup we use
        # the cheaper V_h^e-only targets that don't need all action combinations.
        transition_probs_by_action = None
        q_r_active = 'q_r' in self.config.get_active_networks(self.training_step_count)
        if self.config.use_model_based_targets and q_r_active and hasattr(self.env, 'transition_probabilities'):
            transition_probs_by_action = self._precompute_transition_probs(
                state, human_actions
            )
        
        # Create transition
        transition = Phase2Transition(
            state=state,
            robot_action=robot_action,
            goals=goals.copy(),
            goal_weights=goal_weights.copy(),
            human_actions=human_actions,
            next_state=next_state,
            transition_probs_by_action=transition_probs_by_action,
            terminal=terminal
        )
        
        if self.debug:
            print(f"[DEBUG] collect_transition: done")
        
        return transition, next_state
    
    def _precompute_transition_probs(
        self,
        state: Any,
        human_actions: List[int]
    ) -> Dict[int, List[Tuple[float, Any]]]:
        """
        Pre-compute transition probabilities for all robot actions.
        
        This is called once at transition collection time to cache
        results for efficient reuse during training.
        
        Args:
            state: Current state.
            human_actions: The actual human actions taken.
        
        Returns:
            Dict mapping robot_action_index -> [(prob, next_state), ...].
        """
        num_actions = self.networks.q_r.num_action_combinations
        result = {}
        
        for action_idx in range(num_actions):
            robot_action = self.networks.q_r.action_index_to_tuple(action_idx)
            
            # Build full action vector using ACTUAL human actions
            actions = [0] * self.num_agents  # Default to idle for any gaps
            
            for i, human_idx in enumerate(self.human_agent_indices):
                actions[human_idx] = human_actions[i]
            
            for i, robot_idx in enumerate(self.robot_agent_indices):
                actions[robot_idx] = robot_action[i]
            
            # Get transition probabilities
            trans_probs = self.env.transition_probabilities(state, actions)
            
            if trans_probs is None:
                result[action_idx] = []  # Terminal state
            else:
                result[action_idx] = trans_probs
        
        return result
    
    def _compute_u_r_for_state(self, state: Any) -> torch.Tensor:
        """
        Compute U_r for a state, either from network or directly from X_h.
        
        U_r(s) = -(E_h[X_h(s)^{-ξ}])^η
        
        When u_r_use_network=True, uses the U_r network.
        When u_r_use_network=False, computes directly from X_h values.
        
        Args:
            state: Environment state.
            
        Returns:
            U_r value as a tensor.
        """
        if self.config.u_r_use_network:
            # Use U_r network (or target network depending on context)
            _, u_r = self.networks.u_r.forward(state, None, self.device)
            return u_r
        else:
            # Compute directly from X_h values
            x_h_values = []
            for h in self.human_agent_indices:
                x_h = self.networks.x_h.forward(state, None, h, self.device)
                # Clamp X_h to (0, 1] to prevent explosion when X_h is near 0
                x_h_clamped = torch.clamp(x_h.squeeze(), min=1e-3, max=1.0)
                x_h_values.append(x_h_clamped)
            
            # Stack and compute U_r using the formula:
            # y = E_h[X_h^{-ξ}], U_r = -y^η
            x_h_tensor = torch.stack(x_h_values)  # (num_humans,)
            x_h_powered = x_h_tensor ** (-self.config.xi)
            y = x_h_powered.mean()
            u_r = -(y ** self.config.eta)
            return u_r.unsqueeze(0)  # Return with batch dim
    
    def _compute_u_r_for_state_target(self, state: Any) -> torch.Tensor:
        """
        Compute U_r for a state using TARGET networks, for stable target computation.
        
        U_r(s) = -(E_h[X_h(s)^{-ξ}])^η
        
        When u_r_use_network=True, uses the U_r TARGET network.
        When u_r_use_network=False, computes directly from X_h TARGET values.
        
        Args:
            state: Environment state.
            
        Returns:
            U_r value as a tensor.
        """
        if self.config.u_r_use_network:
            # Use U_r TARGET network
            _, u_r = self.networks.u_r_target.forward(state, None, self.device)
            return u_r
        else:
            # Compute directly from X_h TARGET values
            x_h_values = []
            for h in self.human_agent_indices:
                x_h = self.networks.x_h_target.forward(state, None, h, self.device)
                # Clamp X_h to (0, 1] to prevent explosion when X_h is near 0
                x_h_clamped = torch.clamp(x_h.squeeze(), min=1e-3, max=1.0)
                x_h_values.append(x_h_clamped)
            
            # Stack and compute U_r using the formula:
            # y = E_h[X_h^{-ξ}], U_r = -y^η
            x_h_tensor = torch.stack(x_h_values)  # (num_humans,)
            x_h_powered = x_h_tensor ** (-self.config.xi)
            y = x_h_powered.mean()
            u_r = -(y ** self.config.eta)
            return u_r.unsqueeze(0)  # Return with batch dim
    
    def compute_losses(
        self,
        batch: List[Phase2Transition],
        x_h_batch: Optional[List[Phase2Transition]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        """
        Compute losses for all networks using batched forward passes.
        
        Uses forward_batch methods on all networks for efficient batched
        computation. Works for both neural networks and lookup tables.
        
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
        
        # Collect V_h^e data: (state_idx, state, next_state, human_idx, goal, terminal)
        v_h_e_indices = []  # which transition this came from
        v_h_e_states = []
        v_h_e_next_states = []
        v_h_e_human_indices = []
        v_h_e_goals = []
        v_h_e_achieved = []
        v_h_e_terminal = []  # whether transition is terminal (episode ended)
        
        for i, t in enumerate(batch):
            for h, g_h in t.goals.items():
                v_h_e_indices.append(i)
                v_h_e_states.append(t.state)
                v_h_e_next_states.append(t.next_state)
                v_h_e_human_indices.append(h)
                v_h_e_goals.append(g_h)
                achieved = self.check_goal_achieved(t.next_state, h, g_h)
                v_h_e_achieved.append(achieved)
                v_h_e_terminal.append(t.terminal)
                
                # Debug logging for goal achievement
                if self.debug and achieved:
                    step_count, agent_states, _, _ = t.next_state
                    agent_pos = (int(agent_states[h][0]), int(agent_states[h][1]))
                    goal_target = getattr(g_h, 'target_pos', 'unknown')
                    print(f"[DEBUG] Goal achieved! human={h}, target={goal_target}, "
                          f"agent_pos={agent_pos}, terminal={t.terminal}, goal_hash={hash(g_h)}")
        
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
            
            # Terminal mask: when True, continuation value is 0
            terminal_t = torch.tensor(
                [1.0 if t else 0.0 for t in v_h_e_terminal],
                device=self.device
            )
            
            with torch.no_grad():
                v_h_e_next = self.networks.v_h_e_target.forward_batch(
                    v_h_e_next_states, v_h_e_goals, v_h_e_human_indices,
                    self.env, self.device
                )
            
            # TD target: achieved + (1 - achieved) * (1 - terminal) * gamma_h * V_h^e_next
            target_v_h_e = self.networks.v_h_e.compute_td_target(
                goal_achieved_t, v_h_e_next.squeeze(), terminal=terminal_t
            )
            
            # Debug logging for TD targets
            if self.debug and self.training_step_count % 100 == 0:
                n_achieved = goal_achieved_t.sum().item()
                n_terminal = terminal_t.sum().item()
                n_total = len(goal_achieved_t)
                # Find unique goals and their target values
                goal_targets = {}
                for idx, g in enumerate(v_h_e_goals):
                    goal_key = getattr(g, 'target_pos', hash(g))
                    if goal_key not in goal_targets:
                        goal_targets[goal_key] = {'targets': [], 'achieved': [], 'terminal': [], 'next_v': []}
                    goal_targets[goal_key]['targets'].append(target_v_h_e[idx].item())
                    goal_targets[goal_key]['achieved'].append(goal_achieved_t[idx].item())
                    goal_targets[goal_key]['terminal'].append(terminal_t[idx].item())
                    goal_targets[goal_key]['next_v'].append(v_h_e_next[idx].item())
                print(f"[DEBUG V_h^e] step={self.training_step_count}, achieved={n_achieved}/{n_total}, "
                      f"terminal={n_terminal}/{n_total}, target_mean={target_v_h_e.mean().item():.4f}")
                for goal_key, data in sorted(goal_targets.items()):
                    n = len(data['targets'])
                    mean_target = sum(data['targets']) / n
                    mean_next_v = sum(data['next_v']) / n
                    n_ach = sum(data['achieved'])
                    n_term = sum(data['terminal'])
                    print(f"[DEBUG V_h^e]   Goal {goal_key}: n={n}, achieved={n_ach}, terminal={n_term}, "
                          f"mean_target={mean_target:.4f}, mean_next_v={mean_next_v:.4f}")
            
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
        
        # ----- RND loss (batched, for curiosity-driven exploration) -----
        if self.config.use_rnd and self.networks.rnd is not None:
            # Compute RND loss on all states in batch
            # Need to get state features from the encoder
            rnd_loss = self._compute_rnd_loss_batch(states)
            losses['rnd'] = rnd_loss
            
            with torch.no_grad():
                rnd_stats = self.networks.rnd.get_statistics()
                prediction_stats['rnd'] = {
                    'loss': rnd_loss.item(),
                    'running_mean': rnd_stats['rnd_running_mean'],
                    'running_std': rnd_stats['rnd_running_std'],
                    # Raw novelty values before normalization (more informative)
                    'batch_raw_mean': rnd_stats['rnd_batch_raw_mean'],
                    'batch_raw_std': rnd_stats['rnd_batch_raw_std'],
                }
        
        # Clear caches after each compute_losses call (for neural network encoders)
        if hasattr(self, 'clear_caches'):
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
    
    def _compute_rnd_loss_batch(self, states: List[Any]) -> torch.Tensor:
        """
        Compute RND loss for a batch of states.
        
        Uses the abstract method get_state_features_for_rnd() which must be
        implemented by environment-specific subclasses.
        
        Args:
            states: List of states.
            
        Returns:
            Scalar loss tensor.
        """
        # Get state features for RND computation
        features = self.get_state_features_for_rnd(states)
        
        # Compute RND loss
        return self.networks.rnd.compute_loss(features)
    
    @abstractmethod
    def get_state_features_for_rnd(self, states: List[Any]) -> torch.Tensor:
        """
        Get state features for RND novelty computation.
        
        Must be implemented by environment-specific subclasses to convert
        states to feature tensors suitable for RND.
        
        For neural network trainers, this typically uses the shared state
        encoder (detached) to produce features.
        
        For lookup table trainers, this might use a simple state hash or
        one-hot encoding.
        
        Args:
            states: List of states.
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim).
        """
        pass
    
    def compute_novelty_for_states(self, states: List[Any]) -> torch.Tensor:
        """
        Compute novelty scores for a list of states.
        
        This is used during action selection to compute curiosity bonuses.
        
        Args:
            states: List of states.
            
        Returns:
            Novelty scores tensor of shape (len(states),).
        """
        if not self.config.use_rnd or self.networks.rnd is None:
            return torch.zeros(len(states), device=self.device)
        
        features = self.get_state_features_for_rnd(states)
        return self.networks.rnd.compute_novelty_no_grad(features)
    
    def training_step(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Perform one training step (sample batch, compute losses, update).
        
        Returns:
            Tuple of (loss_values dict, grad_norms dict, prediction_stats dict).
        """
        # Determine X_h batch size (can be larger than regular batch)
        x_h_batch_size = self.config.x_h_batch_size or self.config.batch_size
        min_required = max(self.config.batch_size, x_h_batch_size)
        
        if len(self.replay_buffer) < min_required:
            return {}, {}, {}
        
        # Sample batch for most networks
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Sample potentially larger batch for X_h if configured
        if x_h_batch_size > self.config.batch_size:
            x_h_batch = self.replay_buffer.sample(x_h_batch_size)
        else:
            x_h_batch = batch
        
        # Compute losses (with separate X_h batch)
        losses, prediction_stats = self.compute_losses(batch, x_h_batch)
        
        # Determine which networks are active in this warm-up stage
        active_networks = self.config.get_active_networks(self.training_step_count)
        
        # Combine all losses and do a SINGLE backward pass to avoid
        # in-place modification conflicts when losses share tensors through
        # the shared encoder's outputs
        loss_values = {}
        grad_norms = {}
        
        # Collect trainable losses (only for active networks)
        trainable_losses = []
        for name, loss in losses.items():
            loss_values[name] = loss.item()
            # Only train if network is active in current warm-up stage
            if name in self.optimizers and loss.requires_grad and name in active_networks:
                trainable_losses.append((name, loss))
        
        # Map network names to their network objects
        network_map = {
            'q_r': self.networks.q_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
        }
        if self.config.u_r_use_network:
            network_map['u_r'] = self.networks.u_r
        if self.config.v_r_use_network:
            network_map['v_r'] = self.networks.v_r
        if self.config.use_rnd and self.networks.rnd is not None:
            network_map['rnd'] = self.networks.rnd.predictor  # Only predictor is trained
        
        # Train each network INDEPENDENTLY with its own backward pass
        # This is critical because networks use detached encoder outputs
        # to prevent gradient interference between networks
        #
        # IMPORTANT: We must do ALL backward passes BEFORE any optimizer.step()
        # because step() modifies weights in-place, which would invalidate
        # the computation graph for subsequent backward passes.
        
        # Phase 1: Zero gradients and compute all backward passes
        for name, loss in trainable_losses:
            self.optimizers[name].zero_grad()
        
        for i, (name, loss) in enumerate(trainable_losses):
            # Use retain_graph=True for all but the last backward pass
            retain = (i < len(trainable_losses) - 1)
            loss.backward(retain_graph=retain)
        
        # Phase 2: Apply gradient clipping and optimizer steps
        for name, loss in trainable_losses:
            # Update learning rate using the new warm-up-aware schedule
            self.update_counts[name] += 1
            new_lr = self.config.get_learning_rate(
                name, self.training_step_count, self.update_counts[name]
            )
            for param_group in self.optimizers[name].param_groups:
                param_group['lr'] = new_lr
            
            # Apply gradient clipping (optionally scaled by learning rate)
            if name in network_map:
                net = network_map[name]
                clip_val = self.config.get_effective_grad_clip(name, new_lr)
                if clip_val and clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
            
            grad_norms[name] = self._compute_single_grad_norm(name)
            self.optimizers[name].step()
        
        # Update target networks (each at its own interval)
        self.update_target_networks()
        
        # Add any new lookup table parameters to optimizers
        self._add_new_lookup_params_to_optimizers()
        
        return loss_values, grad_norms, prediction_stats
    
    def _compute_single_grad_norm(self, network_name: str) -> float:
        """Compute gradient L2 norm for a single network."""
        networks = {
            'q_r': self.networks.q_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
        }
        if self.config.u_r_use_network:
            networks['u_r'] = self.networks.u_r
        # Only include V_r if using network mode
        if self.config.v_r_use_network:
            networks['v_r'] = self.networks.v_r
        if network_name not in networks:
            return 0.0
        net = networks[network_name]
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    # ==================== Shared Actor/Learner Logic ====================
    
    class _ActorState:
        """Mutable state for actor (environment interaction)."""
        def __init__(self, state, goals, goal_weights, env_step_count: int = 0):
            self.state = state
            self.goals = goals
            self.goal_weights = goal_weights
            self.env_step_count = env_step_count  # Steps since last env reset
    
    def _sample_goals(self, state) -> Tuple[Dict[int, Any], Dict[int, float]]:
        """Sample goals for all humans using the goal sampler.
        
        Returns:
            Tuple of (goals dict, goal_weights dict).
        """
        goals = {}
        goal_weights = {}
        for h in self.human_agent_indices:
            goal, weight = self.goal_sampler.sample(state, h)
            goals[h] = goal
            goal_weights[h] = weight
        return goals, goal_weights
    
    def _init_actor_state(self) -> "_ActorState":
        """Initialize actor state with fresh environment."""
        state = self.reset_environment()
        goals, goal_weights = self._sample_goals(state)
        return BasePhase2Trainer._ActorState(state, goals, goal_weights, 0)
    
    def _actor_step(self, actor_state: "_ActorState") -> Optional[Phase2Transition]:
        """
        Collect one transition from the environment.
        
        This is the shared actor logic used by both sync and async training.
        Updates actor_state in place (state, goals, env_step_count).
        Handles environment reset when steps_per_episode is reached.
        
        Args:
            actor_state: Mutable actor state to update.
            
        Returns:
            The collected transition, or None if collection failed.
        """
        # Check if this will be the last step of the episode BEFORE collecting
        # This is needed so the transition can be marked as terminal
        is_terminal = (actor_state.env_step_count + 1) >= self.config.steps_per_episode
        
        if self.debug and is_terminal:
            print(f"[DEBUG] Terminal transition! env_step={actor_state.env_step_count}, "
                  f"steps_per_episode={self.config.steps_per_episode}")
        
        # Collect one transition
        transition, next_state = self.collect_transition(
            actor_state.state, actor_state.goals, actor_state.goal_weights,
            terminal=is_terminal
        )
        
        # Check if transition failed (environment ended or error)
        if transition is None:
            # Reset environment and return None
            actor_state.state = self.reset_environment()
            actor_state.goals, actor_state.goal_weights = self._sample_goals(actor_state.state)
            actor_state.env_step_count = 0
            return None
        
        # Update actor state
        actor_state.state = next_state
        actor_state.env_step_count += 1
        
        # Check if any goal was achieved - if so, resample that goal
        # This prevents the agent from repeatedly seeing achieved=1 for the same goal
        goals_to_resample = []
        for h, g in actor_state.goals.items():
            if self.check_goal_achieved(next_state, h, g):
                goals_to_resample.append(h)
        
        if goals_to_resample:
            # Resample only the achieved goals
            new_goals, new_weights = self._sample_goals(next_state)
            for h in goals_to_resample:
                if h in new_goals:
                    actor_state.goals[h] = new_goals[h]
                    actor_state.goal_weights[h] = new_weights[h]
        # Also resample goals with some probability (exploration)
        elif random.random() < self.config.goal_resample_prob:
            actor_state.goals, actor_state.goal_weights = self._sample_goals(actor_state.state)
        
        # Reset environment periodically
        if actor_state.env_step_count >= self.config.steps_per_episode:
            actor_state.state = self.reset_environment()
            actor_state.goals, actor_state.goal_weights = self._sample_goals(actor_state.state)
            actor_state.env_step_count = 0
        
        return transition
    
    class _LearnerState:
        """Mutable state for learner (warmup tracking and progress metrics)."""
        def __init__(self, prev_stage: int, prev_stage_name: str, 
                     prev_param_norms: Optional[Dict[str, float]] = None):
            self.prev_stage = prev_stage
            self.prev_stage_name = prev_stage_name
            # For tracking network parameter changes
            self.prev_param_norms = prev_param_norms or {}
            # For tracking average time per step
            self.start_time: Optional[float] = None
            self.start_step: int = 0
    
    def _init_learner_state(self) -> "_LearnerState":
        """Initialize learner state for warmup tracking."""
        import time
        prev_stage = self.config.get_warmup_stage(self.training_step_count)
        prev_stage_name = self.config.get_warmup_stage_name(self.training_step_count)
        prev_param_norms = self._compute_param_norms()
        state = BasePhase2Trainer._LearnerState(prev_stage, prev_stage_name, prev_param_norms)
        state.start_time = time.time()
        state.start_step = self.training_step_count
        return state
    
    def _learner_step(self, learner_state: "_LearnerState", pbar: Optional[tqdm] = None) -> Dict[str, float]:
        """
        Perform one training step with all logging and warmup handling.
        
        This is the shared learner logic used by both sync and async training.
        Updates learner_state in place for warmup stage tracking.
        
        Args:
            learner_state: Mutable learner state to update.
            pbar: Optional progress bar to update.
            
        Returns:
            Dict of losses from the training step.
        """
        import time
        
        # Do training step
        losses, grad_norms, pred_stats = self.training_step()
        
        # Increment training step counter (this is the gradient update counter)
        self.training_step_count += 1
        
        # Compute current parameter norms to track network changes
        current_param_norms = self._compute_param_norms()
        
        # Update progress bar with meaningful metrics
        if pbar is not None:
            pbar.update(1)
            
            # Compute average seconds per step
            elapsed = time.time() - learner_state.start_time
            steps_done = self.training_step_count - learner_state.start_step
            avg_sec_per_step = elapsed / max(1, steps_done)
            
            # Compute parameter change magnitudes for x_h and q_r
            x_h_change = abs(current_param_norms.get('x_h', 0) - learner_state.prev_param_norms.get('x_h', 0))
            q_r_change = abs(current_param_norms.get('q_r', 0) - learner_state.prev_param_norms.get('q_r', 0))
            
            # Build postfix dict with v_h_e loss and network changes
            postfix = {
                'v_h_e': f"{losses.get('v_h_e', 0):.4f}",
                'Δx_h': f"{x_h_change:.4f}",
                'Δq_r': f"{q_r_change:.4f}",
                's/step': f"{avg_sec_per_step:.2f}",
            }
            pbar.set_postfix(postfix, refresh=False)
            
            # Update prev_param_norms for next step
            learner_state.prev_param_norms = current_param_norms
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Progress/environment_steps', self.total_env_steps, self.training_step_count)
            
            for key, value in losses.items():
                if key == 'u_r' and not self.config.u_r_use_network:
                    continue
                self.writer.add_scalar(f'Loss/{key}', value, self.training_step_count)
            for key, value in grad_norms.items():
                if key == 'u_r' and not self.config.u_r_use_network:
                    continue
                self.writer.add_scalar(f'GradNorm/{key}', value, self.training_step_count)
            for key, stats in pred_stats.items():
                # RND stats have different keys (running_mean/std instead of mean/std)
                if key == 'rnd':
                    # Raw novelty (before normalization) - THIS is what you want to watch
                    # Should decrease over time as predictor learns to recognize states
                    if 'batch_raw_mean' in stats:
                        self.writer.add_scalar('Exploration/rnd_raw_novelty_mean', stats['batch_raw_mean'], self.training_step_count)
                    if 'batch_raw_std' in stats:
                        self.writer.add_scalar('Exploration/rnd_raw_novelty_std', stats['batch_raw_std'], self.training_step_count)
                    # Running normalization stats (less useful for debugging)
                    if 'running_mean' in stats:
                        self.writer.add_scalar('Exploration/rnd_norm_running_mean', stats['running_mean'], self.training_step_count)
                    if 'running_std' in stats:
                        self.writer.add_scalar('Exploration/rnd_norm_running_std', stats['running_std'], self.training_step_count)
                    continue
                self.writer.add_scalar(f'Predictions/{key}_mean', stats['mean'], self.training_step_count)
                self.writer.add_scalar(f'Predictions/{key}_std', stats['std'], self.training_step_count)
                if 'target_mean' in stats:
                    self.writer.add_scalar(f'Targets/{key}_mean', stats['target_mean'], self.training_step_count)
            
            # Log parameter norms
            param_norms = self._compute_param_norms()
            for key, value in param_norms.items():
                self.writer.add_scalar(f'ParamNorm/{key}', value, self.training_step_count)
            
            # Log exploration epsilons side by side
            self.writer.add_scalar('Exploration/epsilon_r', self.config.get_epsilon_r(self.training_step_count), self.training_step_count)
            self.writer.add_scalar('Exploration/epsilon_h', self.config.get_epsilon_h(self.training_step_count), self.training_step_count)
            
            # Log learning rates
            networks_to_log_lr = ['v_h_e', 'x_h', 'q_r']
            if self.config.u_r_use_network:
                networks_to_log_lr.append('u_r')
            if self.config.v_r_use_network:
                networks_to_log_lr.append('v_r')
            for net_name in networks_to_log_lr:
                lr = self.config.get_learning_rate(
                    net_name, self.training_step_count, self.update_counts.get(net_name, 0)
                )
                self.writer.add_scalar(f'LearningRate/{net_name}', lr, self.training_step_count)
            
            # Log warm-up phase information
            self.writer.add_scalar('Warmup/effective_beta_r', 
                                  self.config.get_effective_beta_r(self.training_step_count), self.training_step_count)
            self.writer.add_scalar('Warmup/is_warmup', 
                                  1.0 if self.config.is_in_warmup(self.training_step_count) else 0.0, self.training_step_count)
            active = self.config.get_active_networks(self.training_step_count)
            active_mask = sum(2**i for i, n in enumerate(['v_h_e', 'x_h', 'u_r', 'q_r', 'v_r']) if n in active)
            self.writer.add_scalar('Warmup/active_networks_mask', active_mask, self.training_step_count)
            self.writer.add_scalar('Warmup/stage', 
                                  self.config.get_warmup_stage(self.training_step_count), self.training_step_count)
            
            # Log encoder cache hit rates (if available)
            if hasattr(self, 'get_cache_stats'):
                cache_stats = self.get_cache_stats()
                for encoder_name, (hits, misses) in cache_stats.items():
                    total = hits + misses
                    if total > 0:
                        hit_rate = hits / total
                        self.writer.add_scalar(f'Cache/{encoder_name}_hit_rate', hit_rate, self.training_step_count)
                        self.writer.add_scalar(f'Cache/{encoder_name}_calls', total, self.training_step_count)
                # Reset stats after logging so we get per-step rates
                if hasattr(self, 'reset_cache_stats'):
                    self.reset_cache_stats()
            
            # Log lookup table sizes (if any lookup tables are in use)
            lookup_tables = get_all_lookup_tables(self.networks)
            if lookup_tables:
                total_entries = 0
                for name, net in lookup_tables.items():
                    size = len(net.table)
                    total_entries += size
                    self.writer.add_scalar(f'LookupTable/{name}_size', size, self.training_step_count)
                self.writer.add_scalar('LookupTable/total_entries', total_entries, self.training_step_count)
        
        # Check for warm-up stage transitions
        current_stage = self.config.get_warmup_stage(self.training_step_count)
        if current_stage != learner_state.prev_stage:
            current_stage_name = self.config.get_warmup_stage_name(self.training_step_count)
            active = self.config.get_active_networks(self.training_step_count)
            
            if self.verbose:
                print(f"\n[Warmup] Stage transition at training step {self.training_step_count}:")
                print(f"  {learner_state.prev_stage_name} -> {current_stage_name}")
                print(f"  Active networks: {active}")
                effective_beta = self.config.get_effective_beta_r(self.training_step_count)
                print(f"  Effective beta_r: {effective_beta:.4f}")
            
            if self.writer is not None:
                self.writer.add_scalar('Warmup/stage_transition', 1.0, self.training_step_count)
                self.writer.add_text('Warmup/transitions', 
                                    f"Step {self.training_step_count}: {learner_state.prev_stage_name} -> {current_stage_name}",
                                    global_step=self.training_step_count)
            
            # Clear replay buffer at start of beta_r ramp-up (transition to stage 4)
            if current_stage == 4 and learner_state.prev_stage == 3:
                buffer_size_before = len(self.replay_buffer)
                self.replay_buffer.clear()
                # In async mode, reset shared_env_steps so actors can resume production
                # (otherwise throttling keeps them paused since env_steps >> training_steps)
                if self._shared_env_steps is not None:
                    with self._shared_env_steps.get_lock():
                        self._shared_env_steps.value = 0
                    if self.verbose:
                        print(f"  [Async] Reset shared_env_steps to 0 to unthrottle actors")
                if self.verbose:
                    print(f"  [Training] Cleared replay buffer ({buffer_size_before} transitions) at start of β_r ramp-up")
                if self.writer is not None:
                    self.writer.add_text('Warmup/events', 
                                        f"Cleared replay buffer ({buffer_size_before} transitions) at start of β_r ramp-up",
                                        global_step=self.training_step_count)
            
            # Clear replay buffer after beta_r ramp-up is done (transition to stage 5)
            if current_stage == 5 and learner_state.prev_stage == 4:
                buffer_size_before = len(self.replay_buffer)
                self.replay_buffer.clear()
                # In async mode, reset shared_env_steps so actors can resume production
                if self._shared_env_steps is not None:
                    with self._shared_env_steps.get_lock():
                        self._shared_env_steps.value = 0
                    if self.verbose:
                        print(f"  [Async] Reset shared_env_steps to 0 to unthrottle actors")
                if self.verbose:
                    print(f"  [Training] Cleared replay buffer ({buffer_size_before} transitions) after β_r ramp-up")
                if self.writer is not None:
                    self.writer.add_text('Warmup/events', 
                                        f"Cleared replay buffer ({buffer_size_before} transitions) after β_r ramp-up",
                                        global_step=self.training_step_count)
            
            learner_state.prev_stage = current_stage
            learner_state.prev_stage_name = current_stage_name
        
        # Periodic logging
        if self.training_step_count % 100 == 0:
            if losses and pbar is not None:
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items() if v > 0)
                pbar.set_postfix_str(loss_str[:60])
            
            if self.debug:
                stage = self.config.get_warmup_stage_name(self.training_step_count)
                print(f"Step {self.training_step_count} ({stage}): {losses}")
            
            if self.writer is not None:
                self.writer.flush()
        
        # Step profiler if provided
        if self.profiler is not None:
            self.profiler.step()
        
        return losses

    # ==================== Main Training Entry Point ====================
    
    def train(self, num_training_steps: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Main training loop. Runs until the specified number of training steps.
        
        Args:
            num_training_steps: Number of training steps (gradient updates) to perform.
                               If None, uses self.config.num_training_steps.
        
        Returns:
            List of loss dicts (logged periodically).
        """
        if num_training_steps is None:
            num_training_steps = self.config.num_training_steps
        
        # Use async training if configured
        if self.config.async_training:
            return self._train_async(num_training_steps)
        
        history = []
        
        # Initialize states
        actor_state = self._init_actor_state()
        learner_state = self._init_learner_state()
        
        # Set main networks to train mode (enables dropout)
        self.networks.q_r.train()
        self.networks.v_h_e.train()
        self.networks.x_h.train()
        if self.networks.u_r is not None:
            self.networks.u_r.train()
        if self.networks.v_r is not None:
            self.networks.v_r.train()
        
        # Log initial stage
        if self.verbose:
            active = self.config.get_active_networks(self.training_step_count)
            print(f"[Warmup] Starting in stage {learner_state.prev_stage}: {learner_state.prev_stage_name} (active networks: {active})")
        
        # Log stage transition steps to TensorBoard at start
        if self.writer is not None:
            transition_steps = self.config.get_stage_transition_steps()
            for stage_num, (step, stage_name) in enumerate(transition_steps):
                self.writer.add_text('Warmup/stage_transitions', 
                                     f"Stage {stage_num}: {stage_name} starts at step {step}", 
                                     global_step=0)
        
        # Set up progress bar with proper formatting
        pbar = tqdm(
            total=num_training_steps, 
            desc="Training", 
            unit=" steps",  # Space before 'steps' for readability
            disable=not self.verbose,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]'
        )
        pbar.update(self.training_step_count)  # Start from current position if resuming
        
        while self.training_step_count < num_training_steps:
            # Actor: collect one transition
            transition = self._actor_step(actor_state)
            
            if transition is not None:
                # Push to replay buffer
                self.replay_buffer.push(
                    transition.state,
                    transition.robot_action,
                    transition.goals,
                    transition.goal_weights,
                    transition.human_actions,
                    transition.next_state,
                    transition.transition_probs_by_action,
                    terminal=transition.terminal
                )
                self.total_env_steps += 1
            
            # Learner: perform training updates
            for _ in range(int(self.config.training_steps_per_env_step)):
                if self.training_step_count >= num_training_steps:
                    break
                
                losses = self._learner_step(learner_state, pbar)
                
                # Log to history periodically
                if self.training_step_count % 100 == 0:
                    history.append(losses)
        
        pbar.close()
        
        # Set all networks to eval mode (disables dropout for rollouts)
        self.networks.q_r.eval()
        self.networks.v_h_e.eval()
        self.networks.x_h.eval()
        if self.networks.u_r is not None:
            self.networks.u_r.eval()
        if self.networks.v_r is not None:
            self.networks.v_r.eval()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return history

    # ==================== Async Training Methods ====================
    
    def _get_policy_state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict for policy networks needed by actors.
        
        Override this in subclasses if actors need different networks.
        
        Returns:
            Dict with network state dicts for action selection.
        """
        return {
            'q_r': self.networks.q_r.state_dict(),
        }
    
    def _load_policy_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load policy state dict into networks.
        
        Override this in subclasses if actors need different networks.
        
        Args:
            state_dict: Dict with network state dicts.
        """
        self.networks.q_r.load_state_dict(state_dict['q_r'])
    
    def _train_async(self, num_training_steps: int) -> List[Dict[str, float]]:
        """
        Async actor-learner training loop.
        
        Spawns multiple actor processes that collect transitions in parallel,
        while the main process runs the learner with GPU training.
        
        Args:
            num_training_steps: Total number of training steps (gradient updates) to perform.
            
        Returns:
            List of loss dicts (logged periodically from learner).
        """
        # Use 'spawn' context for CUDA compatibility
        ctx = mp.get_context('spawn')
        
        # Shared queue for transitions from actors to learner
        transition_queue = ctx.Queue(maxsize=self.config.async_queue_size)
        
        # Event to signal actors to stop
        stop_event = ctx.Event()
        
        # Shared counter for training steps (for epsilon/beta_r/warmup)
        # Actors read this to get correct exploration parameters
        shared_training_steps = ctx.Value('i', self.training_step_count)
        
        # Shared counter for environment steps (actors increment, learner reads for logging)
        shared_env_steps = ctx.Value('i', self.total_env_steps)

        # Manager for sharing policy weights
        manager = ctx.Manager()
        policy_lock = manager.Lock()
        # Use a shared dict to hold serialized policy state
        shared_policy = manager.dict()
        shared_policy['state_dict'] = self._serialize_policy_state()
        shared_policy['version'] = 0
        
        # Start actor processes
        actors = []
        for actor_id in range(self.config.num_actors):
            p = ctx.Process(
                target=self._actor_process_entry,
                args=(
                    actor_id,
                    transition_queue,
                    stop_event,
                    shared_training_steps,
                    shared_env_steps,
                    shared_policy,
                    policy_lock,
                ),
                daemon=True
            )
            p.start()
            actors.append(p)
        
        if self.verbose:
            print(f"[Async] Started {self.config.num_actors} actor processes")
        
        # Run learner loop in main process
        history = self._learner_loop(
            transition_queue=transition_queue,
            stop_event=stop_event,
            shared_training_steps=shared_training_steps,
            shared_env_steps=shared_env_steps,
            shared_policy=shared_policy,
            policy_lock=policy_lock,
            num_training_steps=num_training_steps,
        )
        
        # Signal actors to stop and wait for them
        stop_event.set()
        for p in actors:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        
        if self.verbose:
            print(f"[Async] Training complete. All actors stopped.")
        
        return history
    
    def _serialize_policy_state(self) -> bytes:
        """Serialize policy state dict to bytes for sharing."""
        import io
        state = self._get_policy_state_dict()
        # Move tensors to CPU for sharing, leave other types as-is
        def to_cpu(v):
            if isinstance(v, torch.Tensor):
                return v.cpu()
            return v
        cpu_state = {k: {kk: to_cpu(vv) for kk, vv in v.items()} for k, v in state.items()}
        buffer = io.BytesIO()
        torch.save(cpu_state, buffer)
        return buffer.getvalue()
    
    def _deserialize_policy_state(self, data: bytes) -> Dict[str, Any]:
        """Deserialize policy state dict from bytes."""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer, weights_only=False)
    
    def _actor_process_entry(
        self,
        actor_id: int,
        transition_queue: mp.Queue,
        stop_event: mp.Event,
        shared_training_steps: mp.Value,
        shared_env_steps: mp.Value,
        shared_policy: Dict,
        policy_lock,
    ) -> None:
        """
        Entry point for actor process. Runs actor loop until stop_event is set.
        
        This is needed because we can't pickle the full trainer object.
        Subclasses should override _create_actor_env() to provide the environment.
        """
        # Limit PyTorch threads to avoid CPU oversubscription
        # Each actor only needs 1 thread since it does lightweight inference
        torch.set_num_threads(1)
        
        try:
            self._actor_loop_async(
                actor_id=actor_id,
                transition_queue=transition_queue,
                stop_event=stop_event,
                shared_training_steps=shared_training_steps,
                shared_env_steps=shared_env_steps,
                shared_policy=shared_policy,
                policy_lock=policy_lock,
            )
        except Exception as e:
            print(f"[Actor {actor_id}] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _actor_loop_async(
        self,
        actor_id: int,
        transition_queue: mp.Queue,
        stop_event: mp.Event,
        shared_training_steps: mp.Value,
        shared_env_steps: mp.Value,
        shared_policy: Dict,
        policy_lock,
    ) -> None:
        """
        Async actor loop: collect transitions and send to learner via queue.
        
        Uses shared _actor_step() logic, but sends transitions via queue
        instead of pushing directly to replay buffer.
        """
        local_policy_version = -1
        steps_since_sync = 0
        
        # Initialize actor state
        actor_state = self._init_actor_state()
        
        while not stop_event.is_set():
            # Throttle if actors are too far ahead of learner
            if self.config.max_env_steps_per_training_step is not None:
                while not stop_event.is_set():
                    training_steps = shared_training_steps.value
                    env_steps = shared_env_steps.value
                    max_env = training_steps * self.config.max_env_steps_per_training_step
                    if env_steps < max_env or training_steps == 0:
                        break
                    time.sleep(0.01)  # Wait 10ms before checking again
            
            # Sync policy and training_step_count periodically
            steps_since_sync += 1
            if steps_since_sync >= self.config.actor_sync_freq or local_policy_version < 0:
                with policy_lock:
                    current_version = shared_policy['version']
                    if current_version > local_policy_version:
                        policy_state = self._deserialize_policy_state(shared_policy['state_dict'])
                        self._load_policy_state_dict(policy_state)
                        local_policy_version = current_version
                        steps_since_sync = 0
                # Update local training_step_count from shared counter (for epsilon/beta_r)
                self.training_step_count = shared_training_steps.value
            
            # Use shared actor logic to collect one transition
            transition = self._actor_step(actor_state)
            
            if transition is not None:
                # Serialize transition for queue (instead of pushing to buffer directly)
                trans_dict = {
                    'state': transition.state,
                    'robot_action': transition.robot_action,
                    'goals': transition.goals,
                    'goal_weights': transition.goal_weights,
                    'human_actions': transition.human_actions,
                    'next_state': transition.next_state,
                    'transition_probs_by_action': transition.transition_probs_by_action,
                    'terminal': transition.terminal,
                }
                
                try:
                    transition_queue.put(trans_dict, timeout=1.0)
                    # Increment shared env steps counter (for TensorBoard logging)
                    with shared_env_steps.get_lock():
                        shared_env_steps.value += 1
                except Exception as e:
                    # Queue full or serialization error - skip transition
                    pass

    def _learner_loop(
        self,
        transition_queue: mp.Queue,
        stop_event: mp.Event,
        shared_training_steps: mp.Value,
        shared_env_steps: mp.Value,
        shared_policy: Dict,
        policy_lock,
        num_training_steps: int,
    ) -> List[Dict[str, float]]:
        """
        Async learner loop: consume transitions from queue and train.
        
        Uses shared _learner_step() logic for training, but consumes
        transitions from queue instead of collecting them directly.
        
        Args:
            shared_env_steps: Shared counter of env steps produced by actors (for logging).
        """
        history = []
        policy_updates = 0
        
        # Initialize learner state
        learner_state = self._init_learner_state()
        
        # Set main networks to train mode (enables dropout)
        self.networks.q_r.train()
        self.networks.v_h_e.train()
        self.networks.x_h.train()
        if self.networks.u_r is not None:
            self.networks.u_r.train()
        if self.networks.v_r is not None:
            self.networks.v_r.train()
        
        if self.verbose:
            active = self.config.get_active_networks(self.training_step_count)
            print(f"[Learner] Starting in warmup stage {learner_state.prev_stage}: {learner_state.prev_stage_name} (active: {active})")
        
        # Progress bar measured in training steps
        pbar = tqdm(total=num_training_steps, desc="Async Training", unit="steps")
        pbar.update(self.training_step_count)  # Start from current position if resuming
        
        # Wait for minimum buffer size
        # Store shared_env_steps so _learner_step can reset it when buffer is cleared
        self._shared_env_steps = shared_env_steps
        
        if self.verbose:
            print(f"[Learner] Waiting for {self.config.async_min_buffer_size} transitions...")
        
        while len(self.replay_buffer) < self.config.async_min_buffer_size:
            self._consume_transitions(transition_queue, max_items=100)
            if stop_event.is_set():
                break
        
        if self.verbose:
            print(f"[Learner] Buffer ready with {len(self.replay_buffer)} transitions. Starting training.")
        
        # Main training loop
        while self.training_step_count < num_training_steps:
            # Consume new transitions from queue
            self._consume_transitions(transition_queue, max_items=50)
            
            # Do training step if buffer has enough samples
            if len(self.replay_buffer) >= self.config.batch_size:
                # Update total_env_steps from shared counter before logging
                self.total_env_steps = shared_env_steps.value
                
                # Use shared learner logic
                losses = self._learner_step(learner_state, pbar)
                
                # Update shared training_step_count so actors get correct epsilon/beta_r
                shared_training_steps.value = self.training_step_count
                
                # Log to history periodically
                if self.training_step_count % 100 == 0:
                    history.append(losses)
                
                # Update shared policy periodically
                if self.training_step_count % self.config.actor_sync_freq == 0:
                    with policy_lock:
                        shared_policy['state_dict'] = self._serialize_policy_state()
                        shared_policy['version'] += 1
                        policy_updates += 1
        
        pbar.close()
        
        if self.verbose:
            print(f"[Learner] Completed {self.training_step_count} training steps, {policy_updates} policy updates")
        
        # Set all networks to eval mode (disables dropout for rollouts)
        self.networks.q_r.eval()
        self.networks.v_h_e.eval()
        self.networks.x_h.eval()
        if self.networks.u_r is not None:
            self.networks.u_r.eval()
        if self.networks.v_r is not None:
            self.networks.v_r.eval()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return history
    
    def _consume_transitions(self, transition_queue: mp.Queue, max_items: int = 100) -> int:
        """
        Consume transitions from queue and add to replay buffer.
        
        Args:
            transition_queue: Queue with serialized transitions.
            max_items: Maximum number of items to consume.
            
        Returns:
            Number of transitions consumed.
        """
        consumed = 0
        while consumed < max_items:
            try:
                trans_dict = transition_queue.get_nowait()
                
                # Add to buffer using individual fields (matching push signature)
                self.replay_buffer.push(
                    state=trans_dict['state'],
                    robot_action=trans_dict['robot_action'],
                    goals=trans_dict['goals'],
                    goal_weights=trans_dict['goal_weights'],
                    human_actions=trans_dict['human_actions'],
                    next_state=trans_dict['next_state'],
                    transition_probs_by_action=trans_dict.get('transition_probs_by_action'),
                    terminal=trans_dict.get('terminal', False)
                )
                consumed += 1
            except Empty:
                break
            except Exception:
                break
        
        return consumed
    
    # ==================== Save/Load Methods ====================
    
    def save_all_networks(self, path: str) -> str:
        """
        Save all networks to a file.
        
        Saves all trained networks (Q_r, V_h^e, X_h, U_r, V_r) along with their
        target networks. This allows resuming training or using the full model.
        
        Args:
            path: Path to save the checkpoint file.
        
        Returns:
            The actual path where the file was saved (may differ from input
            if a fallback location was used due to permission errors).
        """
        checkpoint = {
            'q_r': self.networks.q_r.state_dict(),
            'v_h_e': self.networks.v_h_e.state_dict(),
            'x_h': self.networks.x_h.state_dict(),
            'total_env_steps': self.total_env_steps,
            'training_step_count': self.training_step_count,
            'config': {
                'gamma_r': self.config.gamma_r,
                'gamma_h': self.config.gamma_h,
                'beta_r': self.config.beta_r,
                'zeta': self.config.zeta,
                'xi': self.config.xi,
                'eta': self.config.eta,
                'u_r_use_network': self.config.u_r_use_network,
                'v_r_use_network': self.config.v_r_use_network,
            }
        }
        
        # Save U_r/V_r networks only if they exist
        if self.networks.u_r is not None:
            checkpoint['u_r'] = self.networks.u_r.state_dict()
        if self.networks.v_r is not None:
            checkpoint['v_r'] = self.networks.v_r.state_dict()
        
        # Save target networks if they exist
        if self.networks.v_r_target is not None:
            checkpoint['v_r_target'] = self.networks.v_r_target.state_dict()
        if self.networks.v_h_e_target is not None:
            checkpoint['v_h_e_target'] = self.networks.v_h_e_target.state_dict()
        if self.networks.x_h_target is not None:
            checkpoint['x_h_target'] = self.networks.x_h_target.state_dict()
        if self.networks.u_r_target is not None:
            checkpoint['u_r_target'] = self.networks.u_r_target.state_dict()
        if self.networks.q_r_target is not None:
            checkpoint['q_r_target'] = self.networks.q_r_target.state_dict()
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Try to save, fall back to tmp file on permission errors
        try:
            torch.save(checkpoint, path)
            return path
        except (IOError, OSError, RuntimeError) as e:
            # Fall back to tmp file to prevent data loss
            basename = os.path.basename(path)
            tmp_path = os.path.join(tempfile.gettempdir(), f'empo_fallback_{basename}')
            print(f"WARNING: Cannot save to {path}: {e}")
            print(f"Saving to fallback location: {tmp_path}")
            torch.save(checkpoint, tmp_path)
            return tmp_path
    
    def load_all_networks(self, path: str, strict: bool = True) -> None:
        """
        Load all networks from a file.
        
        Restores all trained networks and target networks from a checkpoint.
        
        Args:
            path: Path to the checkpoint file.
            strict: If True, requires all keys to match exactly.
        """
        # Note: weights_only=False is required for loading checkpoints with
        # complex nested structures. The checkpoint is trusted since it was
        # created by save_all_networks().
        checkpoint = torch.load(path, weights_only=False)
        
        self.networks.q_r.load_state_dict(checkpoint['q_r'], strict=strict)
        self.networks.v_h_e.load_state_dict(checkpoint['v_h_e'], strict=strict)
        self.networks.x_h.load_state_dict(checkpoint['x_h'], strict=strict)
        
        # Load U_r/V_r only if they exist in checkpoint AND network exists
        if 'u_r' in checkpoint and self.networks.u_r is not None:
            self.networks.u_r.load_state_dict(checkpoint['u_r'], strict=strict)
        if 'v_r' in checkpoint and self.networks.v_r is not None:
            self.networks.v_r.load_state_dict(checkpoint['v_r'], strict=strict)
        
        if 'total_env_steps' in checkpoint:
            self.total_env_steps = checkpoint['total_env_steps']
        if 'training_step_count' in checkpoint:
            self.training_step_count = checkpoint['training_step_count']
        
        # Load target networks if they exist
        if 'v_r_target' in checkpoint and self.networks.v_r_target is not None:
            self.networks.v_r_target.load_state_dict(checkpoint['v_r_target'], strict=strict)
        if 'v_h_e_target' in checkpoint and self.networks.v_h_e_target is not None:
            self.networks.v_h_e_target.load_state_dict(checkpoint['v_h_e_target'], strict=strict)
        if 'x_h_target' in checkpoint and self.networks.x_h_target is not None:
            self.networks.x_h_target.load_state_dict(checkpoint['x_h_target'], strict=strict)
        if 'u_r_target' in checkpoint and self.networks.u_r_target is not None:
            self.networks.u_r_target.load_state_dict(checkpoint['u_r_target'], strict=strict)
        if 'q_r_target' in checkpoint and self.networks.q_r_target is not None:
            self.networks.q_r_target.load_state_dict(checkpoint['q_r_target'], strict=strict)
    
    def save_policy(self, path: str) -> str:
        """
        Save only the robot policy network (Q_r) for deployment/rollouts.
        
        This saves a checkpoint containing:
        - Q_r network state dict (weights)
        - Q_r network config (for reconstruction)
        - beta_r parameter
        
        The policy can be loaded with MultiGridRobotPolicy.from_checkpoint(path).
        
        Args:
            path: Path to save the policy file.
        
        Returns:
            The actual path where the file was saved (may differ from input
            if a fallback location was used due to permission errors).
        """
        checkpoint = {
            'q_r': self.networks.q_r.state_dict(),
            'beta_r': self.config.beta_r,
            # Network config for reconstruction
            'q_r_config': self.networks.q_r.get_config(),
        }
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Try to save, fall back to tmp file on permission errors
        try:
            torch.save(checkpoint, path)
            return path
        except (IOError, OSError, RuntimeError) as e:
            # Fall back to tmp file to prevent data loss
            basename = os.path.basename(path)
            tmp_path = os.path.join(tempfile.gettempdir(), f'empo_fallback_{basename}')
            print(f"WARNING: Cannot save to {path}: {e}")
            print(f"Saving to fallback location: {tmp_path}")
            torch.save(checkpoint, tmp_path)
            return tmp_path

    # ==================== Convenience Evaluation Methods ====================
    
    def get_v_h_e(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int,
        goal: Any
    ) -> float:
        """
        Get V_h^e(s, g_h) - probability human h achieves goal g_h.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            human_agent_idx: Index of the human agent.
            goal: The goal for this human.
        
        Returns:
            V_h^e value in [0, 1].
        """
        self.networks.v_h_e.eval()
        with torch.no_grad():
            v_h_e = self.networks.v_h_e.forward(
                state, world_model, human_agent_idx, goal, self.device
            )
            return v_h_e.squeeze().item()
    
    def get_x_h(
        self,
        state: Any,
        world_model: Any,
        human_agent_idx: int
    ) -> float:
        """
        Get X_h(s) - aggregate goal achievement ability for human h.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            human_agent_idx: Index of the human agent.
        
        Returns:
            X_h value in (0, 1].
        """
        self.networks.x_h.eval()
        with torch.no_grad():
            x_h = self.networks.x_h.forward(
                state, world_model, human_agent_idx, self.device
            )
            x_h_clamped = torch.clamp(x_h.squeeze(), min=1e-3, max=1.0)
            return x_h_clamped.item()
    
    def get_q_r(
        self,
        state: Any,
        world_model: Any
    ) -> np.ndarray:
        """
        Get Q_r(s, a_r) - robot Q-values for all joint actions.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
        
        Returns:
            Array of Q-values for each joint action.
        """
        self.networks.q_r.eval()
        with torch.no_grad():
            q_values = self.networks.q_r.forward(state, world_model, self.device)
            return q_values.squeeze().cpu().numpy()
    
    def get_pi_r(
        self,
        state: Any,
        world_model: Any,
        beta_r: Optional[float] = None
    ) -> np.ndarray:
        """
        Get π_r(a_r|s) - robot policy probabilities.
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            beta_r: Policy concentration parameter. Uses config.beta_r if None.
        
        Returns:
            Array of probabilities for each joint action.
        """
        if beta_r is None:
            beta_r = self.config.beta_r
        
        self.networks.q_r.eval()
        with torch.no_grad():
            q_values = self.networks.q_r.forward(state, world_model, self.device)
            pi_r = self.networks.q_r.get_policy(q_values, beta_r=beta_r)
            return pi_r.squeeze().cpu().numpy()
    
    def get_u_r(
        self,
        state: Any,
        world_model: Any
    ) -> float:
        """
        Get U_r(s) - intrinsic robot reward.
        
        If u_r_use_network is True, uses the U_r network.
        Otherwise, computes directly from X_h values:
            y = E_h[X_h^{-ξ}], U_r = -y^η
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
        
        Returns:
            U_r value (negative).
        """
        with torch.no_grad():
            if self.config.u_r_use_network and self.networks.u_r is not None:
                self.networks.u_r.eval()
                _, u_r = self.networks.u_r.forward(state, world_model, self.device)
                return u_r.item()
            else:
                # Compute directly from X_h values
                x_h_vals = []
                for h in self.human_agent_indices:
                    x_h = self.get_x_h(state, world_model, h)
                    x_h_vals.append(x_h)
                
                x_h_tensor = torch.tensor(x_h_vals, device=self.device)
                y = (x_h_tensor ** (-self.config.xi)).mean()
                u_r_val = -(y ** self.config.eta)
                return u_r_val.item()
    
    def get_v_r(
        self,
        state: Any,
        world_model: Any,
        beta_r: Optional[float] = None
    ) -> float:
        """
        Get V_r(s) - robot state value.
        
        Computed as V_r = U_r + E_{a~π_r}[Q_r(s,a)]
        
        Args:
            state: Environment state.
            world_model: Environment/world model.
            beta_r: Policy concentration parameter. Uses config.beta_r if None.
        
        Returns:
            V_r value.
        """
        if beta_r is None:
            beta_r = self.config.beta_r
        
        with torch.no_grad():
            u_r = self.get_u_r(state, world_model)
            
            self.networks.q_r.eval()
            q_values = self.networks.q_r.forward(state, world_model, self.device)
            pi_r = self.networks.q_r.get_policy(q_values, beta_r=beta_r)
            expected_q = (pi_r * q_values).sum().item()
            
            return u_r + expected_q
