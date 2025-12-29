"""
Base Trainer for Phase 2 Robot Policy Learning.

This module provides the training loop and loss computation for Phase 2
of the EMPO framework (equations 4-9).
"""

import os
import random
import copy
import multiprocessing as mp
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    
    Subclasses must implement environment-specific methods for:
    - State encoding
    - Goal sampling
    - Action execution
    - Goal achievement checking
    
    Args:
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
    """
    
    def __init__(
        self,
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
        self.networks = networks
        self.config = config
        self.human_agent_indices = human_agent_indices
        self.robot_agent_indices = robot_agent_indices
        self.human_policy_prior = human_policy_prior
        self.goal_sampler = goal_sampler
        self.device = device
        self.verbose = verbose
        self.debug = debug
        self.profiler = profiler
        
        # Initialize TensorBoard writer if requested
        self.writer = None
        if tensorboard_dir is not None and HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
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
        
        # Per-network update counters for 1/t learning rate schedules
        self.update_counts = {
            'q_r': 0,
            'v_r': 0,
            'v_h_e': 0,
            'x_h': 0,
            'u_r': 0,
        }
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Initialization complete.")
    
    def __getstate__(self):
        """Exclude unpicklable objects for async training (multiprocessing).
        
        Excludes:
        - TensorBoard writer (contains thread locks)
        - Profiler (contains thread locks)
        - Replay buffer (not needed in actor processes)
        """
        state = self.__dict__.copy()
        # Don't pickle TensorBoard writer - contains thread locks
        state['writer'] = None
        # Don't pickle profiler - may contain locks
        state['profiler'] = None
        # Don't pickle replay buffer - not needed in actor processes
        state['replay_buffer'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling for async training."""
        self.__dict__.update(state)
        # Recreate empty replay buffer if needed
        if self.replay_buffer is None:
            self.replay_buffer = Phase2ReplayBuffer(capacity=self.config.buffer_size)
    
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
        """Initialize optimizers for each network with weight decay."""
        optimizers = {
            'q_r': optim.Adam(
                self.networks.q_r.parameters(), 
                lr=self.config.lr_q_r,
                weight_decay=self.config.q_r_weight_decay
            ),
            'v_h_e': optim.Adam(
                self.networks.v_h_e.parameters(), 
                lr=self.config.lr_v_h_e,
                weight_decay=self.config.v_h_e_weight_decay
            ),
            'x_h': optim.Adam(
                self.networks.x_h.parameters(), 
                lr=self.config.lr_x_h,
                weight_decay=self.config.x_h_weight_decay
            ),
        }
        # Only create U_r optimizer if using network (not direct computation)
        if self.config.u_r_use_network:
            optimizers['u_r'] = optim.Adam(
                self.networks.u_r.parameters(), 
                lr=self.config.lr_u_r,
                weight_decay=self.config.u_r_weight_decay
            )
        # Only create V_r optimizer if using network (not direct computation)
        if self.config.v_r_use_network:
            optimizers['v_r'] = optim.Adam(
                self.networks.v_r.parameters(), 
                lr=self.config.lr_v_r,
                weight_decay=self.config.v_r_weight_decay
            )
        return optimizers
    
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
    
    @abstractmethod
    def check_goal_achieved(self, state: Any, human_idx: int, goal: Any) -> bool:
        """
        Check if a human's goal is achieved in the given state.
        
        Args:
            state: Current environment state.
            human_idx: Index of the human agent.
            goal: The goal to check.
        
        Returns:
            True if goal is achieved, False otherwise.
        """
        pass
    
    @abstractmethod
    def step_environment(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        human_actions: List[int]
    ) -> Any:
        """
        Execute actions in the environment.
        
        Args:
            state: Current state.
            robot_action: Tuple of robot actions.
            human_actions: List of human actions.
        
        Returns:
            Next state.
        """
        pass
    
    @abstractmethod
    def reset_environment(self) -> Any:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state.
        """
        pass
    
    def sample_robot_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action using policy with epsilon-greedy exploration.
        
        Uses q_r_target (frozen copy) for stable action sampling, consistent
        with async mode where actors use a periodically-synced copy.
        
        During warm-up, uses effective beta_r = 0 (uniform random policy).
        After warm-up, beta_r ramps up to nominal value.
        
        Args:
            state: Current state.
        
        Returns:
            Tuple of robot actions.
        """
        epsilon = self.config.get_epsilon(self.training_step_count)
        effective_beta_r = self.config.get_effective_beta_r(self.training_step_count)
        
        with torch.no_grad():
            q_values = self.networks.q_r_target.encode_and_forward(
                state, None, self.device
            )
            return self.networks.q_r_target.sample_action(
                q_values, epsilon, beta_r=effective_beta_r
            )
    
    def sample_human_actions(
        self,
        state: Any,
        goals: Dict[int, Any]
    ) -> List[int]:
        """
        Sample human actions from the human policy prior.
        
        Args:
            state: Current state.
            goals: Dict mapping human index to their goal.
        
        Returns:
            List of human actions.
        """
        actions = []
        for h in self.human_agent_indices:
            goal = goals.get(h)
            action = self.human_policy_prior.sample(state, h, goal)
            actions.append(action)
        return actions
    
    def collect_transition(
        self,
        state: Any,
        goals: Dict[int, Any],
        goal_weights: Dict[int, float]
    ) -> Tuple[Phase2Transition, Any]:
        """
        Collect one transition from the environment.
        
        Args:
            state: Current state.
            goals: Current goal assignments.
            goal_weights: Weights for each goal (from goal sampler).
        
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
            transition_probs_by_action=transition_probs_by_action
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
            _, u_r = self.networks.u_r.encode_and_forward(state, None, self.device)
            return u_r
        else:
            # Compute directly from X_h values
            x_h_values = []
            for h in self.human_agent_indices:
                x_h = self.networks.x_h.encode_and_forward(state, None, h, self.device)
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
            _, u_r = self.networks.u_r_target.encode_and_forward(state, None, self.device)
            return u_r
        else:
            # Compute directly from X_h TARGET values
            x_h_values = []
            for h in self.human_agent_indices:
                x_h = self.networks.x_h_target.encode_and_forward(state, None, h, self.device)
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
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all networks.
        
        Args:
            batch: List of transitions for most networks.
            x_h_batch: Optional larger batch for X_h (defaults to batch).
        
        Returns:
            Dict mapping loss names to loss tensors.
        """
        if x_h_batch is None:
            x_h_batch = batch
        
        # Check which networks are active - determines what we need to compute
        active_networks = self.config.get_active_networks(self.training_step_count)
        x_h_active = 'x_h' in active_networks
        u_r_active = 'u_r' in active_networks
        q_r_active = 'q_r' in active_networks
        
        losses = {
            'v_h_e': torch.tensor(0.0, device=self.device),
            'x_h': torch.tensor(0.0, device=self.device),
            'q_r': torch.tensor(0.0, device=self.device),
        }
        # Only include U_r loss if using network mode
        if self.config.u_r_use_network:
            losses['u_r'] = torch.tensor(0.0, device=self.device)
        # Only include V_r loss if using network mode
        if self.config.v_r_use_network:
            losses['v_r'] = torch.tensor(0.0, device=self.device)
        
        # Track counts for normalization
        x_h_count = 0
        
        for transition in batch:
            # Unpack transition
            s = transition.state
            a_r = transition.robot_action
            goals = transition.goals
            s_prime = transition.next_state
            
            # ===== V_h^e loss (for each human and their goal) =====
            for h, g_h in goals.items():
                # Get V_h^e prediction
                v_h_e_pred = self.networks.v_h_e.encode_and_forward(
                    s, None, h, g_h, self.device
                )
                
                # Check if goal achieved
                goal_achieved = self.check_goal_achieved(s_prime, h, g_h)
                goal_achieved_t = torch.tensor(
                    1.0 if goal_achieved else 0.0, 
                    device=self.device
                )
                
                # Get target V_h^e(s', g_h)
                with torch.no_grad():
                    v_h_e_next = self.networks.v_h_e_target.encode_and_forward(
                        s_prime, None, h, g_h, self.device
                    )
                
                # TD target: U_h(s') + γ_h * V_h^e(s', g_h) if not achieved
                target = self.networks.v_h_e.compute_td_target(
                    goal_achieved_t, v_h_e_next.squeeze()
                )
                
                losses['v_h_e'] = losses['v_h_e'] + (v_h_e_pred.squeeze() - target) ** 2
            
            # ===== U_r loss (only if using network mode AND U_r is active) =====
            if self.config.u_r_use_network and u_r_active:
                # Determine which humans to sample for U_r loss
                if self.config.u_r_sample_humans is None:
                    # Use all humans
                    humans_for_u_r = self.human_agent_indices
                else:
                    # Sample a fixed number of humans
                    n_sample = min(self.config.u_r_sample_humans, len(self.human_agent_indices))
                    humans_for_u_r = random.sample(self.human_agent_indices, n_sample)
                
                y_pred, _ = self.networks.u_r.encode_and_forward(s, None, self.device)
                
                # Accumulate X_h^{-ξ} over sampled humans using X_h target network for stability
                x_h_sum = torch.tensor(0.0, device=self.device)
                for h_u in humans_for_u_r:
                    with torch.no_grad():
                        x_h_for_h = self.networks.x_h_target.encode_and_forward(
                            s, None, h_u, self.device
                        )
                    # Clamp X_h to (0, 1] to prevent X_h^{-ξ} explosion when X_h is near 0
                    x_h_clamped = torch.clamp(x_h_for_h.squeeze(), min=1e-3, max=1.0)
                    # Accumulate X_h^{-ξ}
                    x_h_sum = x_h_sum + x_h_clamped ** (-self.config.xi)
                
                # Average to get E[X_h^{-ξ}] = y (the target)
                x_h_avg = x_h_sum / len(humans_for_u_r)
                
                # target_y = E[X_h^{-ξ}] directly (y is defined as E[X_h^{-ξ}])
                target_y = x_h_avg
                losses['u_r'] = losses['u_r'] + (y_pred.squeeze() - target_y) ** 2
            
            # ===== Q_r loss (only when Q_r is active) =====
            if q_r_active:
                q_r_all = self.networks.q_r.encode_and_forward(s, None, self.device)
                a_r_index = self.networks.q_r.action_tuple_to_index(a_r)
                q_r_pred = q_r_all.squeeze()[a_r_index]
                
                # Use effective beta_r for policy (0 during warm-up for independence)
                effective_beta_r = self.config.get_effective_beta_r(self.training_step_count)
                
                with torch.no_grad():
                    if self.config.v_r_use_network:
                        # Use V_r target network
                        v_r_next = self.networks.v_r_target.encode_and_forward(
                            s_prime, None, self.device
                        )
                    else:
                        # Compute V_r directly: V_r(s') = U_r(s') + π_r(s') · Q_r(s')
                        # Use TARGET networks for stable targets
                        u_r_next = self._compute_u_r_for_state_target(s_prime)
                        q_r_next = self.networks.q_r_target.encode_and_forward(s_prime, None, self.device)
                        pi_r_next = self.networks.q_r_target.get_policy(q_r_next, beta_r=effective_beta_r)
                        v_r_next = compute_v_r_from_components(
                            u_r_next.squeeze(), q_r_next.squeeze(), pi_r_next.squeeze()
                        )
                
                target_q_r = self.config.gamma_r * v_r_next.squeeze()
                losses['q_r'] = losses['q_r'] + (q_r_pred - target_q_r) ** 2
            
            # ===== V_r loss (only if using network mode) =====
            if self.config.v_r_use_network:
                v_r_pred = self.networks.v_r.encode_and_forward(s, None, self.device)
                
                with torch.no_grad():
                    # Use TARGET networks for stable V_r targets
                    u_r = self._compute_u_r_for_state_target(s)
                    q_r_for_v = self.networks.q_r_target.encode_and_forward(s, None, self.device)
                    pi_r = self.networks.q_r_target.get_policy(q_r_for_v, beta_r=effective_beta_r)
                
                target_v_r = compute_v_r_from_components(
                    u_r.squeeze(), q_r_for_v.squeeze(), pi_r.squeeze()
                )
                losses['v_r'] = losses['v_r'] + (v_r_pred.squeeze() - target_v_r) ** 2
        
        # ===== X_h loss (computed on potentially larger batch, only when X_h is active) =====
        if x_h_active:
            for transition in x_h_batch:
                s = transition.state
                goals = transition.goals
                goal_weights = transition.goal_weights
                
                # Determine which human-goal pairs to use for X_h loss
                if self.config.x_h_sample_humans is None:
                    # Use all humans' goals from the transition
                    humans_for_x_h = list(goals.keys())
                else:
                    # Sample a fixed number of humans
                    n_sample = min(self.config.x_h_sample_humans, len(goals))
                    humans_for_x_h = random.sample(list(goals.keys()), n_sample)
                
                for h_x in humans_for_x_h:
                    g_h_x = goals[h_x]
                    w_h_x = goal_weights[h_x]
                    x_h_pred = self.networks.x_h.encode_and_forward(
                        s, None, h_x, self.device
                    )
                    
                    with torch.no_grad():
                        # Use target network for more stable X_h targets
                        v_h_e_for_x = self.networks.v_h_e_target.encode_and_forward(
                            s, None, h_x, g_h_x, self.device
                        )
                    
                    target_x_h = self.networks.x_h.compute_target(v_h_e_for_x.squeeze(), w_h_x)
                    losses['x_h'] = losses['x_h'] + (x_h_pred.squeeze() - target_x_h) ** 2
                    x_h_count += 1
        
        # Average losses over their respective batch sizes
        n = len(batch)
        loss_keys_to_avg = ['v_h_e']
        if q_r_active:
            loss_keys_to_avg.append('q_r')
        if self.config.u_r_use_network and u_r_active:
            loss_keys_to_avg.append('u_r')
        if self.config.v_r_use_network:
            loss_keys_to_avg.append('v_r')
        for k in loss_keys_to_avg:
            losses[k] = losses[k] / n
        # X_h uses its own count (may be from larger batch)
        if x_h_active and x_h_count > 0:
            losses['x_h'] = losses['x_h'] / x_h_count
        
        return losses, {}
    
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
        # Collect one transition
        transition, next_state = self.collect_transition(
            actor_state.state, actor_state.goals, actor_state.goal_weights
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
        
        # Resample goals with some probability
        if random.random() < self.config.goal_resample_prob:
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
                self.writer.add_scalar(f'Predictions/{key}_mean', stats['mean'], self.training_step_count)
                self.writer.add_scalar(f'Predictions/{key}_std', stats['std'], self.training_step_count)
                if 'target_mean' in stats:
                    self.writer.add_scalar(f'Targets/{key}_mean', stats['target_mean'], self.training_step_count)
            
            # Log parameter norms
            param_norms = self._compute_param_norms()
            for key, value in param_norms.items():
                self.writer.add_scalar(f'ParamNorm/{key}', value, self.training_step_count)
            self.writer.add_scalar('Epsilon', self.config.get_epsilon(self.training_step_count), self.training_step_count)
            
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
                    transition.transition_probs_by_action
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
        import pickle
        import io
        state = self._get_policy_state_dict()
        # Move to CPU for sharing
        cpu_state = {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in state.items()}
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
                    transition_probs_by_action=trans_dict.get('transition_probs_by_action')
                )
                consumed += 1
            except Empty:
                break
            except Exception:
                break
        
        return consumed
    
    # ==================== Save/Load Methods ====================
    
    def save_all_networks(self, path: str) -> None:
        """
        Save all networks to a file.
        
        Saves all trained networks (Q_r, V_h^e, X_h, U_r, V_r) along with their
        target networks. This allows resuming training or using the full model.
        
        Args:
            path: Path to save the checkpoint file.
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
        except (IOError, OSError, RuntimeError) as e:
            # Fall back to tmp file to prevent data loss
            basename = os.path.basename(path)
            tmp_path = os.path.join(tempfile.gettempdir(), f'empo_fallback_{basename}')
            print(f"WARNING: Cannot save to {path}: {e}")
            print(f"Saving to fallback location: {tmp_path}")
            torch.save(checkpoint, tmp_path)
    
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
    
    def save_policy(self, path: str) -> None:
        """
        Save only the robot policy network (Q_r) for deployment/rollouts.
        
        This saves a checkpoint containing:
        - Q_r network state dict (weights)
        - Q_r network config (for reconstruction)
        - beta_r parameter
        
        The policy can be loaded with MultiGridRobotPolicy.from_checkpoint(path).
        
        Args:
            path: Path to save the policy file.
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
        except (IOError, OSError, RuntimeError) as e:
            # Fall back to tmp file to prevent data loss
            basename = os.path.basename(path)
            tmp_path = os.path.join(tempfile.gettempdir(), f'empo_fallback_{basename}')
            print(f"WARNING: Cannot save to {path}: {e}")
            print(f"Saving to fallback location: {tmp_path}")
            torch.save(checkpoint, tmp_path)
