"""
Base Trainer for Phase 2 Robot Policy Learning.

This module provides the training loop and loss computation for Phase 2
of the EMPO framework (equations 4-9).
"""

import random
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from .robot_value_network import BaseRobotValueNetwork

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
    u_r: BaseIntrinsicRewardNetwork
    v_r: BaseRobotValueNetwork
    
    # Target networks (frozen copies for stable training)
    v_r_target: Optional[BaseRobotValueNetwork] = None
    v_h_e_target: Optional[BaseHumanGoalAchievementNetwork] = None
    x_h_target: Optional[BaseAggregateGoalAbilityNetwork] = None


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
        human_policy_prior: Callable that returns human action given state and goal.
        goal_sampler: Callable that samples a goal for a human.
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
        tensorboard_dir: Optional[str] = None
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
        
        # Training step counter
        self.total_steps = 0
        
        if self.debug:
            print("[DEBUG] BasePhase2Trainer.__init__: Initialization complete.")
    
    def _init_target_networks(self):
        """Initialize target networks as copies of main networks."""
        self.networks.v_r_target = copy.deepcopy(self.networks.v_r)
        self.networks.v_h_e_target = copy.deepcopy(self.networks.v_h_e)
        self.networks.x_h_target = copy.deepcopy(self.networks.x_h)
        
        # Freeze target networks (no gradients)
        for param in self.networks.v_r_target.parameters():
            param.requires_grad = False
        for param in self.networks.v_h_e_target.parameters():
            param.requires_grad = False
        for param in self.networks.x_h_target.parameters():
            param.requires_grad = False
        
        # Set target networks to eval mode (disables dropout during inference)
        self.networks.v_r_target.eval()
        self.networks.v_h_e_target.eval()
        self.networks.x_h_target.eval()
    
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
            'u_r': optim.Adam(
                self.networks.u_r.parameters(), 
                lr=self.config.lr_u_r,
                weight_decay=self.config.u_r_weight_decay
            ),
        }
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
            'u_r': self.networks.u_r,
        }
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
            'u_r': self.networks.u_r,
        }
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
        """Update target networks (hard copy)."""
        self.networks.v_r_target.load_state_dict(self.networks.v_r.state_dict())
        self.networks.v_h_e_target.load_state_dict(self.networks.v_h_e.state_dict())
        self.networks.x_h_target.load_state_dict(self.networks.x_h.state_dict())
        
        # Ensure target networks stay in eval mode (disables dropout)
        self.networks.v_r_target.eval()
        self.networks.v_h_e_target.eval()
        self.networks.x_h_target.eval()
    
    @abstractmethod
    def encode_state(self, state: Any) -> Dict[str, torch.Tensor]:
        """
        Encode environment state into tensor format.
        
        Args:
            state: Raw environment state.
        
        Returns:
            Dict of encoded state tensors.
        """
        pass
    
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
    
    @abstractmethod
    def encode_states_batch(
        self,
        states: List[Any],
        query_agent_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch of states into tensor format for batched forward passes.
        
        Args:
            states: List of raw environment states.
            query_agent_indices: List of query agent indices (one per state).
        
        Returns:
            Tuple of (grid_tensors, global_features, agent_features, interactive_features)
            all with batch dimension.
        """
        pass
    
    @abstractmethod
    def encode_goals_batch(
        self,
        goals: List[Any]
    ) -> torch.Tensor:
        """
        Encode a batch of goals into tensor format.
        
        Args:
            goals: List of goals.
        
        Returns:
            Goal features tensor with batch dimension.
        """
        pass
    
    def sample_robot_action(self, state: Any) -> Tuple[int, ...]:
        """
        Sample robot action using policy with epsilon-greedy exploration.
        
        Args:
            state: Current state.
        
        Returns:
            Tuple of robot actions.
        """
        epsilon = self.config.get_epsilon(self.total_steps)
        
        with torch.no_grad():
            q_values = self.networks.q_r.encode_and_forward(
                state, None, self.device
            )
            return self.networks.q_r.sample_action(q_values, epsilon)
    
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
            action = self.human_policy_prior(state, h, goal)
            actions.append(action)
        return actions
    
    def collect_transition(
        self,
        state: Any,
        goals: Dict[int, Any]
    ) -> Tuple[Phase2Transition, Any]:
        """
        Collect one transition from the environment.
        
        Args:
            state: Current state.
            goals: Current goal assignments.
        
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
        
        # Create transition
        transition = Phase2Transition(
            state=state,
            robot_action=robot_action,
            goals=goals.copy(),
            human_actions=human_actions,
            next_state=next_state
        )
        
        if self.debug:
            print(f"[DEBUG] collect_transition: done")
        
        return transition, next_state
    
    def compute_losses(
        self,
        batch: List[Phase2Transition],
        x_h_batch: Optional[List[Phase2Transition]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all networks using batched forward passes.
        
        Args:
            batch: List of transitions for most networks.
            x_h_batch: Optional larger batch for X_h (defaults to batch).
        
        Returns:
            Dict mapping loss names to loss tensors.
        """
        if x_h_batch is None:
            x_h_batch = batch
        
        n_batch = len(batch)
        n_humans = len(self.human_agent_indices)
        
        # =====================================================================
        # Step 1: Prepare data for batched encoding
        # =====================================================================
        
        # For V_h^e: we need (s, h, g_h) and (s', h, g_h) for each human in each transition
        # This gives us n_batch * n_humans samples
        v_h_e_states = []
        v_h_e_next_states = []
        v_h_e_humans = []
        v_h_e_goals = []
        v_h_e_goal_achieved = []
        
        for transition in batch:
            for h in self.human_agent_indices:
                g_h = transition.goals.get(h)
                if g_h is not None:
                    v_h_e_states.append(transition.state)
                    v_h_e_next_states.append(transition.next_state)
                    v_h_e_humans.append(h)
                    v_h_e_goals.append(g_h)
                    v_h_e_goal_achieved.append(
                        self.check_goal_achieved(transition.next_state, h, g_h)
                    )
        
        # For U_r: we need (s,) for each transition, and X_h values for target computation
        u_r_states = [t.state for t in batch]
        
        # For Q_r: we need (s,) and (s',) for each transition, plus actions
        q_r_states = [t.state for t in batch]
        q_r_next_states = [t.next_state for t in batch]
        q_r_actions = [t.robot_action for t in batch]
        
        # For X_h: collect all (s, h, g_h) pairs from x_h_batch
        x_h_states = []
        x_h_humans = []
        x_h_goals = []
        
        for transition in x_h_batch:
            if self.config.x_h_sample_humans is None:
                humans_for_x_h = list(transition.goals.keys())
            else:
                n_sample = min(self.config.x_h_sample_humans, len(transition.goals))
                humans_for_x_h = random.sample(list(transition.goals.keys()), n_sample)
            
            for h_x in humans_for_x_h:
                g_h_x = transition.goals.get(h_x)
                if g_h_x is not None:
                    x_h_states.append(transition.state)
                    x_h_humans.append(h_x)
                    x_h_goals.append(g_h_x)
        
        # =====================================================================
        # Step 2: Batch encode all states and goals
        # =====================================================================
        
        # V_h^e encoding (current states)
        if len(v_h_e_states) > 0:
            v_h_e_grid, v_h_e_global, v_h_e_agent, v_h_e_interactive = \
                self.encode_states_batch(v_h_e_states, v_h_e_humans)
            v_h_e_goal_features = self.encode_goals_batch(v_h_e_goals)
            
            # V_h^e encoding (next states for targets)
            v_h_e_next_grid, v_h_e_next_global, v_h_e_next_agent, v_h_e_next_interactive = \
                self.encode_states_batch(v_h_e_next_states, v_h_e_humans)
        
        # U_r encoding (uses robot as query agent, index 0 typically)
        robot_idx = self.robot_agent_indices[0] if self.robot_agent_indices else 0
        u_r_grid, u_r_global, u_r_agent, u_r_interactive = \
            self.encode_states_batch(u_r_states, [robot_idx] * n_batch)
        
        # Q_r encoding (current and next states)
        q_r_grid, q_r_global, q_r_agent, q_r_interactive = \
            self.encode_states_batch(q_r_states, [robot_idx] * n_batch)
        q_r_next_grid, q_r_next_global, q_r_next_agent, q_r_next_interactive = \
            self.encode_states_batch(q_r_next_states, [robot_idx] * n_batch)
        
        # X_h encoding
        if len(x_h_states) > 0:
            x_h_grid, x_h_global, x_h_agent, x_h_interactive = \
                self.encode_states_batch(x_h_states, x_h_humans)
            x_h_goal_features = self.encode_goals_batch(x_h_goals)
        
        # For U_r target computation, we need X_h values for each human
        # Encode states once per human for X_h target network
        x_h_target_values_per_human = {}
        if self.config.u_r_sample_humans is None:
            humans_for_u_r = self.human_agent_indices
        else:
            n_sample = min(self.config.u_r_sample_humans, len(self.human_agent_indices))
            humans_for_u_r = random.sample(self.human_agent_indices, n_sample)
        
        for h_u in humans_for_u_r:
            h_grid, h_global, h_agent, h_interactive = \
                self.encode_states_batch(u_r_states, [h_u] * n_batch)
            with torch.no_grad():
                x_h_vals = self.networks.x_h_target.forward_from_encoded(
                    h_grid, h_global, h_agent, h_interactive
                )
            x_h_target_values_per_human[h_u] = x_h_vals
        
        # =====================================================================
        # Step 3: Batched forward passes
        # =====================================================================
        
        losses = {
            'v_h_e': torch.tensor(0.0, device=self.device),
            'x_h': torch.tensor(0.0, device=self.device),
            'u_r': torch.tensor(0.0, device=self.device),
            'q_r': torch.tensor(0.0, device=self.device),
        }
        if self.config.v_r_use_network:
            losses['v_r'] = torch.tensor(0.0, device=self.device)
        
        # ----- V_h^e loss -----
        if len(v_h_e_states) > 0:
            # Forward pass on current states
            v_h_e_pred = self.networks.v_h_e.forward_with_goal_features(
                v_h_e_grid, v_h_e_global, v_h_e_agent, v_h_e_interactive, v_h_e_goal_features
            )
            
            # Target computation (next states with target network)
            with torch.no_grad():
                v_h_e_next = self.networks.v_h_e_target.forward_with_goal_features(
                    v_h_e_next_grid, v_h_e_next_global, v_h_e_next_agent,
                    v_h_e_next_interactive, v_h_e_goal_features
                )
            
            # Compute TD targets
            goal_achieved_t = torch.tensor(v_h_e_goal_achieved, dtype=torch.float32, device=self.device)
            # TD target: if goal achieved -> 1, else -> 0 + gamma * V_next
            v_h_e_targets = goal_achieved_t + (1.0 - goal_achieved_t) * self.config.gamma_h * v_h_e_next.squeeze()
            
            losses['v_h_e'] = torch.mean((v_h_e_pred.squeeze() - v_h_e_targets) ** 2)
        
        # ----- U_r loss -----
        # Forward pass
        y_pred, _ = self.networks.u_r.forward_from_encoded(
            u_r_grid, u_r_global, u_r_agent, u_r_interactive
        )
        
        # Compute target y from X_h values (averaged over humans)
        # Stack X_h values: shape (n_humans, n_batch)
        x_h_stacked = torch.stack([x_h_target_values_per_human[h].squeeze() for h in humans_for_u_r], dim=0)
        # Compute X_h^{-xi} and average over humans
        x_h_neg_xi = x_h_stacked ** (-self.config.xi)  # (n_humans, n_batch)
        x_h_avg = x_h_neg_xi.mean(dim=0)  # (n_batch,)
        # Target y = E[X_h^{-xi}]^{-1/xi}
        target_y = x_h_avg ** (-1.0 / self.config.xi)
        
        losses['u_r'] = torch.mean((y_pred.squeeze() - target_y) ** 2)
        
        # ----- Q_r loss -----
        # Forward pass - get all Q values
        q_r_all = self.networks.q_r.forward_from_encoded(
            q_r_grid, q_r_global, q_r_agent, q_r_interactive
        )  # (n_batch, n_actions)
        
        # Select Q values for taken actions
        action_indices = torch.tensor(
            [self.networks.q_r.action_tuple_to_index(a) for a in q_r_actions],
            dtype=torch.long, device=self.device
        )
        q_r_pred = q_r_all[torch.arange(n_batch, device=self.device), action_indices]
        
        # Compute V_r(s') for target
        with torch.no_grad():
            if self.config.v_r_use_network:
                v_r_next = self.networks.v_r_target.forward_from_encoded(
                    q_r_next_grid, q_r_next_global, q_r_next_agent, q_r_next_interactive
                )
            else:
                # Compute V_r directly: V_r(s') = U_r(s') + π_r(s') · Q_r(s')
                _, u_r_next = self.networks.u_r.forward_from_encoded(
                    q_r_next_grid, q_r_next_global, q_r_next_agent, q_r_next_interactive
                )
                q_r_next = self.networks.q_r.forward_from_encoded(
                    q_r_next_grid, q_r_next_global, q_r_next_agent, q_r_next_interactive
                )
                pi_r_next = self.networks.q_r.get_policy(q_r_next)
                # V_r = U_r + sum(pi * Q) - compute for each batch element
                v_r_next = u_r_next.squeeze() + (pi_r_next * q_r_next).sum(dim=-1)
        
        target_q_r = self.config.gamma_r * v_r_next.squeeze()
        losses['q_r'] = torch.mean((q_r_pred - target_q_r) ** 2)
        
        # ----- V_r loss (only if using network mode) -----
        if self.config.v_r_use_network:
            v_r_pred = self.networks.v_r.forward_from_encoded(
                q_r_grid, q_r_global, q_r_agent, q_r_interactive
            )
            
            with torch.no_grad():
                _, u_r_for_v = self.networks.u_r.forward_from_encoded(
                    q_r_grid, q_r_global, q_r_agent, q_r_interactive
                )
                q_r_for_v = self.networks.q_r.forward_from_encoded(
                    q_r_grid, q_r_global, q_r_agent, q_r_interactive
                )
                pi_r = self.networks.q_r.get_policy(q_r_for_v)
            
            target_v_r = u_r_for_v.squeeze() + (pi_r * q_r_for_v).sum(dim=-1)
            losses['v_r'] = torch.mean((v_r_pred.squeeze() - target_v_r) ** 2)
        
        # ----- X_h loss -----
        if len(x_h_states) > 0:
            # Forward pass
            x_h_pred = self.networks.x_h.forward_from_encoded(
                x_h_grid, x_h_global, x_h_agent, x_h_interactive
            )
            
            # Target computation using V_h^e target network
            with torch.no_grad():
                v_h_e_for_x = self.networks.v_h_e_target.forward_with_goal_features(
                    x_h_grid, x_h_global, x_h_agent, x_h_interactive, x_h_goal_features
                )
            
            # X_h target = V_h^e^zeta
            target_x_h = v_h_e_for_x.squeeze() ** self.config.zeta
            
            losses['x_h'] = torch.mean((x_h_pred.squeeze() - target_x_h) ** 2)
        
        return losses
    
    def training_step(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform one training step (sample batch, compute losses, update).
        
        Returns:
            Tuple of (loss_values dict, grad_norms dict).
        """
        # Determine X_h batch size (can be larger than regular batch)
        x_h_batch_size = self.config.x_h_batch_size or self.config.batch_size
        min_required = max(self.config.batch_size, x_h_batch_size)
        
        if len(self.replay_buffer) < min_required:
            return {}, {}
        
        # Sample batch for most networks
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Sample potentially larger batch for X_h if configured
        if x_h_batch_size > self.config.batch_size:
            x_h_batch = self.replay_buffer.sample(x_h_batch_size)
        else:
            x_h_batch = batch
        
        # Compute losses (with separate X_h batch)
        losses = self.compute_losses(batch, x_h_batch)
        
        # Update each network - need retain_graph=True for all but last backward
        # since losses may share computational graphs through state encodings
        loss_values = {}
        grad_norms = {}
        
        # Map network names to their grad clip configs and network objects
        grad_clip_configs = {
            'q_r': (self.config.q_r_grad_clip, self.networks.q_r),
            'v_h_e': (self.config.v_h_e_grad_clip, self.networks.v_h_e),
            'x_h': (self.config.x_h_grad_clip, self.networks.x_h),
            'u_r': (self.config.u_r_grad_clip, self.networks.u_r),
        }
        if self.config.v_r_use_network:
            grad_clip_configs['v_r'] = (self.config.v_r_grad_clip, self.networks.v_r)
        
        loss_names = list(losses.keys())
        for i, name in enumerate(loss_names):
            loss = losses[name]
            if name not in self.optimizers:
                # Skip losses for networks we're not training (e.g., v_r when not using network)
                loss_values[name] = loss.item()
                continue
            if loss.requires_grad:
                self.optimizers[name].zero_grad()
                # Retain graph for all but the last loss to allow multiple backwards
                retain = (i < len(loss_names) - 1)
                loss.backward(retain_graph=retain)
                
                # Apply gradient clipping for this network
                if name in grad_clip_configs:
                    clip_val, net = grad_clip_configs[name]
                    if clip_val and clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), clip_val)
                
                # Compute gradient norm before step
                grad_norms[name] = self._compute_single_grad_norm(name)
                self.optimizers[name].step()
            loss_values[name] = loss.item()
        
        # Update target networks periodically
        if self.total_steps % self.config.v_r_target_update_freq == 0:
            self.update_target_networks()
        
        return loss_values, grad_norms
    
    def _compute_single_grad_norm(self, network_name: str) -> float:
        """Compute gradient L2 norm for a single network."""
        networks = {
            'q_r': self.networks.q_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
            'u_r': self.networks.u_r,
        }
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
    
    def train_episode(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train for one episode.
        
        Returns:
            Tuple of (avg_losses, avg_grad_norms) for the episode.
        """
        if self.debug:
            print(f"[DEBUG] train_episode: resetting environment...")
        
        state = self.reset_environment()
        
        if self.debug:
            print(f"[DEBUG] train_episode: sampling initial goals...")
        
        # Sample initial goals
        goals = {h: self.goal_sampler(state, h) for h in self.human_agent_indices}
        
        if self.debug:
            print(f"[DEBUG] train_episode: goals sampled, starting {self.config.steps_per_episode} steps...")
        
        episode_losses = {
            'v_h_e': [], 'x_h': [], 'u_r': [], 'q_r': []
        }
        episode_grad_norms = {
            'v_h_e': [], 'x_h': [], 'u_r': [], 'q_r': []
        }
        # Only track V_r if using network mode
        if self.config.v_r_use_network:
            episode_losses['v_r'] = []
            episode_grad_norms['v_r'] = []
        
        for step in range(self.config.steps_per_episode):
            if self.debug and step % 5 == 0:
                print(f"[DEBUG] train_episode: step {step}/{self.config.steps_per_episode}")
            
            # Collect transition
            transition, next_state = self.collect_transition(state, goals)
            
            if self.debug and step == 0:
                print(f"[DEBUG] train_episode: pushing to replay buffer...")
            
            self.replay_buffer.push(
                transition.state,
                transition.robot_action,
                transition.goals,
                transition.human_actions,
                transition.next_state
            )
            
            if self.debug and step == 0:
                print(f"[DEBUG] train_episode: starting training updates...")
            
            # Training updates
            for _ in range(self.config.updates_per_step):
                losses, grad_norms = self.training_step()
                for k, v in losses.items():
                    episode_losses[k].append(v)
                for k, v in grad_norms.items():
                    episode_grad_norms[k].append(v)
            
            self.total_steps += 1
            state = next_state
            
            # Resample goals with some probability
            if random.random() < self.config.goal_resample_prob:
                goals = {h: self.goal_sampler(state, h) for h in self.human_agent_indices}
        
        if self.debug:
            print(f"[DEBUG] train_episode: episode complete, averaging losses...")
        
        # Average losses and grad norms
        avg_losses = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in episode_losses.items()
        }
        avg_grad_norms = {
            k: sum(v) / len(v) if v else 0.0
            for k, v in episode_grad_norms.items()
        }
        return avg_losses, avg_grad_norms
    
    def train(self, num_episodes: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Main training loop.
        
        Args:
            num_episodes: Number of episodes to train (default: from config).
        
        Returns:
            List of episode loss dicts.
        """
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        history = []
        
        # Set up progress bar
        pbar = tqdm(total=num_episodes, desc="Training", unit="episodes", disable=not self.verbose)
        
        for episode in range(num_episodes):
            episode_losses, episode_grad_norms = self.train_episode()
            history.append(episode_losses)
            
            # Log to TensorBoard
            if self.writer is not None:
                for key, value in episode_losses.items():
                    self.writer.add_scalar(f'Loss/{key}', value, episode)
                for key, value in episode_grad_norms.items():
                    self.writer.add_scalar(f'GradNorm/{key}', value, episode)
                # Also log parameter norms
                param_norms = self._compute_param_norms()
                for key, value in param_norms.items():
                    self.writer.add_scalar(f'ParamNorm/{key}', value, episode)
                self.writer.add_scalar('Epsilon', self.config.get_epsilon(self.total_steps), episode)
                # Flush to ensure data is written to disk for real-time monitoring
                self.writer.flush()
            
            # Update progress bar
            pbar.update(1)
            if episode_losses:
                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in episode_losses.items() if v > 0)
                pbar.set_postfix_str(loss_str[:60])
            
            # Print occasional summary if verbose
            if self.debug and episode % 100 == 0:
                print(f"Episode {episode}: {episode_losses}")
        
        pbar.close()
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return history
