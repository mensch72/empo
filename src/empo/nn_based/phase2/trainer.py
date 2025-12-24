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
        
        # Freeze target networks
        for param in self.networks.v_r_target.parameters():
            param.requires_grad = False
        for param in self.networks.v_h_e_target.parameters():
            param.requires_grad = False
        for param in self.networks.x_h_target.parameters():
            param.requires_grad = False
    
    def _init_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for each network."""
        return {
            'q_r': optim.Adam(self.networks.q_r.parameters(), lr=self.config.lr_q_r),
            'v_r': optim.Adam(self.networks.v_r.parameters(), lr=self.config.lr_v_r),
            'v_h_e': optim.Adam(self.networks.v_h_e.parameters(), lr=self.config.lr_v_h_e),
            'x_h': optim.Adam(
                self.networks.x_h.parameters(), 
                lr=self.config.lr_x_h,
                weight_decay=self.config.x_h_weight_decay
            ),
            'u_r': optim.Adam(self.networks.u_r.parameters(), lr=self.config.lr_u_r),
        }
    
    def _compute_param_norms(self) -> Dict[str, float]:
        """Compute L2 norms of network parameters for monitoring."""
        norms = {}
        networks = {
            'q_r': self.networks.q_r,
            'v_r': self.networks.v_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
            'u_r': self.networks.u_r,
        }
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
            'v_r': self.networks.v_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
            'u_r': self.networks.u_r,
        }
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
        Compute losses for all networks.
        
        Args:
            batch: List of transitions for most networks.
            x_h_batch: Optional larger batch for X_h (defaults to batch).
        
        Returns:
            Dict mapping loss names to loss tensors.
        """
        if x_h_batch is None:
            x_h_batch = batch
        
        losses = {
            'v_h_e': torch.tensor(0.0, device=self.device),
            'x_h': torch.tensor(0.0, device=self.device),
            'u_r': torch.tensor(0.0, device=self.device),
            'q_r': torch.tensor(0.0, device=self.device),
            'v_r': torch.tensor(0.0, device=self.device),
        }
        
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
            
            # ===== U_r loss (averaged over sampled or all humans) =====
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
                # Accumulate X_h^{-ξ}
                x_h_sum = x_h_sum + x_h_for_h.squeeze() ** (-self.config.xi)
            
            # Average to get E[X_h^{-ξ}]
            x_h_avg = x_h_sum / len(humans_for_u_r)
            
            # Compute y target: y = E[X_h^{-ξ}]^{1/ξ} (this gives X_h^{-1} if ξ=1)
            target_y = x_h_avg ** (-1.0 / self.config.xi)
            losses['u_r'] = losses['u_r'] + (y_pred.squeeze() - target_y) ** 2
            
            # ===== Q_r loss =====
            q_r_all = self.networks.q_r.encode_and_forward(s, None, self.device)
            a_r_index = self.networks.q_r.action_tuple_to_index(a_r)
            q_r_pred = q_r_all.squeeze()[a_r_index]
            
            with torch.no_grad():
                v_r_next = self.networks.v_r_target.encode_and_forward(
                    s_prime, None, self.device
                )
            
            target_q_r = self.config.gamma_r * v_r_next.squeeze()
            losses['q_r'] = losses['q_r'] + (q_r_pred - target_q_r) ** 2
            
            # ===== V_r loss =====
            v_r_pred = self.networks.v_r.encode_and_forward(s, None, self.device)
            
            with torch.no_grad():
                _, u_r = self.networks.u_r.encode_and_forward(s, None, self.device)
                q_r_for_v = self.networks.q_r.encode_and_forward(s, None, self.device)
                pi_r = self.networks.q_r.get_policy(q_r_for_v)
            
            target_v_r = self.networks.v_r.compute_from_components(
                u_r.squeeze(), q_r_for_v.squeeze(), pi_r.squeeze()
            )
            losses['v_r'] = losses['v_r'] + (v_r_pred.squeeze() - target_v_r) ** 2
        
        # ===== X_h loss (computed on potentially larger batch) =====
        for transition in x_h_batch:
            s = transition.state
            goals = transition.goals
            
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
                x_h_pred = self.networks.x_h.encode_and_forward(
                    s, None, h_x, self.device
                )
                
                with torch.no_grad():
                    # Use target network for more stable X_h targets
                    v_h_e_for_x = self.networks.v_h_e_target.encode_and_forward(
                        s, None, h_x, g_h_x, self.device
                    )
                
                target_x_h = self.networks.x_h.compute_target(v_h_e_for_x.squeeze())
                losses['x_h'] = losses['x_h'] + (x_h_pred.squeeze() - target_x_h) ** 2
                x_h_count += 1
        
        # Average losses over their respective batch sizes
        n = len(batch)
        for k in ['v_h_e', 'u_r', 'q_r', 'v_r']:
            losses[k] = losses[k] / n
        # X_h uses its own count (may be from larger batch)
        if x_h_count > 0:
            losses['x_h'] = losses['x_h'] / x_h_count
        
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
        loss_names = list(losses.keys())
        for i, name in enumerate(loss_names):
            loss = losses[name]
            if loss.requires_grad:
                self.optimizers[name].zero_grad()
                # Retain graph for all but the last loss to allow multiple backwards
                retain = (i < len(loss_names) - 1)
                loss.backward(retain_graph=retain)
                
                # Apply gradient clipping for X_h if configured
                if name == 'x_h' and self.config.x_h_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.networks.x_h.parameters(), 
                        self.config.x_h_grad_clip
                    )
                
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
            'v_r': self.networks.v_r,
            'v_h_e': self.networks.v_h_e,
            'x_h': self.networks.x_h,
            'u_r': self.networks.u_r,
        }
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
            'v_h_e': [], 'x_h': [], 'u_r': [], 'q_r': [], 'v_r': []
        }
        episode_grad_norms = {
            'v_h_e': [], 'x_h': [], 'u_r': [], 'q_r': [], 'v_r': []
        }
        
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
