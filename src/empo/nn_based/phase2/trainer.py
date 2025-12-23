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

from .config import Phase2Config
from .replay_buffer import Phase2Transition, Phase2ReplayBuffer
from .robot_q_network import BaseRobotQNetwork
from .human_goal_ability import BaseHumanGoalAchievementNetwork
from .aggregate_goal_ability import BaseAggregateGoalAbilityNetwork
from .intrinsic_reward_network import BaseIntrinsicRewardNetwork
from .robot_value_network import BaseRobotValueNetwork


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
    """
    
    def __init__(
        self,
        networks: Phase2Networks,
        config: Phase2Config,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        human_policy_prior: Callable,
        goal_sampler: Callable,
        device: str = 'cpu'
    ):
        self.networks = networks
        self.config = config
        self.human_agent_indices = human_agent_indices
        self.robot_agent_indices = robot_agent_indices
        self.human_policy_prior = human_policy_prior
        self.goal_sampler = goal_sampler
        self.device = device
        
        # Initialize target networks
        self._init_target_networks()
        
        # Initialize optimizers
        self.optimizers = self._init_optimizers()
        
        # Replay buffer
        self.replay_buffer = Phase2ReplayBuffer(capacity=config.buffer_size)
        
        # Training step counter
        self.total_steps = 0
    
    def _init_target_networks(self):
        """Initialize target networks as copies of main networks."""
        self.networks.v_r_target = copy.deepcopy(self.networks.v_r)
        self.networks.v_h_e_target = copy.deepcopy(self.networks.v_h_e)
        
        # Freeze target networks
        for param in self.networks.v_r_target.parameters():
            param.requires_grad = False
        for param in self.networks.v_h_e_target.parameters():
            param.requires_grad = False
    
    def _init_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Initialize optimizers for each network."""
        return {
            'q_r': optim.Adam(self.networks.q_r.parameters(), lr=self.config.lr_q_r),
            'v_r': optim.Adam(self.networks.v_r.parameters(), lr=self.config.lr_v_r),
            'v_h_e': optim.Adam(self.networks.v_h_e.parameters(), lr=self.config.lr_v_h_e),
            'x_h': optim.Adam(self.networks.x_h.parameters(), lr=self.config.lr_x_h),
            'u_r': optim.Adam(self.networks.u_r.parameters(), lr=self.config.lr_u_r),
        }
    
    def update_target_networks(self):
        """Update target networks (hard copy)."""
        self.networks.v_r_target.load_state_dict(self.networks.v_r.state_dict())
        self.networks.v_h_e_target.load_state_dict(self.networks.v_h_e.state_dict())
    
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
        # Sample actions
        robot_action = self.sample_robot_action(state)
        human_actions = self.sample_human_actions(state, goals)
        
        # Step environment
        next_state = self.step_environment(state, robot_action, human_actions)
        
        # Create transition
        transition = Phase2Transition(
            state=state,
            robot_action=robot_action,
            goals=goals.copy(),
            human_actions=human_actions,
            next_state=next_state
        )
        
        return transition, next_state
    
    def compute_losses(
        self,
        batch: List[Phase2Transition]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all networks.
        
        Args:
            batch: List of transitions.
        
        Returns:
            Dict mapping loss names to loss tensors.
        """
        losses = {
            'v_h_e': torch.tensor(0.0, device=self.device),
            'x_h': torch.tensor(0.0, device=self.device),
            'u_r': torch.tensor(0.0, device=self.device),
            'q_r': torch.tensor(0.0, device=self.device),
            'v_r': torch.tensor(0.0, device=self.device),
        }
        
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
            
            # ===== X_h loss (sample one human) =====
            h_sampled = random.choice(self.human_agent_indices)
            g_h_sampled = self.goal_sampler(s, h_sampled)
            
            x_h_pred = self.networks.x_h.encode_and_forward(
                s, None, h_sampled, self.device
            )
            
            with torch.no_grad():
                v_h_e_for_x = self.networks.v_h_e.encode_and_forward(
                    s, None, h_sampled, g_h_sampled, self.device
                )
            
            target_x_h = self.networks.x_h.compute_target(v_h_e_for_x.squeeze())
            losses['x_h'] = losses['x_h'] + (x_h_pred.squeeze() - target_x_h) ** 2
            
            # ===== U_r loss (based on y) =====
            y_pred, _ = self.networks.u_r.encode_and_forward(s, None, self.device)
            
            with torch.no_grad():
                x_h_for_u = self.networks.x_h.encode_and_forward(
                    s, None, h_sampled, self.device
                )
            
            target_y = self.networks.u_r.compute_target_y(x_h_for_u.squeeze())
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
        
        # Average losses over batch
        n = len(batch)
        return {k: v / n for k, v in losses.items()}
    
    def training_step(self) -> Dict[str, float]:
        """
        Perform one training step (sample batch, compute losses, update).
        
        Returns:
            Dict of loss values.
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Compute losses
        losses = self.compute_losses(batch)
        
        # Update each network
        loss_values = {}
        for name, loss in losses.items():
            if loss.requires_grad:
                self.optimizers[name].zero_grad()
                loss.backward()
                self.optimizers[name].step()
            loss_values[name] = loss.item()
        
        # Update target networks periodically
        if self.total_steps % self.config.v_r_target_update_freq == 0:
            self.update_target_networks()
        
        return loss_values
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode.
        
        Returns:
            Dict of average loss values for the episode.
        """
        state = self.reset_environment()
        
        # Sample initial goals
        goals = {h: self.goal_sampler(state, h) for h in self.human_agent_indices}
        
        episode_losses = {
            'v_h_e': [], 'x_h': [], 'u_r': [], 'q_r': [], 'v_r': []
        }
        
        for step in range(self.config.steps_per_episode):
            # Collect transition
            transition, next_state = self.collect_transition(state, goals)
            self.replay_buffer.push(
                transition.state,
                transition.robot_action,
                transition.goals,
                transition.human_actions,
                transition.next_state
            )
            
            # Training updates
            for _ in range(self.config.updates_per_step):
                losses = self.training_step()
                for k, v in losses.items():
                    episode_losses[k].append(v)
            
            self.total_steps += 1
            state = next_state
            
            # Resample goals with some probability
            if random.random() < self.config.goal_resample_prob:
                goals = {h: self.goal_sampler(state, h) for h in self.human_agent_indices}
        
        # Average losses
        return {
            k: sum(v) / len(v) if v else 0.0
            for k, v in episode_losses.items()
        }
    
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
        for episode in range(num_episodes):
            episode_losses = self.train_episode()
            history.append(episode_losses)
            
            if episode % 100 == 0:
                print(f"Episode {episode}: {episode_losses}")
        
        return history
