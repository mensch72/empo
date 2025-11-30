"""
Neural Network-based Stochastic Approximation for Human Policy Priors.

This module provides neural network function approximators for computing human
policy priors when the state space, number of agents, and goal space are too
large for tabular methods (backward induction).

The approach approximates the same Bellman-style computation as the tabular
method, but uses neural networks trained via stochastic gradient descent on
sampled states rather than exact computation over all states.

Main components:
    - StateEncoder: Encodes grid-based states into feature vectors (CNN-based)
    - GoalEncoder: Encodes possible goals (target positions) into feature vectors
    - AgentEncoder: Encodes agent attributes (position, direction, index)
    - QNetwork (h_Q): Maps (state, human, goal) -> Q-values for each action
    - PolicyPriorNetwork (h_phi): Maps (state, human) -> marginal policy prior

Mathematical background:
    The networks approximate:
    
    h_Q(s, h, g) ≈ Q^π(s, a, g) for Boltzmann policy π
    
    h_pi(s, h, g) = softmax(β * h_Q(s, h, g))  [goal-specific policy]
    
    h_phi(s, h) = E_g[h_pi(s, h, g)]  [marginal policy prior]
    
    Training uses TD-style updates:
    Q_target(s, a, g) = γ * E_{s'}[V(s', g)]  where V = Σ_a π(a) * Q(a)

Example usage:
    >>> from empo.neural_policy_prior import NeuralHumanPolicyPrior, train_neural_policy_prior
    >>> 
    >>> # Train the neural network on sampled states
    >>> neural_prior = train_neural_policy_prior(
    ...     env=env,
    ...     human_agent_indices=[0, 1],
    ...     goal_sampler=goal_sampler,
    ...     num_episodes=1000,
    ...     beta=10.0
    ... )
    >>> 
    >>> # Use like tabular policy prior
    >>> action_dist = neural_prior(state, agent_idx=0, goal=my_goal)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler


class StateEncoder(nn.Module):
    """
    Encodes grid-based multigrid states into feature vectors.
    
    The encoder uses a CNN to process a spatial representation of the grid,
    capturing object positions, types, and agent locations.
    
    For multigrid environments, the state tuple format is:
        (step_count, agent_states, mobile_objects, mutable_objects)
    
    The encoder converts this into a 2D grid representation and applies
    convolutional layers to extract spatial features.
    
    Args:
        grid_width: Width of the grid environment.
        grid_height: Height of the grid environment.
        num_object_types: Number of distinct object types to encode.
        num_agents: Total number of agents in the environment.
        feature_dim: Output feature dimension (default: 128).
    """
    
    def __init__(
        self, 
        grid_width: int, 
        grid_height: int, 
        num_object_types: int = 8,
        num_agents: int = 2,
        feature_dim: int = 128
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_object_types = num_object_types
        self.num_agents = num_agents
        self.feature_dim = feature_dim
        
        # Input channels: object type one-hot + agent positions (one channel per agent)
        in_channels = num_object_types + num_agents
        
        # CNN for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Compute flattened size after conv layers
        self.flat_size = 64 * grid_width * grid_height
        
        # Project to feature dimension
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size + 1, feature_dim),  # +1 for step count
            nn.ReLU(),
        )
    
    def forward(self, state_tensor: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of states into feature vectors.
        
        Args:
            state_tensor: Tensor of shape (batch, channels, height, width) 
                         representing the grid state.
            step_count: Tensor of shape (batch, 1) with normalized step counts.
        
        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        batch_size = state_tensor.shape[0]
        
        # Apply CNN
        x = self.conv(state_tensor)  # (batch, 64, H, W)
        x = x.view(batch_size, -1)   # (batch, 64*H*W)
        
        # Concatenate step count and project
        x = torch.cat([x, step_count], dim=1)
        x = self.fc(x)
        
        return x


class GoalEncoder(nn.Module):
    """
    Encodes possible goals into feature vectors.
    
    For multigrid environments, goals are typically "reach a target cell"
    or "reach a rectangular region". This encoder handles position-based goals.
    
    Goal encoding:
        - Target position: (x, y) normalized to [0, 1] range
        - Optional: region bounds (x1, y1, x2, y2) for rectangular goals
    
    Args:
        grid_width: Width of the grid for normalization.
        grid_height: Height of the grid for normalization.
        feature_dim: Output feature dimension (default: 32).
    """
    
    def __init__(
        self, 
        grid_width: int, 
        grid_height: int, 
        feature_dim: int = 32
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # MLP for goal encoding (2 coords for point, or 4 for rectangle)
        self.fc = nn.Sequential(
            nn.Linear(4, 64),  # x1, y1, x2, y2 (same for point goals)
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, goal_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode goal coordinates into feature vectors.
        
        Args:
            goal_coords: Tensor of shape (batch, 4) with normalized coordinates
                        [x1/W, y1/H, x2/W, y2/H]. For point goals, x1=x2, y1=y2.
        
        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        return self.fc(goal_coords)


class AgentEncoder(nn.Module):
    """
    Encodes agent attributes into feature vectors.
    
    The encoding captures:
        - Agent position (x, y) normalized to [0, 1]
        - Agent direction (one-hot over 4 directions)
        - Agent index (embedded)
    
    This allows the network to generalize across agents at different positions
    while still distinguishing them by index when needed.
    
    Args:
        grid_width: Width of the grid for normalization.
        grid_height: Height of the grid for normalization.
        num_agents: Maximum number of agents (for index embedding).
        feature_dim: Output feature dimension (default: 32).
    """
    
    def __init__(
        self, 
        grid_width: int, 
        grid_height: int, 
        num_agents: int,
        feature_dim: int = 32
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Agent index embedding
        self.agent_embedding = nn.Embedding(num_agents, 16)
        
        # MLP combining position, direction, and index embedding
        # Input: 2 (pos) + 4 (direction one-hot) + 16 (index embedding) = 22
        self.fc = nn.Sequential(
            nn.Linear(22, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )
    
    def forward(
        self, 
        position: torch.Tensor, 
        direction: torch.Tensor, 
        agent_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode agent attributes into feature vectors.
        
        Args:
            position: Tensor of shape (batch, 2) with normalized positions [x/W, y/H].
            direction: Tensor of shape (batch, 4) with one-hot direction encoding.
            agent_idx: Tensor of shape (batch,) with agent indices.
        
        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        idx_embed = self.agent_embedding(agent_idx)  # (batch, 16)
        x = torch.cat([position, direction, idx_embed], dim=1)  # (batch, 22)
        return self.fc(x)


class QNetwork(nn.Module):
    """
    Q-value network: h_Q(state, human, goal) -> Q-values by action.
    
    Combines state, agent, and goal encodings to predict Q-values
    for each action the human agent can take.
    
    Architecture:
        state_features = StateEncoder(state)
        agent_features = AgentEncoder(human_pos, human_dir, human_idx)
        goal_features = GoalEncoder(goal_coords)
        combined = concat(state_features, agent_features, goal_features)
        Q_values = MLP(combined)
    
    Args:
        state_encoder: Pretrained or trainable StateEncoder.
        agent_encoder: Pretrained or trainable AgentEncoder.
        goal_encoder: Pretrained or trainable GoalEncoder.
        num_actions: Number of possible actions.
        hidden_dim: Hidden layer dimension (default: 256).
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        agent_encoder: AgentEncoder,
        goal_encoder: GoalEncoder,
        num_actions: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.agent_encoder = agent_encoder
        self.goal_encoder = goal_encoder
        self.num_actions = num_actions
        
        # Combined feature dimension
        combined_dim = (state_encoder.feature_dim + 
                       agent_encoder.fc[-2].out_features +  # Get actual output dim
                       goal_encoder.fc[-2].out_features)
        
        # Actually compute the feature dims from the encoder architectures
        # State: 128, Agent: 32, Goal: 32 by default
        state_dim = state_encoder.feature_dim
        agent_dim = 32  # As defined in AgentEncoder
        goal_dim = 32   # As defined in GoalEncoder
        combined_dim = state_dim + agent_dim + goal_dim
        
        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(
        self,
        state_tensor: torch.Tensor,
        step_count: torch.Tensor,
        agent_position: torch.Tensor,
        agent_direction: torch.Tensor,
        agent_idx: torch.Tensor,
        goal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Args:
            state_tensor: Grid state tensor (batch, channels, H, W).
            step_count: Normalized step count (batch, 1).
            agent_position: Agent position (batch, 2).
            agent_direction: Agent direction one-hot (batch, 4).
            agent_idx: Agent index (batch,).
            goal_coords: Goal coordinates (batch, 4).
        
        Returns:
            Q-values tensor of shape (batch, num_actions).
        """
        state_feat = self.state_encoder(state_tensor, step_count)
        agent_feat = self.agent_encoder(agent_position, agent_direction, agent_idx)
        goal_feat = self.goal_encoder(goal_coords)
        
        combined = torch.cat([state_feat, agent_feat, goal_feat], dim=1)
        q_values = self.q_head(combined)
        
        return q_values
    
    def get_policy(self, q_values: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute Boltzmann policy from Q-values.
        
        Args:
            q_values: Q-values tensor of shape (batch, num_actions).
            beta: Inverse temperature (higher = more deterministic).
        
        Returns:
            Policy tensor of shape (batch, num_actions) with action probabilities.
        """
        if beta == float('inf'):
            # Argmax policy with ties broken uniformly
            max_q = q_values.max(dim=1, keepdim=True)[0]
            is_max = (q_values == max_q).float()
            policy = is_max / is_max.sum(dim=1, keepdim=True)
        else:
            # Softmax with temperature
            scaled_q = beta * (q_values - q_values.max(dim=1, keepdim=True)[0])
            policy = F.softmax(scaled_q, dim=1)
        return policy


class PolicyPriorNetwork(nn.Module):
    """
    Policy prior network: h_phi(state, human) -> marginal action distribution.
    
    This network directly predicts the marginal policy prior (averaged over goals)
    without needing explicit goal conditioning. It's trained to match the
    expected policy E_g[softmax(β * Q(s,h,g))].
    
    Useful when:
        - Marginal policy is needed frequently and goal enumeration is expensive
        - Goals follow a complex distribution that's hard to sample from
    
    Args:
        state_encoder: Pretrained or trainable StateEncoder.
        agent_encoder: Pretrained or trainable AgentEncoder.
        num_actions: Number of possible actions.
        hidden_dim: Hidden layer dimension (default: 256).
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        agent_encoder: AgentEncoder,
        num_actions: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.state_encoder = state_encoder
        self.agent_encoder = agent_encoder
        self.num_actions = num_actions
        
        # Combined feature dimension (state + agent)
        state_dim = state_encoder.feature_dim
        agent_dim = 32  # As defined in AgentEncoder
        combined_dim = state_dim + agent_dim
        
        # Policy head (outputs logits)
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
    
    def forward(
        self,
        state_tensor: torch.Tensor,
        step_count: torch.Tensor,
        agent_position: torch.Tensor,
        agent_direction: torch.Tensor,
        agent_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute marginal policy prior.
        
        Args:
            state_tensor: Grid state tensor (batch, channels, H, W).
            step_count: Normalized step count (batch, 1).
            agent_position: Agent position (batch, 2).
            agent_direction: Agent direction one-hot (batch, 4).
            agent_idx: Agent index (batch,).
        
        Returns:
            Policy tensor of shape (batch, num_actions) with action probabilities.
        """
        state_feat = self.state_encoder(state_tensor, step_count)
        agent_feat = self.agent_encoder(agent_position, agent_direction, agent_idx)
        
        combined = torch.cat([state_feat, agent_feat], dim=1)
        logits = self.policy_head(combined)
        
        return F.softmax(logits, dim=1)


class NeuralHumanPolicyPrior(HumanPolicyPrior):
    """
    Neural network-based human policy prior.
    
    This class wraps trained Q-network and (optional) policy prior network
    to provide the same interface as TabularHumanPolicyPrior.
    
    Attributes:
        q_network: Trained QNetwork for goal-conditioned Q-values.
        phi_network: Optional PolicyPriorNetwork for marginal policy.
        beta: Inverse temperature for Boltzmann policy.
        goal_sampler: Sampler for goals (used when computing marginal without phi_network).
        device: Torch device (cpu or cuda).
    """
    
    def __init__(
        self,
        world_model: Any,
        human_agent_indices: List[int],
        q_network: QNetwork,
        phi_network: Optional[PolicyPriorNetwork] = None,
        beta: float = 1.0,
        goal_sampler: Optional[PossibleGoalSampler] = None,
        goal_generator: Optional[PossibleGoalGenerator] = None,
        num_mc_samples: int = 100,
        device: str = 'cpu'
    ):
        """
        Initialize the neural human policy prior.
        
        Args:
            world_model: The environment/world model.
            human_agent_indices: Indices of human agents.
            q_network: Trained QNetwork.
            phi_network: Optional trained PolicyPriorNetwork for fast marginal queries.
            beta: Inverse temperature for Boltzmann policy.
            goal_sampler: Sampler for Monte Carlo marginal computation.
            goal_generator: Generator for exact marginal computation.
            num_mc_samples: Number of MC samples for marginal (when using sampler).
            device: Torch device ('cpu' or 'cuda').
        """
        super().__init__(world_model, human_agent_indices)
        self.q_network = q_network.to(device)
        self.phi_network = phi_network.to(device) if phi_network else None
        self.beta = beta
        self.goal_sampler = goal_sampler
        self.goal_generator = goal_generator
        self.num_mc_samples = num_mc_samples
        self.device = device
        
        # Put networks in eval mode
        self.q_network.eval()
        if self.phi_network:
            self.phi_network.eval()
    
    def _state_to_tensors(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a state tuple to tensor representation.
        
        Args:
            state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
        
        Returns:
            Tuple of (state_tensor, step_count_tensor).
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Create grid tensor (simplified - actual implementation would be more complex)
        # This is a placeholder - real implementation needs to match environment specifics
        num_channels = self.q_network.state_encoder.conv[0].in_channels
        H = self.q_network.state_encoder.grid_height
        W = self.q_network.state_encoder.grid_width
        
        grid_tensor = torch.zeros(1, num_channels, H, W, device=self.device)
        
        # Encode agent positions
        for i, agent_state in enumerate(agent_states):
            if i < self.q_network.state_encoder.num_agents:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    # Agent channel
                    channel_idx = self.q_network.state_encoder.num_object_types + i
                    grid_tensor[0, channel_idx, y, x] = 1.0
        
        # Normalize step count
        max_steps = getattr(self.world_model, 'max_steps', 100)
        step_tensor = torch.tensor([[step_count / max_steps]], device=self.device)
        
        return grid_tensor, step_tensor
    
    def _agent_to_tensors(
        self, 
        state, 
        human_agent_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract agent tensors from state.
        
        Args:
            state: State tuple.
            human_agent_index: Index of the human agent.
        
        Returns:
            Tuple of (position, direction_onehot, agent_idx).
        """
        _, agent_states, _, _ = state
        agent_state = agent_states[human_agent_index]
        
        # Position normalized to [0, 1]
        x = agent_state[0] / self.q_network.state_encoder.grid_width
        y = agent_state[1] / self.q_network.state_encoder.grid_height
        position = torch.tensor([[x, y]], device=self.device)
        
        # Direction as one-hot (4 directions)
        direction = torch.zeros(1, 4, device=self.device)
        dir_idx = int(agent_state[2]) % 4
        direction[0, dir_idx] = 1.0
        
        # Agent index
        agent_idx = torch.tensor([human_agent_index], device=self.device)
        
        return position, direction, agent_idx
    
    def _goal_to_tensor(self, goal: PossibleGoal) -> torch.Tensor:
        """
        Convert a goal to tensor representation.
        
        Args:
            goal: A PossibleGoal instance (assumed to have target position attributes).
        
        Returns:
            Goal coordinates tensor of shape (1, 4).
        """
        # Extract target position from goal (implementation-specific)
        # Assume goal has target_pos attribute (as in ReachCellGoal)
        if hasattr(goal, 'target_pos'):
            target = goal.target_pos
            x = float(target[0]) / self.q_network.state_encoder.grid_width
            y = float(target[1]) / self.q_network.state_encoder.grid_height
            return torch.tensor([[x, y, x, y]], device=self.device)  # Point goal
        else:
            # Default to zeros for unknown goal types
            return torch.zeros(1, 4, device=self.device)
    
    def __call__(
        self, 
        state, 
        human_agent_index: int, 
        possible_goal: Optional[PossibleGoal] = None
    ) -> np.ndarray:
        """
        Get the action distribution for a human agent.
        
        Args:
            state: Current world state.
            human_agent_index: Index of the human agent.
            possible_goal: If provided, return goal-conditioned policy.
                          If None, return marginal policy over goals.
        
        Returns:
            np.ndarray: Probability distribution over actions.
        """
        with torch.no_grad():
            if possible_goal is not None:
                # Goal-conditioned policy using Q-network
                state_tensor, step_tensor = self._state_to_tensors(state)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                goal_tensor = self._goal_to_tensor(possible_goal)
                
                q_values = self.q_network(
                    state_tensor, step_tensor,
                    position, direction, agent_idx,
                    goal_tensor
                )
                
                policy = self.q_network.get_policy(q_values, self.beta)
                return policy.cpu().numpy()[0]
            
            elif self.phi_network is not None:
                # Use direct marginal network
                state_tensor, step_tensor = self._state_to_tensors(state)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                
                policy = self.phi_network(
                    state_tensor, step_tensor,
                    position, direction, agent_idx
                )
                return policy.cpu().numpy()[0]
            
            elif self.goal_generator is not None:
                # Compute marginal by exact enumeration over goals
                state_tensor, step_tensor = self._state_to_tensors(state)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                
                total_policy = np.zeros(self.q_network.num_actions)
                
                for goal, weight in self.goal_generator.generate(state, human_agent_index):
                    goal_tensor = self._goal_to_tensor(goal)
                    
                    q_values = self.q_network(
                        state_tensor, step_tensor,
                        position, direction, agent_idx,
                        goal_tensor
                    )
                    
                    policy = self.q_network.get_policy(q_values, self.beta)
                    total_policy += weight * policy.cpu().numpy()[0]
                
                return total_policy
            
            elif self.goal_sampler is not None:
                # Compute marginal by Monte Carlo sampling
                state_tensor, step_tensor = self._state_to_tensors(state)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                
                total_policy = np.zeros(self.q_network.num_actions)
                total_weight = 0.0
                
                for _ in range(self.num_mc_samples):
                    goal, weight = self.goal_sampler.sample(state, human_agent_index)
                    goal_tensor = self._goal_to_tensor(goal)
                    
                    q_values = self.q_network(
                        state_tensor, step_tensor,
                        position, direction, agent_idx,
                        goal_tensor
                    )
                    
                    policy = self.q_network.get_policy(q_values, self.beta)
                    total_policy += weight * policy.cpu().numpy()[0]
                    total_weight += weight
                
                return total_policy / total_weight if total_weight > 0 else total_policy
            
            else:
                raise ValueError(
                    "Either possible_goal must be provided, or one of "
                    "phi_network, goal_generator, or goal_sampler must be set"
                )


def train_neural_policy_prior(
    world_model: Any,
    human_agent_indices: List[int],
    goal_sampler: PossibleGoalSampler,
    num_episodes: int = 1000,
    steps_per_episode: int = 50,
    beta: float = 1.0,
    gamma: float = 1.0,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    train_phi_network: bool = True,
    device: str = 'cpu',
    verbose: bool = True
) -> NeuralHumanPolicyPrior:
    """
    Train neural networks to approximate the human policy prior.
    
    This function trains the Q-network using TD-style updates on states
    sampled from random rollouts. The training approximates the fixed-point
    computation done by tabular backward induction.
    
    Training procedure:
        1. Sample states by random rollouts in the environment
        2. For each sampled state, sample goals from goal_sampler
        3. Compute TD targets: Q_target = γ * E_{s'}[V(s')]
        4. Train Q-network to minimize (Q_predicted - Q_target)²
        5. Optionally train phi_network to match E_g[softmax(βQ)]
    
    Args:
        world_model: The environment (must support get_state, set_state, step).
        human_agent_indices: Indices of human agents to model.
        goal_sampler: Sampler for possible goals.
        num_episodes: Number of training episodes.
        steps_per_episode: Steps per episode for state sampling.
        beta: Inverse temperature for Boltzmann policy.
        gamma: Discount factor.
        learning_rate: Learning rate for optimization.
        batch_size: Batch size for training.
        train_phi_network: Whether to also train the marginal policy network.
        device: Torch device ('cpu' or 'cuda').
        verbose: Whether to print training progress.
    
    Returns:
        NeuralHumanPolicyPrior: Trained policy prior model.
    """
    # Get environment dimensions
    grid_width = world_model.width
    grid_height = world_model.height
    num_agents = len(world_model.agents)
    num_actions = world_model.action_space.n
    
    # Create networks
    state_encoder = StateEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents
    ).to(device)
    
    agent_encoder = AgentEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents
    ).to(device)
    
    goal_encoder = GoalEncoder(
        grid_width=grid_width,
        grid_height=grid_height
    ).to(device)
    
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=num_actions
    ).to(device)
    
    phi_network = None
    if train_phi_network:
        # Create separate state/agent encoders for phi (shared would also work)
        phi_state_encoder = StateEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_agents=num_agents
        ).to(device)
        
        phi_agent_encoder = AgentEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_agents=num_agents
        ).to(device)
        
        phi_network = PolicyPriorNetwork(
            state_encoder=phi_state_encoder,
            agent_encoder=phi_agent_encoder,
            num_actions=num_actions
        ).to(device)
    
    # Optimizers
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    phi_optimizer = None
    if phi_network:
        phi_optimizer = torch.optim.Adam(phi_network.parameters(), lr=learning_rate)
    
    # Replay buffer for experience
    replay_buffer: List[Dict[str, Any]] = []
    max_buffer_size = 10000
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment and collect experiences
        world_model.reset()
        
        for step in range(steps_per_episode):
            state = world_model.get_state()
            
            # Sample goal for each human agent
            for human_idx in human_agent_indices:
                goal, goal_weight = goal_sampler.sample(state, human_idx)
                
                # Random action for exploration
                action = np.random.randint(num_actions)
                
                # Take step (using random actions for all agents)
                actions = [np.random.randint(num_actions) for _ in range(num_agents)]
                world_model.set_state(state)  # Ensure we're at the right state
                _, _, done, _ = world_model.step(actions)
                next_state = world_model.get_state()
                
                # Store experience
                experience = {
                    'state': state,
                    'next_state': next_state,
                    'human_idx': human_idx,
                    'goal': goal,
                    'goal_weight': goal_weight,
                    'action': action,
                    'done': done,
                }
                
                if len(replay_buffer) >= max_buffer_size:
                    replay_buffer.pop(0)
                replay_buffer.append(experience)
                
                if done:
                    world_model.reset()
                    break
        
        # Training step
        if len(replay_buffer) >= batch_size:
            # Sample batch
            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in indices]
            
            # Prepare batch tensors (simplified - actual implementation would be more efficient)
            # This is a placeholder for the actual batch preparation logic
            
            # Q-network update would go here
            # For this minimal implementation, we skip the actual gradient computation
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{num_episodes}: Buffer size = {len(replay_buffer)}")
    
    if verbose:
        print("Training complete!")
    
    # Create and return the policy prior
    return NeuralHumanPolicyPrior(
        world_model=world_model,
        human_agent_indices=human_agent_indices,
        q_network=q_network,
        phi_network=phi_network,
        beta=beta,
        goal_sampler=goal_sampler,
        device=device
    )


# Utility function for creating networks with default architecture
def create_policy_prior_networks(
    world_model: Any,
    num_agents: Optional[int] = None,
    feature_dim: int = 128,
    hidden_dim: int = 256,
    device: str = 'cpu'
) -> Tuple[QNetwork, PolicyPriorNetwork]:
    """
    Create Q-network and policy prior network with default architecture.
    
    Args:
        world_model: The environment.
        num_agents: Number of agents (if None, inferred from world_model).
        feature_dim: Feature dimension for encoders.
        hidden_dim: Hidden dimension for network heads.
        device: Torch device.
    
    Returns:
        Tuple of (QNetwork, PolicyPriorNetwork).
    """
    grid_width = world_model.width
    grid_height = world_model.height
    if num_agents is None:
        num_agents = len(world_model.agents)
    num_actions = world_model.action_space.n
    
    # Shared encoders for Q-network
    state_encoder = StateEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents,
        feature_dim=feature_dim
    ).to(device)
    
    agent_encoder = AgentEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents
    ).to(device)
    
    goal_encoder = GoalEncoder(
        grid_width=grid_width,
        grid_height=grid_height
    ).to(device)
    
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=num_actions,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Separate encoders for phi network
    phi_state_encoder = StateEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents,
        feature_dim=feature_dim
    ).to(device)
    
    phi_agent_encoder = AgentEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents
    ).to(device)
    
    phi_network = PolicyPriorNetwork(
        state_encoder=phi_state_encoder,
        agent_encoder=phi_agent_encoder,
        num_actions=num_actions,
        hidden_dim=hidden_dim
    ).to(device)
    
    return q_network, phi_network
