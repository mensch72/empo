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
        self.feature_dim = feature_dim
        
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
        self.feature_dim = feature_dim
        
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
        
        # Combined feature dimension from encoder outputs
        combined_dim = (state_encoder.feature_dim + 
                       agent_encoder.feature_dim +
                       goal_encoder.feature_dim)
        
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
        
        # Combined feature dimension from encoder outputs
        combined_dim = state_encoder.feature_dim + agent_encoder.feature_dim
        
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


def _state_to_tensors_static(
    state,
    grid_width: int,
    grid_height: int,
    num_agents: int,
    num_object_types: int = 8,
    max_steps: int = 100,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a state tuple to tensor representation.
    
    Args:
        state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
        grid_width: Width of the grid.
        grid_height: Height of the grid.
        num_agents: Number of agents.
        num_object_types: Number of object types for encoding.
        max_steps: Maximum steps for normalization.
        device: Torch device.
    
    Returns:
        Tuple of (grid_tensor, step_count_tensor).
    """
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    num_channels = num_object_types + num_agents
    grid_tensor = torch.zeros(1, num_channels, grid_height, grid_width, device=device)
    
    # Encode agent positions
    for i, agent_state in enumerate(agent_states):
        if i < num_agents:
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                channel_idx = num_object_types + i
                grid_tensor[0, channel_idx, y, x] = 1.0
    
    # Normalize step count
    step_tensor = torch.tensor([[step_count / max_steps]], device=device, dtype=torch.float32)
    
    return grid_tensor, step_tensor


def _agent_to_tensors_static(
    state,
    human_agent_index: int,
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract agent tensors from state.
    
    Args:
        state: State tuple.
        human_agent_index: Index of the human agent.
        grid_width: Width of the grid (must be > 0).
        grid_height: Height of the grid (must be > 0).
        device: Torch device.
    
    Returns:
        Tuple of (position, direction_onehot, agent_idx).
    
    Raises:
        ValueError: If grid_width or grid_height <= 0.
    """
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(f"Grid dimensions must be positive, got width={grid_width}, height={grid_height}")
    
    _, agent_states, _, _ = state
    agent_state = agent_states[human_agent_index]
    
    # Position normalized to [0, 1]
    x = float(agent_state[0]) / grid_width
    y = float(agent_state[1]) / grid_height
    position = torch.tensor([[x, y]], device=device, dtype=torch.float32)
    
    # Direction as one-hot (4 directions)
    direction = torch.zeros(1, 4, device=device, dtype=torch.float32)
    dir_idx = int(agent_state[2]) % 4
    direction[0, dir_idx] = 1.0
    
    # Agent index
    agent_idx = torch.tensor([human_agent_index], device=device, dtype=torch.long)
    
    return position, direction, agent_idx


def _goal_to_tensor_static(
    goal: PossibleGoal,
    grid_width: int,
    grid_height: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convert a goal to tensor representation.
    
    Args:
        goal: A PossibleGoal instance (assumed to have target position attributes).
        grid_width: Width of the grid (must be > 0).
        grid_height: Height of the grid (must be > 0).
        device: Torch device.
    
    Returns:
        Goal coordinates tensor of shape (1, 4).
    
    Raises:
        ValueError: If grid_width or grid_height <= 0.
    """
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError(f"Grid dimensions must be positive, got width={grid_width}, height={grid_height}")
    
    if hasattr(goal, 'target_pos'):
        target = goal.target_pos
        x = float(target[0]) / grid_width
        y = float(target[1]) / grid_height
        return torch.tensor([[x, y, x, y]], device=device, dtype=torch.float32)
    else:
        return torch.zeros(1, 4, device=device, dtype=torch.float32)


def train_neural_policy_prior(
    world_model: Any,
    human_agent_indices: List[int],
    goal_sampler: PossibleGoalSampler,
    num_episodes: int = 1000,
    steps_per_episode: int = 50,
    beta: float = 1.0,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    train_phi_network: bool = True,
    epsilon: float = 0.3,
    device: str = 'cpu',
    verbose: bool = True
) -> NeuralHumanPolicyPrior:
    """
    Train neural networks to approximate the human policy prior.
    
    This function trains the Q-network using Monte Carlo returns on trajectories
    collected from rollouts. The training approximates the fixed-point
    computation done by tabular backward induction.
    
    Training procedure:
        1. For each episode, sample random goals for each human agent
        2. Collect trajectory using epsilon-greedy exploration
        3. Compute Monte Carlo returns: G_t = r_t + γ*G_{t+1}
        4. Train Q-network to minimize MSE loss: L = (Q(s,a,g) - G_t)²
        5. Optionally train phi_network to match E_g[softmax(βQ)]
    
    Loss function:
        Q-network: L_Q = E[(Q(s,a,g) - G_t)²] where G_t is Monte Carlo return
        Phi-network: L_phi = KL(phi(s,h) || E_g[softmax(β*Q(s,h,g))])
    
    Args:
        world_model: The environment (must support get_state, set_state, step).
        human_agent_indices: Indices of human agents to model.
        goal_sampler: Sampler for possible goals.
        num_episodes: Number of training episodes.
        steps_per_episode: Steps per episode for state sampling.
        beta: Inverse temperature for Boltzmann policy.
        gamma: Discount factor for returns.
        learning_rate: Learning rate for optimization.
        batch_size: Not used in MC training (kept for API compatibility).
        train_phi_network: Whether to also train the marginal policy network.
        epsilon: Exploration rate for epsilon-greedy policy.
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
    max_steps = getattr(world_model, 'max_steps', 100)
    
    # Create Q-network with encoders
    state_encoder = StateEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents,
        feature_dim=64
    ).to(device)
    
    agent_encoder = AgentEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        num_agents=num_agents,
        feature_dim=32
    ).to(device)
    
    goal_encoder = GoalEncoder(
        grid_width=grid_width,
        grid_height=grid_height,
        feature_dim=32
    ).to(device)
    
    q_network = QNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        num_actions=num_actions,
        hidden_dim=128
    ).to(device)
    
    # Create phi network if requested
    phi_network = None
    phi_optimizer = None
    if train_phi_network:
        phi_state_encoder = StateEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_agents=num_agents,
            feature_dim=64
        ).to(device)
        
        phi_agent_encoder = AgentEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_agents=num_agents,
            feature_dim=32
        ).to(device)
        
        phi_network = PolicyPriorNetwork(
            state_encoder=phi_state_encoder,
            agent_encoder=phi_agent_encoder,
            num_actions=num_actions,
            hidden_dim=128
        ).to(device)
        phi_optimizer = torch.optim.Adam(phi_network.parameters(), lr=learning_rate)
    
    # Q-network optimizer
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Training statistics
    q_losses = []
    phi_losses = []
    
    # Training loop
    for episode in range(num_episodes):
        world_model.reset()
        
        # Sample a random goal for each human agent
        initial_state = world_model.get_state()
        human_goals = {}
        for human_idx in human_agent_indices:
            goal, _ = goal_sampler.sample(initial_state, human_idx)
            human_goals[human_idx] = goal
        
        # Track which humans have reached their goals
        humans_at_goal = {h_idx: False for h_idx in human_agent_indices}
        
        # Collect trajectory
        trajectories = {h_idx: [] for h_idx in human_agent_indices}
        state = initial_state
        
        for step in range(steps_per_episode):
            # Check if humans are at their goals at the START of this step
            _, curr_agent_states, _, _ = state
            for human_idx in human_agent_indices:
                goal = human_goals[human_idx]
                if hasattr(goal, 'target_pos'):
                    curr_pos = curr_agent_states[human_idx]
                    if int(curr_pos[0]) == goal.target_pos[0] and int(curr_pos[1]) == goal.target_pos[1]:
                        humans_at_goal[human_idx] = True
            
            # Get actions for all agents
            actions = []
            
            for agent_idx in range(num_agents):
                if agent_idx in human_agent_indices:
                    # Human uses epsilon-greedy policy based on Q-network
                    goal = human_goals[agent_idx]
                    
                    # Convert state to tensors
                    grid_tensor, step_tensor = _state_to_tensors_static(
                        state, grid_width, grid_height, num_agents,
                        max_steps=max_steps, device=device
                    )
                    position, direction, agent_idx_t = _agent_to_tensors_static(
                        state, agent_idx, grid_width, grid_height, device
                    )
                    goal_tensor = _goal_to_tensor_static(goal, grid_width, grid_height, device)
                    
                    with torch.no_grad():
                        q_values = q_network(
                            grid_tensor, step_tensor,
                            position, direction, agent_idx_t,
                            goal_tensor
                        )
                    
                    # Epsilon-greedy action selection
                    if np.random.random() < epsilon:
                        action = np.random.choice(range(num_actions),p=[0.02,0.19,0.19,0.6])
#                        action = np.random.randint(num_actions)
                    else:
                        policy = F.softmax(beta * q_values, dim=1)
                        action = torch.multinomial(policy, 1).item()
                    
                    # Store transition with at_goal flag
                    trajectories[agent_idx].append({
                        'state': state,
                        'action': action,
                        'goal': goal,
                        'at_goal_before_action': humans_at_goal[agent_idx],
                    })
                else:
                    # Non-human agents use random policy
                    action = np.random.randint(num_actions)
                
                actions.append(action)
            
            # Take step
            _, _, done, _ = world_model.step(actions)
            next_state = world_model.get_state()
            
            # Store rewards with reward shaping for denser feedback
            _, next_agent_states, _, _ = next_state
            _, curr_agent_states, _, _ = state
            
            for human_idx in human_agent_indices:
                goal = human_goals[human_idx]
                was_at_goal = humans_at_goal[human_idx]
                goal_achieved = float(goal.is_achieved(next_state))
                
                if was_at_goal:
                    # If already at goal, give reward 1.0 (goal achieved)
                    reward = 1.0
                    is_terminal = True
                else:
                    # Base reward: 1.0 when goal is reached, 0 otherwise
                    base_reward = goal_achieved
                    
                    # Potential-based reward shaping (Ng et al. 1999):
                    # F(s,a,s') = γ * Φ(s') - Φ(s)
                    # This preserves the optimal policy while providing denser feedback.
                    # We use Φ(s) = -distance(agent, goal) / max_dist as potential function
                    # which is maximal (0) when agent is at the goal.
                    shaping_reward = 0.0
                    if hasattr(goal, 'target_pos'):
                        target = goal.target_pos
                        max_dist = grid_width + grid_height
                        
                        # Potential at current state: Φ(s) = -d(s)/max_dist
                        curr_pos = curr_agent_states[human_idx]
                        curr_dist = abs(curr_pos[0] - target[0]) + abs(curr_pos[1] - target[1])
                        phi_s = -curr_dist / max_dist
                        
                        # Potential at next state: Φ(s') = -d(s')/max_dist  
                        next_pos = next_agent_states[human_idx]
                        next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                        phi_s_prime = -next_dist / max_dist
                        
                        # Shaping: γ * Φ(s') - Φ(s) = γ * (-d(s')/max) - (-d(s)/max)
                        #        = (d(s) - γ*d(s')) / max_dist
                        shaping_reward = gamma * phi_s_prime - phi_s

                        # TODO: generalize this for other goal types, especially for rectangular goals
                    else:
                        print("Warning: Goal does not have target_pos attribute for shaping reward.")

                    reward = base_reward + shaping_reward
                    is_terminal = goal_achieved > 0
                
                # Update goal reached status
                if goal_achieved > 0:
                    humans_at_goal[human_idx] = True
                
                if trajectories[human_idx]:
                    trajectories[human_idx][-1]['reward'] = reward
                    trajectories[human_idx][-1]['next_state'] = next_state
                    trajectories[human_idx][-1]['done'] = done or is_terminal
            
            if done:
                break
            
            state = next_state
        
        # ====================================================================
        # GRADIENT UPDATE: Train Q-network using TD(0) learning
        # This is more stable than Monte Carlo for continuous control
        # ====================================================================
        episode_q_loss = 0.0
        num_updates = 0
        
        for human_idx in human_agent_indices:
            trajectory = trajectories[human_idx]
            if len(trajectory) == 0:
                continue
            
            # TD(0) update for each transition
            q_optimizer.zero_grad()
            losses = []
            
            for i, t in enumerate(trajectory):
                reward = t.get('reward', 0.0)
                done_flag = t.get('done', False)
                was_at_goal = t.get('at_goal_before_action', False)
                
                # Convert current state to tensors
                grid_tensor, step_tensor = _state_to_tensors_static(
                    t['state'], grid_width, grid_height, num_agents,
                    max_steps=max_steps, device=device
                )
                position, direction, agent_idx_t = _agent_to_tensors_static(
                    t['state'], human_idx, grid_width, grid_height, device
                )
                goal_tensor = _goal_to_tensor_static(t['goal'], grid_width, grid_height, device)
                
                # Forward pass through Q-network for current state
                q_values = q_network(
                    grid_tensor, step_tensor,
                    position, direction, agent_idx_t,
                    goal_tensor
                )
                q_value = q_values[0, t['action']]
                
                # Compute TD target
                if was_at_goal:
                    # If at goal, Q-value should be 1.0 (goal already achieved)
                    target = torch.tensor(1.0, device=device, dtype=torch.float32)
                elif done_flag or i == len(trajectory) - 1:
                    # Terminal state - target is just the reward
                    target = torch.tensor(reward, device=device, dtype=torch.float32)
                else:
                    # Non-terminal - use bootstrap
                    next_state = t.get('next_state', t['state'])
                    next_grid, next_step = _state_to_tensors_static(
                        next_state, grid_width, grid_height, num_agents,
                        max_steps=max_steps, device=device
                    )
                    next_pos, next_dir, next_idx = _agent_to_tensors_static(
                        next_state, human_idx, grid_width, grid_height, device
                    )
                    
                    with torch.no_grad():
                        next_q_values = q_network(
                            next_grid, next_step,
                            next_pos, next_dir, next_idx,
                            goal_tensor
                        )
                        # For Boltzmann policy: V(s') = sum_a π(a|s') Q(s',a)
                        next_policy = F.softmax(beta * next_q_values, dim=1)
                        next_v = (next_policy * next_q_values).sum()
                        target = reward + gamma * next_v
                
                # MSE loss: L = (Q(s,a,g) - target)²
                loss = F.mse_loss(q_value, target)
                losses.append(loss)
            
            # Average loss over trajectory and perform gradient update
            if len(losses) > 0:
                total_loss = torch.stack(losses).mean()
                
                # Gradient clipping for stability
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                q_optimizer.step()
                
                episode_q_loss += total_loss.item()
                num_updates += 1
        
        if num_updates > 0:
            q_losses.append(episode_q_loss / num_updates)
        
        # ====================================================================
        # GRADIENT UPDATE: Train phi-network (optional)
        # ====================================================================
        if phi_network is not None and phi_optimizer is not None:
            episode_phi_loss = 0.0
            phi_updates = 0
            
            for human_idx in human_agent_indices:
                trajectory = trajectories[human_idx]
                if len(trajectory) == 0:
                    continue
                
                phi_optimizer.zero_grad()
                phi_losses_list = []
                
                for t in trajectory:
                    # Convert state to tensors
                    grid_tensor, step_tensor = _state_to_tensors_static(
                        t['state'], grid_width, grid_height, num_agents,
                        max_steps=max_steps, device=device
                    )
                    position, direction, agent_idx_t = _agent_to_tensors_static(
                        t['state'], human_idx, grid_width, grid_height, device
                    )
                    goal_tensor = _goal_to_tensor_static(t['goal'], grid_width, grid_height, device)
                    
                    # Get target policy from Q-network (detached)
                    with torch.no_grad():
                        q_values = q_network(
                            grid_tensor, step_tensor,
                            position, direction, agent_idx_t,
                            goal_tensor
                        )
                        target_policy = F.softmax(beta * q_values, dim=1)
                    
                    # Get predicted policy from phi-network
                    predicted_policy = phi_network(
                        grid_tensor, step_tensor,
                        position, direction, agent_idx_t
                    )
                    
                    # KL divergence loss: KL(target || predicted)
                    # Using cross-entropy since KL = H(p,q) - H(p) and H(p) is constant
                    phi_loss = F.kl_div(
                        predicted_policy.log(),
                        target_policy,
                        reduction='batchmean'
                    )
                    phi_losses_list.append(phi_loss)
                
                if len(phi_losses_list) > 0:
                    avg_phi_loss = torch.stack(phi_losses_list).mean()
                    avg_phi_loss.backward()
                    phi_optimizer.step()
                    
                    episode_phi_loss += avg_phi_loss.item()
                    phi_updates += 1
            
            if phi_updates > 0:
                phi_losses.append(episode_phi_loss / phi_updates)
        
        # Logging
        if verbose and (episode + 1) % 100 == 0:
            avg_q_loss = np.mean(q_losses[-100:]) if q_losses else 0.0
            avg_phi_loss = np.mean(phi_losses[-100:]) if phi_losses else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Q-Loss = {avg_q_loss:.4f}, Phi-Loss = {avg_phi_loss:.4f}")
    
    if verbose:
        print("Training complete!")
        if q_losses:
            print(f"  Final Q-Loss: {np.mean(q_losses[-100:]):.4f}")
        if phi_losses:
            print(f"  Final Phi-Loss: {np.mean(phi_losses[-100:]):.4f}")
    
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
