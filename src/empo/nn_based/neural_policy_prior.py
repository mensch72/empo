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
    >>>
    >>> # Save trained model for reuse
    >>> neural_prior.save("my_policy_prior.pt")
    >>>
    >>> # Load for use with different action space
    >>> loaded_prior = NeuralHumanPolicyPrior.load(
    ...     "my_policy_prior.pt",
    ...     world_model=new_env,
    ...     human_agent_indices=[0, 1, 2],
    ...     infeasible_actions_become=0  # Map unknown actions to 'still'
    ... )
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
from abc import ABC, abstractmethod

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler


"""
Object type to channel index mapping for grid encoding.

This defines a consistent mapping from object types (wall, door, key, etc.)
to channel indices, used during both training and inference.

Objects are organized by their properties:
- Overlappable: Goal, Floor, Switch, etc. (agents can stand on them)
- Non-overlappable immobile: Wall, MagicWall, Lava, Door (fixed obstacles)
- Non-overlappable mobile: Block, Rock (can be pushed)
"""
OBJECT_TYPE_TO_CHANNEL = {
    'wall': 0,
    'magicwall': 0,     # Treat as wall variant
    'door': 1,
    'key': 2,
    'ball': 3,
    'box': 4,
    'goal': 5,
    'lava': 6,
    'block': 7,
    'rock': 8,
    'unsteadyground': 9,
    'switch': 10,
    'killbutton': 11,
    'pauseswitch': 12,
    'disablingswitch': 13,
    'controlbutton': 14,
    'floor': 15,
}
NUM_OBJECT_TYPE_CHANNELS = 16  # Total object type channels

# Object property categories for "other objects" channels
OVERLAPPABLE_OBJECTS = {'goal', 'floor', 'switch', 'killbutton', 'pauseswitch', 
                        'disablingswitch', 'controlbutton', 'unsteadyground', 'objectgoal'}
NON_OVERLAPPABLE_IMMOBILE_OBJECTS = {'wall', 'magicwall', 'lava', 'door'}
NON_OVERLAPPABLE_MOBILE_OBJECTS = {'block', 'rock'}

# Default action encoding (can be customized per environment)
DEFAULT_ACTION_ENCODING = {
    0: 'still',
    1: 'left',
    2: 'right', 
    3: 'forward',
    4: 'pickup',
    5: 'drop',
    6: 'toggle',
    7: 'done',
}

SMALL_ACTION_ENCODING = {
    0: 'still',
    1: 'left',
    2: 'right',
    3: 'forward',
}


class StateEncoder(nn.Module):
    """
    Encodes grid-based multigrid states into feature vectors.
    
    The encoder uses a CNN to process a spatial representation of the grid,
    capturing object positions, types, and agent locations.
    
    For multigrid environments, the state tuple format is:
        (step_count, agent_states, mobile_objects, mutable_objects)
    
    The encoder converts this into a 2D grid representation and applies
    convolutional layers to extract spatial features.
    
    Input channels (in order):
        1. Object type channels (walls, doors, lava, etc.): num_object_types
        2. "Other overlappable objects" channel: 1 (for objects not in object_types_list)
        3. "Other non-overlappable immobile objects" channel: 1
        4. "Other non-overlappable mobile objects" channel: 1
        5. Per-agent position channels: num_agents_per_color * num_colors
        6. Query agent channel: 1 (marks the agent specified by agent_idx)
        7. "Other humans" channel: 1 (marks all human agents other than the query agent)
    
    Args:
        grid_width: Width of the grid environment.
        grid_height: Height of the grid environment.
        num_object_types: Number of object type channels in the network (default: NUM_OBJECT_TYPE_CHANNELS).
            Each channel is a binary indicator for a specific object type presence.
        num_agents: Total number of agents in the environment.
        num_agents_per_color: Dict mapping color string to number of agents of that color.
            Used for color-specific agent channels. If None, uses num_agents for backward compatibility.
        feature_dim: Output feature dimension (default: 128).
        object_types_list: List of object type strings that have explicit channels.
            Objects not in this list go into the "other objects" channels.
            If None, uses all types in OBJECT_TYPE_TO_CHANNEL.
    """
    
    def __init__(
        self, 
        grid_width: int, 
        grid_height: int, 
        num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
        num_agents: int = 2,
        num_agents_per_color: Optional[Dict[str, int]] = None,
        feature_dim: int = 128,
        object_types_list: Optional[List[str]] = None
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_object_types = num_object_types
        self.num_agents = num_agents
        self.feature_dim = feature_dim
        
        # Store object types list for save/load compatibility
        if object_types_list is None:
            self.object_types_list = list(OBJECT_TYPE_TO_CHANNEL.keys())
        else:
            self.object_types_list = object_types_list
        
        # Store agent counts per color for enhanced agent encoding
        if num_agents_per_color is None:
            # Backward compatibility: single set of agent channels
            self.num_agents_per_color = None
            self.agent_color_order = None
            num_agent_channels = num_agents
        else:
            self.num_agents_per_color = num_agents_per_color
            # Consistent ordering of colors for channel assignment
            self.agent_color_order = sorted(num_agents_per_color.keys())
            num_agent_channels = sum(num_agents_per_color.values())
        
        # Calculate total input channels:
        # - num_object_types: explicit object type channels
        # - 3: "other" object channels (overlappable, immobile, mobile)
        # - num_agent_channels: per-agent position channels (or per-color lists)
        # - 1: query agent channel
        # - 1: "other humans" channel
        self.num_other_object_channels = 3  # overlappable, immobile, mobile
        in_channels = (num_object_types + 
                      self.num_other_object_channels +
                      num_agent_channels + 
                      1 +  # query agent channel
                      1)   # other humans channel
        
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
        - Agent index (embedded) - for distinguishing agents within trained capacity
        - Query agent features - separate encoding for the agent being queried
        - Agent color (embedded, optional)
    
    The query agent encoding is crucial for policy transfer: when the environment
    has more agents than the network was trained with, the query agent's identity
    is preserved through its dedicated feature encoding, not just the index embedding.
    
    Args:
        grid_width: Width of the grid for normalization.
        grid_height: Height of the grid for normalization.
        num_agents: Maximum number of agents (for index embedding).
        num_agents_per_color: Optional dict mapping color to agent count.
        feature_dim: Output feature dimension (default: 32).
        agent_colors: Optional list of agent color strings in order of agent index.
    """
    
    def __init__(
        self, 
        grid_width: int, 
        grid_height: int, 
        num_agents: int,
        num_agents_per_color: Optional[Dict[str, int]] = None,
        feature_dim: int = 32,
        agent_colors: Optional[List[str]] = None
    ):
        super().__init__()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_agents = num_agents
        self.feature_dim = feature_dim
        self.num_agents_per_color = num_agents_per_color
        self.agent_colors = agent_colors
        
        # Agent index embedding (for agents within trained capacity)
        self.agent_embedding = nn.Embedding(num_agents, 16)
        
        # Query agent encoder - encodes the query agent's features separately
        # This provides a dedicated pathway that doesn't depend on agent index
        # Input: 2 (position) + 4 (direction) = 6
        self.query_agent_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
        )
        
        # Color embedding if we have color info
        if agent_colors is not None:
            unique_colors = sorted(set(agent_colors))
            self.color_to_idx = {c: i for i, c in enumerate(unique_colors)}
            self.color_embedding = nn.Embedding(len(unique_colors), 8)
            color_embed_dim = 8
        else:
            self.color_to_idx = None
            self.color_embedding = None
            color_embed_dim = 0
        
        # MLP combining all features:
        # - position (2) + direction (4) + index embedding (16) + query agent features (16) + color (optional)
        input_dim = 2 + 4 + 16 + 16 + color_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )
    
    def forward(
        self, 
        position: torch.Tensor, 
        direction: torch.Tensor, 
        agent_idx: torch.Tensor,
        agent_color_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode agent attributes into feature vectors.
        
        Args:
            position: Tensor of shape (batch, 2) with normalized positions [x/W, y/H].
            direction: Tensor of shape (batch, 4) with one-hot direction encoding.
            agent_idx: Tensor of shape (batch,) with agent indices.
                Indices >= num_agents are clamped to num_agents-1 for the embedding,
                but the query agent features still capture the actual position/direction.
            agent_color_idx: Optional tensor of shape (batch,) with color indices.
                If None and color_embedding exists, will use zeros.
        
        Returns:
            Feature tensor of shape (batch, feature_dim).
        """
        # Clamp agent indices to valid embedding range
        clamped_idx = torch.clamp(agent_idx, 0, self.num_agents - 1)
        idx_embed = self.agent_embedding(clamped_idx)  # (batch, 16)
        
        # Query agent encoding - captures position and direction in dedicated pathway
        query_input = torch.cat([position, direction], dim=1)  # (batch, 6)
        query_features = self.query_agent_encoder(query_input)  # (batch, 16)
        
        if self.color_embedding is not None:
            if agent_color_idx is None:
                agent_color_idx = torch.zeros_like(agent_idx)
            color_embed = self.color_embedding(agent_color_idx)  # (batch, 8)
            x = torch.cat([position, direction, idx_embed, query_features, color_embed], dim=1)
        else:
            x = torch.cat([position, direction, idx_embed, query_features], dim=1)
        
        return self.fc(x)

class SoftClamp(nn.Module):
    """
    Implements a soft clamp to [a - (b-a), b + (b-a)] that is linear in [a,b] and exponential outside.
    
    Q(Z) = 1 - ReLU(1 - ReLU(Z)) - exp(-ReLU(Z - 1)) + exp(-ReLU(-Z))
    """
    def __init__(self, a: float = 0.5, b: float = 1.5):
        super().__init__()
        # Define the linear boundaries
        self.a = float(a)
        self.b = float(b)
        # Define the range size R = b - a
        self.R = self.b - self.a
        
        if self.R <= 0:
            raise ValueError("b must be greater than a for the linear region [a, b].")

        self.relu = nn.ReLU()

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        
        # --- 1. Term 1: Linear Core (The Clip Function) ---
        # T1 = ReLU(Z - a) - ReLU(Z - b) + a
        # This function is Z in [a, b], 'a' for Z < a, and 'b' for Z > b.
        term1_linear_core = self.relu(Z - self.a) - self.relu(Z - self.b) + self.a
        
        # --- 2. Term 2: Upper Exponential Tail (Z > b) ---
        # T2 = R * (1 - exp(-(1/R) * ReLU(Z - b)))
        # This term is 0 for Z <= b, and approaches R for Z >> b.
        # Adding T1 (which is 'b' for Z > b) and T2 (which approaches R) 
        # gives a total saturation at b + R.
        term2_tail_gt_b = self.R * (1.0 - torch.exp(-(1.0 / self.R) * self.relu(Z - self.b)))
        
        # --- 3. Term 3: Lower Exponential Tail (Z < a) ---
        # T3 = R * (exp(-(1/R) * ReLU(a - Z)) - 1)
        # This term is 0 for Z >= a, and approaches -R for Z << a.
        # Adding T1 (which is 'a' for Z < a) and T3 (which approaches -R) 
        # gives a total saturation at a - R.
        term3_tail_lt_a = self.R * (torch.exp(-(1.0 / self.R) * self.relu(self.a - Z)) - 1.0)
        
        return term1_linear_core + term2_tail_gt_b + term3_tail_lt_a
        
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
        feasible_range: Optional tuple (a, b) for theoretical bounds for the Q values. Will be used for clamping (either soft or hard).
    """
    
    def __init__(
        self,
        state_encoder: StateEncoder,
        agent_encoder: AgentEncoder,
        goal_encoder: GoalEncoder,
        num_actions: int,
        hidden_dim: int = 256,
        feasible_range: Optional[tuple] = None
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
        
        # Q-value head - outputs logits, sigmoid applied in forward() to bound Q in [0,1]
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )
        if feasible_range is not None:
            # Add soft clamp to the output layer to prevent gradient explosion outside feasible range and nudge Q-values into feasible range:
            self.q_head.add_module("soft_clamp", SoftClamp(*feasible_range))

        self.feasible_range = feasible_range
    
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
            Q-values tensor of shape (batch, num_actions), bounded to [0,1] if bounded_q=True.
        """
        state_feat = self.state_encoder(state_tensor, step_count)
        agent_feat = self.agent_encoder(agent_position, agent_direction, agent_idx)
        goal_feat = self.goal_encoder(goal_coords)
        
        combined = torch.cat([state_feat, agent_feat, goal_feat], dim=1)
        q_values_predicted = self.q_head(combined)
        
        return q_values_predicted  # might be unbounded if soft_clamp is None!
    
    def get_policy(self, q_values: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute Boltzmann policy from Q-values.
        
        Args:
            q_values: Q-values tensor of shape (batch, num_actions).
                     If bounded_q=True, these are already in [0,1].
            beta: Inverse temperature (higher = more deterministic).
        
        Returns:
            Policy tensor of shape (batch, num_actions) with action probabilities.
        """
        if self.feasible_range is not None:
            # hard clamp here because differentiability is not needed for policy extraction:
            q_values = torch.clamp(q_values, self.feasible_range[0] - (self.feasible_range[1]-self.feasible_range[0]), self.feasible_range[1])
        if beta == float('inf'):
            # Argmax policy with ties broken uniformly
            max_q = q_values.max(dim=1, keepdim=True)[0]
            is_max = (q_values == max_q).float()
            policy = is_max / is_max.sum(dim=1, keepdim=True)
        else:
            # Softmax with temperature
            # Since Q-values are bounded to [0,1], beta*Q is bounded to [0, beta]
            # Subtract max for numerical stability (standard softmax trick)
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
    
    def _state_to_tensors(self, state, query_agent_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a state tuple to tensor representation with full grid encoding.
        
        This method populates:
        1. Object-type channels (walls, doors, lava, etc.) from the world grid
        2. "Other objects" channels (overlappable, immobile, mobile)
        3. Per-agent position channels
        4. Query agent channel
        5. "Other humans" channel (all human agents except query_agent_index)
        
        Args:
            state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
            query_agent_index: The index of the agent making the query. If provided,
                              this agent is excluded from the "other humans" channel.
        
        Returns:
            Tuple of (state_tensor, step_count_tensor).
        """
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        num_channels = self.q_network.state_encoder.conv[0].in_channels
        H = self.q_network.state_encoder.grid_height
        W = self.q_network.state_encoder.grid_width
        num_object_types = self.q_network.state_encoder.num_object_types
        num_agents = self.q_network.state_encoder.num_agents
        
        # Channel indices (must match StateEncoder structure)
        num_other_object_channels = 3
        other_overlappable_idx = num_object_types
        other_immobile_idx = num_object_types + 1
        other_mobile_idx = num_object_types + 2
        agent_channels_start = num_object_types + num_other_object_channels
        query_agent_channel_idx = agent_channels_start + num_agents
        other_humans_channel_idx = query_agent_channel_idx + 1
        
        grid_tensor = torch.zeros(1, num_channels, H, W, device=self.device)
        
        # 1. Encode object-type channels from the persistent world grid
        if hasattr(self.world_model, 'grid') and self.world_model.grid is not None:
            for y in range(H):
                for x in range(W):
                    cell = self.world_model.grid.get(x, y)
                    if cell is not None:
                        cell_type = getattr(cell, 'type', None)
                        if cell_type is not None:
                            if cell_type in OBJECT_TYPE_TO_CHANNEL:
                                channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                                if channel_idx < num_object_types:
                                    grid_tensor[0, channel_idx, y, x] = 1.0
                            else:
                                # Object type not in explicit channels - use "other" channels
                                if cell_type in OVERLAPPABLE_OBJECTS:
                                    grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                                elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                                    grid_tensor[0, other_mobile_idx, y, x] = 1.0
                                elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                                    grid_tensor[0, other_immobile_idx, y, x] = 1.0
                                else:
                                    if hasattr(cell, 'can_overlap') and cell.can_overlap():
                                        grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                                    else:
                                        grid_tensor[0, other_immobile_idx, y, x] = 1.0
        
        # 2. Encode mobile objects (if any) into their respective channels
        if mobile_objects:
            for mobile_obj in mobile_objects:
                obj_type = mobile_obj[0]
                obj_x = mobile_obj[1]
                obj_y = mobile_obj[2]
                if 0 <= obj_x < W and 0 <= obj_y < H:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        if channel_idx < num_object_types:
                            grid_tensor[0, channel_idx, int(obj_y), int(obj_x)] = 1.0
                    else:
                        grid_tensor[0, other_mobile_idx, int(obj_y), int(obj_x)] = 1.0
        
        # 3. Encode agent positions (per-agent channels)
        for i, agent_state in enumerate(agent_states):
            if i < num_agents:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    channel_idx = agent_channels_start + i
                    grid_tensor[0, channel_idx, y, x] = 1.0
        
        # 4. Encode query agent channel
        if query_agent_index is not None and query_agent_index < len(agent_states):
            agent_state = agent_states[query_agent_index]
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < W and 0 <= y < H:
                grid_tensor[0, query_agent_channel_idx, y, x] = 1.0
        
        # 5. Encode "other humans" channel (anonymous channel for all other human agents)
        for i, agent_state in enumerate(agent_states):
            # Skip the query agent (if specified)
            if query_agent_index is not None and i == query_agent_index:
                continue
            # Only include human agents (in human_agent_indices)
            if i in self.human_agent_indices:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < W and 0 <= y < H:
                    grid_tensor[0, other_humans_channel_idx, y, x] = 1.0
        
        # Normalize step count
        max_steps = getattr(self.world_model, 'max_steps', 100)
        step_tensor = torch.tensor([[step_count / max_steps]], device=self.device)
        
        return grid_tensor, step_tensor
    
    def get_human_positions_by_id(self, state) -> Dict[int, Tuple[int, int]]:
        """
        Get human agent positions indexed by agent ID.
        
        Args:
            state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
        
        Returns:
            Dictionary mapping agent index to (x, y) position for all human agents.
        """
        _, agent_states, _, _ = state
        positions = {}
        for agent_idx in self.human_agent_indices:
            if agent_idx < len(agent_states):
                agent_state = agent_states[agent_idx]
                positions[agent_idx] = (int(agent_state[0]), int(agent_state[1]))
        return positions
    
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
        
        # Agent index - AgentEncoder handles clamping internally
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
                state_tensor, step_tensor = self._state_to_tensors(state, query_agent_index=human_agent_index)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                goal_tensor = self._goal_to_tensor(possible_goal)
                
                q_values = self.q_network(
                    state_tensor, step_tensor,
                    position, direction, agent_idx,
                    goal_tensor
                )
                
                policy = self.q_network.get_policy(q_values, self.beta)
                saved_policy = policy.cpu().numpy()[0]
                return self._remap_policy(saved_policy)
            
            elif self.phi_network is not None:
                # Use direct marginal network
                state_tensor, step_tensor = self._state_to_tensors(state, query_agent_index=human_agent_index)
                position, direction, agent_idx = self._agent_to_tensors(state, human_agent_index)
                
                policy = self.phi_network(
                    state_tensor, step_tensor,
                    position, direction, agent_idx
                )
                saved_policy = policy.cpu().numpy()[0]
                return self._remap_policy(saved_policy)
            
            elif self.goal_generator is not None:
                # Compute marginal by exact enumeration over goals
                state_tensor, step_tensor = self._state_to_tensors(state, query_agent_index=human_agent_index)
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
                
                return self._remap_policy(total_policy)
            
            elif self.goal_sampler is not None:
                # Compute marginal by Monte Carlo sampling
                state_tensor, step_tensor = self._state_to_tensors(state, query_agent_index=human_agent_index)
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
                
                saved_policy = total_policy / total_weight if total_weight > 0 else total_policy
                return self._remap_policy(saved_policy)
            
            else:
                raise ValueError(
                    "Either possible_goal must be provided, or one of "
                    "phi_network, goal_generator, or goal_sampler must be set"
                )
    
    def save(self, filepath: str) -> None:
        """
        Save the trained policy prior to a file.
        
        Saves the Q-network weights and all input-relevant architectural parameters
        needed for loading the network in a different environment.
        
        Saved metadata includes:
            - grid_dimensions: (width, height) of the grid
            - object_types_list: List of object types with explicit channels
            - action_encoding: Mapping of action index to action string
            - num_agents: Number of agents the network was trained with
            - num_agents_per_color: Dict of agent counts per color (if available)
            - num_actions: Number of actions in action space
            - beta: Inverse temperature parameter
            - feature_dims: Feature dimensions for encoders
        
        Args:
            filepath: Path to save the model file (.pt format recommended).
        """
        # Build action encoding from world_model
        action_encoding = {}
        if hasattr(self.world_model, 'actions') and hasattr(self.world_model.actions, 'available'):
            for i, action_name in enumerate(self.world_model.actions.available):
                action_encoding[i] = action_name
        else:
            # Fallback to default encoding
            num_actions = self.q_network.num_actions
            for i in range(num_actions):
                if i in DEFAULT_ACTION_ENCODING:
                    action_encoding[i] = DEFAULT_ACTION_ENCODING[i]
                else:
                    action_encoding[i] = f'action_{i}'
        
        # Collect metadata
        state_encoder = self.q_network.state_encoder
        
        # Get hidden_dim from q_head (first linear layer output dimension)
        hidden_dim = self.q_network.q_head[0].out_features
        
        metadata = {
            'grid_width': state_encoder.grid_width,
            'grid_height': state_encoder.grid_height,
            'num_object_types': state_encoder.num_object_types,
            'object_types_list': getattr(state_encoder, 'object_types_list', list(OBJECT_TYPE_TO_CHANNEL.keys())),
            'action_encoding': action_encoding,
            'num_agents': state_encoder.num_agents,
            'num_agents_per_color': getattr(state_encoder, 'num_agents_per_color', None),
            'agent_color_order': getattr(state_encoder, 'agent_color_order', None),
            'num_actions': self.q_network.num_actions,
            'beta': self.beta,
            'human_agent_indices': self.human_agent_indices,
            'state_encoder_feature_dim': state_encoder.feature_dim,
            'agent_encoder_feature_dim': self.q_network.agent_encoder.feature_dim,
            'goal_encoder_feature_dim': self.q_network.goal_encoder.feature_dim,
            'hidden_dim': hidden_dim,
            'feasible_range': self.q_network.feasible_range,
        }
        
        # Collect agent encoder specific metadata
        agent_encoder = self.q_network.agent_encoder
        if hasattr(agent_encoder, 'agent_colors') and agent_encoder.agent_colors is not None:
            metadata['agent_colors'] = agent_encoder.agent_colors
        if hasattr(agent_encoder, 'color_to_idx') and agent_encoder.color_to_idx is not None:
            metadata['agent_color_to_idx'] = agent_encoder.color_to_idx
        
        # Save state dict and metadata
        save_dict = {
            'metadata': metadata,
            'q_network_state_dict': self.q_network.state_dict(),
        }
        
        if self.phi_network is not None:
            save_dict['phi_network_state_dict'] = self.phi_network.state_dict()
        
        torch.save(save_dict, filepath)
    
    @classmethod
    def load(
        cls,
        filepath: str,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        goal_generator: Optional[PossibleGoalGenerator] = None,
        infeasible_actions_become: Optional[int] = None,
        beta: Optional[float] = None,
        device: str = 'cpu'
    ) -> 'NeuralHumanPolicyPrior':
        """
        Load a saved policy prior for use with a potentially different environment.
        
        This method handles compatibility between the saved network and the new
        environment, including:
        - Grid dimension matching (must match exactly)
        - Action encoding compatibility (no conflicts allowed)
        - Object type handling (unrecognized types go to "other" channels)
        - Agent count handling (padding/truncation as needed)
        
        Args:
            filepath: Path to the saved model file.
            world_model: The new environment/world model.
            human_agent_indices: Indices of human agents in the new environment.
            goal_sampler: Optional goal sampler for marginal computation.
            goal_generator: Optional goal generator for marginal computation.
            infeasible_actions_become: How to handle actions in the saved network
                that don't exist in the new world_model's action space:
                - If an integer (e.g., 0 for 'still'), probability is mapped to that action.
                - If None, these actions are conditioned out (probability set to 0).
            beta: Inverse temperature. If None, uses the saved value.
            device: Torch device ('cpu' or 'cuda').
        
        Returns:
            NeuralHumanPolicyPrior: A loaded policy prior adapted for the new environment.
        
        Raises:
            ValueError: If grid dimensions don't match or action encodings conflict.
        """
        # Load saved data
        save_dict = torch.load(filepath, map_location=device, weights_only=False)
        metadata = save_dict['metadata']
        
        # Validate grid dimensions
        if world_model.width != metadata['grid_width'] or world_model.height != metadata['grid_height']:
            raise ValueError(
                f"Grid dimensions mismatch: saved ({metadata['grid_width']}x{metadata['grid_height']}) "
                f"vs current ({world_model.width}x{world_model.height}). "
                "Grid dimensions must match exactly for loading."
            )
        
        # Get current action encoding from world_model
        current_action_encoding = {}
        if hasattr(world_model, 'actions') and hasattr(world_model.actions, 'available'):
            for i, action_name in enumerate(world_model.actions.available):
                current_action_encoding[i] = action_name
        else:
            num_current_actions = world_model.action_space.n
            for i in range(num_current_actions):
                if i in DEFAULT_ACTION_ENCODING:
                    current_action_encoding[i] = DEFAULT_ACTION_ENCODING[i]
                else:
                    current_action_encoding[i] = f'action_{i}'
        
        saved_action_encoding = metadata['action_encoding']
        # Convert string keys back to int (JSON serialization issue)
        saved_action_encoding = {int(k): v for k, v in saved_action_encoding.items()}
        
        # Check for action encoding conflicts
        # A conflict is when the same action ID maps to different action names
        for action_id, action_name in current_action_encoding.items():
            if action_id in saved_action_encoding and saved_action_encoding[action_id] != action_name:
                raise ValueError(
                    f"Action encoding conflict at index {action_id}: "
                    f"saved='{saved_action_encoding[action_id]}' vs current='{action_name}'. "
                    "Cannot load network with conflicting action meanings."
                )
        
        # Build action mapping
        # - saved_to_current: maps saved action index to current action index (for valid actions)
        # - actions_to_mask: list of saved action indices that should be masked (prob=0)
        # - actions_to_remap: list of saved action indices that should be remapped
        saved_action_to_name = saved_action_encoding
        current_name_to_action = {v: k for k, v in current_action_encoding.items()}
        
        num_saved_actions = metadata['num_actions']
        num_current_actions = len(current_action_encoding)
        
        action_mapping = {}  # saved_idx -> current_idx or None
        for saved_idx in range(num_saved_actions):
            saved_name = saved_action_to_name.get(saved_idx, f'action_{saved_idx}')
            if saved_name in current_name_to_action:
                action_mapping[saved_idx] = current_name_to_action[saved_name]
            else:
                # Action exists in saved but not in current
                if infeasible_actions_become is not None:
                    action_mapping[saved_idx] = infeasible_actions_become
                else:
                    action_mapping[saved_idx] = None  # Will be masked out
        
        # Reconstruct the network architecture
        grid_width = metadata['grid_width']
        grid_height = metadata['grid_height']
        num_agents = metadata['num_agents']
        num_object_types = metadata['num_object_types']
        
        # Recreate encoders with saved architecture
        state_encoder = StateEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_object_types=num_object_types,
            num_agents=num_agents,
            num_agents_per_color=metadata.get('num_agents_per_color'),
            feature_dim=metadata.get('state_encoder_feature_dim', 128),
            object_types_list=metadata.get('object_types_list')
        ).to(device)
        
        agent_colors = metadata.get('agent_colors')
        agent_encoder = AgentEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            num_agents=num_agents,
            num_agents_per_color=metadata.get('num_agents_per_color'),
            feature_dim=metadata.get('agent_encoder_feature_dim', 32),
            agent_colors=agent_colors
        ).to(device)
        
        goal_encoder = GoalEncoder(
            grid_width=grid_width,
            grid_height=grid_height,
            feature_dim=metadata.get('goal_encoder_feature_dim', 32)
        ).to(device)
        
        hidden_dim = metadata.get('hidden_dim', 128)
        
        q_network = QNetwork(
            state_encoder=state_encoder,
            agent_encoder=agent_encoder,
            goal_encoder=goal_encoder,
            num_actions=num_saved_actions,  # Use saved num_actions
            hidden_dim=hidden_dim,
            feasible_range=metadata.get('feasible_range')
        ).to(device)
        
        # Load weights
        q_network.load_state_dict(save_dict['q_network_state_dict'])
        
        # Load phi_network if available
        phi_network = None
        if 'phi_network_state_dict' in save_dict:
            phi_state_encoder = StateEncoder(
                grid_width=grid_width,
                grid_height=grid_height,
                num_object_types=num_object_types,
                num_agents=num_agents,
                num_agents_per_color=metadata.get('num_agents_per_color'),
                feature_dim=metadata.get('state_encoder_feature_dim', 128),
                object_types_list=metadata.get('object_types_list')
            ).to(device)
            
            phi_agent_encoder = AgentEncoder(
                grid_width=grid_width,
                grid_height=grid_height,
                num_agents=num_agents,
                num_agents_per_color=metadata.get('num_agents_per_color'),
                feature_dim=metadata.get('agent_encoder_feature_dim', 32),
                agent_colors=agent_colors
            ).to(device)
            
            phi_network = PolicyPriorNetwork(
                state_encoder=phi_state_encoder,
                agent_encoder=phi_agent_encoder,
                num_actions=num_saved_actions
            ).to(device)
            
            phi_network.load_state_dict(save_dict['phi_network_state_dict'])
        
        # Use saved beta if not overridden
        if beta is None:
            beta = metadata.get('beta', 1.0)
        
        # Create wrapper that handles action mapping
        prior = cls(
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            q_network=q_network,
            phi_network=phi_network,
            beta=beta,
            goal_sampler=goal_sampler,
            goal_generator=goal_generator,
            device=device
        )
        
        # Store action mapping info for runtime use
        prior._action_mapping = action_mapping
        prior._num_current_actions = num_current_actions
        prior._saved_action_encoding = saved_action_encoding
        prior._current_action_encoding = current_action_encoding
        
        return prior
    
    def _remap_policy(self, saved_policy: np.ndarray) -> np.ndarray:
        """
        Remap policy from saved action space to current action space.
        
        This method handles:
        - Actions that exist in both: probability transferred directly
        - Actions in saved but not current: masked (prob=0) or remapped
        - Actions in current but not saved: get probability 0
        
        Args:
            saved_policy: Policy over saved action space (sums to 1).
        
        Returns:
            Policy over current action space (renormalized to sum to 1).
        """
        if not hasattr(self, '_action_mapping'):
            # No mapping needed - same action space
            return saved_policy
        
        current_policy = np.zeros(self._num_current_actions)
        
        for saved_idx, prob in enumerate(saved_policy):
            if saved_idx in self._action_mapping:
                current_idx = self._action_mapping[saved_idx]
                if current_idx is not None:
                    current_policy[current_idx] += prob
        
        # Renormalize to sum to 1 (handles masked actions)
        total = current_policy.sum()
        if total > 0:
            current_policy /= total
        else:
            # All actions masked - uniform over current actions
            current_policy = np.ones(self._num_current_actions) / self._num_current_actions
        
        return current_policy


def _get_action_encoding(world_model: Any) -> Dict[int, str]:
    """
    Extract action encoding from a world model.
    
    Args:
        world_model: Environment with actions attribute.
    
    Returns:
        Dictionary mapping action index to action string.
    """
    if hasattr(world_model, 'actions') and hasattr(world_model.actions, 'available'):
        return {i: name for i, name in enumerate(world_model.actions.available)}
    elif hasattr(world_model, 'action_space'):
        num_actions = world_model.action_space.n
        encoding = {}
        for i in range(num_actions):
            if i in DEFAULT_ACTION_ENCODING:
                encoding[i] = DEFAULT_ACTION_ENCODING[i]
            else:
                encoding[i] = f'action_{i}'
        return encoding
    return {}


def _state_to_tensors_static(
    state,
    grid_width: int,
    grid_height: int,
    num_agents: int,
    num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
    max_steps: int = 100,
    device: str = 'cpu',
    world_model: Any = None,
    human_agent_indices: Optional[List[int]] = None,
    query_agent_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a state tuple to tensor representation with full grid encoding.
    
    This function populates:
    1. Object-type channels (walls, doors, lava, etc.) from the world grid
    2. "Other objects" channels (overlappable, immobile, mobile) for unknown types
    3. Per-agent position channels
    4. Query agent channel
    5. "Other humans" channel (all human agents except query_agent_index)
    
    Args:
        state: State tuple (step_count, agent_states, mobile_objects, mutable_objects).
        grid_width: Width of the grid.
        grid_height: Height of the grid.
        num_agents: Number of agents.
        num_object_types: Number of object types for encoding.
        max_steps: Maximum steps for normalization.
        device: Torch device.
        world_model: Optional world model for accessing persistent grid objects.
        human_agent_indices: Optional list of human agent indices for "other humans" channel.
        query_agent_index: Optional index of query agent to exclude from "other humans".
    
    Returns:
        Tuple of (grid_tensor, step_count_tensor).
    """
    step_count, agent_states, mobile_objects, mutable_objects = state
    
    # Channel structure:
    # - num_object_types: explicit object type channels
    # - 3: "other" object channels (overlappable, immobile, mobile)
    # - num_agents: per-agent position channels
    # - 1: query agent channel
    # - 1: "other humans" channel
    num_other_object_channels = 3
    num_channels = num_object_types + num_other_object_channels + num_agents + 1 + 1
    grid_tensor = torch.zeros(1, num_channels, grid_height, grid_width, device=device)
    
    # Channel indices
    other_overlappable_idx = num_object_types
    other_immobile_idx = num_object_types + 1
    other_mobile_idx = num_object_types + 2
    agent_channels_start = num_object_types + num_other_object_channels
    query_agent_channel_idx = agent_channels_start + num_agents
    other_humans_channel_idx = query_agent_channel_idx + 1
    
    # 1. Encode object-type channels from the persistent world grid
    if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
        for y in range(grid_height):
            for x in range(grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is not None:
                        if cell_type in OBJECT_TYPE_TO_CHANNEL:
                            channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                            if channel_idx < num_object_types:
                                grid_tensor[0, channel_idx, y, x] = 1.0
                        else:
                            # Object type not in explicit channels - use "other" channels
                            if cell_type in OVERLAPPABLE_OBJECTS:
                                grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                                grid_tensor[0, other_mobile_idx, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                                grid_tensor[0, other_immobile_idx, y, x] = 1.0
                            else:
                                # Unknown type - check can_overlap
                                if hasattr(cell, 'can_overlap') and cell.can_overlap():
                                    grid_tensor[0, other_overlappable_idx, y, x] = 1.0
                                else:
                                    grid_tensor[0, other_immobile_idx, y, x] = 1.0
    
    # 2. Encode mobile objects (if any) into their respective channels
    # Mobile objects format: (obj_type, x, y) - no color
    if mobile_objects:
        for mobile_obj in mobile_objects:
            obj_type = mobile_obj[0]
            obj_x = mobile_obj[1]
            obj_y = mobile_obj[2]
            if 0 <= obj_x < grid_width and 0 <= obj_y < grid_height:
                if obj_type in OBJECT_TYPE_TO_CHANNEL:
                    channel_idx = OBJECT_TYPE_TO_CHANNEL[obj_type]
                    if channel_idx < num_object_types:
                        grid_tensor[0, channel_idx, int(obj_y), int(obj_x)] = 1.0
                else:
                    # Mobile object not in explicit channels
                    grid_tensor[0, other_mobile_idx, int(obj_y), int(obj_x)] = 1.0
    
    # 3. Encode agent positions (per-agent channels)
    for i, agent_state in enumerate(agent_states):
        if i < num_agents:
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                channel_idx = agent_channels_start + i
                grid_tensor[0, channel_idx, y, x] = 1.0
    
    # 4. Encode query agent channel (specific agent being queried)
    if query_agent_index is not None and query_agent_index < len(agent_states):
        agent_state = agent_states[query_agent_index]
        x, y = int(agent_state[0]), int(agent_state[1])
        if 0 <= x < grid_width and 0 <= y < grid_height:
            grid_tensor[0, query_agent_channel_idx, y, x] = 1.0
    
    # 5. Encode "other humans" channel (anonymous channel for all other human agents)
    if human_agent_indices is not None:
        for i, agent_state in enumerate(agent_states):
            # Skip the query agent (if specified)
            if query_agent_index is not None and i == query_agent_index:
                continue
            # Only include human agents
            if i in human_agent_indices:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    grid_tensor[0, other_humans_channel_idx, y, x] = 1.0
    
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


def _batch_states_to_tensors(
    transitions: List[Dict[str, Any]],
    grid_width: int,
    grid_height: int,
    num_agents: int,
    num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
    max_steps: int = 100,
    device: str = 'cpu',
    world_model: Any = None,
    human_agent_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a batch of transitions to batched tensor representations.
    
    This function populates:
    1. Object-type channels (walls, doors, lava, etc.) from the world grid
    2. "Other objects" channels (overlappable, immobile, mobile)
    3. Per-agent position channels
    4. Query agent channel
    5. "Other humans" channel (all human agents except the query agent for each transition)
    
    Args:
        transitions: List of transition dictionaries with 'state', 'human_idx', 'goal' keys.
        grid_width: Width of the grid.
        grid_height: Height of the grid.
        num_agents: Number of agents.
        num_object_types: Number of object types for encoding.
        max_steps: Maximum steps for normalization.
        device: Torch device.
        world_model: Optional world model for accessing persistent grid objects.
        human_agent_indices: Optional list of human agent indices for "other humans" channel.
    
    Returns:
        Tuple of batched tensors:
            - grid_tensors: (batch, channels, H, W)
            - step_tensors: (batch, 1)
            - positions: (batch, 2)
            - directions: (batch, 4)
            - agent_indices: (batch,)
            - goal_coords: (batch, 4)
    """
    batch_size = len(transitions)
    
    # Channel structure:
    # - num_object_types: explicit object type channels
    # - 3: "other" object channels (overlappable, immobile, mobile)
    # - num_agents: per-agent position channels
    # - 1: query agent channel
    # - 1: "other humans" channel
    num_other_object_channels = 3
    num_channels = num_object_types + num_other_object_channels + num_agents + 1 + 1
    
    # Channel indices
    other_overlappable_idx = num_object_types
    other_immobile_idx = num_object_types + 1
    other_mobile_idx = num_object_types + 2
    agent_channels_start = num_object_types + num_other_object_channels
    query_agent_channel_idx = agent_channels_start + num_agents
    other_humans_channel_idx = query_agent_channel_idx + 1
    
    # Pre-allocate tensors
    grid_tensors = torch.zeros(batch_size, num_channels, grid_height, grid_width, device=device)
    step_tensors = torch.zeros(batch_size, 1, device=device)
    positions = torch.zeros(batch_size, 2, device=device)
    directions = torch.zeros(batch_size, 4, device=device)
    agent_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    goal_coords = torch.zeros(batch_size, 4, device=device)
    
    # Pre-compute grid objects (shared across batch since grid is static)
    grid_objects_tensor = None
    other_objects_tensor = None
    if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
        grid_objects_tensor = torch.zeros(num_object_types, grid_height, grid_width, device=device)
        other_objects_tensor = torch.zeros(3, grid_height, grid_width, device=device)  # overlappable, immobile, mobile
        for y in range(grid_height):
            for x in range(grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is not None:
                        if cell_type in OBJECT_TYPE_TO_CHANNEL:
                            channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                            if channel_idx < num_object_types:
                                grid_objects_tensor[channel_idx, y, x] = 1.0
                        else:
                            # Object type not in explicit channels - use "other" channels
                            if cell_type in OVERLAPPABLE_OBJECTS:
                                other_objects_tensor[0, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                                other_objects_tensor[2, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                                other_objects_tensor[1, y, x] = 1.0
                            else:
                                if hasattr(cell, 'can_overlap') and cell.can_overlap():
                                    other_objects_tensor[0, y, x] = 1.0
                                else:
                                    other_objects_tensor[1, y, x] = 1.0
    
    for i, t in enumerate(transitions):
        state = t['state']
        human_idx = t['human_idx']
        goal = t['goal']
        
        step_count, agent_states, mobile_objects, _ = state
        
        # 1. Copy grid object channels (if precomputed)
        if grid_objects_tensor is not None:
            grid_tensors[i, :num_object_types] = grid_objects_tensor
        if other_objects_tensor is not None:
            grid_tensors[i, other_overlappable_idx:other_mobile_idx+1] = other_objects_tensor
        
        # 2. Encode mobile objects (if any)
        if mobile_objects:
            for mobile_obj in mobile_objects:
                obj_type = mobile_obj[0]
                obj_x = mobile_obj[1]
                obj_y = mobile_obj[2]
                if 0 <= obj_x < grid_width and 0 <= obj_y < grid_height:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        if channel_idx < num_object_types:
                            grid_tensors[i, channel_idx, int(obj_y), int(obj_x)] = 1.0
                    else:
                        grid_tensors[i, other_mobile_idx, int(obj_y), int(obj_x)] = 1.0
        
        # 3. Encode agent positions in grid (per-agent channels)
        for j, agent_state in enumerate(agent_states):
            if j < num_agents:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    channel_idx = agent_channels_start + j
                    grid_tensors[i, channel_idx, y, x] = 1.0
        
        # 4. Encode query agent channel
        if human_idx < len(agent_states):
            agent_state = agent_states[human_idx]
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid_tensors[i, query_agent_channel_idx, y, x] = 1.0
        
        # 5. Encode "other humans" channel
        if human_agent_indices is not None:
            for j, agent_state in enumerate(agent_states):
                if j == human_idx:
                    continue
                if j in human_agent_indices:
                    x, y = int(agent_state[0]), int(agent_state[1])
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        grid_tensors[i, other_humans_channel_idx, y, x] = 1.0
        
        # Step count normalized
        step_tensors[i, 0] = step_count / max_steps
        
        # Agent position and direction
        agent_state = agent_states[human_idx]
        positions[i, 0] = float(agent_state[0]) / grid_width
        positions[i, 1] = float(agent_state[1]) / grid_height
        dir_idx = int(agent_state[2]) % 4
        directions[i, dir_idx] = 1.0
        
        # Agent index
        agent_indices[i] = human_idx
        
        # Goal coordinates
        if hasattr(goal, 'target_pos'):
            target = goal.target_pos
            goal_coords[i, 0] = float(target[0]) / grid_width
            goal_coords[i, 1] = float(target[1]) / grid_height
            goal_coords[i, 2] = float(target[0]) / grid_width
            goal_coords[i, 3] = float(target[1]) / grid_height
    
    return grid_tensors, step_tensors, positions, directions, agent_indices, goal_coords


def _batch_next_states_to_tensors(
    transitions: List[Dict[str, Any]],
    grid_width: int,
    grid_height: int,
    num_agents: int,
    num_object_types: int = NUM_OBJECT_TYPE_CHANNELS,
    max_steps: int = 100,
    device: str = 'cpu',
    world_model: Any = None,
    human_agent_indices: Optional[List[int]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert next states from a batch of transitions to batched tensor representations.
    
    Args:
        transitions: List of transition dictionaries with 'next_state', 'human_idx' keys.
        grid_width: Width of the grid.
        grid_height: Height of the grid.
        num_agents: Number of agents.
        num_object_types: Number of object types for encoding.
        max_steps: Maximum steps for normalization.
        device: Torch device.
        world_model: Optional world model for accessing persistent grid objects.
        human_agent_indices: Optional list of human agent indices for "other humans" channel.
    
    Returns:
        Tuple of batched tensors for next states:
            - grid_tensors: (batch, channels, H, W)
            - step_tensors: (batch, 1)
            - positions: (batch, 2)
            - directions: (batch, 4)
            - agent_indices: (batch,)
    """
    batch_size = len(transitions)
    
    # Channel structure matches _batch_states_to_tensors
    num_other_object_channels = 3
    num_channels = num_object_types + num_other_object_channels + num_agents + 1 + 1
    
    # Channel indices
    other_overlappable_idx = num_object_types
    other_immobile_idx = num_object_types + 1
    other_mobile_idx = num_object_types + 2
    agent_channels_start = num_object_types + num_other_object_channels
    query_agent_channel_idx = agent_channels_start + num_agents
    other_humans_channel_idx = query_agent_channel_idx + 1
    
    # Pre-allocate tensors
    grid_tensors = torch.zeros(batch_size, num_channels, grid_height, grid_width, device=device)
    step_tensors = torch.zeros(batch_size, 1, device=device)
    positions = torch.zeros(batch_size, 2, device=device)
    directions = torch.zeros(batch_size, 4, device=device)
    agent_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    # Pre-compute grid objects (shared across batch since grid is static)
    grid_objects_tensor = None
    other_objects_tensor = None
    if world_model is not None and hasattr(world_model, 'grid') and world_model.grid is not None:
        grid_objects_tensor = torch.zeros(num_object_types, grid_height, grid_width, device=device)
        other_objects_tensor = torch.zeros(3, grid_height, grid_width, device=device)
        for y in range(grid_height):
            for x in range(grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type is not None:
                        if cell_type in OBJECT_TYPE_TO_CHANNEL:
                            channel_idx = OBJECT_TYPE_TO_CHANNEL[cell_type]
                            if channel_idx < num_object_types:
                                grid_objects_tensor[channel_idx, y, x] = 1.0
                        else:
                            if cell_type in OVERLAPPABLE_OBJECTS:
                                other_objects_tensor[0, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_MOBILE_OBJECTS:
                                other_objects_tensor[2, y, x] = 1.0
                            elif cell_type in NON_OVERLAPPABLE_IMMOBILE_OBJECTS:
                                other_objects_tensor[1, y, x] = 1.0
                            else:
                                if hasattr(cell, 'can_overlap') and cell.can_overlap():
                                    other_objects_tensor[0, y, x] = 1.0
                                else:
                                    other_objects_tensor[1, y, x] = 1.0
    
    for i, t in enumerate(transitions):
        state = t['next_state']
        human_idx = t['human_idx']
        
        step_count, agent_states, mobile_objects, _ = state
        
        # 1. Copy grid object channels (if precomputed)
        if grid_objects_tensor is not None:
            grid_tensors[i, :num_object_types] = grid_objects_tensor
        if other_objects_tensor is not None:
            grid_tensors[i, other_overlappable_idx:other_mobile_idx+1] = other_objects_tensor
        
        # 2. Encode mobile objects (if any)
        if mobile_objects:
            for mobile_obj in mobile_objects:
                obj_type = mobile_obj[0]
                obj_x = mobile_obj[1]
                obj_y = mobile_obj[2]
                if 0 <= obj_x < grid_width and 0 <= obj_y < grid_height:
                    if obj_type in OBJECT_TYPE_TO_CHANNEL:
                        channel_idx = OBJECT_TYPE_TO_CHANNEL[obj_type]
                        if channel_idx < num_object_types:
                            grid_tensors[i, channel_idx, int(obj_y), int(obj_x)] = 1.0
                    else:
                        grid_tensors[i, other_mobile_idx, int(obj_y), int(obj_x)] = 1.0
        
        # 3. Encode agent positions in grid
        for j, agent_state in enumerate(agent_states):
            if j < num_agents:
                x, y = int(agent_state[0]), int(agent_state[1])
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    channel_idx = agent_channels_start + j
                    grid_tensors[i, channel_idx, y, x] = 1.0
        
        # 4. Encode query agent channel
        if human_idx < len(agent_states):
            agent_state = agent_states[human_idx]
            x, y = int(agent_state[0]), int(agent_state[1])
            if 0 <= x < grid_width and 0 <= y < grid_height:
                grid_tensors[i, query_agent_channel_idx, y, x] = 1.0
        
        # 5. Encode "other humans" channel
        if human_agent_indices is not None:
            for j, agent_state in enumerate(agent_states):
                if j == human_idx:
                    continue
                if j in human_agent_indices:
                    x, y = int(agent_state[0]), int(agent_state[1])
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        grid_tensors[i, other_humans_channel_idx, y, x] = 1.0
        
        # Step count normalized
        step_tensors[i, 0] = step_count / max_steps
        
        # Agent position and direction
        agent_state = agent_states[human_idx]
        positions[i, 0] = float(agent_state[0]) / grid_width
        positions[i, 1] = float(agent_state[1]) / grid_height
        dir_idx = int(agent_state[2]) % 4
        directions[i, dir_idx] = 1.0
        
        # Agent index
        agent_indices[i] = human_idx
    
    return grid_tensors, step_tensors, positions, directions, agent_indices


class ReplayBuffer:
    """
    Experience replay buffer for storing transitions.
    
    Stores transitions as dictionaries and supports random sampling for
    mini-batch training. This reduces variance and decorrelates samples
    compared to online per-transition updates.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, transition: Dict[str, Any]):
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a random batch of transitions."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class PathDistanceCalculator:
    """
    Computes shortest path distances between cells using precomputed paths on a stripped grid.
    
    At initialization, creates a "stripped down" grid version that only contains impassable
    obstacles (walls, magic walls, lava) and precomputes shortest paths between all 
    passable cells using BFS.
    
    At runtime, calculates a "distance indicator" by walking along the precomputed shortest
    path and summing parameterized "passing difficulty" scores based on what's currently
    on each cell.
    
    Default passing difficulty scores:
        - Empty cell or open door: 1
        - Closed door, another agent, or block: 2
        - Pickable object (key, ball, box): 3
        - Locked door: 25 (need to find a key)
        - Rock: 50
        - Wall/MagicWall/Lava (impassable): inf
    
    Args:
        world_model: The environment with grid, width, height attributes.
        passing_costs: Optional dict mapping object types to passing costs.
                      Keys are object type strings ('empty', 'door_open', 'door_closed',
                      'door_locked', 'agent', 'block', 'pickable', 'rock', 'wall', 'lava').
    """
    
    # Default passing costs for different object types
    DEFAULT_PASSING_COSTS = {
        'empty': 1,
        'door_open': 1,
        'door_closed': 2,      # Can be opened easily
        'door_locked': 25,     # Need to find a key first
        'agent': 2,
        'block': 2,
        'pickable': 3,  # key, ball, box
        'rock': 50,
        'wall': float('inf'),
        'magicwall': float('inf'),
        'lava': float('inf'),  # Deadly - impassable
        'unsteadyground': 2,
        'unknown': 2,
    }
    
    def __init__(self, world_model: Any, passing_costs: Optional[Dict[str, float]] = None):
        """
        Initialize the path distance calculator.
        
        Precomputes shortest paths between all pairs of empty cells on a stripped
        grid containing only walls.
        """
        self.grid_width = world_model.width
        self.grid_height = world_model.height
        self.passing_costs = passing_costs or self.DEFAULT_PASSING_COSTS.copy()
        
        # Create wall-only grid and precompute shortest paths
        self._wall_grid = self._create_wall_grid(world_model)
        self._shortest_paths = self._precompute_shortest_paths()
        self.feasible_range = self._compute_feasible_range()
    
    def _create_wall_grid(self, world_model: Any) -> np.ndarray:
        """
        Create a boolean grid where True = impassable, False = passable.
        
        Considers Wall, MagicWall, and Lava as impassable obstacles.
        """
        wall_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = world_model.grid.get(x, y)
                if cell is not None:
                    # Check if it's an impassable type (Wall, MagicWall, or Lava)
                    cell_type = getattr(cell, 'type', None)
                    if cell_type in ('wall', 'magicwall', 'lava'):
                        wall_grid[y, x] = True
        
        return wall_grid
    
    def _precompute_shortest_paths(self) -> Dict[Tuple[int, int], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        """
        Precompute shortest paths between all pairs of empty cells using BFS.
        
        Returns a dict: source_pos -> {target_pos -> path} where path is a list
        of (x, y) coordinates from source to target (inclusive).
        """
        from collections import deque
        
        paths = {}
        
        # Find all empty cells (not walls)
        empty_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not self._wall_grid[y, x]:
                    empty_cells.append((x, y))
        
        # For each empty cell, compute shortest paths to all other empty cells
        for source in empty_cells:
            paths[source] = {}
            paths[source][source] = [source]  # Path to self is just the source
            
            # BFS from source
            visited = {source: [source]}
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                cx, cy = current
                
                # Check 4 neighbors (up, down, left, right)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    
                    # Check bounds
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        neighbor = (nx, ny)
                        
                        # Check if not a wall and not visited
                        if not self._wall_grid[ny, nx] and neighbor not in visited:
                            # Path to neighbor is path to current + neighbor
                            visited[neighbor] = visited[current] + [neighbor]
                            paths[source][neighbor] = visited[neighbor]
                            queue.append(neighbor)
        
        return paths
    
    def get_shortest_path(self, source: Tuple[int, int], target: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Get the precomputed shortest path from source to target.
        
        Returns None if no path exists (one or both positions are walls).
        """
        if source in self._shortest_paths:
            return self._shortest_paths[source].get(target)
        return None
    
    def compute_path_cost(self, source: Tuple[int, int], target: Tuple[int, int], 
                          world_model: Any) -> float:
        """
        Compute the path cost from source to target based on current grid state.
        
        Walks along the precomputed shortest path and sums up passing difficulty
        scores based on what's currently on each cell.
        
        Args:
            source: (x, y) starting position
            target: (x, y) target position
            world_model: Current environment state for checking cell contents
        
        Returns:
            Total path cost (sum of passing difficulties), or inf if no path exists.
        """
        path = self.get_shortest_path(source, target)
        
        if path is None:
            return float('inf')
        
        if source == target:
            return 0.0
        
        # Build agent position lookup once for efficiency
        agent_positions = set()
        for agent in world_model.agents:
            if agent.pos is not None:
                agent_positions.add((int(agent.pos[0]), int(agent.pos[1])))
        
        total_cost = 0.0
        
        # Sum passing costs for each cell along the path (excluding source)
        for pos in path[1:]:  # Skip source position
            x, y = pos
            cell = world_model.grid.get(x, y)
            cost = self._get_cell_passing_cost(cell, pos, agent_positions)
            total_cost += cost
        
        return total_cost
    
    def _get_cell_passing_cost(self, cell: Any, pos: Tuple[int, int], 
                               agent_positions: set) -> float:
        """
        Get the passing cost for a cell based on its current contents.
        
        Args:
            cell: The object at this cell (or None for empty)
            pos: (x, y) position of the cell
            agent_positions: Set of (x, y) positions where agents are located
        
        Returns:
            Passing cost for this cell.
        """
        # Check for agent at this position first (O(1) lookup)
        if pos in agent_positions:
            return self.passing_costs.get('agent', 2)
        
        if cell is None:
            return self.passing_costs.get('empty', 1)
        
        cell_type = getattr(cell, 'type', None)
        
        # Handle different object types
        if cell_type == 'door':
            is_open = getattr(cell, 'is_open', False)
            is_locked = getattr(cell, 'is_locked', False)
            if is_open:
                return self.passing_costs.get('door_open', 1)
            elif is_locked:
                return self.passing_costs.get('door_locked', 25)
            else:
                return self.passing_costs.get('door_closed', 2)
        
        elif cell_type == 'block':
            return self.passing_costs.get('block', 2)
        
        elif cell_type == 'rock':
            return self.passing_costs.get('rock', 50)
        
        elif cell_type in ('key', 'ball', 'box'):
            return self.passing_costs.get('pickable', 3)
        
        elif cell_type in ('wall', 'magicwall'):
            return self.passing_costs.get('wall', float('inf'))
        
        elif cell_type in ('goal', 'floor', 'switch', 'objectgoal'):
            # These are typically passable like empty cells
            return self.passing_costs.get('empty', 1)
        
        elif cell_type == 'lava':
            # Lava is deadly - impassable
            return self.passing_costs.get('lava', float('inf'))
        
        elif cell_type == 'unsteadyground':
            return self.passing_costs.get('unsteadyground', 2)
        
        else:
            # Unknown type - treat as mildly difficult
            return self.passing_costs.get('unknown', 2)
    
    def _compute_feasible_range(self) -> Tuple[float, float]:
        """
        Compute the feasible range of path costs based on grid size and passing costs.
        
        Returns:
            Tuple of (-max_cost, max_cost) for clamping purposes.
        """
        max_cost = (self.grid_width + self.grid_height) * max(
            cost for cost in self.passing_costs.values() if cost < float('inf')
        )
        return (-max_cost, max_cost)
    
    def compute_potential(self, agent_pos: Tuple[int, int], target_pos: Tuple[int, int],
                         world_model: Any, max_cost: Optional[float] = None) -> float:
        """
        Compute the potential function value for reward shaping.
        
        Φ(s) = -path_cost(agent_pos, target_pos) / max_cost
        
        The potential is maximal (0) when agent is at target.
        
        Args:
            agent_pos: (x, y) current agent position
            target_pos: (x, y) target/goal position
            world_model: Current environment for checking cell contents
            max_cost: Maximum possible cost for normalization (default: grid_width + grid_height)
        
        Returns:
            Potential value in range [-1, 0] (approximately).
        """
        if max_cost is None:
            max_cost = (self.grid_width + self.grid_height) * 50  # Rough upper bound
        
        path_cost = self.compute_path_cost(agent_pos, target_pos, world_model)
        
        if path_cost == float('inf'):
            return -1.0  # No path - minimum potential
        
        return -path_cost / max_cost


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
    replay_buffer_size: int = 10000,
    updates_per_episode: int = 4,
    train_phi_network: bool = True,
    epsilon: float = 0.3,
    exploration_policy: np.ndarray = None,
    use_path_based_shaping: bool = False,
    passing_costs: Optional[Dict[str, float]] = None,
    device: str = 'cpu',
    verbose: bool = True,
    world_model_generator: Optional[Callable[[int], Any]] = None,
    episodes_per_model: int = 1
) -> NeuralHumanPolicyPrior:
    """
    Train neural networks to approximate the human policy prior.
    
    This function trains the Q-network using TD learning with experience replay.
    The training approximates the fixed-point computation done by tabular backward induction.
    
    Training procedure:
        1. For each episode, sample random goals for each human agent
        2. Collect trajectory using epsilon-greedy exploration
        3. Store transitions in replay buffer
        4. Sample random mini-batches and train Q-network with TD(0):
           L_Q = E[(Q(s,a,g) - (r + γ * V(s')))²]
        5. Optionally train phi_network to match E_g[softmax(βQ)]
    
    Loss function:
        Q-network: L_Q = E[(Q(s,a,g) - (r + γ*V(s')))²] with TD(0) targets
        Phi-network: L_phi = KL(phi(s,h) || E_g[softmax(β*Q(s,h,g))])
    
    Args:
        world_model: The environment (must support get_state, set_state, step).
            Used as the base environment for network initialization, and as the
            training environment when world_model_generator is not provided.
        human_agent_indices: Indices of human agents to model.
        goal_sampler: Sampler for possible goals. When using world_model_generator,
            the sampler should support being updated with new environments via
            set_world_model() or similar method, or should work with any environment
            of the same structure.
        num_episodes: Number of training episodes.
        steps_per_episode: Steps per episode for state sampling.
        beta: Inverse temperature for Boltzmann policy.
        gamma: Discount factor for returns.
        learning_rate: Learning rate for optimization.
        batch_size: Size of mini-batches for training.
        replay_buffer_size: Maximum number of transitions to store.
        updates_per_episode: Number of gradient updates per episode.
        train_phi_network: Whether to also train the marginal policy network.
        epsilon: Exploration rate for epsilon-greedy policy.
        exploration_policy: Optional fixed exploration policy (action probabilities).
        use_path_based_shaping: If True, use path-based reward shaping with precomputed
            shortest paths and passing difficulty scores. If False (default), use simple 
            Manhattan distance. Note: when using world_model_generator, path-based
            shaping uses Manhattan distance as fallback since paths vary per environment.
        passing_costs: Optional dict mapping object types to passing costs for path-based
            shaping. Keys: 'empty', 'door_open', 'door_closed', 'agent', 'block',
            'pickable', 'rock', 'wall'. See PathDistanceCalculator for defaults.
        device: Torch device ('cpu' or 'cuda').
        verbose: Whether to print training progress.
        world_model_generator: Optional callable that takes an episode index and returns
            a new world model. When provided, a new environment is generated periodically
            (controlled by episodes_per_model), enabling the Q-network to learn policies
            that generalize across different grid layouts. The generated environments must
            have the same dimensions (width, height, num_agents, num_actions) as world_model.
            The episode index enables curriculum learning (e.g., increasing difficulty).
            Example: lambda episode: RandomMultigridEnv(seed=episode)
        episodes_per_model: Number of episodes to run on each generated environment before
            creating a new one. Only used when world_model_generator is provided.
            Default is 1 (new environment each episode). Higher values are more efficient
            but may reduce diversity. For curriculum learning, consider values like 10-50.
    
    Returns:
        NeuralHumanPolicyPrior: Trained policy prior model.
    """
    # Get environment dimensions
    grid_width = world_model.width
    grid_height = world_model.height
    num_agents = len(world_model.agents)
    num_actions = world_model.action_space.n
    max_steps = getattr(world_model, 'max_steps', 100)
    
    # Initialize path distance calculator for reward shaping if enabled
    path_calculator = None
    if use_path_based_shaping:
        path_calculator = PathDistanceCalculator(world_model, passing_costs)
        feasible_range = tuple(np.array(path_calculator.feasible_range) + np.array([0,1]))
        if verbose:
            print("Initialized path-based reward shaping with precomputed shortest paths")
    else:
        feasible_range = (0, 1)  # Q values bounded in [0,1] due to rewards 0 or 1 (at most once)
    
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
        hidden_dim=128,
        feasible_range=feasible_range
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
    
    # Experience replay buffer for batch learning
    replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
    
    # Training statistics
    q_losses = []
    phi_losses = []
    
    # For ensemble training: track current environment
    current_world_model = world_model
    current_path_calculator = path_calculator
    
    if world_model_generator is not None and verbose:
        print(f"Ensemble training enabled: new environment every {episodes_per_model} episode(s)")
    
    # Training loop
    for episode in range(num_episodes):
        # Generate new environment if using world_model_generator
        if world_model_generator is not None and episode % episodes_per_model == 0:
            current_world_model = world_model_generator(episode)
            current_world_model.reset()
            # Update goal sampler if it has a method to do so
            if hasattr(goal_sampler, 'set_world_model'):
                goal_sampler.set_world_model(current_world_model)
            elif hasattr(goal_sampler, 'world_model'):
                goal_sampler.world_model = current_world_model
            # Recompute path calculator for new environment if using path-based shaping
            if use_path_based_shaping:
                current_path_calculator = PathDistanceCalculator(current_world_model, passing_costs)
        else:
            current_world_model.reset()
        
        # Sample a random goal for each human agent
        initial_state = current_world_model.get_state()
        human_goals = {}
        for human_idx in human_agent_indices:
            goal, _ = goal_sampler.sample(initial_state, human_idx)
            human_goals[human_idx] = goal
        
        # Track which humans have reached their goals
        humans_at_goal = {h_idx: False for h_idx in human_agent_indices}
        
        # Collect trajectory
        trajectories = {h_idx: [] for h_idx in human_agent_indices}
        state = initial_state
        
        # Check initial state: if any human starts at their goal, mark as done
        _, init_agent_states, _, _ = initial_state
        for human_idx in human_agent_indices:
            goal = human_goals[human_idx]
            if hasattr(goal, 'target_pos'):
                curr_pos = init_agent_states[human_idx]
                if int(curr_pos[0]) == goal.target_pos[0] and int(curr_pos[1]) == goal.target_pos[1]:
                    humans_at_goal[human_idx] = True
        
        for step in range(steps_per_episode):
            # Get actions for all agents
            actions = []
            
            for agent_idx in range(num_agents):
                if agent_idx in human_agent_indices:
                    # Human uses epsilon-greedy policy based on Q-network
                    goal = human_goals[agent_idx]
                    
                    # Check if this agent's personal episode has already ended
                    # (they reached their goal in a previous step)
                    agent_episode_ended = humans_at_goal[agent_idx]
                    
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
                        # Random action with uniform distribution
                        if exploration_policy is not None:
                            action = np.random.choice(
                                np.arange(num_actions), p=exploration_policy)
                        else:
                            action = np.random.randint(num_actions)
                    else:
                        policy = q_network.get_policy(q_values, beta)
                        action = torch.multinomial(policy, 1).item()
                    
                    # Only store transition if agent's personal episode hasn't ended
                    # Once an agent reaches their goal, their episode is done and we
                    # should not use subsequent steps for learning
                    if not agent_episode_ended:
                        trajectories[agent_idx].append({
                            'state': state,
                            'action': action,
                            'goal': goal,
                            'at_goal_before_action': False,  # We know they weren't at goal yet
                        })
                else:
                    # Non-human agents use random policy
                    action = np.random.randint(num_actions)
                
                actions.append(action)
            
            # Take step
            _, _, done, _ = current_world_model.step(actions)
            next_state = current_world_model.get_state()
            
            # Store rewards with reward shaping for denser feedback
            _, next_agent_states, _, _ = next_state
            _, curr_agent_states, _, _ = state
            
            for human_idx in human_agent_indices:
                # Skip agents whose personal episode has already ended
                was_at_goal = humans_at_goal[human_idx]
                if was_at_goal:
                    # This agent's episode ended in a previous step, don't add more data
                    continue
                
                goal = human_goals[human_idx]
                goal_achieved = float(goal.is_achieved(next_state))
                
                # Base reward: 1.0 when goal is reached, 0 otherwise
                base_reward = goal_achieved
                
                # Potential-based reward shaping (Ng et al. 1999):
                # F(s,a,s') = γ * Φ(s') - Φ(s)
                # This preserves the optimal policy while providing denser feedback.
                shaping_reward = 0.0
                if hasattr(goal, 'target_pos'):
                    target = goal.target_pos
                    
                    # Get agent positions
                    curr_pos = curr_agent_states[human_idx]
                    curr_pos_tuple = (int(curr_pos[0]), int(curr_pos[1]))
                    next_pos = next_agent_states[human_idx]
                    next_pos_tuple = (int(next_pos[0]), int(next_pos[1]))
                    target_tuple = (int(target[0]), int(target[1]))
                    
                    if current_path_calculator is not None:
                        # Use path-based distance with passing difficulty scores
                        # Φ(s) = -path_cost(agent_pos, target) / max_cost
                        max_cost = (grid_width + grid_height) * 50  # Upper bound
                        
                        # Potential at current state
                        phi_s = current_path_calculator.compute_potential(
                            curr_pos_tuple, target_tuple, current_world_model, max_cost)
                        
                        # Potential at next state
                        phi_s_prime = current_path_calculator.compute_potential(
                            next_pos_tuple, target_tuple, current_world_model, max_cost)
                    else:
                        # Fall back to simple Manhattan distance
                        max_dist = grid_width + grid_height
                        
                        # Potential at current state: Φ(s) = -d(s)/max_dist
                        curr_dist = abs(curr_pos[0] - target[0]) + abs(curr_pos[1] - target[1])
                        phi_s = -curr_dist / max_dist
                        
                        # Potential at next state: Φ(s') = -d(s')/max_dist  
                        next_dist = abs(next_pos[0] - target[0]) + abs(next_pos[1] - target[1])
                        phi_s_prime = -next_dist / max_dist
                    
                    # Shaping: γ * Φ(s') - Φ(s)
                    shaping_reward = gamma * phi_s_prime - phi_s
                else:
                    print("Warning: Goal does not have target_pos attribute for shaping reward.")

                reward = base_reward + shaping_reward
                is_terminal = goal_achieved > 0
                
                # Update goal reached status BEFORE storing - this marks end of personal episode
                if goal_achieved > 0:
                    humans_at_goal[human_idx] = True
                
                # Store reward and next state for the last transition
                if trajectories[human_idx]:
                    trajectories[human_idx][-1]['reward'] = reward
                    trajectories[human_idx][-1]['next_state'] = next_state
                    trajectories[human_idx][-1]['done'] = done or is_terminal
            
            if done:
                break
            
            state = next_state
        
        # ====================================================================
        # Add completed transitions to replay buffer
        # ====================================================================
        for human_idx in human_agent_indices:
            trajectory = trajectories[human_idx]
            for i, t in enumerate(trajectory):
                # Only add transitions that have reward information
                if 'reward' in t and 'next_state' in t:
                    # Store transition with all necessary info for batch learning
                    replay_buffer.push({
                        'state': t['state'],
                        'action': t['action'],
                        'goal': t['goal'],
                        'reward': t['reward'],
                        'next_state': t['next_state'],
                        'done': t.get('done', False),
                        'human_idx': human_idx
                    })
        
        # ====================================================================
        # BATCH LEARNING: Train Q-network using mini-batches from replay buffer
        # Uses proper vectorized batch processing for efficiency
        # ====================================================================
        episode_q_loss = 0.0
        num_updates = 0
        
        # Only train if we have enough samples
        if len(replay_buffer) >= batch_size:
            for _ in range(updates_per_episode):
                # Sample random batch from replay buffer
                batch = replay_buffer.sample(batch_size)
                actual_batch_size = len(batch)
                
                q_optimizer.zero_grad()
                
                # Convert batch to tensors using vectorized helper
                # Note: When using world_model_generator, grid encoding from current_world_model
                # may not match older transitions in buffer. This is acceptable as the network
                # learns to generalize across grid layouts.
                (grid_tensors, step_tensors, positions, directions, 
                 agent_indices, goal_coords) = _batch_states_to_tensors(
                    batch, grid_width, grid_height, num_agents,
                    num_object_types=NUM_OBJECT_TYPE_CHANNELS,
                    max_steps=max_steps, device=device,
                    world_model=current_world_model,
                    human_agent_indices=human_agent_indices
                )
                
                # Single batched forward pass for current states
                q_values = q_network(
                    grid_tensors, step_tensors,
                    positions, directions, agent_indices,
                    goal_coords
                )  # Shape: (batch_size, num_actions)
                
                # Extract Q-values for the actions that were taken
                actions = torch.tensor([t['action'] for t in batch], device=device, dtype=torch.long)
                q_values_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: (batch_size,)
                
                # Compute TD targets
                rewards = torch.tensor([t['reward'] for t in batch], device=device, dtype=torch.float32)
                dones = torch.tensor([t['done'] for t in batch], device=device, dtype=torch.bool)
                
                # For non-terminal states, we need to bootstrap from next state
                with torch.no_grad():
                    # Get indices of non-terminal transitions
                    non_terminal_mask = ~dones
                    non_terminal_indices = non_terminal_mask.nonzero(as_tuple=True)[0]
                    
                    # Initialize targets with rewards (correct for terminal states)
                    targets = rewards.clone()
                    
                    if len(non_terminal_indices) > 0:
                        # Get non-terminal transitions
                        non_terminal_batch = [batch[i] for i in non_terminal_indices.tolist()]
                        
                        # Convert next states to tensors
                        (next_grids, next_steps, next_positions, next_directions,
                         next_agent_indices) = _batch_next_states_to_tensors(
                            non_terminal_batch, grid_width, grid_height, num_agents,
                            num_object_types=NUM_OBJECT_TYPE_CHANNELS,
                            max_steps=max_steps, device=device,
                            world_model=current_world_model,
                            human_agent_indices=human_agent_indices
                        )
                        
                        # Get goal coords for non-terminal transitions
                        non_terminal_goal_coords = goal_coords[non_terminal_indices]
                        
                        # Single batched forward pass for next states
                        next_q_values = q_network(
                            next_grids, next_steps,
                            next_positions, next_directions, next_agent_indices,
                            non_terminal_goal_coords
                        )  # Shape: (num_non_terminal, num_actions)
                        
                        # Compute V(s') = sum_a π(a|s') Q(s',a) for Boltzmann policy
                        # Q-values are bounded to [0,1] by sigmoid, so this is stable
                        next_policy = q_network.get_policy(next_q_values, beta)
                        next_v = (next_policy * next_q_values).sum(dim=1)  # Shape: (num_non_terminal,)
                        
                        # Update targets for non-terminal states: r + γ * V(s')
                        targets[non_terminal_indices] = rewards[non_terminal_indices] + gamma * next_v
                
                # Clamp targets to feasible range since Q-values are bounded
                if feasible_range is not None:  
                    targets = torch.clamp(targets, feasible_range[0], feasible_range[1])
                
                # Compute MSE loss over the batch
                loss = F.mse_loss(q_values_selected, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
                q_optimizer.step()
                
                episode_q_loss += loss.item()
                num_updates += 1
        
        if num_updates > 0:
            q_losses.append(episode_q_loss / num_updates)
        
        # ====================================================================
        # BATCH LEARNING: Train phi-network using mini-batches (optional)
        # Uses proper vectorized batch processing for efficiency
        # ====================================================================
        if phi_network is not None and phi_optimizer is not None and len(replay_buffer) >= batch_size:
            episode_phi_loss = 0.0
            phi_updates = 0
            
            for _ in range(updates_per_episode):
                batch = replay_buffer.sample(batch_size)
                
                phi_optimizer.zero_grad()
                
                # Convert batch to tensors using vectorized helper
                # Pass world_model and human_agent_indices for full grid encoding
                (grid_tensors, step_tensors, positions, directions, 
                 agent_indices, goal_coords) = _batch_states_to_tensors(
                    batch, grid_width, grid_height, num_agents,
                    num_object_types=NUM_OBJECT_TYPE_CHANNELS,
                    max_steps=max_steps, device=device,
                    world_model=world_model,
                    human_agent_indices=human_agent_indices
                )
                
                # Get target policy from Q-network (detached) - single batched forward pass
                with torch.no_grad():
                    q_values = q_network(
                        grid_tensors, step_tensors,
                        positions, directions, agent_indices,
                        goal_coords
                    )
                    target_policy = q_network.get_policy(q_values, beta)
                
                # Get predicted policy from phi-network - single batched forward pass
                predicted_policy = phi_network(
                    grid_tensors, step_tensors,
                    positions, directions, agent_indices
                )
                
                # KL divergence loss over the batch
                phi_loss = F.kl_div(
                    predicted_policy.log(),
                    target_policy,
                    reduction='batchmean'
                )
                
                phi_loss.backward()
                phi_optimizer.step()
                
                episode_phi_loss += phi_loss.item()
                phi_updates += 1
            
            if phi_updates > 0:
                phi_losses.append(episode_phi_loss / phi_updates)
        
        # Logging
        if verbose and (episode + 1) % 100 == 0:
            avg_q_loss = np.mean(q_losses[-100:]) if q_losses else 0.0
            avg_phi_loss = np.mean(phi_losses[-100:]) if phi_losses else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Q-Loss = {avg_q_loss:.4f}, Phi-Loss = {avg_phi_loss:.4f}, "
                  f"Buffer = {len(replay_buffer)}")
    
    if verbose:
        print("Training complete!")
        print(f"  Replay buffer size: {len(replay_buffer)}")
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
