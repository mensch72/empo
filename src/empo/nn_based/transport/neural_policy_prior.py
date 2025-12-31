"""
Neural human policy prior for transport environment.

Extends BaseNeuralHumanPolicyPrior with transport-specific implementation
using GNN-based encoding for network state.
"""

import torch
import torch.optim as optim
import random
from typing import Any, Dict, List, Optional

from empo.possible_goal import PossibleGoal, PossibleGoalSampler

from ..neural_policy_prior import BaseNeuralHumanPolicyPrior
from ..replay_buffer import ReplayBuffer
from .q_network import TransportQNetwork
from .policy_prior_network import TransportPolicyPriorNetwork
from .constants import NUM_TRANSPORT_ACTIONS


# Default action encoding for transport environment
DEFAULT_TRANSPORT_ACTION_ENCODING = {
    0: 'pass',
    1: 'unboard',
    # 2-11: board_0 to board_9
    # 12-31: dest_0 to dest_19
    # 32-41: depart_0 to depart_9
}
for i in range(10):
    DEFAULT_TRANSPORT_ACTION_ENCODING[2 + i] = f'board_{i}'
for i in range(20):
    DEFAULT_TRANSPORT_ACTION_ENCODING[12 + i] = f'dest_{i}'
for i in range(10):
    DEFAULT_TRANSPORT_ACTION_ENCODING[32 + i] = f'depart_{i}'


class TransportNeuralHumanPolicyPrior(BaseNeuralHumanPolicyPrior):
    """
    Neural policy prior for transport environment.
    
    Extends BaseNeuralHumanPolicyPrior with transport-specific:
    - GNN-based network encoding via TransportQNetwork
    - Support for cluster-based goals
    - Action masking for step-type-specific actions
    
    This class provides the interface for computing action probabilities
    given the current state and optional goal. It uses a learned Q-network
    to estimate action values and converts them to probabilities.
    
    Args:
        q_network: TransportQNetwork instance
        world_model: TransportEnvWrapper instance
        human_agent_indices: List of human agent indices
        goal_sampler: Optional goal sampler for marginal computation
        action_encoding: Optional action name mapping
        device: Torch device
    
    Example:
        >>> q_network = TransportQNetwork(max_nodes=100, num_clusters=10)
        >>> prior = TransportNeuralHumanPolicyPrior(
        ...     q_network=q_network,
        ...     world_model=env,
        ...     human_agent_indices=[0, 1, 2, 3],
        ... )
        >>> 
        >>> # Get action probabilities for an agent
        >>> probs = prior(state=None, agent_idx=0, goal=goal)
        >>> print(probs)
    """
    
    def __init__(
        self,
        q_network: TransportQNetwork,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = 'cpu'
    ):
        policy_network = TransportPolicyPriorNetwork(q_network)
        super().__init__(
            q_network=q_network,
            policy_network=policy_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            action_encoding=action_encoding or DEFAULT_TRANSPORT_ACTION_ENCODING,
            device=device
        )
    
    def _compute_marginal_policy(
        self,
        state: Any,
        agent_idx: int
    ) -> torch.Tensor:
        """
        Compute marginal policy over goals.
        
        If a goal sampler is available, samples goals and averages
        the Q-network policies. Otherwise, returns uniform distribution.
        
        Args:
            state: Ignored (uses current env state)
            agent_idx: Index of the agent
        
        Returns:
            Action probabilities (num_actions,)
        """
        # Sample goals if sampler available
        goals = []
        if self.goal_sampler is not None:
            try:
                # Sample multiple goals
                for _ in range(10):
                    goal, weight = self.goal_sampler.sample(state, agent_idx)
                    if goal is not None:
                        goals.append(goal)
            except (ValueError, RuntimeError, IndexError):
                goals = []
        
        if not goals:
            # Uniform distribution if no goals
            probs = torch.ones(self.q_network.num_actions, device=self.device)
            return probs / probs.sum()
        
        return self.policy_network.compute_marginal(
            state, self.world_model, agent_idx, goals,
            device=self.device
        )
    
    def get_action_mask(self, agent_idx: int) -> torch.Tensor:
        """
        Get action mask for the given agent.
        
        This uses the environment's action_masks() method to determine
        which actions are valid for the current step type and agent state.
        
        Args:
            agent_idx: Index of the agent
        
        Returns:
            Boolean tensor (num_actions,) where True = valid action
        """
        masks = self.world_model.action_masks()
        return torch.tensor(masks[agent_idx], dtype=torch.bool, device=self.device)
    
    def sample(
        self,
        state: Any,
        agent_idx: int,
        goal: Optional['PossibleGoal'] = None,
        apply_action_mask: bool = True,
        beta: float = 5.0
    ) -> int:
        """
        Sample an action from the learned policy.
        
        Uses the Q-network to compute action probabilities and samples
        from the resulting Boltzmann distribution.
        
        Args:
            state: Current state (ignored, uses current env state)
            agent_idx: Index of the agent
            goal: Goal to condition the policy on. If None, uses marginal policy.
            apply_action_mask: If True, mask invalid actions before sampling
            beta: Boltzmann temperature for sampling (higher = more greedy)
        
        Returns:
            Sampled action index
        """
        import torch.nn.functional as F
        
        self.q_network.eval()
        
        with torch.no_grad():
            if goal is not None:
                # Goal-conditioned Q-values
                q_values = self.q_network.forward(
                    state, self.world_model, agent_idx, goal, self.device
                )
            else:
                # Use marginal policy (average over sampled goals)
                probs = self._compute_marginal_policy(state, agent_idx)
                if apply_action_mask:
                    action_mask = self.get_action_mask(agent_idx)
                    probs = probs * action_mask.float()
                    probs = probs / probs.sum()  # Renormalize
                return torch.multinomial(probs.unsqueeze(0), 1).item()
            
            # Apply action mask
            if apply_action_mask:
                action_mask = self.get_action_mask(agent_idx)
                masked_q = q_values.clone()
                masked_q[0, ~action_mask] = float('-inf')
            else:
                masked_q = q_values
            
            # Boltzmann policy
            if beta == float('inf'):
                action = torch.argmax(masked_q, dim=1).item()
            else:
                probs = F.softmax(beta * masked_q, dim=1)
                probs = probs / probs.sum()  # Renormalize
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    @classmethod
    def _validate_network_params(
        cls,
        config: Dict[str, Any],
        world_model: Any
    ) -> None:
        """Validate that network parameters match (transport-specific)."""
        # Check max_nodes
        env_num_nodes = len(world_model.env.network.nodes())
        if env_num_nodes > config.get('max_nodes', 100):
            raise ValueError(
                f"Network too large: saved max_nodes={config.get('max_nodes')}, "
                f"environment nodes={env_num_nodes}"
            )
        
        # Check clusters
        env_num_clusters = world_model.num_clusters if world_model.use_clusters else 0
        if env_num_clusters > config.get('num_clusters', 0):
            raise ValueError(
                f"Too many clusters: saved num_clusters={config.get('num_clusters')}, "
                f"environment clusters={env_num_clusters}"
            )
    
    @classmethod
    def load(
        cls,
        filepath: str,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        infeasible_actions_become: Optional[int] = None,
        device: str = 'cpu'
    ) -> 'TransportNeuralHumanPolicyPrior':
        """
        Load a model from file.
        
        Args:
            filepath: Path to saved model.
            world_model: TransportEnvWrapper instance.
            human_agent_indices: Human agent indices.
            goal_sampler: Goal sampler.
            infeasible_actions_become: Action to remap unsupported actions to.
            device: Torch device.
        
        Returns:
            Loaded TransportNeuralHumanPolicyPrior instance.
        
        Raises:
            ValueError: If network parameters don't match.
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Validate network parameters (transport-specific)
        cls._validate_network_params(config, world_model)
        
        # Validate using base class method
        saved_encoding = config.get('action_encoding', DEFAULT_TRANSPORT_ACTION_ENCODING)
        cls._validate_action_encoding(saved_encoding, world_model)
        
        # Create Q-network from saved configuration
        state_config = config.get('state_encoder_config', {})
        goal_config = config.get('goal_encoder_config', {})
        
        q_network = TransportQNetwork(
            max_nodes=config.get('max_nodes', state_config.get('max_nodes', 100)),
            num_clusters=config.get('num_clusters', state_config.get('num_clusters', 0)),
            num_actions=config.get('num_actions', NUM_TRANSPORT_ACTIONS),
            hidden_dim=config.get('hidden_dim', state_config.get('hidden_dim', 128)),
            beta=config.get('beta', 1.0),
            feasible_range=config.get('feasible_range'),
            state_feature_dim=state_config.get('feature_dim', 128),
            goal_feature_dim=goal_config.get('feature_dim', 32),
            num_gnn_layers=state_config.get('num_gnn_layers', 3),
            gnn_type=state_config.get('gnn_type', 'gcn'),
        )
        
        # Load state dict
        try:
            q_network.load_state_dict(checkpoint['q_network_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                # Partial loading for transfer
                saved_state = checkpoint['q_network_state_dict']
                current_state = q_network.state_dict()
                compatible_state = {}
                for key, value in saved_state.items():
                    if key in current_state and current_state[key].shape == value.shape:
                        compatible_state[key] = value
                current_state.update(compatible_state)
                q_network.load_state_dict(current_state)
            else:
                raise
        
        prior = cls(
            q_network=q_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            action_encoding=saved_encoding,
            device=device
        )
        
        if infeasible_actions_become is not None:
            prior._infeasible_actions_become = infeasible_actions_become
        
        return prior
    
    @classmethod
    def create(
        cls,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        max_nodes: int = 100,
        num_clusters: int = 0,
        num_actions: int = NUM_TRANSPORT_ACTIONS,
        state_feature_dim: int = 128,
        goal_feature_dim: int = 32,
        hidden_dim: int = 128,
        beta: float = 1.0,
        feasible_range: Optional[tuple] = None,
        num_gnn_layers: int = 3,
        gnn_type: str = 'gcn',
        device: str = 'cpu'
    ) -> 'TransportNeuralHumanPolicyPrior':
        """
        Create a new TransportNeuralHumanPolicyPrior with fresh networks.
        
        This is a convenience method for creating a policy prior with
        default or specified network configuration.
        
        Args:
            world_model: TransportEnvWrapper instance
            human_agent_indices: Human agent indices
            goal_sampler: Optional goal sampler
            max_nodes: Maximum number of nodes
            num_clusters: Number of clusters (0 for node-based routing)
            num_actions: Number of actions
            state_feature_dim: State encoder output dimension
            goal_feature_dim: Goal encoder output dimension
            hidden_dim: Hidden layer dimension
            beta: Boltzmann temperature
            feasible_range: Optional Q-value bounds
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN layer type ('gcn', 'gat', 'gin')
            device: Torch device
        
        Returns:
            New TransportNeuralHumanPolicyPrior instance
        """
        q_network = TransportQNetwork(
            max_nodes=max_nodes,
            num_clusters=num_clusters,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            beta=beta,
            feasible_range=feasible_range,
            state_feature_dim=state_feature_dim,
            goal_feature_dim=goal_feature_dim,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
        )
        
        return cls(
            q_network=q_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            device=device
        )


def train_transport_neural_policy_prior(
    env: Any,
    human_agent_indices: List[int],
    goal_sampler: PossibleGoalSampler,
    num_episodes: int = 1000,
    steps_per_episode: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    beta: float = 1.0,
    buffer_capacity: int = 100000,
    target_update_freq: int = 100,
    state_feature_dim: int = 128,
    goal_feature_dim: int = 32,
    hidden_dim: int = 128,
    num_gnn_layers: int = 3,
    gnn_type: str = 'gcn',
    device: str = 'cpu',
    verbose: bool = True,
    reward_shaping: bool = False,
    epsilon: float = 0.3,
    exploration_policy: Optional[List[float]] = None,
    updates_per_episode: int = 1,
    max_nodes: Optional[int] = None,
    num_clusters: Optional[int] = None,
) -> TransportNeuralHumanPolicyPrior:
    """
    Train a neural policy prior for transport environments.
    
    Uses Q-learning with experience replay. The Q-network uses GNN-based
    encoding for the road network state.
    
    Goals can be either node-based (TransportGoal) or cluster-based 
    (TransportClusterGoal) depending on the goal_sampler provided.
    
    Reward shaping (optional):
        When reward_shaping=True, uses potential-based shaping where the potential
        is based on the shortest path distance in the road network, normalized to [-1, 0]:
            Φ(s) = -d(s, goal) / max_distance
        The shaping reward is: F(s, a, s') = γ * Φ(s') - Φ(s)
        
        This ensures the optimal policy is unchanged (potential-based shaping
        preserves optimality) while providing denser reward signals.
    
    Q-value bounds:
        - Without reward_shaping: Q ∈ [0, 1] (goal achievement reward only)
        - With reward_shaping: Q ∈ [-1, 2] (base reward + potential-based shaping)
    
    Args:
        env: TransportEnvWrapper instance
        human_agent_indices: Indices of human agents to train policies for
        goal_sampler: Sampler for training goals (node or cluster goals)
        num_episodes: Number of training episodes
        steps_per_episode: Steps per episode
        batch_size: Training batch size
        learning_rate: Learning rate for Q-network
        gamma: Discount factor
        beta: Boltzmann temperature for soft Q-learning
        buffer_capacity: Replay buffer capacity
        target_update_freq: Steps between target network updates
        state_feature_dim: State encoder output dimension
        goal_feature_dim: Goal encoder output dimension
        hidden_dim: Hidden layer dimension
        num_gnn_layers: Number of GNN layers
        gnn_type: GNN layer type ('gcn', 'gat', 'gin')
        device: Torch device
        verbose: Print progress
        reward_shaping: Use distance-based reward shaping (default False)
        epsilon: Exploration rate for epsilon-greedy
        exploration_policy: Optional action probability weights for exploration
        updates_per_episode: Number of training updates per episode
        max_nodes: Maximum network nodes (default: env's node count)
        num_clusters: Number of clusters (default: from env)
    
    Returns:
        Trained TransportNeuralHumanPolicyPrior
    
    Example:
        >>> from empo.transport import create_transport_env, TransportGoalSampler
        >>> env = create_transport_env(num_humans=4, num_vehicles=2, num_nodes=20)
        >>> goal_sampler = TransportGoalSampler(env, seed=42)
        >>> 
        >>> prior = train_transport_neural_policy_prior(
        ...     env=env,
        ...     human_agent_indices=[0, 1, 2, 3],
        ...     goal_sampler=goal_sampler,
        ...     num_episodes=100,
        ...     verbose=True,
        ... )
        >>> 
        >>> # Save the trained model
        >>> prior.save("transport_prior.pt")
    """
    import networkx as nx
    
    # Get network parameters from environment
    network = env.env.network
    env_num_nodes = len(network.nodes())
    
    if max_nodes is None:
        max_nodes = max(env_num_nodes, 100)
    
    if num_clusters is None:
        num_clusters = env.num_clusters if env.use_clusters else 0
    
    # Compute shortest path lengths for all pairs (used for reward shaping)
    # This is the maximum possible path length in the network
    all_pairs_lengths = dict(nx.all_pairs_shortest_path_length(network))
    max_path_length = 0
    for source in all_pairs_lengths:
        for target, length in all_pairs_lengths[source].items():
            if length > max_path_length:
                max_path_length = length
    
    # Set feasible range based on whether reward shaping is used
    # Following multigrid pattern:
    # - Shaping reward class returns (-1, 1) as the potential range
    # - We then add 1.0 to upper bound for base reward [0, 1]
    # Without shaping: Q ∈ [0, 1] (base reward only)
    # With shaping: Q ∈ [-1, 2] = shaping_range + (0, 1) for base reward
    if reward_shaping:
        # Shaping potential range is (-1, 1), add 1.0 for base reward
        shaping_range = (-1.0, 1.0)
        feasible_range = (shaping_range[0], shaping_range[1] + 1.0)
    else:
        feasible_range = (0.0, 1.0)
    
    # Create Q-network
    q_network = TransportQNetwork(
        max_nodes=max_nodes,
        num_clusters=num_clusters,
        num_actions=NUM_TRANSPORT_ACTIONS,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        num_gnn_layers=num_gnn_layers,
        gnn_type=gnn_type,
    ).to(device)
    
    # Target network
    target_network = TransportQNetwork(
        max_nodes=max_nodes,
        num_clusters=num_clusters,
        num_actions=NUM_TRANSPORT_ACTIONS,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        num_gnn_layers=num_gnn_layers,
        gnn_type=gnn_type,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Convert numpy array exploration_policy to list if needed
    if exploration_policy is not None:
        if hasattr(exploration_policy, 'tolist'):
            exploration_policy = exploration_policy.tolist()
    
    def get_agent_node(positions, agent_name):
        """Get agent's current node or nearest node if on edge."""
        pos = positions.get(agent_name)
        if pos is None:
            return None
        if isinstance(pos, tuple):
            # On edge - return source node
            edge_info, progress = pos
            return edge_info[0]
        return pos
    
    def compute_potential(agent_pos, target_node):
        """
        Compute potential function Φ(s) = -d(s, goal) / max_distance.
        
        Returns value in [-1, 0] where:
        - 0 when at goal
        - -1 when maximally far from goal
        """
        if agent_pos is None or target_node is None:
            return -1.0
        
        if agent_pos == target_node:
            return 0.0
        
        # Get shortest path length
        if agent_pos in all_pairs_lengths and target_node in all_pairs_lengths[agent_pos]:
            dist = all_pairs_lengths[agent_pos][target_node]
        else:
            # No path - return minimum potential
            return -1.0
        
        # Normalize to [-1, 0]
        if max_path_length > 0:
            return -dist / max_path_length
        return 0.0
    
    def compute_reward(pre_positions, post_positions, agent_name, goal):
        """Compute reward with optional potential-based shaping."""
        # Base reward: goal achievement
        base_reward = 1.0 if goal.is_achieved(None) else 0.0
        
        if not reward_shaping:
            return base_reward
        
        # Get target node from goal
        target_node = None
        if hasattr(goal, 'target_node'):
            target_node = goal.target_node
        elif hasattr(goal, 'target_cluster') and hasattr(goal, 'get_target_nodes'):
            # For cluster goals, use centroid
            target_nodes = list(goal.get_target_nodes())
            if target_nodes:
                target_node = target_nodes[0]
        
        if target_node is None:
            return base_reward
        
        # Compute potential-based shaping: F(s,a,s') = γ * Φ(s') - Φ(s)
        curr_node = get_agent_node(pre_positions, agent_name)
        next_node = get_agent_node(post_positions, agent_name)
        
        phi_s = compute_potential(curr_node, target_node)
        phi_s_prime = compute_potential(next_node, target_node)
        
        shaping_reward = gamma * phi_s_prime - phi_s
        
        return base_reward + shaping_reward
    
    # Custom replay buffer that keeps env reference
    replay_buffer = ReplayBuffer(buffer_capacity)
    total_steps = 0
    
    # Training loop
    for episode in range(num_episodes):
        env.reset()
        
        for step in range(steps_per_episode):
            agent_idx = random.choice(human_agent_indices)
            
            # Sample goal using sampler
            try:
                goal, _ = goal_sampler.sample(None, agent_idx)
            except (ValueError, RuntimeError, IndexError):
                continue
            
            if goal is None:
                continue
            
            # Get action using epsilon-greedy
            if random.random() < epsilon:
                # Explore: sample from action mask or exploration policy
                action_mask = env.action_masks()[agent_idx]
                valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                if valid_actions:
                    if exploration_policy is not None:
                        weights = [exploration_policy[a] for a in valid_actions]
                        action = random.choices(valid_actions, weights=weights)[0]
                    else:
                        action = random.choice(valid_actions)
                else:
                    action = 0  # PASS
            else:
                # Exploit: use Q-network
                q_network.eval()
                with torch.no_grad():
                    q_values = q_network.forward(
                        None, env, agent_idx, goal, device
                    )
                    # Apply action mask
                    action_mask = torch.tensor(
                        env.action_masks()[agent_idx], 
                        dtype=torch.bool, 
                        device=device
                    )
                    masked_q = q_values.clone()
                    masked_q[0, ~action_mask] = float('-inf')
                    probs = q_network.get_policy(masked_q).squeeze(0)
                    action = torch.multinomial(probs, 1).item()
            
            # Store pre-step state info for reward computation
            pre_step_positions = dict(env.env.agent_positions)
            agent_name = env.agents[agent_idx]
            
            # Execute action - use integers directly (TransportActions.PASS = 0)
            from empo.transport import TransportActions
            actions = [TransportActions.PASS] * env.num_agents
            actions[agent_idx] = action  # Just use the integer action
            
            env.step(actions)
            
            # Compute reward
            post_step_positions = dict(env.env.agent_positions)
            reward = compute_reward(pre_step_positions, post_step_positions, agent_name, goal)
            
            # Store transition in replay buffer
            # For transport, we store the full env state snapshot since
            # the Q-network needs the env to extract graph features
            replay_buffer.push(
                state={'env_snapshot': True, 'agent_idx': agent_idx},  # Placeholder
                action=action,
                next_state={'env_snapshot': True},  # Placeholder
                agent_idx=agent_idx,
                goal=goal
            )
            
            total_steps += 1
        
        # Training updates at end of episode
        if len(replay_buffer) >= batch_size:
            for _ in range(updates_per_episode):
                # Sample batch - but we need to train with current env state
                # since we can't snapshot the entire environment
                # Instead, use the current state for TD learning (online learning)
                q_network.train()
                
                batch = replay_buffer.sample(batch_size)
                
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                for transition in batch:
                    action = transition['action']
                    agent_idx = transition['agent_idx']
                    goal = transition['goal']
                    
                    # Get current Q-value using current env state
                    # This is simplified online learning that uses current state
                    # In practice, for graph-based envs with state snapshots,
                    # we'd need to serialize/deserialize the full state
                    q_values = q_network.forward(
                        None, env, agent_idx, goal, device
                    )
                    current_q = q_values[0, action]
                    
                    # Compute target
                    # Goal achievement (reward=1) is treated as terminal state
                    # so no future value is added when goal is reached
                    goal_achieved = goal.is_achieved(None)
                    
                    if goal_achieved:
                        # Terminal state: target = reward = 1.0
                        target = torch.tensor(1.0, device=device)
                    else:
                        # Non-terminal: target = 0 + γ * V(s')
                        with torch.no_grad():
                            next_q = target_network.forward(
                                None, env, agent_idx, goal, device
                            )
                            next_v = q_network.get_value(next_q)
                        target = gamma * next_v
                    
                    loss = (current_q - target) ** 2
                    total_loss = total_loss + loss
                
                avg_loss = total_loss / len(batch)
                
                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()
                
                # Update target network periodically
                if total_steps % target_update_freq == 0:
                    target_network.load_state_dict(q_network.state_dict())
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    return TransportNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        device=device
    )


def _get_env_state(env) -> Dict[str, Any]:
    """
    Extract state dict from transport environment.
    
    The state includes:
    - agent_positions: dict mapping agent names to positions
    - step_type: current step type
    - real_time: simulation time
    - network: the road network graph
    
    Args:
        env: TransportEnvWrapper instance
    
    Returns:
        State dictionary
    """
    return {
        'agent_positions': dict(env.env.agent_positions),
        'step_type': env.step_type,
        'step_type_idx': env.step_type_idx,
        'real_time': env.env.real_time,
        'vehicle_destinations': dict(env.env.vehicle_destinations),
        'human_aboard': dict(env.env.human_aboard),
    }
