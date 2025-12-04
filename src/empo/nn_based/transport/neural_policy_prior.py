"""
Neural human policy prior for transport environment.

Extends BaseNeuralHumanPolicyPrior with transport-specific implementation
using GNN-based encoding for network state.
"""

import torch
from typing import Any, Dict, List, Optional

from empo.possible_goal import PossibleGoalSampler

from ..neural_policy_prior import BaseNeuralHumanPolicyPrior
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
