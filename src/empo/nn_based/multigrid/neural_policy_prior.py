"""
Neural human policy prior for multigrid environments.

Extends BaseNeuralHumanPolicyPrior with multigrid-specific implementation.
"""

import torch
import torch.optim as optim
from typing import Any, Dict, List, Optional
import random

from empo.possible_goal import PossibleGoalSampler

from ..neural_policy_prior import BaseNeuralHumanPolicyPrior
from ..replay_buffer import ReplayBuffer
from .constants import DEFAULT_ACTION_ENCODING
from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .feature_extraction import get_num_agents_per_color
from .path_distance import PathDistanceCalculator


class MultiGridNeuralHumanPolicyPrior(BaseNeuralHumanPolicyPrior):
    """
    Neural policy prior for multigrid environments.
    
    Extends BaseNeuralHumanPolicyPrior with multigrid-specific:
    - Network creation from multigrid world_model
    - Load with multigrid-specific validation
    """
    
    def __init__(
        self,
        q_network: MultiGridQNetwork,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = 'cpu'
    ):
        policy_network = MultiGridPolicyPriorNetwork(q_network)
        super().__init__(
            q_network=q_network,
            policy_network=policy_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            action_encoding=action_encoding or DEFAULT_ACTION_ENCODING,
            device=device
        )
    
    def _compute_marginal_policy(
        self,
        state: Any,
        agent_idx: int
    ) -> torch.Tensor:
        """Compute marginal policy over goals."""
        if self.goal_sampler is not None:
            goals = list(self.goal_sampler.sample_goals(state, agent_idx, n=10))
        else:
            goals = []
        
        if not goals:
            probs = torch.ones(self.q_network.num_actions, device=self.device)
            return probs / probs.sum()
        
        return self.policy_network.compute_marginal(
            state, self.world_model, agent_idx, goals,
            device=self.device
        )
    
    @classmethod
    def _validate_grid_dimensions(
        cls,
        config: Dict[str, Any],
        world_model: Any
    ) -> None:
        """Validate that grid dimensions match (multigrid-specific)."""
        env_height = getattr(world_model, 'height', None)
        env_width = getattr(world_model, 'width', None)
        
        if env_height is not None and env_height != config.get('grid_height'):
            raise ValueError(
                f"Grid dimensions mismatch: saved height={config.get('grid_height')}, "
                f"environment height={env_height}"
            )
        if env_width is not None and env_width != config.get('grid_width'):
            raise ValueError(
                f"Grid dimensions mismatch: saved width={config.get('grid_width')}, "
                f"environment width={env_width}"
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
    ) -> 'MultiGridNeuralHumanPolicyPrior':
        """
        Load a model from file.
        
        Args:
            filepath: Path to saved model.
            world_model: New environment.
            human_agent_indices: Human agent indices.
            goal_sampler: Goal sampler.
            infeasible_actions_become: Action to remap unsupported actions to.
            device: Torch device.
        
        Returns:
            Loaded MultiGridNeuralHumanPolicyPrior instance.
        
        Raises:
            ValueError: If grid dimensions don't match or actions conflict.
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Validate grid dimensions (multigrid-specific)
        cls._validate_grid_dimensions(config, world_model)
        # Validate using base class method
        saved_encoding = config.get('action_encoding', DEFAULT_ACTION_ENCODING)
        cls._validate_action_encoding(saved_encoding, world_model)
        
        # Get agent configuration from world_model
        num_agents_per_color = get_num_agents_per_color(world_model)
        if not num_agents_per_color:
            num_agents_per_color = config['num_agents_per_color']
        
        # Create Q-network with saved configuration
        q_network = MultiGridQNetwork(
            grid_height=config['grid_height'],
            grid_width=config['grid_width'],
            num_actions=config['num_actions'],
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=config.get('num_agent_colors', 7),
            state_feature_dim=config.get('state_feature_dim', 256),
            goal_feature_dim=config.get('goal_feature_dim', 32),
            hidden_dim=config.get('hidden_dim', 256),
            beta=config.get('beta', 1.0),
            max_kill_buttons=config.get('max_kill_buttons', 4),
            max_pause_switches=config.get('max_pause_switches', 4),
            max_disabling_switches=config.get('max_disabling_switches', 4),
            max_control_buttons=config.get('max_control_buttons', 4),
        )
        
        # Load state dict with strict=False to allow size mismatches for agent encoder
        # (enables policy transfer across different agent configurations)
        try:
            q_network.load_state_dict(checkpoint['q_network_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                # Partial loading for policy transfer - compatible layers only
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


def train_multigrid_neural_policy_prior(
    world_model: Any = None,
    env: Any = None,
    human_agent_indices: List[int] = None,
    goal_sampler: PossibleGoalSampler = None,
    num_episodes: int = 1000,
    steps_per_episode: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    beta: float = 1.0,
    buffer_capacity: int = 100000,
    replay_buffer_size: int = None,  # Alias for buffer_capacity
    target_update_freq: int = 100,
    state_feature_dim: int = 256,
    goal_feature_dim: int = 32,
    hidden_dim: int = 256,
    device: str = 'cpu',
    verbose: bool = True,
    reward_shaping: bool = True,
    use_path_based_shaping: bool = None,  # Alias for reward_shaping
    epsilon: float = 0.3,
    exploration_policy: Optional[List[float]] = None,
    updates_per_episode: int = 1,
    train_phi_network: bool = False,  # Kept for API compatibility
    world_model_generator: Optional[Any] = None,
    episodes_per_model: int = 1
) -> MultiGridNeuralHumanPolicyPrior:
    """
    Train a neural policy prior for multigrid environments.
    
    Uses Q-learning with experience replay.
    
    Args:
        world_model: Multigrid environment (alias for env).
        env: Multigrid environment.
        human_agent_indices: Indices of human agents.
        goal_sampler: Sampler for training goals.
        num_episodes: Number of training episodes.
        steps_per_episode: Steps per episode.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        gamma: Discount factor.
        beta: Boltzmann temperature.
        buffer_capacity: Replay buffer capacity.
        replay_buffer_size: Alias for buffer_capacity.
        target_update_freq: Steps between target network updates.
        state_feature_dim: State encoder feature dim (includes grid, agent, interactive).
        goal_feature_dim: Goal encoder feature dim.
        hidden_dim: Hidden layer dim.
        device: Torch device.
        verbose: Print progress.
        reward_shaping: Use distance-based reward shaping.
        use_path_based_shaping: Alias for reward_shaping.
        epsilon: Exploration rate for epsilon-greedy.
        exploration_policy: Optional action probability weights for exploration.
        updates_per_episode: Number of training updates per episode.
        train_phi_network: Kept for API compatibility (unused).
        world_model_generator: Optional generator for environment ensemble training.
        episodes_per_model: Episodes per environment when using ensemble.
    
    Returns:
        Trained MultiGridNeuralHumanPolicyPrior.
    """
    # Handle parameter aliases
    if world_model is not None and env is None:
        env = world_model
    if env is None:
        raise ValueError("Must provide either 'env' or 'world_model'")
    
    if replay_buffer_size is not None:
        buffer_capacity = replay_buffer_size
    
    if use_path_based_shaping is not None:
        reward_shaping = use_path_based_shaping
    
    # Get environment info
    grid_height = getattr(env, 'height', 10)
    grid_width = getattr(env, 'width', 10)
    
    # Get number of actions - handle action_space, action enum classes, and instances
    num_actions = 8  # Default
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        actions = getattr(env, 'actions', None)
        if actions is not None:
            if hasattr(actions, '__len__'):
                num_actions = len(actions)
            elif hasattr(actions, '__members__'):
                # It's an enum class
                num_actions = len(actions.__members__)
    
    num_agents_per_color = get_num_agents_per_color(env)
    
    if not num_agents_per_color:
        num_agents_per_color = {'grey': len(human_agent_indices)}
    
    num_agent_colors = len(set(num_agents_per_color.keys()))
    
    # Create Q-network with unified state encoder
    q_network = MultiGridQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_actions=num_actions,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=num_agent_colors,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta
    ).to(device)
    
    # Target network
    target_network = MultiGridQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_actions=num_actions,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=num_agent_colors,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Convert numpy array exploration_policy to list if needed
    if exploration_policy is not None:
        if hasattr(exploration_policy, 'tolist'):
            exploration_policy = exploration_policy.tolist()
    
    # Use generic Trainer with exploration_policy
    from ..trainer import Trainer
    trainer = Trainer(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        target_update_freq=target_update_freq,
        device=device,
        exploration_policy=exploration_policy
    )
    
    path_calc = PathDistanceCalculator(grid_height, grid_width) if reward_shaping else None
    
    # Handle environment ensemble training
    current_env = env
    episode_count_for_model = 0
    
    for episode in range(num_episodes):
        # Switch environment if using ensemble
        if world_model_generator is not None:
            episode_count_for_model += 1
            if episode_count_for_model >= episodes_per_model:
                # world_model_generator is a function that takes episode number
                if callable(world_model_generator):
                    current_env = world_model_generator(episode)
                else:
                    # It's an iterator
                    try:
                        current_env = next(world_model_generator)
                    except StopIteration:
                        pass  # Keep using current env
                episode_count_for_model = 0
                
                # Update goal sampler if it has set_world_model method
                if hasattr(goal_sampler, 'set_world_model'):
                    goal_sampler.set_world_model(current_env)
        
        current_env.reset()
        state = current_env.get_state()
        
        for step in range(steps_per_episode):
            agent_idx = random.choice(human_agent_indices)
            
            # Sample goal using sampler
            try:
                goal, _ = goal_sampler.sample(state, agent_idx)
            except Exception:
                continue
            
            if goal is None:
                continue
            
            # Get action using trainer's sample_action with epsilon exploration
            action = trainer.sample_action(state, current_env, agent_idx, goal, epsilon=epsilon)
            
            # Execute action and get next state
            # Build action list with 'still' for other agents
            num_agents = len(current_env.agents) if hasattr(current_env, 'agents') else 1
            actions = [0] * num_agents  # 0 = still
            actions[agent_idx] = action
            
            current_env.step(actions)
            next_state = current_env.get_state()
            
            # Store transition
            trainer.store_transition(state, action, next_state, agent_idx, goal)
            
            state = next_state
        
        # Training updates at end of episode
        for _ in range(updates_per_episode):
            trainer.train_step(batch_size)
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    # Get action encoding
    action_encoding = DEFAULT_ACTION_ENCODING
    if hasattr(env, 'actions'):
        actions = env.actions
        if hasattr(actions, '__members__'):
            # It's an enum class
            action_encoding = {i: name.lower() for i, name in enumerate(actions.__members__.keys())}
        elif hasattr(actions, '__iter__'):
            action_encoding = {i: a.name.lower() for i, a in enumerate(actions)}
    
    return MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        action_encoding=action_encoding,
        device=device
    )
