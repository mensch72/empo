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
        
        # Validate using base class methods
        cls._validate_grid_dimensions(config, world_model)
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
            state_feature_dim=config.get('state_feature_dim', 128),
            agent_feature_dim=config.get('agent_feature_dim', 64),
            goal_feature_dim=config.get('goal_feature_dim', 32),
            interactive_feature_dim=config.get('interactive_feature_dim', 32),
            hidden_dim=config.get('hidden_dim', 256),
            beta=config.get('beta', 1.0),
            max_kill_buttons=config.get('max_kill_buttons', 4),
            max_pause_switches=config.get('max_pause_switches', 4),
            max_disabling_switches=config.get('max_disabling_switches', 4),
            max_control_buttons=config.get('max_control_buttons', 4),
        )
        
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        
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
    agent_feature_dim: int = 64,
    goal_feature_dim: int = 32,
    interactive_feature_dim: int = 32,
    hidden_dim: int = 256,
    device: str = 'cpu',
    verbose: bool = True,
    reward_shaping: bool = True
) -> MultiGridNeuralHumanPolicyPrior:
    """
    Train a neural policy prior for multigrid environments.
    
    Uses Q-learning with experience replay.
    
    Args:
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
        target_update_freq: Steps between target network updates.
        state_feature_dim: State encoder feature dim.
        agent_feature_dim: Agent encoder feature dim.
        goal_feature_dim: Goal encoder feature dim.
        interactive_feature_dim: Interactive encoder feature dim.
        hidden_dim: Hidden layer dim.
        device: Torch device.
        verbose: Print progress.
        reward_shaping: Use distance-based reward shaping.
    
    Returns:
        Trained MultiGridNeuralHumanPolicyPrior.
    """
    # Get environment info
    grid_height = getattr(env, 'height', 10)
    grid_width = getattr(env, 'width', 10)
    num_actions = len(getattr(env, 'actions', range(8)))
    num_agents_per_color = get_num_agents_per_color(env)
    
    if not num_agents_per_color:
        num_agents_per_color = {'grey': len(human_agent_indices)}
    
    num_agent_colors = len(set(num_agents_per_color.keys()))
    
    # Create Q-network
    q_network = MultiGridQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_actions=num_actions,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=num_agent_colors,
        state_feature_dim=state_feature_dim,
        agent_feature_dim=agent_feature_dim,
        goal_feature_dim=goal_feature_dim,
        interactive_feature_dim=interactive_feature_dim,
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
        agent_feature_dim=agent_feature_dim,
        goal_feature_dim=goal_feature_dim,
        interactive_feature_dim=interactive_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    path_calc = PathDistanceCalculator(grid_height, grid_width) if reward_shaping else None
    
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.get_state()
        
        for step in range(steps_per_episode):
            agent_idx = random.choice(human_agent_indices)
            goals = list(goal_sampler.sample_goals(state, agent_idx, n=1))
            if not goals:
                continue
            goal = goals[0]
            
            # Get action from current policy
            q_network.eval()
            with torch.no_grad():
                q_values = q_network.encode_and_forward(
                    state, env, agent_idx, goal, device
                )
                probs = q_network.get_policy(q_values).squeeze(0)
                action = torch.multinomial(probs, 1).item()
            
            next_state = state  # Placeholder
            
            replay_buffer.push(state, action, next_state, agent_idx, goal)
            
            # Train
            if len(replay_buffer) >= batch_size:
                q_network.train()
                batch = replay_buffer.sample(batch_size)
                
                loss = _compute_td_loss(
                    q_network, target_network, batch, gamma, device
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_steps += 1
            
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())
            
            state = next_state
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    # Get action encoding
    action_encoding = DEFAULT_ACTION_ENCODING
    if hasattr(env, 'actions'):
        action_encoding = {i: a.name.lower() for i, a in enumerate(env.actions)}
    
    return MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        action_encoding=action_encoding,
        device=device
    )


def _compute_td_loss(
    q_network: MultiGridQNetwork,
    target_network: MultiGridQNetwork,
    batch: List[Dict],
    gamma: float,
    device: str
) -> torch.Tensor:
    """Compute TD loss for a batch of transitions."""
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for transition in batch:
        state = transition['state']
        action = transition['action']
        next_state = transition['next_state']
        agent_idx = transition['agent_idx']
        goal = transition['goal']
        
        q_values = q_network.encode_and_forward(
            state, None, agent_idx, goal, device
        )
        current_q = q_values[0, action]
        
        with torch.no_grad():
            next_q = target_network.encode_and_forward(
                next_state, None, agent_idx, goal, device
            )
            next_v = q_network.get_value(next_q)
        
        target = gamma * next_v
        loss = (current_q - target) ** 2
        total_loss = total_loss + loss
    
    return total_loss / len(batch)
