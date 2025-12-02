"""
Neural human policy prior for multigrid environments.

This module provides the main NeuralHumanPolicyPrior implementation for multigrid,
including save/load functionality for policy transfer across different environment
configurations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler

from .constants import (
    STANDARD_COLORS,
    COLOR_TO_IDX,
    NUM_OBJECT_TYPE_CHANNELS,
    NUM_STANDARD_COLORS,
    AGENT_FEATURE_SIZE,
    NUM_GLOBAL_WORLD_FEATURES,
    DEFAULT_ACTION_ENCODING,
)
from .state_encoder import MultiGridStateEncoder
from .agent_encoder import MultiGridAgentEncoder
from .goal_encoder import MultiGridGoalEncoder
from .interactive_encoder import MultiGridInteractiveObjectEncoder
from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .feature_extraction import (
    extract_agent_features,
    extract_interactive_objects,
    extract_global_world_features,
    extract_agent_colors,
)


class MultiGridNeuralHumanPolicyPrior(HumanPolicyPrior):
    """
    Neural network-based human policy prior for multigrid environments.
    
    This class implements the HumanPolicyPrior interface using neural networks
    trained on sampled states. It supports:
    - Complete encoding of ALL multigrid state features
    - Save/load for policy transfer across different configurations
    - Handling of different action spaces via remapping
    
    Args:
        world_model: Multigrid environment.
        human_agent_indices: Indices of human agents.
        q_network: Trained Q-network.
        policy_prior_network: Policy prior network wrapping the Q-network.
        goal_sampler: Sampler for possible goals.
        beta: Inverse temperature for Boltzmann policy.
        action_encoding: Mapping from action index to action name.
        device: Torch device.
    """
    
    def __init__(
        self,
        world_model: Any,
        human_agent_indices: List[int],
        q_network: MultiGridQNetwork,
        policy_prior_network: MultiGridPolicyPriorNetwork,
        goal_sampler: PossibleGoalSampler,
        beta: float = 1.0,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = 'cpu'
    ):
        super().__init__(world_model)
        self.world_model = world_model
        self.human_agent_indices = human_agent_indices
        self.q_network = q_network
        self.policy_prior_network = policy_prior_network
        self.goal_sampler = goal_sampler
        self.beta = beta
        self.action_encoding = action_encoding or DEFAULT_ACTION_ENCODING
        self.device = device
        
        # Extract agent colors for encoding
        self.agent_colors = extract_agent_colors(world_model)
        
        # Move networks to device
        self.q_network.to(device)
        self.policy_prior_network.to(device)
    
    def __call__(
        self,
        state: Tuple,
        agent_idx: int,
        goal: Optional[PossibleGoal] = None,
        num_goal_samples: int = 10
    ) -> np.ndarray:
        """
        Compute action probabilities for the given state and agent.
        
        Args:
            state: Environment state tuple.
            agent_idx: Index of the agent to compute policy for.
            goal: Optional specific goal. If None, marginalizes over sampled goals.
            num_goal_samples: Number of goals to sample for marginalization.
        
        Returns:
            Action probability distribution as numpy array.
        """
        self.q_network.eval()
        
        with torch.no_grad():
            # Encode state
            grid_tensor, global_features = self.q_network.state_encoder.encode_state(
                state, self.world_model, agent_idx, self.device
            )
            
            # Encode agents
            (query_pos, query_dir, query_abil, query_carr, query_stat,
             all_pos, all_dir, all_abil, all_carr, all_stat, colors) = \
                self.q_network.agent_encoder.encode_agents(
                    state, self.world_model, agent_idx, self.device
                )
            
            # Encode interactive objects
            kb, ps, ds, cb = self.q_network.interactive_encoder.encode_interactive_objects(
                self.world_model, state, self.device
            )
            
            if goal is not None:
                # Single goal - compute Q-values directly
                goal_coords = self.q_network.goal_encoder.encode_goal(goal, self.device)
                
                q_values = self.q_network(
                    grid_tensor, global_features,
                    query_pos, query_dir, query_abil, query_carr, query_stat,
                    all_pos, all_dir, all_abil, all_carr, all_stat, colors,
                    goal_coords,
                    kb, ps, ds, cb
                )
                
                policy = self.q_network.get_policy(q_values, self.beta)
            else:
                # Sample goals and marginalize
                goals = [self.goal_sampler.sample() for _ in range(num_goal_samples)]
                goal_coords_list = [
                    self.q_network.goal_encoder.encode_goal(g, self.device)
                    for g in goals
                ]
                
                policy = self.policy_prior_network(
                    grid_tensor, global_features,
                    query_pos, query_dir, query_abil, query_carr, query_stat,
                    all_pos, all_dir, all_abil, all_carr, all_stat, colors,
                    goal_coords_list, None,
                    kb, ps, ds, cb,
                    beta=self.beta
                )
        
        return policy.cpu().numpy().squeeze()
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the trained model and metadata for later reuse.
        
        Saves:
        - Q-network weights
        - Architectural parameters (grid dims, feature dims, etc.)
        - Action encoding
        - Agent configuration
        
        Args:
            filepath: Path to save the model file.
        """
        filepath = Path(filepath)
        
        metadata = {
            'grid_width': self.q_network.state_encoder.grid_width,
            'grid_height': self.q_network.state_encoder.grid_height,
            'num_object_types': self.q_network.state_encoder.num_object_types,
            'num_agent_colors': self.q_network.state_encoder.num_agent_colors,
            'state_feature_dim': self.q_network.state_encoder.feature_dim,
            'agent_feature_dim': self.q_network.agent_encoder.feature_dim,
            'goal_feature_dim': self.q_network.goal_encoder.feature_dim,
            'interactive_feature_dim': self.q_network.interactive_encoder.feature_dim,
            'num_actions': self.q_network.num_actions,
            'num_agents_per_color': self.q_network.agent_encoder.num_agents_per_color,
            'max_kill_buttons': self.q_network.interactive_encoder.max_kill_buttons,
            'max_pause_switches': self.q_network.interactive_encoder.max_pause_switches,
            'max_disabling_switches': self.q_network.interactive_encoder.max_disabling_switches,
            'max_control_buttons': self.q_network.interactive_encoder.max_control_buttons,
            'action_encoding': self.action_encoding,
            'beta': self.beta,
            'feasible_range': self.q_network.feasible_range,
        }
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'metadata': metadata,
        }, filepath)
    
    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: PossibleGoalSampler,
        action_encoding: Optional[Dict[int, str]] = None,
        infeasible_actions_become: Optional[int] = None,
        device: str = 'cpu'
    ) -> 'MultiGridNeuralHumanPolicyPrior':
        """
        Load a saved model for use with a (potentially different) environment.
        
        Handles policy transfer by:
        - Validating grid dimensions match
        - Detecting and handling action encoding conflicts
        - Clamping agent indices for environments with more agents
        
        Args:
            filepath: Path to saved model file.
            world_model: Target environment.
            human_agent_indices: Human agent indices in target environment.
            goal_sampler: Goal sampler for target environment.
            action_encoding: Optional action encoding for target environment.
            infeasible_actions_become: Action index to map infeasible actions to.
            device: Torch device.
        
        Returns:
            Loaded MultiGridNeuralHumanPolicyPrior instance.
        
        Raises:
            ValueError: If grid dimensions don't match.
            ValueError: If action encodings conflict and no fallback provided.
        """
        filepath = Path(filepath)
        
        # Load with weights_only=False for compatibility, but be aware of security implications
        # Only load models from trusted sources
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        metadata = checkpoint['metadata']
        
        # Validate grid dimensions
        if (world_model.width != metadata['grid_width'] or 
            world_model.height != metadata['grid_height']):
            raise ValueError(
                f"Grid dimensions mismatch: saved ({metadata['grid_width']}x{metadata['grid_height']}) "
                f"vs target ({world_model.width}x{world_model.height})"
            )
        
        # Check action encoding compatibility
        saved_encoding = metadata['action_encoding']
        target_encoding = action_encoding or DEFAULT_ACTION_ENCODING
        
        action_remapping = {}
        for action_idx, action_name in target_encoding.items():
            # Find if this action exists in saved encoding
            found = False
            for saved_idx, saved_name in saved_encoding.items():
                if saved_name == action_name:
                    if saved_idx != action_idx:
                        action_remapping[action_idx] = saved_idx
                    found = True
                    break
            
            if not found:
                if infeasible_actions_become is not None:
                    action_remapping[action_idx] = infeasible_actions_become
                else:
                    raise ValueError(
                        f"Action encoding conflict: action '{action_name}' (idx {action_idx}) "
                        f"not found in saved model. Provide infeasible_actions_become to handle."
                    )
        
        # Reconstruct networks
        state_encoder = MultiGridStateEncoder(
            grid_height=metadata['grid_height'],
            grid_width=metadata['grid_width'],
            num_object_types=metadata['num_object_types'],
            num_agent_colors=metadata['num_agent_colors'],
            feature_dim=metadata['state_feature_dim']
        )
        
        agent_encoder = MultiGridAgentEncoder(
            num_agents_per_color=metadata['num_agents_per_color'],
            feature_dim=metadata['agent_feature_dim']
        )
        
        goal_encoder = MultiGridGoalEncoder(
            feature_dim=metadata['goal_feature_dim']
        )
        
        interactive_encoder = MultiGridInteractiveObjectEncoder(
            max_kill_buttons=metadata['max_kill_buttons'],
            max_pause_switches=metadata['max_pause_switches'],
            max_disabling_switches=metadata['max_disabling_switches'],
            max_control_buttons=metadata['max_control_buttons'],
            feature_dim=metadata['interactive_feature_dim']
        )
        
        q_network = MultiGridQNetwork(
            state_encoder=state_encoder,
            agent_encoder=agent_encoder,
            goal_encoder=goal_encoder,
            interactive_encoder=interactive_encoder,
            num_actions=metadata['num_actions'],
            feasible_range=metadata.get('feasible_range')
        )
        
        # Load weights
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        
        policy_prior_network = MultiGridPolicyPriorNetwork(
            q_network=q_network,
            num_actions=metadata['num_actions']
        )
        
        instance = cls(
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            q_network=q_network,
            policy_prior_network=policy_prior_network,
            goal_sampler=goal_sampler,
            beta=metadata['beta'],
            action_encoding=target_encoding,
            device=device
        )
        
        # Store remapping for use in __call__
        instance._action_remapping = action_remapping
        
        return instance


def train_multigrid_neural_policy_prior(
    env: Any,
    human_agent_indices: List[int],
    goal_sampler: PossibleGoalSampler,
    num_episodes: int = 1000,
    beta: float = 1.0,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_steps_per_episode: int = 100,
    state_feature_dim: int = 128,
    agent_feature_dim: int = 64,
    goal_feature_dim: int = 32,
    interactive_feature_dim: int = 64,
    hidden_dim: int = 256,
    max_kill_buttons: int = 4,
    max_pause_switches: int = 4,
    max_disabling_switches: int = 4,
    max_control_buttons: int = 4,
    device: str = 'cpu',
    verbose: bool = True
) -> MultiGridNeuralHumanPolicyPrior:
    """
    Train a neural policy prior on a multigrid environment.
    
    Uses experience replay and TD-learning to train the Q-network on
    trajectories generated by a random policy.
    
    Args:
        env: Multigrid environment.
        human_agent_indices: Indices of human agents.
        goal_sampler: Sampler for possible goals.
        num_episodes: Number of training episodes.
        beta: Inverse temperature for Boltzmann policy.
        gamma: Discount factor.
        learning_rate: Learning rate for optimizer.
        batch_size: Batch size for training.
        max_steps_per_episode: Maximum steps per episode.
        state_feature_dim: State encoder output dimension.
        agent_feature_dim: Agent encoder output dimension.
        goal_feature_dim: Goal encoder output dimension.
        interactive_feature_dim: Interactive object encoder output dimension.
        hidden_dim: Q-network hidden layer dimension.
        max_kill_buttons: Maximum KillButtons to track.
        max_pause_switches: Maximum PauseSwitches to track.
        max_disabling_switches: Maximum DisablingSwitches to track.
        max_control_buttons: Maximum ControlButtons to track.
        device: Torch device.
        verbose: Whether to print progress.
    
    Returns:
        Trained MultiGridNeuralHumanPolicyPrior instance.
    """
    # Get environment info
    grid_width = env.width
    grid_height = env.height
    num_actions = env.action_space.n
    num_agents = len(env.agents)
    
    # Get agent colors and build num_agents_per_color
    agent_colors = extract_agent_colors(env)
    num_agents_per_color = {}
    for color in agent_colors:
        num_agents_per_color[color] = num_agents_per_color.get(color, 0) + 1
    
    # Get action encoding from environment
    action_encoding = {}
    if hasattr(env, 'actions'):
        for i, action in enumerate(env.actions):
            if hasattr(action, 'name'):
                action_encoding[i] = action.name.lower()
            else:
                action_encoding[i] = str(action).lower()
    else:
        action_encoding = DEFAULT_ACTION_ENCODING
    
    # Create encoders
    state_encoder = MultiGridStateEncoder(
        grid_height=grid_height,
        grid_width=grid_width,
        num_object_types=NUM_OBJECT_TYPE_CHANNELS,
        num_agent_colors=NUM_STANDARD_COLORS,
        feature_dim=state_feature_dim
    )
    
    agent_encoder = MultiGridAgentEncoder(
        num_agents_per_color=num_agents_per_color,
        feature_dim=agent_feature_dim
    )
    
    goal_encoder = MultiGridGoalEncoder(feature_dim=goal_feature_dim)
    
    interactive_encoder = MultiGridInteractiveObjectEncoder(
        max_kill_buttons=max_kill_buttons,
        max_pause_switches=max_pause_switches,
        max_disabling_switches=max_disabling_switches,
        max_control_buttons=max_control_buttons,
        feature_dim=interactive_feature_dim
    )
    
    # Create Q-network
    q_network = MultiGridQNetwork(
        state_encoder=state_encoder,
        agent_encoder=agent_encoder,
        goal_encoder=goal_encoder,
        interactive_encoder=interactive_encoder,
        num_actions=num_actions,
        hidden_dim=hidden_dim
    )
    q_network.to(device)
    
    # Create policy prior network
    policy_prior_network = MultiGridPolicyPriorNetwork(
        q_network=q_network,
        num_actions=num_actions
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Training loop
    replay_buffer = []
    buffer_capacity = 10000
    
    for episode in range(num_episodes):
        env.reset()
        goal = goal_sampler.sample()
        
        episode_loss = 0.0
        num_steps = 0
        
        for step in range(max_steps_per_episode):
            state = env.get_state()
            
            # Random action for data collection
            actions = [np.random.randint(num_actions) for _ in range(num_agents)]
            
            # Take step
            _, rewards, done, truncated, _ = env.step(actions)
            
            next_state = env.get_state()
            
            # Store transitions for each human agent
            for human_idx in human_agent_indices:
                transition = {
                    'state': state,
                    'action': actions[human_idx],
                    'reward': rewards[human_idx] if isinstance(rewards, (list, np.ndarray)) else rewards,
                    'next_state': next_state,
                    'done': done,
                    'goal': goal,
                    'human_idx': human_idx
                }
                
                if len(replay_buffer) < buffer_capacity:
                    replay_buffer.append(transition)
                else:
                    replay_buffer[np.random.randint(buffer_capacity)] = transition
            
            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                batch_indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch = [replay_buffer[i] for i in batch_indices]
                
                loss = _train_step(
                    q_network, optimizer, batch, env, gamma, beta, device
                )
                episode_loss += loss
                num_steps += 1
            
            if done or truncated:
                break
        
        if verbose and (episode + 1) % 100 == 0:
            avg_loss = episode_loss / max(num_steps, 1)
            print(f"Episode {episode + 1}/{num_episodes}, Avg Loss: {avg_loss:.4f}")
    
    # Create and return policy prior
    return MultiGridNeuralHumanPolicyPrior(
        world_model=env,
        human_agent_indices=human_agent_indices,
        q_network=q_network,
        policy_prior_network=policy_prior_network,
        goal_sampler=goal_sampler,
        beta=beta,
        action_encoding=action_encoding,
        device=device
    )


def _train_step(
    q_network: MultiGridQNetwork,
    optimizer: torch.optim.Optimizer,
    batch: List[Dict],
    env: Any,
    gamma: float,
    beta: float,
    device: str
) -> float:
    """Perform a single training step on a batch of transitions."""
    q_network.train()
    optimizer.zero_grad()
    
    batch_size = len(batch)
    
    # Encode batch
    grid_tensors = []
    global_features_list = []
    query_pos_list = []
    query_dir_list = []
    query_abil_list = []
    query_carr_list = []
    query_stat_list = []
    all_pos_list = []
    all_dir_list = []
    all_abil_list = []
    all_carr_list = []
    all_stat_list = []
    colors_list = []
    goal_coords_list = []
    kb_list = []
    ps_list = []
    ds_list = []
    cb_list = []
    
    actions = []
    rewards = []
    dones = []
    
    for t in batch:
        state = t['state']
        human_idx = t['human_idx']
        goal = t['goal']
        
        # Encode state
        grid_tensor, global_features = q_network.state_encoder.encode_state(
            state, env, human_idx, device
        )
        grid_tensors.append(grid_tensor)
        global_features_list.append(global_features)
        
        # Encode agents
        (query_pos, query_dir, query_abil, query_carr, query_stat,
         all_pos, all_dir, all_abil, all_carr, all_stat, colors) = \
            q_network.agent_encoder.encode_agents(state, env, human_idx, device)
        
        query_pos_list.append(query_pos)
        query_dir_list.append(query_dir)
        query_abil_list.append(query_abil)
        query_carr_list.append(query_carr)
        query_stat_list.append(query_stat)
        all_pos_list.append(all_pos)
        all_dir_list.append(all_dir)
        all_abil_list.append(all_abil)
        all_carr_list.append(all_carr)
        all_stat_list.append(all_stat)
        colors_list.append(colors)
        
        # Encode goal
        goal_coords = q_network.goal_encoder.encode_goal(goal, device)
        goal_coords_list.append(goal_coords)
        
        # Encode interactive objects
        kb, ps, ds, cb = q_network.interactive_encoder.encode_interactive_objects(
            env, state, device
        )
        kb_list.append(kb)
        ps_list.append(ps)
        ds_list.append(ds)
        cb_list.append(cb)
        
        actions.append(t['action'])
        rewards.append(t['reward'])
        dones.append(t['done'])
    
    # Stack batch
    grid_tensors = torch.cat(grid_tensors, dim=0)
    global_features = torch.cat(global_features_list, dim=0)
    query_pos = torch.cat(query_pos_list, dim=0)
    query_dir = torch.cat(query_dir_list, dim=0)
    query_abil = torch.cat(query_abil_list, dim=0)
    query_carr = torch.cat(query_carr_list, dim=0)
    query_stat = torch.cat(query_stat_list, dim=0)
    all_pos = torch.cat(all_pos_list, dim=0)
    all_dir = torch.cat(all_dir_list, dim=0)
    all_abil = torch.cat(all_abil_list, dim=0)
    all_carr = torch.cat(all_carr_list, dim=0)
    all_stat = torch.cat(all_stat_list, dim=0)
    colors = torch.cat(colors_list, dim=0)
    goal_coords = torch.cat(goal_coords_list, dim=0)
    kb = torch.cat(kb_list, dim=0)
    ps = torch.cat(ps_list, dim=0)
    ds = torch.cat(ds_list, dim=0)
    cb = torch.cat(cb_list, dim=0)
    
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    dones = torch.tensor(dones, device=device, dtype=torch.bool)
    
    # Forward pass
    q_values = q_network(
        grid_tensors, global_features,
        query_pos, query_dir, query_abil, query_carr, query_stat,
        all_pos, all_dir, all_abil, all_carr, all_stat, colors,
        goal_coords,
        kb, ps, ds, cb
    )
    
    # Get Q-values for taken actions
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Compute targets (simplified - using rewards only for now)
    # Full implementation would bootstrap from next state
    targets = rewards
    
    # Mask terminal states
    targets = torch.where(dones, rewards, targets)
    
    # Loss
    loss = nn.functional.mse_loss(q_selected, targets)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
