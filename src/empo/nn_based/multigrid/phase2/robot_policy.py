"""
Multigrid Robot Policy for Phase 2.

This module provides deployable robot policy classes for multigrid environments.

Includes:
- MultiGridRobotPolicy: Learned policy from trained Q network
- MultiGridRobotExplorationPolicy: Simple exploration policy for epsilon-greedy training
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ...phase2.robot_policy import RobotPolicy
from ...phase2.lookup.robot_q_network import LookupTableRobotQNetwork
from ..state_encoder import MultiGridStateEncoder
from .robot_q_network import MultiGridRobotQNetwork


class MultiGridRobotPolicy(RobotPolicy):
    """
    Robot policy for multigrid environments.
    
    This provides a minimal interface for running a trained robot policy
    in rollouts without the full training infrastructure.
    
    The policy loads from a checkpoint containing the Q network, its config,
    and the beta_r parameter. At deployment time:
    1. Call reset(world_model) at the start of each episode
    2. Call sample(state) to get actions during the episode
    
    Example usage:
        # Load policy from checkpoint
        policy = MultiGridRobotPolicy(path="policy.pt")
        
        # At start of episode
        policy.reset(env)
        
        # During episode
        state = env.get_state()
        action = policy.sample(state)
    """
    
    def __init__(
        self,
        q_network: Optional[Union[MultiGridRobotQNetwork, LookupTableRobotQNetwork]] = None,
        beta_r: float = 10.0,
        device: str = 'cpu',
        path: Optional[str] = None
    ):
        """
        Initialize the multigrid robot policy.
        
        Args:
            q_network: Optional Q_r network (neural or lookup table). If None, must provide path to load from.
            beta_r: Power-law policy exponent (default if not in checkpoint).
            device: Torch device for inference.
            path: Path to policy checkpoint saved by trainer.save_policy().
                  If provided without q_network, reconstructs network from config.
        
        Examples:
            # From checkpoint (recommended):
            policy = MultiGridRobotPolicy(path="policy.pt")
            
            # With custom Q network:
            policy = MultiGridRobotPolicy(q_network=my_q_network)
        """
        self.device = device
        self._world_model = None  # Set by reset()
        self._is_lookup_table = False  # Track policy type for sample()
        
        if q_network is None:
            if path is None:
                raise ValueError(
                    "Either q_network or path must be provided. "
                    "Use: MultiGridRobotPolicy(path='policy.pt')"
                )
            
            # Load checkpoint and reconstruct Q network
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
            if 'q_r_config' not in checkpoint:
                raise ValueError(
                    "Checkpoint missing 'q_r_config'. Was it saved with trainer.save_policy()?"
                )
            
            config = checkpoint['q_r_config']
            self.beta_r = checkpoint.get('beta_r', beta_r)
            
            # Check if this is a lookup table policy
            if config.get('type') == 'lookup_table':
                self._is_lookup_table = True
                self.q_network = LookupTableRobotQNetwork(
                    num_actions=config['num_actions'],
                    num_robots=config['num_robots'],
                    beta_r=self.beta_r,
                    default_q_r=config.get('default_q_r', -1.0),
                    feasible_range=config.get('feasible_range'),
                )
                self.q_network.load_state_dict(checkpoint['q_r'])
            else:
                # Neural network policy - get state encoder config
                state_enc_config = config['state_encoder_config']
                own_state_enc_config = config.get('own_state_encoder_config', state_enc_config)
                
                # Check if networks were trained with use_encoders=False (identity mode)
                use_encoders = state_enc_config.get('use_encoders', True)
                
                # Create state encoders with correct use_encoders flag
                # This ensures identity mode produces correct feature_dim
                state_encoder = MultiGridStateEncoder(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_agents_per_color=state_enc_config['num_agents_per_color'],
                    num_agent_colors=state_enc_config.get('num_agent_colors', 7),
                    feature_dim=state_enc_config.get('feature_dim', config['hidden_dim']),
                    max_kill_buttons=state_enc_config.get('max_kill_buttons', 4),
                    max_pause_switches=state_enc_config.get('max_pause_switches', 4),
                    max_disabling_switches=state_enc_config.get('max_disabling_switches', 4),
                    max_control_buttons=state_enc_config.get('max_control_buttons', 4),
                    use_encoders=use_encoders,
                )
                
                own_use_encoders = own_state_enc_config.get('use_encoders', True)
                own_state_encoder = MultiGridStateEncoder(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_agents_per_color=own_state_enc_config['num_agents_per_color'],
                    num_agent_colors=own_state_enc_config.get('num_agent_colors', 7),
                    feature_dim=own_state_enc_config.get('feature_dim', config['hidden_dim']),
                    max_kill_buttons=own_state_enc_config.get('max_kill_buttons', 4),
                    max_pause_switches=own_state_enc_config.get('max_pause_switches', 4),
                    max_disabling_switches=own_state_enc_config.get('max_disabling_switches', 4),
                    max_control_buttons=own_state_enc_config.get('max_control_buttons', 4),
                    use_encoders=own_use_encoders,
                    share_cache_with=state_encoder,
                )
                
                # Reconstruct Q network from config with pre-built state encoders
                self.q_network = MultiGridRobotQNetwork(
                    grid_height=config['grid_height'],
                    grid_width=config['grid_width'],
                    num_robot_actions=config['num_robot_actions'],
                    num_robots=config['num_robots'],
                    hidden_dim=config['hidden_dim'],
                    beta_r=self.beta_r,
                    feasible_range=config.get('feasible_range'),
                    dropout=config.get('dropout', 0.0),
                    # Pass pre-built state encoders instead of individual params
                    num_agents_per_color=state_enc_config['num_agents_per_color'],
                    num_agent_colors=state_enc_config.get('num_agent_colors', 7),
                    state_feature_dim=state_enc_config.get('feature_dim', config['hidden_dim']),
                    state_encoder=state_encoder,
                    own_state_encoder=own_state_encoder,
                )
                
                # Load weights with validation
                checkpoint_keys = set(checkpoint['q_r'].keys())
                model_keys = set(self.q_network.state_dict().keys())
                
                if checkpoint_keys != model_keys:
                    missing_in_checkpoint = model_keys - checkpoint_keys
                    extra_in_checkpoint = checkpoint_keys - model_keys
                    
                    # Check if this is an encoder mismatch
                    encoder_keys_missing = any('encoder' in k for k in missing_in_checkpoint)
                    
                    if encoder_keys_missing:
                        raise ValueError(
                            f"State dict mismatch - likely encoder configuration issue.\n"
                            f"Config says use_encoders={use_encoders}, own_use_encoders={own_use_encoders}\n"
                            f"Missing in checkpoint: {sorted(missing_in_checkpoint)[:5]}{'...' if len(missing_in_checkpoint) > 5 else ''}\n"
                            f"Extra in checkpoint: {sorted(extra_in_checkpoint)[:5]}{'...' if len(extra_in_checkpoint) > 5 else ''}\n"
                            f"This may happen if the checkpoint was saved with different use_encoders settings.\n"
                            f"Try deleting old checkpoint files and re-training."
                        )
                
                self.q_network.load_state_dict(checkpoint['q_r'])
        else:
            self.q_network = q_network
            self.beta_r = beta_r
            self._is_lookup_table = isinstance(q_network, LookupTableRobotQNetwork)
            
            # Load weights if path provided
            if path is not None:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                self.q_network.load_state_dict(checkpoint['q_r'])
                self.beta_r = checkpoint.get('beta_r', beta_r)
        
        self.q_network.to(device)
        self.q_network.eval()
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at the start of an episode.
        
        Caches the world model reference needed for state tensorization.
        The world model provides static grid information (walls, doors, etc.)
        that is needed to convert states to tensor form.
        
        Args:
            world_model: MultiGridEnv or similar with grid information.
        """
        self._world_model = world_model
    
    def sample(self, state: Any) -> Any:
        """
        Sample an action for the given state.
        
        Args:
            state: Multigrid state tuple (step_count, agent_states, mobile_objects, mutable_objects).
        
        Returns:
            Tuple of actions, one per robot. Each action is an integer.
        
        Raises:
            RuntimeError: If reset() was not called before sample().
        """
        if self._world_model is None:
            raise RuntimeError(
                "Must call reset(world_model) before sample(). "
                "The world model is needed to tensorize the state."
            )
        
        with torch.no_grad():
            if self._is_lookup_table:
                # Lookup table uses forward() with raw state directly
                q_values = self.q_network.forward(state, self._world_model, self.device)
                # Sample action from power-law policy
                actions = self.q_network.sample_action(q_values)
            else:
                # Neural network needs tensorized state
                shared_tensors = self.q_network.state_encoder.tensorize_state(
                    state, self._world_model, self.device
                )
                own_tensors = self.q_network.own_state_encoder.tensorize_state(
                    state, self._world_model, self.device
                )
                
                # Combine into input format expected by Q network
                # shared: (grid, global, agent, interactive)
                # own: (grid, global, agent, interactive)
                all_tensors = (*shared_tensors, *own_tensors)
                
                # Sample action using the power-law policy
                actions = self.q_network.sample_action(*all_tensors)
            
            return actions


# Direction vectors for forward movement (matching DIR_TO_VEC in multigrid.py)
DIR_TO_VEC = [
    (1, 0),   # right (positive X)
    (0, 1),   # down (positive Y)
    (-1, 0),  # left (negative X)
    (0, -1),  # up (negative Y)
]


class MultiGridRobotExplorationPolicy(RobotPolicy):
    """
    Simple exploration policy for epsilon-greedy robot action selection.
    
    This policy samples actions according to fixed probabilities, but avoids
    attempting "forward" when a robot cannot move forward (blocked by wall,
    object that can't be pushed, etc.).
    
    When forward is blocked for a robot, its probability mass is redistributed 
    proportionally to the other actions for that robot.
    
    Supports multiple robots - each robot's action is sampled independently.
    
    Designed for use with SmallActions (0=still, 1=left, 2=right, 3=forward).
    
    Example usage:
        # Prefer forward (0.6), then right (0.2), with low chance for still/left
        exploration = MultiGridRobotExplorationPolicy(
            action_probs=[0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        )
        trainer = MultiGridPhase2Trainer(
            ...,
            robot_exploration_policy=exploration
        )
    """
    
    def __init__(
        self,
        action_probs: Optional[List[float]] = None,
        robot_agent_indices: Optional[List[int]] = None
    ):
        """
        Initialize the exploration policy.
        
        Args:
            action_probs: Probabilities for each action [still, left, right, forward].
                         Default: [0.1, 0.1, 0.2, 0.6] (bias toward forward/right).
            robot_agent_indices: List of robot agent indices. If None, will be 
                                 detected from world model on reset().
        """
        if action_probs is None:
            action_probs = [0.1, 0.1, 0.2, 0.6]  # still, left, right, forward
        
        if len(action_probs) != 4:
            raise ValueError(f"action_probs must have 4 elements, got {len(action_probs)}")
        
        total = sum(action_probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"action_probs must sum to 1.0, got {total}")
        
        self.action_probs = np.array(action_probs, dtype=np.float64)
        self._robot_agent_indices = robot_agent_indices
        self._world_model = None
    
    def reset(self, world_model: Any) -> None:
        """
        Reset the policy at episode start.
        
        Args:
            world_model: The MultiGrid environment.
        """
        self._world_model = world_model
        # Auto-detect robot indices from world model if not provided
        if self._robot_agent_indices is None and hasattr(world_model, 'get_robot_agent_indices'):
            self._robot_agent_indices = world_model.get_robot_agent_indices()
    
    @property
    def robot_agent_indices(self) -> List[int]:
        """Get the robot agent indices."""
        if self._robot_agent_indices is None:
            return [1]  # Default: single robot at index 1
        return self._robot_agent_indices
    
    def sample(self, state: Any) -> Tuple[int, ...]:
        """
        Sample exploration actions for all robots, avoiding forward when blocked.
        
        Each robot's action is sampled independently.
        
        Args:
            state: Environment state tuple from get_state().
        
        Returns:
            Tuple of action indices, one per robot: (action_robot0, action_robot1, ...)
            where each action is 0=still, 1=left, 2=right, 3=forward
        """
        # Parse state to get agent states
        # State format: (step_count, agent_states, mobile_objects, mutable_objects)
        step_count, agent_states, *rest = state
        
        actions = []
        for robot_idx in self.robot_agent_indices:
            action = self._sample_single_robot_action(robot_idx, agent_states)
            actions.append(action)
        
        return tuple(actions)
    
    def _sample_single_robot_action(self, robot_idx: int, agent_states: Tuple) -> int:
        """
        Sample an action for a single robot.
        
        Uses the environment's can_forward() method to check if forward movement
        is possible, accounting for robot capabilities (can push rocks, can enter
        magic walls).
        
        Args:
            robot_idx: Index of the robot agent
            agent_states: Agent states from the state tuple
        
        Returns:
            Action index: 0=still, 1=left, 2=right, 3=forward
        """
        if self._world_model is None:
            # No world model - can't check if forward is blocked
            return int(np.random.choice(4, p=self.action_probs))
        
        # Get the current state from the world model to use can_forward
        state = self._world_model.get_state()
        
        # Use the environment's can_forward method which accounts for agent capabilities
        can_move_forward = self._world_model.can_forward(state, robot_idx)
        
        if not can_move_forward:
            # Redistribute forward probability to other actions
            probs = self.action_probs.copy()
            forward_prob = probs[3]
            probs[3] = 0.0
            
            # Distribute proportionally to remaining actions
            remaining = probs.sum()
            if remaining > 0:
                probs *= (1.0 + forward_prob / remaining)
            else:
                # All probabilities were zero except forward - use uniform
                probs = np.array([1/3, 1/3, 1/3, 0.0])
            
            return int(np.random.choice(4, p=probs))
        else:
            return int(np.random.choice(4, p=self.action_probs))
