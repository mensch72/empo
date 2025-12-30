"""
Multigrid Robot Policy for Phase 2.

This module provides a deployable robot policy class for multigrid environments.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ...phase2.robot_policy import RobotPolicy
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
        q_network: Optional[MultiGridRobotQNetwork] = None,
        beta_r: float = 10.0,
        device: str = 'cpu',
        path: Optional[str] = None
    ):
        """
        Initialize the multigrid robot policy.
        
        Args:
            q_network: Optional Q_r network. If None, must provide path to load from.
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
            
            # Get state encoder config - it has 'feature_dim', not 'state_feature_dim'
            state_enc_config = config['state_encoder_config']
            
            # Reconstruct Q network from config
            self.q_network = MultiGridRobotQNetwork(
                grid_height=config['grid_height'],
                grid_width=config['grid_width'],
                num_robot_actions=config['num_robot_actions'],
                num_robots=config['num_robots'],
                hidden_dim=config['hidden_dim'],
                beta_r=self.beta_r,
                feasible_range=config.get('feasible_range'),
                dropout=config.get('dropout', 0.0),
                # State encoder config - note: key is 'feature_dim' not 'state_feature_dim'
                num_agents_per_color=state_enc_config['num_agents_per_color'],
                num_agent_colors=state_enc_config.get('num_agent_colors', 7),
                state_feature_dim=state_enc_config.get('feature_dim', config['hidden_dim']),
                max_kill_buttons=state_enc_config.get('max_kill_buttons', 4),
                max_pause_switches=state_enc_config.get('max_pause_switches', 4),
                max_disabling_switches=state_enc_config.get('max_disabling_switches', 4),
                max_control_buttons=state_enc_config.get('max_control_buttons', 4),
            )
            
            # Load weights
            self.q_network.load_state_dict(checkpoint['q_r'])
        else:
            self.q_network = q_network
            self.beta_r = beta_r
            
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
            # Tensorize state using both encoders
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
