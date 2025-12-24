"""
Multigrid-specific Phase 2 Trainer.

This module provides the training function for Phase 2 of the EMPO framework
(equations 4-9) specialized for multigrid environments.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from gym_multigrid.multigrid import MultiGridEnv

from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.trainer import BasePhase2Trainer, Phase2Networks
from empo.nn_based.multigrid import MultiGridStateEncoder
from empo.nn_based.multigrid.goal_encoder import MultiGridGoalEncoder

from .robot_q_network import MultiGridRobotQNetwork
from .human_goal_ability import MultiGridHumanGoalAchievementNetwork
from .aggregate_goal_ability import MultiGridAggregateGoalAbilityNetwork
from .intrinsic_reward_network import MultiGridIntrinsicRewardNetwork
from .robot_value_network import MultiGridRobotValueNetwork


class MultiGridPhase2Trainer(BasePhase2Trainer):
    """
    Phase 2 trainer for multigrid environments.
    
    Implements environment-specific methods for:
    - State encoding using MultiGridStateEncoder
    - Goal achievement checking
    - Environment stepping
    
    Args:
        env: MultiGridEnv instance.
        networks: Phase2Networks container.
        config: Phase2Config.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: Callable returning human action given state, index, goal.
        goal_sampler: Callable returning a goal for a human.
        device: Torch device.
    """
    
    def __init__(
        self,
        env: MultiGridEnv,
        networks: Phase2Networks,
        config: Phase2Config,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        human_policy_prior: Callable,
        goal_sampler: Callable,
        device: str = 'cpu'
    ):
        self.env = env
        super().__init__(
            networks=networks,
            config=config,
            human_agent_indices=human_agent_indices,
            robot_agent_indices=robot_agent_indices,
            human_policy_prior=human_policy_prior,
            goal_sampler=goal_sampler,
            device=device
        )
    
    def encode_state(self, state: Any) -> Dict[str, torch.Tensor]:
        """Encode multigrid state to tensors."""
        # The networks handle their own encoding
        return {'state': state}
    
    def check_goal_achieved(self, state: Any, human_idx: int, goal: Any) -> bool:
        """Check if human's goal is achieved."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[human_idx]
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        
        # Goals should have a target_pos attribute
        if hasattr(goal, 'target_pos'):
            goal_pos = goal.target_pos
            return agent_x == goal_pos[0] and agent_y == goal_pos[1]
        elif hasattr(goal, 'is_achieved'):
            return goal.is_achieved(state)
        else:
            return False
    
    def step_environment(
        self,
        state: Any,
        robot_action: Tuple[int, ...],
        human_actions: List[int]
    ) -> Any:
        """Execute actions and return next state."""
        # Build action list for all agents
        actions = []
        human_idx = 0
        robot_idx = 0
        
        for agent_idx in range(len(self.env.agents)):
            if agent_idx in self.human_agent_indices:
                actions.append(human_actions[human_idx])
                human_idx += 1
            elif agent_idx in self.robot_agent_indices:
                actions.append(robot_action[robot_idx])
                robot_idx += 1
            else:
                actions.append(0)  # Idle for unknown agents
        
        # Step environment
        self.env.step(actions)
        return self.env.get_state()
    
    def reset_environment(self) -> Any:
        """Reset environment and return initial state."""
        self.env.reset()
        return self.env.get_state()


def create_phase2_networks(
    env: MultiGridEnv,
    config: Phase2Config,
    num_robots: int,
    num_actions: int,
    hidden_dim: int = 256,
    device: str = 'cpu'
) -> Phase2Networks:
    """
    Create all Phase 2 networks for a multigrid environment.
    
    Args:
        env: MultiGridEnv instance.
        config: Phase2Config.
        num_robots: Number of robot agents.
        num_actions: Number of actions per robot.
        hidden_dim: Hidden dimension for networks.
        device: Torch device.
    
    Returns:
        Phase2Networks container with all networks.
    """
    # Count agents per color
    num_agents_per_color = {}
    for agent in env.agents:
        color = agent.color
        num_agents_per_color[color] = num_agents_per_color.get(color, 0) + 1
    
    # Common parameters
    grid_height = env.height
    grid_width = env.width
    
    # Create networks
    q_r = MultiGridRobotQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_robot_actions=num_actions,
        num_robots=num_robots,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        beta_r=config.beta_r,
    ).to(device)
    
    v_h_e = MultiGridHumanGoalAchievementNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        gamma_h=config.gamma_h,
    ).to(device)
    
    x_h = MultiGridAggregateGoalAbilityNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        zeta=config.zeta,
    ).to(device)
    
    u_r = MultiGridIntrinsicRewardNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        xi=config.xi,
        eta=config.eta,
    ).to(device)
    
    v_r = MultiGridRobotValueNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_agents_per_color=num_agents_per_color,
        state_feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    
    return Phase2Networks(
        q_r=q_r,
        v_h_e=v_h_e,
        x_h=x_h,
        u_r=u_r,
        v_r=v_r
    )


def train_multigrid_phase2(
    world_model: MultiGridEnv,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    human_policy_prior: Callable,
    goal_sampler: Callable,
    config: Optional[Phase2Config] = None,
    num_episodes: Optional[int] = None,
    hidden_dim: int = 256,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[MultiGridRobotQNetwork, Phase2Networks, List[Dict[str, float]]]:
    """
    Train Phase 2 robot policy for a multigrid environment.
    
    This function trains neural networks to learn the robot policy that maximizes
    aggregate human power, as defined in equations (4)-(9) of the EMPO paper.
    
    Args:
        world_model: MultiGridEnv instance.
        human_agent_indices: List of human agent indices.
        robot_agent_indices: List of robot agent indices.
        human_policy_prior: Callable(state, human_idx, goal) -> action.
        goal_sampler: Callable(state, human_idx) -> goal.
        config: Phase2Config (uses defaults if None).
        num_episodes: Override config.num_episodes if provided.
        hidden_dim: Hidden dimension for networks.
        device: Torch device.
        verbose: Print progress.
    
    Returns:
        Tuple of (robot_q_network, all_networks, training_history).
    """
    if config is None:
        config = Phase2Config()
    
    if num_episodes is not None:
        config.num_episodes = num_episodes
    
    # Determine number of actions from environment
    if hasattr(world_model.actions, 'n'):
        num_actions = world_model.actions.n
    elif hasattr(world_model.actions, 'available'):
        num_actions = len(world_model.actions.available)
    else:
        num_actions = len(world_model.actions)
    num_robots = len(robot_agent_indices)
    
    if verbose:
        print(f"Creating Phase 2 networks:")
        print(f"  Grid: {world_model.width}x{world_model.height}")
        print(f"  Humans: {len(human_agent_indices)}")
        print(f"  Robots: {num_robots}")
        print(f"  Actions per robot: {num_actions}")
        print(f"  Joint action space: {num_actions ** num_robots} actions")
    
    # Create networks
    networks = create_phase2_networks(
        env=world_model,
        config=config,
        num_robots=num_robots,
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        device=device
    )
    
    # Create trainer
    trainer = MultiGridPhase2Trainer(
        env=world_model,
        networks=networks,
        config=config,
        human_agent_indices=human_agent_indices,
        robot_agent_indices=robot_agent_indices,
        human_policy_prior=human_policy_prior,
        goal_sampler=goal_sampler,
        device=device
    )
    
    if verbose:
        print(f"\nTraining for {config.num_episodes} episodes...")
        print(f"  Steps per episode: {config.steps_per_episode}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Buffer size: {config.buffer_size}")
    
    # Train
    history = trainer.train(config.num_episodes)
    
    if verbose:
        print(f"\nTraining completed!")
        if history:
            final_losses = history[-1]
            print(f"  Final losses: {final_losses}")
    
    return networks.q_r, networks, history
