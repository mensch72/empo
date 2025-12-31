"""
Neural human policy prior for multigrid environments.

Extends BaseNeuralHumanPolicyPrior with multigrid-specific implementation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Any, Dict, List, Optional
import random
from tqdm import tqdm

from empo.possible_goal import PossibleGoalSampler

from ..neural_policy_prior import BaseNeuralHumanPolicyPrior
from ..replay_buffer import ReplayBuffer
from .constants import DEFAULT_ACTION_ENCODING
from .q_network import MultiGridQNetwork
from .policy_prior_network import MultiGridPolicyPriorNetwork
from .direct_phi_network import DirectPhiNetwork
from .feature_extraction import get_num_agents_per_color
from .path_distance import PathDistanceCalculator

# Numerical stability constant for log operations
LOG_EPS = 1e-10


class MultiGridNeuralHumanPolicyPrior(BaseNeuralHumanPolicyPrior):
    """
    Neural policy prior for multigrid environments.
    
    Extends BaseNeuralHumanPolicyPrior with multigrid-specific:
    - Network creation from multigrid world_model
    - Load with multigrid-specific validation
    - Optional direct phi network for fast marginal queries
    """
    
    def __init__(
        self,
        q_network: MultiGridQNetwork,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = 'cpu',
        direct_phi_network: Optional[DirectPhiNetwork] = None
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
        self.direct_phi_network = direct_phi_network
        if self.direct_phi_network is not None:
            self.direct_phi_network.to(device)
            self.direct_phi_network.eval()
    
    def _compute_marginal_policy(
        self,
        state: Any,
        agent_idx: int
    ) -> torch.Tensor:
        """Compute marginal policy over goals.
        
        If a direct phi network is available, uses it for fast inference.
        Otherwise, computes marginal by averaging over sampled goals.
        """
        # Use direct phi network if available (fast path)
        if self.direct_phi_network is not None:
            with torch.no_grad():
                probs = self.direct_phi_network.forward(
                    state, self.world_model, agent_idx, self.device
                )
                return probs.squeeze(0)
        
        # Fall back to averaging over sampled goals (slow path)
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
        """Validate grid dimensions for policy loading.
        
        Allows loading policies trained on larger grids for use on smaller grids
        (by padding with walls). Rejects loading policies from smaller grids for
        larger grids (coordinates would be out of bounds).
        """
        env_height = getattr(world_model, 'height', None)
        env_width = getattr(world_model, 'width', None)
        saved_height = config.get('grid_height')
        saved_width = config.get('grid_width')
        
        # Only reject if environment grid is LARGER than saved grid
        if env_height is not None and env_height > saved_height:
            raise ValueError(
                f"Cannot load policy trained on smaller grid: "
                f"saved height={saved_height}, environment height={env_height}. "
                f"Policies can only be transferred from larger to smaller grids."
            )
        if env_width is not None and env_width > saved_width:
            raise ValueError(
                f"Cannot load policy trained on smaller grid: "
                f"saved width={saved_width}, environment width={env_width}. "
                f"Policies can only be transferred from larger to smaller grids."
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
        
        Supports cross-grid loading: policies trained on larger grids can be loaded
        for use on smaller grids. The encoder will pad the smaller grid with walls
        to match the trained policy's expected dimensions. This enables transfer
        learning and efficient policy reuse across different grid sizes.
        
        Args:
            filepath: Path to saved model.
            world_model: New environment. Can have smaller dimensions than the saved
                policy (will be padded with walls), but not larger dimensions.
            human_agent_indices: Human agent indices.
            goal_sampler: Goal sampler.
            infeasible_actions_become: Action to remap unsupported actions to.
            device: Torch device.
        
        Returns:
            Loaded MultiGridNeuralHumanPolicyPrior instance.
        
        Raises:
            ValueError: If environment grid is larger than saved grid, or if
                action encodings conflict.
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
            feasible_range=config.get('feasible_range', None),
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
    robot_shaping_exponent: float = 0.0,
    button_toggle_bias: float = 0.0,
    epsilon: float = 0.3,
    exploration_policy: Optional[List[float]] = None,
    updates_per_episode: int = 1,
    train_phi_network: bool = False,
    phi_learning_rate: float = 1e-3,
    phi_num_goal_samples: int = 10,
    world_model_generator: Optional[Any] = None,
    episodes_per_model: int = 1
) -> MultiGridNeuralHumanPolicyPrior:
    """
    Train a neural policy prior for multigrid environments.
    
    Uses Q-learning with experience replay. Optionally trains a direct phi network
    that can predict marginal policies without iterating over goals.
    
    All goals are represented as bounding boxes (x1, y1, x2, y2).
    Point goals are (x, y, x, y).
    
    The phi network (h_phi) is trained jointly with the Q-network in the same
    training loop. This is more accurate than distillation because:
    1. Both networks see the same states from the same exploration policy
    2. The phi network learns from fresh Q-network policies at each step
    3. No lag between Q-network improvement and phi network learning
    
    Robot Shaping (for control button scenarios):
        When robot_shaping_exponent > 0, uses multiplicative potential shaping:
        Φ(s) = Φ_human(s) * (0.5 * |Φ_robot(s)|^a)
        
        The robot factor is normalized to [0, 0.5] so that:
        - When robot is at goal: factor=0, Φ=0 (best possible)
        - When robot is far: factor=0.5, Φ=0.5*Φ_human
        
        This creates an intentional discontinuity when a robot becomes button-controlled:
        Before: Φ=Φ_human, After: Φ≤0.5*Φ_human (less negative = better)
        This gives a positive shaping reward, incentivizing the human to let the
        robot program buttons. Only considers robots that are controlled_agent
        of some ControlButton in the environment.
    
    Button-Aware Exploration:
        When button_toggle_bias > 0, the exploration policy is biased toward
        toggling when the human agent faces a programmed control button.
        This helps discover the button-control mechanism faster.
    
    Args:
        world_model: Multigrid environment (alias for env).
        env: Multigrid environment.
        human_agent_indices: Indices of human agents.
        goal_sampler: Sampler for training goals.
        num_episodes: Number of training episodes.
        steps_per_episode: Steps per episode.
        batch_size: Training batch size.
        learning_rate: Learning rate for Q-network.
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
        robot_shaping_exponent: Exponent 'a' for multiplicative robot shaping.
            When > 0, combines human and robot potentials as Φ_h * |Φ_r|^a.
            Only uses robots controlled by ControlButtons. Default 0.0 (disabled).
        button_toggle_bias: Probability of toggling when facing a programmed button
            during exploration. Default 0.0 (disabled, uniform exploration).
        epsilon: Exploration rate for epsilon-greedy.
        exploration_policy: Optional action probability weights for exploration.
            Can be a list (static) or callable(state, world_model, agent_idx) -> weights.
        updates_per_episode: Number of training updates per episode.
        train_phi_network: Whether to train a direct phi network for fast marginal queries.
            When True, trains h_phi(s, agent) -> P(action) to approximate E_g[π(a|s,g)].
        phi_learning_rate: Learning rate for phi network.
        phi_num_goal_samples: Number of goals to sample per state for phi network training.
        world_model_generator: Optional generator for environment ensemble training.
        episodes_per_model: Episodes per environment when using ensemble.
    
    Returns:
        Trained MultiGridNeuralHumanPolicyPrior (with optional direct_phi_network).
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
    
    # Create path calculator for reward shaping (with world_model for path precomputation)
    path_calc = PathDistanceCalculator(
        grid_height, grid_width, 
        world_model=env,
        passing_costs=None  # Use default passing costs
    ) if reward_shaping else None
    
    # Compute feasible range for Q-values based on reward shaping
    # When using path-based shaping, the feasible range is determined by the
    # maximum possible path cost (grid diagonal * max passing cost)
    if reward_shaping and path_calc is not None:
        # Use the feasible_range computed by PathDistanceCalculator
        # which accounts for passing difficulty scores
        feasible_range = path_calc.feasible_range
        # Add small margin for the base reward [0, 1]
        feasible_range = (feasible_range[0], feasible_range[1] + 1.0)
    else:
        # Without shaping, Q-values are bounded by discounted rewards in [0, 1]
        feasible_range = (0.0, 1.0)
    
    # Create Q-network with unified state encoder
    # All goals are rectangles (x1, y1, x2, y2). Point goals are (x, y, x, y).
    q_network = MultiGridQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_actions=num_actions,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=num_agent_colors,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range
    ).to(device)
    
    # Target network (same feasible_range)
    target_network = MultiGridQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        num_actions=num_actions,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=num_agent_colors,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # Create phi network if requested
    phi_network = None
    phi_optimizer = None
    if train_phi_network:
        phi_network = DirectPhiNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            num_actions=num_actions,
            num_agents_per_color=num_agents_per_color,
            num_agent_colors=num_agent_colors,
            state_feature_dim=state_feature_dim,
            hidden_dim=hidden_dim
        ).to(device)
        phi_optimizer = optim.Adam(phi_network.parameters(), lr=phi_learning_rate)
    
    # Convert numpy array exploration_policy to list if needed
    if exploration_policy is not None:
        if hasattr(exploration_policy, 'tolist'):
            exploration_policy = exploration_policy.tolist()
    
    # Track current world_model for reward shaping (updated in ensemble training)
    current_world_model = [env]  # Use list to allow modification in closure
    
    # Detect button-controlled robots for robot shaping
    # Scan grid for ControlButton cells and collect their controlled_agent indices
    button_controlled_robots = set()
    button_positions = {}  # Maps (x, y) -> controlled_agent index
    if hasattr(env, 'grid'):
        for j in range(grid_height):
            for i in range(grid_width):
                cell = env.grid.get(i, j)
                if cell is not None and cell.type == 'controlbutton':
                    if hasattr(cell, 'controlled_agent') and cell.controlled_agent is not None:
                        button_controlled_robots.add(cell.controlled_agent)
                        if hasattr(cell, 'triggered_action') and cell.triggered_action is not None:
                            button_positions[(i, j)] = cell.controlled_agent
    
    # Create button-aware exploration policy if button_toggle_bias > 0
    # Direction vectors: 0=east(+x), 1=south(+y), 2=west(-x), 3=north(-y)
    DIR_TO_DELTA = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    TOGGLE_ACTION = 6  # Standard toggle action in multigrid
    
    def button_aware_exploration(state, world_model, agent_idx):
        """
        State-dependent exploration policy that biases toward toggle when facing a button.
        """
        # Get agent position and direction from state
        _, agent_states, _, _ = state
        if agent_idx >= len(agent_states):
            return [1.0] * num_actions  # Uniform fallback
        
        agent_state = agent_states[agent_idx]
        if len(agent_state) < 3:
            return [1.0] * num_actions
        
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        agent_dir = int(agent_state[2])
        
        # Check if there's a programmed button in front of the agent
        dx, dy = DIR_TO_DELTA.get(agent_dir, (0, 0))
        front_pos = (agent_x + dx, agent_y + dy)
        
        # Check current world_model's grid for button at front_pos
        wm = current_world_model[0]
        facing_programmed_button = False
        if hasattr(wm, 'grid') and front_pos in button_positions:
            cell = wm.grid.get(front_pos[0], front_pos[1])
            if cell is not None and cell.type == 'controlbutton':
                if hasattr(cell, 'triggered_action') and cell.triggered_action is not None:
                    facing_programmed_button = True
        
        if facing_programmed_button and TOGGLE_ACTION < num_actions:
            # Bias toward toggle action
            weights = [1.0] * num_actions
            # Redistribute probability: toggle gets button_toggle_bias, rest share remaining
            remaining = 1.0 - button_toggle_bias
            for i in range(num_actions):
                if i == TOGGLE_ACTION:
                    weights[i] = button_toggle_bias
                else:
                    weights[i] = remaining / (num_actions - 1)
            return weights
        else:
            # Uniform exploration
            return [1.0] * num_actions
    
    # Use button-aware exploration if enabled, otherwise use provided or None
    effective_exploration_policy = exploration_policy
    if button_toggle_bias > 0 and button_positions:
        effective_exploration_policy = button_aware_exploration
    
    # Create reward function with potential-based shaping (Ng et al. 1999)
    # F(s,a,s') = γ * Φ(s') - Φ(s) where Φ(s) = -path_cost(agent, goal) / max_cost
    # Path cost sums passing difficulty scores along the shortest path
    def compute_shaped_reward(state, action, next_state, agent_idx, goal):
        """Compute reward with path-based potential shaping."""
        # Extract goal target (point or rectangle)
        if hasattr(goal, 'target_rect'):
            target = goal.target_rect  # Rectangle goal (x1, y1, x2, y2)
        elif hasattr(goal, 'target_pos'):
            target = goal.target_pos  # Point goal (x, y)
        else:
            # No target position - just use goal achievement
            if hasattr(goal, 'is_achieved'):
                return float(goal.is_achieved(next_state))
            return 0.0
        
        # Base reward: goal achievement
        base_reward = 0.0
        if hasattr(goal, 'is_achieved'):
            base_reward = float(goal.is_achieved(next_state))
        else:
            # Check if agent is at/in the goal
            _, next_agent_states, _, _ = next_state
            if agent_idx < len(next_agent_states):
                next_pos = (int(next_agent_states[agent_idx][0]), 
                           int(next_agent_states[agent_idx][1]))
                if path_calc is not None:
                    base_reward = 1.0 if path_calc.is_in_goal(next_pos, target) else 0.0
        
        # If no path calculator, just return base reward
        if path_calc is None:
            return base_reward
        
        # Compute potential-based shaping reward using path costs
        _, curr_agent_states, _, _ = state
        _, next_agent_states, _, _ = next_state
        
        if agent_idx >= len(curr_agent_states) or agent_idx >= len(next_agent_states):
            return base_reward
        
        curr_pos = (int(curr_agent_states[agent_idx][0]), 
                   int(curr_agent_states[agent_idx][1]))
        next_pos = (int(next_agent_states[agent_idx][0]), 
                   int(next_agent_states[agent_idx][1]))
        
        # Use world_model for path cost computation with passing difficulty scores
        wm = current_world_model[0]
        
        # Compute human potential: Φ_human = -path_cost(human, goal) / max_cost
        phi_human_s = path_calc.compute_potential(curr_pos, target, wm)
        phi_human_s_prime = path_calc.compute_potential(next_pos, target, wm)
        
        # Compute robot potential if robot_shaping_exponent > 0
        # Uses multiplicative combination: Φ = Φ_human * (0.5 * |Φ_robot|^a)
        # The robot factor is normalized to [0, 0.5] so that:
        # - When robot is at goal (Φ_robot=0): factor=0, Φ=0 (best)
        # - When robot is far (Φ_robot=-1): factor=0.5, Φ=0.5*Φ_human
        # This creates a discontinuity when a robot becomes button-controlled:
        # Before: Φ=Φ_human, After: Φ=0.5*Φ_human (at most)
        # Since Φ_human<0, this makes Φ less negative (better), giving positive
        # shaping reward - an incentive to let the robot program the button.
        if robot_shaping_exponent > 0 and button_controlled_robots:
            # Find the best (closest to 0) robot potential
            min_robot_phi_s = -1.0
            min_robot_phi_s_prime = -1.0
            
            for robot_idx in button_controlled_robots:
                if robot_idx < len(curr_agent_states):
                    robot_curr_pos = (int(curr_agent_states[robot_idx][0]),
                                     int(curr_agent_states[robot_idx][1]))
                    robot_phi_s = path_calc.compute_potential(robot_curr_pos, target, wm)
                    if robot_phi_s > min_robot_phi_s:
                        min_robot_phi_s = robot_phi_s
                
                if robot_idx < len(next_agent_states):
                    robot_next_pos = (int(next_agent_states[robot_idx][0]),
                                     int(next_agent_states[robot_idx][1]))
                    robot_phi_s_prime = path_calc.compute_potential(robot_next_pos, target, wm)
                    if robot_phi_s_prime > min_robot_phi_s_prime:
                        min_robot_phi_s_prime = robot_phi_s_prime
            
            # Multiplicative potential with robot factor normalized to [0, 0.5]:
            # Φ = Φ_human * (0.5 * |Φ_robot|^a)
            # Both Φ_human and Φ_robot are in [-1, 0], so |Φ_robot| in [0, 1]
            robot_factor_s = 0.5 * (abs(min_robot_phi_s) ** robot_shaping_exponent)
            robot_factor_s_prime = 0.5 * (abs(min_robot_phi_s_prime) ** robot_shaping_exponent)
            phi_s = phi_human_s * robot_factor_s
            phi_s_prime = phi_human_s_prime * robot_factor_s_prime
        else:
            # Standard human-only potential
            phi_s = phi_human_s
            phi_s_prime = phi_human_s_prime
        
        # Shaping: F(s,a,s') = γ * Φ(s') - Φ(s)
        shaping_reward = gamma * phi_s_prime - phi_s
        
        return base_reward + shaping_reward
    
    reward_fn = compute_shaped_reward if reward_shaping else None
    
    # Use generic Trainer with exploration_policy and reward function
    from ..trainer import Trainer
    trainer = Trainer(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        target_update_freq=target_update_freq,
        device=device,
        exploration_policy=effective_exploration_policy,
        reward_fn=reward_fn
    )
    
    # Handle environment ensemble training
    current_env = env
    episode_count_for_model = 0
    
    # State buffer for phi network training (uses deque for O(1) append/remove)
    phi_state_buffer = deque(maxlen=buffer_capacity) if train_phi_network else None
    
    # Set up progress bar
    pbar = tqdm(total=num_episodes, desc="Training", unit="episodes", disable=not verbose)
    
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
        
        # Update current_world_model for reward shaping closure
        current_world_model[0] = current_env
        
        current_env.reset()
        state = current_env.get_state()
        
        for step in range(steps_per_episode):
            agent_idx = random.choice(human_agent_indices)
            
            # Sample goal using sampler
            try:
                goal, _ = goal_sampler.sample(state, agent_idx)
            except (ValueError, RuntimeError, IndexError):
                # Goal sampling can fail for invalid states or edge cases
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
            
            # Store transition for Q-network training
            trainer.store_transition(state, action, next_state, agent_idx, goal)
            
            # Store state for phi network training (deque auto-removes oldest when full)
            if train_phi_network:
                phi_state_buffer.append({
                    'state': state,
                    'agent_idx': agent_idx,
                    'world_model': current_env
                })
            
            state = next_state
        
        # Training updates at end of episode - train both networks together
        for _ in range(updates_per_episode):
            # Train Q-network
            trainer.train_step(batch_size)
            
            # Train phi network in the same loop (joint training)
            if train_phi_network and len(phi_state_buffer) >= batch_size:
                _train_phi_network_step(
                    phi_network=phi_network,
                    phi_optimizer=phi_optimizer,
                    q_network=q_network,
                    state_buffer=phi_state_buffer,
                    goal_sampler=goal_sampler,
                    batch_size=batch_size,
                    num_goal_samples=phi_num_goal_samples,
                    device=device
                )
        
        pbar.update(1)
    
    pbar.close()
    
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
        device=device,
        direct_phi_network=phi_network
    )


def _train_phi_network_step(
    phi_network: DirectPhiNetwork,
    phi_optimizer: optim.Optimizer,
    q_network: MultiGridQNetwork,
    state_buffer: List[Dict],
    goal_sampler: PossibleGoalSampler,
    batch_size: int,
    num_goal_samples: int,
    device: str
) -> float:
    """
    Train the phi network jointly with Q-network (called each training step).
    
    The phi network (h_phi) learns to approximate E_g[π(a|s,g)] - the marginal
    policy prior over goals. Since we use a goal *sampler* (not generator),
    this is a stochastic approximation trained via gradient descent.
    
    Training procedure:
    1. Sample a batch of states from the buffer
    2. For each state, sample multiple goals using weight-proportional sampling
       (goals are sampled with probability proportional to their area weight)
    3. Average the Q-network policies (simple average since sampling already accounts for weights)
    4. Train phi to match this marginal via cross-entropy loss
    
    IMPORTANT: The goal sampler should sample goals with probability proportional
    to their weight (1+x2-x1)*(1+y2-y1). If it does, we use simple averaging.
    If it samples uniformly, the marginal will be biased toward smaller goals.
    
    Joint training with Q-network ensures:
    - Both networks see the same state distribution
    - Phi learns from fresh Q-network policies (no lag)
    - Better convergence than separate distillation phase
    
    Args:
        phi_network: The direct phi network to train.
        phi_optimizer: Optimizer for phi network.
        q_network: Current Q-network (used to compute target marginals).
        state_buffer: Buffer of (state, agent_idx, world_model) tuples.
        goal_sampler: Goal sampler that samples with weight-proportional probabilities.
        batch_size: Batch size.
        num_goal_samples: Number of goals to sample per state for marginal computation.
        device: Torch device.
    
    Returns:
        Training loss.
    """
    if len(state_buffer) < batch_size:
        return 0.0
    
    # Sample batch of states
    indices = random.sample(range(len(state_buffer)), batch_size)
    
    phi_network.train()
    q_network.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    
    for idx in indices:
        item = state_buffer[idx]
        state = item['state']
        agent_idx = item['agent_idx']
        world_model = item['world_model']
        
        # Compute Q-network's marginal policy over sampled goals
        # Goals are sampled with weight-proportional probabilities, so we use simple averaging
        with torch.no_grad():
            marginal_probs = torch.zeros(q_network.num_actions, device=device)
            valid_samples = 0
            
            # Sample multiple goals and average their policies
            # The sampler samples with P(goal) ∝ weight, so simple average gives correct marginal
            for _ in range(num_goal_samples):
                try:
                    goal, _ = goal_sampler.sample(state, agent_idx)
                    if goal is None:
                        continue
                    
                    q_values = q_network.forward(
                        state, world_model, agent_idx, goal, device
                    )
                    policy = q_network.get_policy(q_values).squeeze(0)
                    marginal_probs += policy
                    valid_samples += 1
                except (ValueError, RuntimeError, IndexError):
                    # Goal sampling or Q-network forward can fail for edge cases
                    continue
            
            if valid_samples == 0:
                continue
            
            # Simple average to get marginal (target for phi network)
            # This is correct because goals are sampled with weight-proportional probabilities
            target_marginal = marginal_probs / valid_samples
        
        # Get phi network's prediction
        phi_probs = phi_network.forward(
            state, world_model, agent_idx, device
        ).squeeze(0)
        
        # Cross-entropy loss (target is fixed): -sum(target * log(phi))
        loss = -torch.sum(target_marginal * torch.log(phi_probs + LOG_EPS))
        total_loss = total_loss + loss
    
    # Backpropagate
    avg_loss = total_loss / batch_size
    phi_optimizer.zero_grad()
    avg_loss.backward()
    phi_optimizer.step()
    
    return avg_loss.item()
