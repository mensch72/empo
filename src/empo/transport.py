"""
Transport Environment Wrapper for EMPO Framework.

This module provides a wrapper around the ai_transport PettingZoo environment
that makes it compatible with the gym-style interface used by the EMPO framework
(similar to the multigrid environment).

Key Features:
1. step() accepts a list of actions by agent index (not a dict)
2. Returns observations as a list by agent index
3. Provides action masking for handling invalid actions across step types
4. Action mask included in observations AND available via action_masks() method
5. Compatible with existing training code used for multigrid

Goal Classes:
- TransportGoal: Goal that an agent reaches a particular node
- TransportGoalGenerator: Iterates over all node goals for an agent
- TransportGoalSampler: Samples node goals uniformly

Usage:
    >>> from empo.transport import TransportEnvWrapper
    >>> env = TransportEnvWrapper(num_humans=4, num_vehicles=2)
    >>> obs = env.reset(seed=42)
    >>> # Action mask is in observation AND available via method
    >>> masks = env.action_masks()
    >>> print(obs[0]['action_mask'])  # Same as masks[0]
    >>> # Step with a list of actions
    >>> obs, rewards, done, info = env.step([0, 1, 0, 2, 0, 1])
"""

from typing import List, Dict, Tuple, Any, Optional, Union, Iterator
import numpy as np
import networkx as nx

# Import the ai_transport parallel environment
from ai_transport import parallel_env as TransportParallelEnv

# Import goal base classes
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler


# Fixed action space definition for consistent interface across all step types
# This allows training code to use a fixed action space with masking
class TransportActions:
    """
    Fixed action space definition for the transport environment.
    
    Actions are defined to cover all possible actions across all step types:
    - PASS (0): Do nothing / stay in place
    - UNBOARD (1): Human unboards from vehicle
    - BOARD_0 to BOARD_9 (2-11): Board vehicle 0-9 at current node
    - DEST_0 to DEST_N (12-31): Vehicle sets destination to node 0-19
    - DEPART_0 to DEPART_9 (32-41): Depart into outgoing edge 0-9
    
    Note: The actual validity of actions depends on step_type and agent state.
    Use action_masks() to determine which actions are valid.
    """
    PASS = 0
    UNBOARD = 1
    BOARD_START = 2
    BOARD_END = 11  # Support up to 10 vehicles at a node
    DEST_START = 12
    DEST_END = 31  # Support up to 20 destination nodes
    DEPART_START = 32
    DEPART_END = 41  # Support up to 10 outgoing edges
    
    # Total number of actions in the fixed action space
    NUM_ACTIONS = 42
    
    @classmethod
    def board_vehicle(cls, vehicle_idx: int) -> int:
        """Get action index for boarding vehicle at given index."""
        return cls.BOARD_START + vehicle_idx
    
    @classmethod
    def set_destination(cls, node_idx: int) -> int:
        """Get action index for setting destination to given node index."""
        return cls.DEST_START + node_idx
    
    @classmethod
    def depart_edge(cls, edge_idx: int) -> int:
        """Get action index for departing into outgoing edge at given index."""
        return cls.DEPART_START + edge_idx


# Step type indices for observation encoding
class StepType:
    """Step type encoding for observations."""
    ROUTING = 0
    UNBOARDING = 1
    BOARDING = 2
    DEPARTING = 3
    
    NAMES = ['routing', 'unboarding', 'boarding', 'departing']
    
    @classmethod
    def from_name(cls, name: str) -> int:
        """Convert step type name to index."""
        return cls.NAMES.index(name)
    
    @classmethod
    def to_name(cls, idx: int) -> str:
        """Convert step type index to name."""
        return cls.NAMES[idx]


class TransportEnvWrapper:
    """
    Wrapper for ai_transport environment providing gym-style interface.
    
    This wrapper adapts the PettingZoo ParallelEnv interface to match
    the multigrid-style interface used by EMPO training code:
    
    1. step() accepts a list of actions by agent index
    2. Returns observations as a list by agent index
    3. Provides action_masks() for valid action checking
    4. Includes step_type in observations
    
    Attributes:
        env: The underlying ai_transport parallel environment
        agents: List of agent IDs in fixed order
        num_agents: Total number of agents
        action_space: gymnasium.spaces.Discrete with fixed action count
        observation_space: Dictionary observation space
    """
    
    def __init__(
        self,
        num_humans: int = 4,
        num_vehicles: int = 2,
        network: Optional[nx.DiGraph] = None,
        human_speeds: Optional[List[float]] = None,
        vehicle_speeds: Optional[List[float]] = None,
        vehicle_capacities: Optional[List[int]] = None,
        vehicle_fuel_uses: Optional[List[float]] = None,
        observation_scenario: str = 'full',
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        max_nodes: int = 20,
        max_edges_per_node: int = 10,
        max_vehicles_per_node: int = 10,
    ):
        """
        Initialize the transport environment wrapper.
        
        Args:
            num_humans: Number of human (passenger) agents
            num_vehicles: Number of vehicle agents
            network: NetworkX DiGraph for the road network (generated if None)
            human_speeds: Walking speed for each human
            vehicle_speeds: Speed for each vehicle
            vehicle_capacities: Capacity for each vehicle
            vehicle_fuel_uses: Fuel consumption for each vehicle
            observation_scenario: One of 'full', 'local', or 'statistical'
            render_mode: 'human' for rendering, None for no rendering
            max_steps: Maximum steps per episode
            max_nodes: Maximum number of nodes (for action space sizing)
            max_edges_per_node: Maximum outgoing edges per node
            max_vehicles_per_node: Maximum vehicles at a node for boarding
        """
        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.max_vehicles_per_node = max_vehicles_per_node
        
        # Create underlying environment
        self.env = TransportParallelEnv(
            num_humans=num_humans,
            num_vehicles=num_vehicles,
            network=network,
            human_speeds=human_speeds,
            vehicle_speeds=vehicle_speeds,
            vehicle_capacities=vehicle_capacities,
            vehicle_fuel_uses=vehicle_fuel_uses,
            observation_scenario=observation_scenario,
            render_mode=render_mode,
        )
        
        # Store agent list in fixed order (humans first, then vehicles)
        self.agents = list(self.env.possible_agents)
        self.num_agents = len(self.agents)
        
        # Create agent index mapping
        self._agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}
        self._idx_to_agent = {i: agent for i, agent in enumerate(self.agents)}
        
        # Human and vehicle agent indices
        self.human_agent_indices = [i for i, a in enumerate(self.agents) if a.startswith('human_')]
        self.vehicle_agent_indices = [i for i, a in enumerate(self.agents) if a.startswith('vehicle_')]
        
        # Step counter
        self.step_count = 0
        
        # Store last observations for action mask computation
        self._last_obs = None
    
    @property
    def network(self) -> nx.DiGraph:
        """Get the road network."""
        return self.env.network
    
    @property
    def step_type(self) -> str:
        """Get the current step type name."""
        return self.env.step_type
    
    @property
    def step_type_idx(self) -> int:
        """Get the current step type as an index."""
        return StepType.from_name(self.env.step_type)
    
    @property
    def real_time(self) -> float:
        """Get the current simulation time."""
        return self.env.real_time
    
    def create_random_2d_network(self, num_nodes: int = 12, **kwargs) -> nx.DiGraph:
        """Create a random 2D road network."""
        return self.env.create_random_2d_network(num_nodes=num_nodes, **kwargs)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> List[Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (passed to underlying env)
            
        Returns:
            List of observations for each agent (by index)
        """
        self.step_count = 0
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        # Convert dict to list ordered by agent index
        obs_list = self._obs_dict_to_list(obs_dict)
        self._last_obs = obs_list
        
        return obs_list
    
    def step(self, actions: Union[List[int], np.ndarray]) -> Tuple[List[Dict], np.ndarray, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            actions: List of actions for each agent (by index).
                     Actions are from the fixed action space (TransportActions).
                     Invalid actions are converted to PASS.
        
        Returns:
            observations: List of observations for each agent
            rewards: numpy array of rewards for each agent
            done: Whether the episode is done
            info: Additional information dictionary
        """
        self.step_count += 1
        
        # Convert list actions to dict format with action mapping
        actions_dict = self._convert_actions(actions)
        
        # Step the underlying environment
        obs_dict, rewards_dict, terms_dict, truncs_dict, info_dict = self.env.step(actions_dict)
        
        # Convert outputs to list/array format
        obs_list = self._obs_dict_to_list(obs_dict)
        rewards = np.array([rewards_dict.get(agent, 0.0) for agent in self.agents])
        
        # Episode is done if any agent terminated/truncated or max steps reached
        done = any(terms_dict.values()) or any(truncs_dict.values()) or self.step_count >= self.max_steps
        
        # Build info dict
        info = {
            'step_type': self.env.step_type,
            'step_type_idx': self.step_type_idx,
            'real_time': self.env.real_time,
            'step_count': self.step_count,
        }
        
        self._last_obs = obs_list
        
        return obs_list, rewards, done, info
    
    def action_masks(self) -> np.ndarray:
        """
        Get action masks for all agents.
        
        Returns:
            Boolean array of shape (num_agents, NUM_ACTIONS) where True means valid action.
        """
        masks = np.zeros((self.num_agents, TransportActions.NUM_ACTIONS), dtype=bool)
        
        # PASS is always valid for all agents
        masks[:, TransportActions.PASS] = True
        
        step_type = self.env.step_type
        
        for i, agent in enumerate(self.agents):
            pos = self.env.agent_positions.get(agent)
            is_at_node = pos is not None and not isinstance(pos, tuple)
            is_human = agent in self.env.human_agents
            is_vehicle = agent in self.env.vehicle_agents
            
            if step_type == 'routing':
                # Vehicles at nodes can set destinations
                if is_vehicle and is_at_node:
                    nodes = list(self.env.network.nodes())
                    # DEST_0 = None destination, DEST_1..N = node 0..N-1
                    masks[i, TransportActions.DEST_START] = True  # None destination
                    for node_idx in range(min(len(nodes), self.max_nodes)):
                        masks[i, TransportActions.DEST_START + 1 + node_idx] = True
            
            elif step_type == 'unboarding':
                # Humans aboard vehicles at nodes can unboard
                if is_human:
                    aboard = self.env.human_aboard.get(agent)
                    if aboard is not None:
                        vehicle_pos = self.env.agent_positions.get(aboard)
                        if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
                            masks[i, TransportActions.UNBOARD] = True
            
            elif step_type == 'boarding':
                # Humans at nodes not aboard can board vehicles
                if is_human and is_at_node:
                    aboard = self.env.human_aboard.get(agent)
                    if aboard is None:
                        # Find vehicles at same node
                        vehicles_at_node = [
                            v for v in self.env.vehicle_agents
                            if not isinstance(self.env.agent_positions.get(v), tuple) 
                            and self.env.agent_positions.get(v) == pos
                        ]
                        for v_idx in range(min(len(vehicles_at_node), self.max_vehicles_per_node)):
                            masks[i, TransportActions.BOARD_START + v_idx] = True
            
            elif step_type == 'departing':
                # Agents at nodes can depart into outgoing edges
                if is_at_node:
                    can_depart = False
                    if is_vehicle:
                        can_depart = True
                    elif is_human:
                        aboard = self.env.human_aboard.get(agent)
                        can_depart = aboard is None
                    
                    if can_depart:
                        outgoing_edges = list(self.env.network.out_edges(pos))
                        for e_idx in range(min(len(outgoing_edges), self.max_edges_per_node)):
                            masks[i, TransportActions.DEPART_START + e_idx] = True
        
        return masks
    
    def action_mask(self, agent_idx: int) -> np.ndarray:
        """Get action mask for a single agent by index."""
        return self.action_masks()[agent_idx]
    
    def _convert_actions(self, actions: Union[List[int], np.ndarray]) -> Dict[str, int]:
        """
        Convert fixed action space actions to environment-specific actions.
        
        Maps from the fixed TransportActions to the dynamic action space
        of the underlying environment based on current step_type.
        """
        actions_dict = {}
        step_type = self.env.step_type
        
        for i, action in enumerate(actions):
            agent = self.agents[i]
            pos = self.env.agent_positions.get(agent)
            is_at_node = pos is not None and not isinstance(pos, tuple)
            is_human = agent in self.env.human_agents
            is_vehicle = agent in self.env.vehicle_agents
            
            # Default to PASS
            env_action = 0
            
            if action == TransportActions.PASS:
                env_action = 0
            
            elif action == TransportActions.UNBOARD:
                if step_type == 'unboarding' and is_human:
                    # Unboard action is 1 in the env
                    env_action = 1
            
            elif TransportActions.BOARD_START <= action <= TransportActions.BOARD_END:
                if step_type == 'boarding' and is_human and is_at_node:
                    aboard = self.env.human_aboard.get(agent)
                    if aboard is None:
                        vehicle_idx = action - TransportActions.BOARD_START
                        vehicles_at_node = [
                            v for v in self.env.vehicle_agents
                            if not isinstance(self.env.agent_positions.get(v), tuple)
                            and self.env.agent_positions.get(v) == pos
                        ]
                        if vehicle_idx < len(vehicles_at_node):
                            # Board action is vehicle_idx + 1 in the env
                            env_action = vehicle_idx + 1
            
            elif TransportActions.DEST_START <= action <= TransportActions.DEST_END:
                if step_type == 'routing' and is_vehicle and is_at_node:
                    dest_idx = action - TransportActions.DEST_START
                    # dest_idx 0 = None, 1..N = node 0..N-1
                    # In env: action 0 = None, action 1..N = node 0..N-1
                    nodes = list(self.env.network.nodes())
                    if dest_idx == 0:
                        env_action = 0  # None destination
                    elif dest_idx - 1 < len(nodes):
                        env_action = dest_idx  # Node destination
            
            elif TransportActions.DEPART_START <= action <= TransportActions.DEPART_END:
                if step_type == 'departing' and is_at_node:
                    can_depart = False
                    if is_vehicle:
                        can_depart = True
                    elif is_human:
                        aboard = self.env.human_aboard.get(agent)
                        can_depart = aboard is None
                    
                    if can_depart:
                        edge_idx = action - TransportActions.DEPART_START
                        outgoing_edges = list(self.env.network.out_edges(pos))
                        if edge_idx < len(outgoing_edges):
                            # Depart action is edge_idx + 1 in the env
                            env_action = edge_idx + 1
            
            actions_dict[agent] = env_action
        
        return actions_dict
    
    def _obs_dict_to_list(self, obs_dict: Dict[str, Any]) -> List[Dict]:
        """Convert observation dictionary to list ordered by agent index.
        
        Each observation includes:
        - All fields from the underlying environment
        - step_type_idx: Integer encoding of current step type
        - action_mask: Boolean array of valid actions for this agent
        """
        # Compute action masks once for all agents
        masks = self.action_masks()
        
        obs_list = []
        for i, agent in enumerate(self.agents):
            if agent in obs_dict:
                obs = dict(obs_dict[agent])
                # Add step_type_idx to observation for neural network encoding
                obs['step_type_idx'] = StepType.from_name(obs.get('step_type', 'routing'))
                # Add action mask to observation
                obs['action_mask'] = masks[i]
                obs_list.append(obs)
            else:
                # Agent not present (shouldn't happen in parallel env)
                obs_list.append({
                    'step_type': self.env.step_type, 
                    'step_type_idx': self.step_type_idx,
                    'action_mask': masks[i]
                })
        return obs_list
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def get_agent_type(self, agent_idx: int) -> str:
        """Get the type of agent at given index ('human' or 'vehicle')."""
        agent = self.agents[agent_idx]
        if agent in self.env.human_agents:
            return 'human'
        else:
            return 'vehicle'
    
    def get_agent_position(self, agent_idx: int) -> Any:
        """Get the position of agent at given index."""
        agent = self.agents[agent_idx]
        return self.env.agent_positions.get(agent)
    
    def is_human_aboard(self, agent_idx: int) -> Optional[str]:
        """Check if human agent is aboard a vehicle. Returns vehicle ID or None."""
        agent = self.agents[agent_idx]
        if agent in self.env.human_agents:
            return self.env.human_aboard.get(agent)
        return None
    
    def get_vehicle_destination(self, agent_idx: int) -> Optional[int]:
        """Get destination node for vehicle agent. Returns node ID or None."""
        agent = self.agents[agent_idx]
        if agent in self.env.vehicle_agents:
            return self.env.vehicle_destinations.get(agent)
        return None


def create_transport_env(
    num_humans: int = 4,
    num_vehicles: int = 2,
    num_nodes: int = 12,
    seed: Optional[int] = None,
    **kwargs
) -> TransportEnvWrapper:
    """
    Convenience function to create a transport environment with random network.
    
    Args:
        num_humans: Number of human agents
        num_vehicles: Number of vehicle agents  
        num_nodes: Number of nodes in random network
        seed: Random seed
        **kwargs: Additional arguments for TransportEnvWrapper
        
    Returns:
        TransportEnvWrapper with random network
    """
    # Create wrapper without network first
    wrapper = TransportEnvWrapper(
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        **kwargs
    )
    
    # Generate random network
    network = wrapper.create_random_2d_network(num_nodes=num_nodes, seed=seed)
    
    # Recreate with the network
    wrapper = TransportEnvWrapper(
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        network=network,
        **kwargs
    )
    
    return wrapper


# =============================================================================
# Transport Goal Classes
# =============================================================================

class TransportGoal(PossibleGoal):
    """
    Goal that an agent reaches a particular node in the transport network.
    
    This goal is achieved when the specified agent is at the target node
    (not on an edge, but actually at the node).
    
    Attributes:
        env: The TransportEnvWrapper environment (inherited from PossibleGoal)
        agent_idx: Index of the agent this goal applies to
        target_node: The node ID that the agent should reach
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5)
        >>> goal = TransportGoal(env, agent_idx=0, target_node=3)
        >>> # Goal is achieved when human_0 is at node 3
    """
    
    def __init__(self, env: TransportEnvWrapper, agent_idx: int, target_node: Any):
        """
        Initialize the transport goal.
        
        Args:
            env: The TransportEnvWrapper environment
            agent_idx: Index of the agent this goal applies to
            target_node: The target node ID the agent should reach
        """
        super().__init__(env=env)
        self.agent_idx = agent_idx
        self.target_node = target_node
        # Store target_pos for compatibility with existing goal rendering code
        self.target_pos = target_node
    
    def is_achieved(self, state) -> int:
        """
        Check if this goal is achieved.
        
        For TransportEnvWrapper, we check if the agent at agent_idx
        is currently at the target_node (not on an edge).
        
        Note: Since TransportEnvWrapper doesn't have a get_state/set_state
        interface like WorldModel, this method checks the current env state.
        The state parameter is ignored for now.
        
        Args:
            state: Ignored (uses current env state)
        
        Returns:
            int: 1 if the agent is at the target node, 0 otherwise
        """
        pos = self.env.get_agent_position(self.agent_idx)
        # Agent is at a node (not on an edge) and it's the target node
        if pos is not None and not isinstance(pos, tuple) and pos == self.target_node:
            return 1
        return 0
    
    def __hash__(self) -> int:
        """Return hash based on agent index and target node."""
        return hash((self.agent_idx, self.target_node))
    
    def __eq__(self, other) -> bool:
        """Check equality with another TransportGoal."""
        return (isinstance(other, TransportGoal) and 
                self.agent_idx == other.agent_idx and 
                self.target_node == other.target_node)
    
    def __repr__(self) -> str:
        return f"TransportGoal(agent={self.agent_idx}, node={self.target_node})"


class TransportGoalGenerator(PossibleGoalGenerator):
    """
    Generator that yields all possible node goals for an agent.
    
    Iterates over all nodes in the transport network, yielding a
    TransportGoal for each node with weight 1.0.
    
    This is useful for exact computation of integrals over the goal space
    when the network is small enough to enumerate all nodes.
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5)
        >>> generator = TransportGoalGenerator(env)
        >>> for goal, weight in generator.generate(state=None, human_agent_index=0):
        ...     print(f"{goal} with weight {weight}")
    """
    
    def __init__(self, env: TransportEnvWrapper):
        """
        Initialize the goal generator.
        
        Args:
            env: The TransportEnvWrapper environment
        """
        super().__init__(env=env)
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        """
        Generate all possible node goals for the specified agent.
        
        Yields one TransportGoal for each node in the network, all with
        weight 1.0 (uniform weighting).
        
        Args:
            state: Current state (ignored, uses env's network)
            human_agent_index: Index of the agent whose goals to generate
        
        Yields:
            Tuple[TransportGoal, float]: Pairs of (goal, weight=1.0)
        """
        nodes = list(self.env.network.nodes())
        for node in nodes:
            goal = TransportGoal(self.env, agent_idx=human_agent_index, target_node=node)
            yield goal, 1.0


class TransportGoalSampler(PossibleGoalSampler):
    """
    Sampler that uniformly samples node goals for an agent.
    
    Randomly selects a node from the transport network and returns
    a TransportGoal for that node with weight 1.0.
    
    This is useful for Monte Carlo approximation when the network
    is too large for exact enumeration.
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=100)
        >>> sampler = TransportGoalSampler(env)
        >>> goal, weight = sampler.sample(state=None, human_agent_index=0)
        >>> print(f"Sampled {goal} with weight {weight}")
    """
    
    def __init__(self, env: TransportEnvWrapper, seed: Optional[int] = None):
        """
        Initialize the goal sampler.
        
        Args:
            env: The TransportEnvWrapper environment
            seed: Optional random seed for reproducibility
        """
        super().__init__(env=env)
        self.rng = np.random.RandomState(seed)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """
        Sample a random node goal for the specified agent.
        
        Uniformly samples a node from the network and returns a
        TransportGoal for that node with weight 1.0.
        
        Args:
            state: Current state (ignored, uses env's network)
            human_agent_index: Index of the agent whose goal to sample
        
        Returns:
            Tuple[TransportGoal, float]: (goal, weight=1.0)
        """
        nodes = list(self.env.network.nodes())
        target_node = self.rng.choice(nodes)
        goal = TransportGoal(self.env, agent_idx=human_agent_index, target_node=target_node)
        return goal, 1.0
