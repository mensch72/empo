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
6. Optional cluster-based routing: vehicles announce destination clusters rather than nodes

Goal Classes:
- TransportGoal: Goal that an agent reaches a particular node
- TransportGoalGenerator: Iterates over all node goals for an agent
- TransportGoalSampler: Samples node goals uniformly
- TransportClusterGoal: Goal that an agent reaches any node in a cluster
- TransportClusterGoalGenerator: Iterates over all cluster goals for an agent
- TransportClusterGoalSampler: Samples cluster goals uniformly

Usage:
    >>> from empo.transport import TransportEnvWrapper
    >>> # Node-based routing (default)
    >>> env = TransportEnvWrapper(num_humans=4, num_vehicles=2)
    >>> # Cluster-based routing
    >>> env = TransportEnvWrapper(num_humans=4, num_vehicles=2, num_clusters=10)
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
from ai_transport.envs.clustering import cluster_network, get_cluster_for_node, get_nodes_in_cluster

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
    - DEST_0 to DEST_N (12-31): Vehicle sets destination to node/cluster 0-19
      - When use_clusters=False: DEST_0 = None, DEST_1..N = node 0..N-1
      - When use_clusters=True: DEST_0 = None, DEST_1..N = cluster 0..N-1
    - DEPART_0 to DEPART_9 (32-41): Depart into outgoing edge 0-9
    
    Note: The actual validity of actions depends on step_type and agent state.
    Use action_masks() to determine which actions are valid.
    """
    PASS = 0
    UNBOARD = 1
    BOARD_START = 2
    BOARD_END = 11  # Support up to 10 vehicles at a node
    DEST_START = 12
    DEST_END = 31  # Support up to 20 destination nodes/clusters
    DEPART_START = 32
    DEPART_END = 41  # Support up to 10 outgoing edges
    
    # Total number of actions in the fixed action space
    NUM_ACTIONS = 42
    
    @classmethod
    def board_vehicle(cls, vehicle_idx: int) -> int:
        """Get action index for boarding vehicle at given index."""
        return cls.BOARD_START + vehicle_idx
    
    @classmethod
    def set_destination(cls, idx: int) -> int:
        """Get action index for setting destination to given node/cluster index."""
        return cls.DEST_START + idx
    
    @classmethod
    def set_destination_cluster(cls, cluster_idx: int) -> int:
        """Get action index for setting destination to given cluster index."""
        return cls.DEST_START + cluster_idx + 1  # +1 because DEST_START is None
    
    @classmethod
    def set_destination_node(cls, node_idx: int) -> int:
        """Get action index for setting destination to given node index."""
        return cls.DEST_START + node_idx + 1  # +1 because DEST_START is None
    
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
    5. Optional cluster-based routing (num_clusters > 0)
    
    Cluster-Based Routing:
    When num_clusters > 0, the network is clustered using k-means and vehicles
    announce destination clusters rather than specific nodes. This increases
    flexibility and potentially passenger empowerment, as vehicles are committed
    to regions rather than fixed destinations.
    
    Attributes:
        env: The underlying ai_transport parallel environment
        agents: List of agent IDs in fixed order
        num_agents: Total number of agents
        action_space: gymnasium.spaces.Discrete with fixed action count
        observation_space: Dictionary observation space
        use_clusters: Whether cluster-based routing is enabled
        cluster_info: Clustering information (if clusters enabled)
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
        num_clusters: int = 0,
        clustering_method: str = 'kmeans',
        clustering_seed: int = 42,
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
            num_clusters: Number of clusters for cluster-based routing (0 = use nodes)
            clustering_method: Clustering method ('kmeans' or 'spectral')
            clustering_seed: Random seed for clustering
        """
        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        self.max_vehicles_per_node = max_vehicles_per_node
        self.num_clusters = num_clusters
        self.clustering_method = clustering_method
        self.clustering_seed = clustering_seed
        
        # Cluster-based routing is enabled if num_clusters > 0
        self.use_clusters = num_clusters > 0
        self.cluster_info = None
        
        # Vehicle destination clusters (for cluster-based routing)
        # Maps vehicle agent ID to cluster ID (or None)
        self.vehicle_destination_clusters: Dict[str, Optional[int]] = {}
        
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
        
        # Initialize clustering if enabled (requires network)
        if self.use_clusters and self.env.network is not None:
            self._init_clustering()
        
        # Store agent list in fixed order (humans first, then vehicles)
        self.agents = list(self.env.possible_agents)
        self.num_agents = len(self.agents)
        
        # Create agent index mapping
        self._agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}
        self._idx_to_agent = {i: agent for i, agent in enumerate(self.agents)}
        
        # Human and vehicle agent indices
        self.human_agent_indices = [i for i, a in enumerate(self.agents) if a.startswith('human_')]
        self.vehicle_agent_indices = [i for i, a in enumerate(self.agents) if a.startswith('vehicle_')]
        
        # Initialize vehicle destination clusters
        for agent in self.env.vehicle_agents:
            self.vehicle_destination_clusters[agent] = None
        
        # Step counter
        self.step_count = 0
        
        # Store last observations for action mask computation
        self._last_obs = None
    
    def _init_clustering(self):
        """Initialize clustering on the network."""
        if self.env.network is None or len(self.env.network.nodes()) == 0:
            self.cluster_info = None
            return
        
        # Check if network has coordinates
        sample_node = list(self.env.network.nodes())[0]
        node_data = self.env.network.nodes[sample_node]
        has_coords = ('x' in node_data and 'y' in node_data) or 'pos' in node_data
        
        if not has_coords:
            # Network doesn't have coordinates yet, skip clustering
            self.cluster_info = None
            return
        
        self.cluster_info = cluster_network(
            self.env.network,
            k=self.num_clusters,
            method=self.clustering_method,
            random_state=self.clustering_seed
        )
        # Update num_clusters to actual number (may be less if fewer nodes)
        self.num_clusters = self.cluster_info['num_clusters']
    
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
        
        # Reinitialize clustering if enabled (network may have changed)
        if self.use_clusters:
            self._init_clustering()
        
        # Reset vehicle destination clusters
        for agent in self.env.vehicle_agents:
            self.vehicle_destination_clusters[agent] = None
        
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
                    masks[i, TransportActions.DEST_START] = True  # None destination
                    
                    if self.use_clusters and self.cluster_info is not None:
                        # Cluster-based routing: DEST_1..N = cluster 0..N-1
                        for cluster_idx in range(min(self.num_clusters, self.max_nodes)):
                            masks[i, TransportActions.DEST_START + 1 + cluster_idx] = True
                    else:
                        # Node-based routing: DEST_1..N = node 0..N-1
                        nodes = list(self.env.network.nodes())
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
                    
                    if self.use_clusters and self.cluster_info is not None:
                        # Cluster-based routing
                        # dest_idx 0 = None, 1..N = cluster 0..N-1
                        if dest_idx == 0:
                            env_action = 0  # None destination
                            self.vehicle_destination_clusters[agent] = None
                        else:
                            cluster_idx = dest_idx - 1
                            if cluster_idx < self.num_clusters:
                                # Store the destination cluster
                                self.vehicle_destination_clusters[agent] = cluster_idx
                                # Set destination to cluster centroid node
                                centroid = self.cluster_info['centroids'].get(cluster_idx)
                                if centroid is not None:
                                    nodes = list(self.env.network.nodes())
                                    try:
                                        node_action = nodes.index(centroid) + 1
                                        env_action = node_action
                                    except ValueError:
                                        env_action = 0  # Centroid not found, pass
                    else:
                        # Node-based routing
                        # dest_idx 0 = None, 1..N = node 0..N-1
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
        - use_clusters: Whether cluster-based routing is enabled
        - num_clusters: Number of clusters (if enabled)
        - my_cluster: Current cluster ID (if at a node and clusters enabled)
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
                
                # Add cluster information
                obs['use_clusters'] = self.use_clusters
                obs['num_clusters'] = self.num_clusters if self.use_clusters else 0
                
                if self.use_clusters and self.cluster_info is not None:
                    # Add agent's current cluster (if at a node)
                    pos = self.env.agent_positions.get(agent)
                    if pos is not None and not isinstance(pos, tuple):
                        obs['my_cluster'] = get_cluster_for_node(pos, self.cluster_info)
                    else:
                        obs['my_cluster'] = None
                    
                    # Add destination cluster for vehicles
                    if agent in self.env.vehicle_agents:
                        obs['destination_cluster'] = self.vehicle_destination_clusters.get(agent)
                
                obs_list.append(obs)
            else:
                # Agent not present (shouldn't happen in parallel env)
                obs_list.append({
                    'step_type': self.env.step_type, 
                    'step_type_idx': self.step_type_idx,
                    'action_mask': masks[i],
                    'use_clusters': self.use_clusters,
                    'num_clusters': self.num_clusters if self.use_clusters else 0,
                })
        return obs_list
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def render_clusters(
        self,
        ax=None,
        node_size: int = 100,
        show_labels: bool = True,
        show_centroids: bool = True,
        show_agents: bool = True,
        cmap: str = 'tab20'
    ):
        """
        Render the network with cluster coloring.
        
        This method visualizes the transport network colored by cluster assignments.
        Useful for understanding the cluster structure and agent positions.
        
        Args:
            ax: Matplotlib axes (created if None).
            node_size: Size of node markers.
            show_labels: Whether to show node labels.
            show_centroids: Whether to highlight centroid nodes with stars.
            show_agents: Whether to show agent positions.
            cmap: Colormap name for cluster colors.
        
        Returns:
            Matplotlib axes object.
        
        Raises:
            ValueError: If clustering is not enabled (num_clusters=0).
        
        Example:
            >>> env = create_transport_env(num_humans=4, num_vehicles=2, num_nodes=20, num_clusters=5)
            >>> env.reset(seed=42)
            >>> ax = env.render_clusters()
            >>> plt.show()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not self.use_clusters or self.cluster_info is None:
            raise ValueError(
                "Clustering is not enabled. Create environment with num_clusters > 0."
            )
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        
        G = self.env.network
        nodes = list(G.nodes())
        
        # Extract coordinates
        coords = []
        for node in nodes:
            node_data = G.nodes[node]
            if 'x' in node_data and 'y' in node_data:
                coords.append([float(node_data['x']), float(node_data['y'])])
            elif 'pos' in node_data:
                pos = node_data['pos']
                coords.append([float(pos[0]), float(pos[1])])
            else:
                coords.append([0.0, 0.0])
        coords = np.array(coords)
        
        node_to_cluster = self.cluster_info['node_to_cluster']
        num_clusters = self.cluster_info['num_clusters']
        centroids = self.cluster_info['centroids']
        
        # Create color map
        colormap = plt.colormaps.get_cmap(cmap)
        colors = [colormap(node_to_cluster.get(node, 0) / max(1, num_clusters - 1)) 
                  for node in nodes]
        
        # Draw edges
        for u, v in G.edges():
            if u in nodes and v in nodes:
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                ax.plot(
                    [coords[u_idx, 0], coords[v_idx, 0]],
                    [coords[u_idx, 1], coords[v_idx, 1]],
                    'k-', alpha=0.2, linewidth=0.5
                )
        
        # Draw nodes
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=node_size, zorder=2)
        
        # Highlight centroids
        if show_centroids:
            for cluster_id, centroid_node in centroids.items():
                if centroid_node in nodes:
                    idx = nodes.index(centroid_node)
                    ax.scatter(
                        coords[idx, 0], coords[idx, 1],
                        c='black', s=node_size * 2, marker='*', zorder=3,
                        edgecolors='white', linewidths=1
                    )
        
        # Draw agents
        if show_agents:
            for i, agent in enumerate(self.agents):
                pos = self.env.agent_positions.get(agent)
                if pos is None:
                    continue
                
                agent_type = self.get_agent_type(i)
                
                # Determine agent coordinates
                if isinstance(pos, tuple):
                    # Agent is on an edge - interpolate position
                    edge, progress = pos
                    if len(edge) >= 2:
                        u, v = edge[0], edge[1]
                        if u in nodes and v in nodes:
                            u_idx = nodes.index(u)
                            v_idx = nodes.index(v)
                            # Get edge length
                            edge_data = G.edges.get((u, v), {})
                            edge_length = edge_data.get('length', 1.0)
                            frac = progress / edge_length if edge_length > 0 else 0
                            frac = min(max(frac, 0), 1)
                            ax_pos = coords[u_idx] + frac * (coords[v_idx] - coords[u_idx])
                        else:
                            continue
                    else:
                        continue
                else:
                    # Agent is at a node
                    if pos not in nodes:
                        continue
                    idx = nodes.index(pos)
                    ax_pos = coords[idx]
                
                # Draw agent marker
                if agent_type == 'human':
                    marker = 'o'
                    color = 'blue'
                    size = node_size * 1.5
                else:  # vehicle
                    marker = 's'
                    color = 'red'
                    size = node_size * 2
                
                ax.scatter(
                    ax_pos[0], ax_pos[1],
                    c=color, s=size, marker=marker, zorder=4,
                    edgecolors='white', linewidths=2, label=agent
                )
        
        # Add labels
        if show_labels:
            for i, node in enumerate(nodes):
                cluster_id = node_to_cluster.get(node, '?')
                ax.annotate(
                    f'{node}\n(C{cluster_id})',
                    (coords[i, 0], coords[i, 1]),
                    fontsize=6, ha='center', va='bottom'
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Transport Network Clusters (k={num_clusters})')
        ax.set_aspect('equal')
        
        # Add legend for agents if shown
        if show_agents and len(self.agents) > 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles[:min(6, len(handles))], labels[:min(6, len(labels))],
                         loc='upper left', fontsize=8)
        
        return ax
    
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
    
    def get_vehicle_destination_cluster(self, agent_idx: int) -> Optional[int]:
        """
        Get destination cluster for vehicle agent (cluster-based routing only).
        
        Returns cluster ID or None. Only valid when use_clusters=True.
        """
        agent = self.agents[agent_idx]
        if agent in self.env.vehicle_agents:
            return self.vehicle_destination_clusters.get(agent)
        return None
    
    def get_cluster_for_position(self, position: Any) -> Optional[int]:
        """
        Get the cluster ID for a given position.
        
        Args:
            position: Node ID or (edge, coord) tuple
            
        Returns:
            Cluster ID or None if position is on an edge or clusters not enabled
        """
        if not self.use_clusters or self.cluster_info is None:
            return None
        
        # If position is a node (not on an edge)
        if position is not None and not isinstance(position, tuple):
            return get_cluster_for_node(position, self.cluster_info)
        
        return None
    
    def get_nodes_in_cluster(self, cluster_id: int) -> List:
        """Get all nodes in a cluster."""
        if not self.use_clusters or self.cluster_info is None:
            return []
        return get_nodes_in_cluster(cluster_id, self.cluster_info)


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


# =============================================================================
# Cluster-Based Goal Classes
# =============================================================================

class TransportClusterGoal(PossibleGoal):
    """
    Goal that an agent reaches any node within a particular cluster.
    
    This goal is achieved when the specified agent is at any node
    belonging to the target cluster. This is more flexible than
    TransportGoal as it allows reaching any node in a region.
    
    Attributes:
        env: The TransportEnvWrapper environment (inherited from PossibleGoal)
        agent_idx: Index of the agent this goal applies to
        target_cluster: The cluster ID that the agent should reach
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=20, num_clusters=5)
        >>> goal = TransportClusterGoal(env, agent_idx=0, target_cluster=2)
        >>> # Goal is achieved when human_0 is at any node in cluster 2
    """
    
    def __init__(self, env: TransportEnvWrapper, agent_idx: int, target_cluster: int):
        """
        Initialize the cluster goal.
        
        Args:
            env: The TransportEnvWrapper environment
            agent_idx: Index of the agent this goal applies to
            target_cluster: The target cluster ID the agent should reach
        """
        super().__init__(env=env)
        self.agent_idx = agent_idx
        self.target_cluster = target_cluster
        # Store target_pos for compatibility (use cluster centroid if available)
        if env.use_clusters and env.cluster_info is not None:
            self.target_pos = env.cluster_info['centroids'].get(target_cluster)
        else:
            self.target_pos = None
    
    def is_achieved(self, state) -> int:
        """
        Check if this goal is achieved.
        
        The goal is achieved if the agent at agent_idx is currently at
        any node belonging to the target_cluster.
        
        Args:
            state: Ignored (uses current env state)
        
        Returns:
            int: 1 if the agent is at any node in the target cluster, 0 otherwise
        """
        pos = self.env.get_agent_position(self.agent_idx)
        
        # Agent must be at a node (not on an edge)
        if pos is None or isinstance(pos, tuple):
            return 0
        
        # Check if the node belongs to the target cluster
        if self.env.use_clusters and self.env.cluster_info is not None:
            node_cluster = get_cluster_for_node(pos, self.env.cluster_info)
            if node_cluster == self.target_cluster:
                return 1
        
        return 0
    
    def __hash__(self) -> int:
        """Return hash based on agent index and target cluster."""
        return hash((self.agent_idx, self.target_cluster, 'cluster'))
    
    def __eq__(self, other) -> bool:
        """Check equality with another TransportClusterGoal."""
        return (isinstance(other, TransportClusterGoal) and 
                self.agent_idx == other.agent_idx and 
                self.target_cluster == other.target_cluster)
    
    def __repr__(self) -> str:
        return f"TransportClusterGoal(agent={self.agent_idx}, cluster={self.target_cluster})"


class TransportClusterGoalGenerator(PossibleGoalGenerator):
    """
    Generator that yields all possible cluster goals for an agent.
    
    Iterates over all clusters in the transport network, yielding a
    TransportClusterGoal for each cluster with weight 1.0.
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=20, num_clusters=5)
        >>> generator = TransportClusterGoalGenerator(env)
        >>> for goal, weight in generator.generate(state=None, human_agent_index=0):
        ...     print(f"{goal} with weight {weight}")
    """
    
    def __init__(self, env: TransportEnvWrapper):
        """
        Initialize the cluster goal generator.
        
        Args:
            env: The TransportEnvWrapper environment
        """
        super().__init__(env=env)
    
    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        """
        Generate all possible cluster goals for the specified agent.
        
        Yields one TransportClusterGoal for each cluster, all with weight 1.0.
        
        Args:
            state: Current state (ignored, uses env's clustering)
            human_agent_index: Index of the agent whose goals to generate
        
        Yields:
            Tuple[TransportClusterGoal, float]: Pairs of (goal, weight=1.0)
        """
        if not self.env.use_clusters or self.env.cluster_info is None:
            return
        
        for cluster_id in range(self.env.num_clusters):
            goal = TransportClusterGoal(self.env, agent_idx=human_agent_index, target_cluster=cluster_id)
            yield goal, 1.0


class TransportClusterGoalSampler(PossibleGoalSampler):
    """
    Sampler that uniformly samples cluster goals for an agent.
    
    Randomly selects a cluster and returns a TransportClusterGoal
    for that cluster with weight 1.0.
    
    Example:
        >>> env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=50, num_clusters=10)
        >>> sampler = TransportClusterGoalSampler(env)
        >>> goal, weight = sampler.sample(state=None, human_agent_index=0)
        >>> print(f"Sampled {goal} with weight {weight}")
    """
    
    def __init__(self, env: TransportEnvWrapper, seed: Optional[int] = None):
        """
        Initialize the cluster goal sampler.
        
        Args:
            env: The TransportEnvWrapper environment
            seed: Optional random seed for reproducibility
        """
        super().__init__(env=env)
        self.rng = np.random.RandomState(seed)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """
        Sample a random cluster goal for the specified agent.
        
        Args:
            state: Current state (ignored, uses env's clustering)
            human_agent_index: Index of the agent whose goal to sample
        
        Returns:
            Tuple[TransportClusterGoal, float]: (goal, weight=1.0)
        """
        if not self.env.use_clusters or self.env.num_clusters == 0:
            # Fall back to node-based sampling if clusters not enabled
            nodes = list(self.env.network.nodes())
            target_node = self.rng.choice(nodes)
            return TransportGoal(self.env, agent_idx=human_agent_index, target_node=target_node), 1.0
        
        target_cluster = self.rng.randint(0, self.env.num_clusters)
        goal = TransportClusterGoal(self.env, agent_idx=human_agent_index, target_cluster=target_cluster)
        return goal, 1.0
