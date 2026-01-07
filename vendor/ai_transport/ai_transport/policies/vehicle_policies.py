"""
Vehicle agent policies for AI Transport environment.

Provides abstract base class and concrete implementations for vehicle decision-making.
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional


class VehiclePolicy(ABC):
    """Abstract base class for vehicle agent policies."""
    
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        """
        Initialize vehicle policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action for the current observation.
        
        Args:
            observation: Current observation for the agent
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
    
    @abstractmethod
    def reset(self):
        """Reset policy state (e.g., current destination)."""


class RandomVehiclePolicy(VehiclePolicy):
    """
    Completely random policy for vehicles.
    
    The vehicle takes random actions with configurable passing probabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        pass_prob_routing: float = 0.3,
        pass_prob_departing: float = 0.2,
        seed: Optional[int] = None
    ):
        """
        Initialize random vehicle policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            pass_prob_routing: Probability of passing (no destination) in routing step
            pass_prob_departing: Probability of passing (staying at node) in departing step
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.pass_prob_routing = pass_prob_routing
        self.pass_prob_departing = pass_prob_departing
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get random action with step-type-specific passing probability.
        
        Args:
            observation: Current observation for the agent (must contain 'step_type')
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        
        # Determine pass probability based on step type
        if step_type == 'routing':
            pass_prob = self.pass_prob_routing
        elif step_type == 'departing':
            pass_prob = self.pass_prob_departing
        else:
            pass_prob = 1.0  # Always pass in unboarding/boarding
        
        # Decide whether to pass
        if self.rng.random() < pass_prob:
            if step_type == 'routing':
                return 0, "Passing (keeping current destination, random choice)"
            elif step_type == 'departing':
                return 0, "Passing (staying at node, random choice)"
            else:
                return 0, "Passing (no action in this step)"
        
        # Otherwise, take random action from non-pass options
        if action_space_size <= 1:
            return 0, "Passing (only option)"
        
        action = self.rng.randint(1, action_space_size)
        
        # Get action description from action_mapping if available
        action_mapping = observation.get('action_mapping', {})
        desc = action_mapping.get('description', {}).get(action, 'unknown')
        detail = action_mapping.get('details', {}).get(action, '')
        
        if desc == 'set_destination_none':
            justification = "Clearing destination (random choice)"
        elif desc == 'set_destination_node':
            justification = f"Setting destination to node {detail} (random choice)"
        elif desc == 'depart_edge':
            justification = f"Departing to edge {detail} (random choice)"
        else:
            justification = f"Action {action} (random choice)"
            
        return action, justification
    
    def reset(self):
        """Reset policy state (no state to reset for random policy)."""


class ShortestPathVehiclePolicy(VehiclePolicy):
    """
    Policy where vehicle takes fastest path to destination.
    
    The vehicle:
    - Takes the fastest path to its current destination
    - When reaching destination, chooses new destination with probability proportional
      to Euclidean distance from current node
    - Uses edge speed to determine fastest path
    """
    
    def __init__(
        self,
        agent_id: str,
        network,
        seed: Optional[int] = None
    ):
        """
        Initialize shortest path vehicle policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            network: NetworkX graph with edge 'speed' attribute
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.network = network
        self.current_destination = None
        self.nodes = list(network.nodes())
        
        # Pre-compute node coordinates for distance calculations
        self.node_coords = {}
        for node in self.nodes:
            self.node_coords[node] = (
                network.nodes[node].get('x', 0.0),
                network.nodes[node].get('y', 0.0)
            )
        
        # Create travel time graph (weight = length / speed)
        self.time_graph = nx.DiGraph()
        for u, v, data in network.edges(data=True):
            length = data.get('length', 1.0)
            speed = data.get('speed', 1.0)
            travel_time = length / speed if speed > 0 else float('inf')
            self.time_graph.add_edge(u, v, weight=travel_time)
    
    def _euclidean_distance(self, node1, node2) -> float:
        """Compute Euclidean distance between two nodes."""
        x1, y1 = self.node_coords.get(node1, (0, 0))
        x2, y2 = self.node_coords.get(node2, (0, 0))
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _choose_new_destination(self, current_node) -> Optional[int]:
        """
        Choose new destination with probability proportional to distance.
        
        Args:
            current_node: Current node ID
            
        Returns:
            New destination node ID
        """
        # Get all other nodes
        other_nodes = [n for n in self.nodes if n != current_node]
        if not other_nodes:
            return None
        
        # Compute distances
        distances = np.array([self._euclidean_distance(current_node, n) for n in other_nodes])
        
        # Avoid division by zero
        distances = distances + 1e-6
        
        # Compute probabilities proportional to distance
        probabilities = distances / distances.sum()
        
        # Sample new destination
        return self.rng.choice(other_nodes, p=probabilities)
    
    def _get_next_node_on_path(self, current_node, destination) -> Optional[int]:
        """
        Get next node on shortest (fastest) path to destination.
        
        Args:
            current_node: Current node ID
            destination: Destination node ID
            
        Returns:
            Next node on path, or None if no path exists
        """
        if current_node == destination:
            return None
        
        try:
            path = nx.shortest_path(self.time_graph, current_node, destination, weight='weight')
            if len(path) >= 2:
                return path[1]  # Next node after current
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action based on shortest path to destination.
        
        For routing: set or update destination
        For departing: choose edge on shortest path to destination
        Otherwise: pass
        
        Args:
            observation: Current observation containing step_type, my_position, action_mapping
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        
        if step_type == 'routing':
            return self._get_routing_action(observation, action_space_size)
        elif step_type == 'departing':
            return self._get_departing_action(observation, action_space_size)
        else:
            # Unboarding, boarding, or on edge - pass
            if step_type == 'unboarding':
                return 0, "Passing (no action in unboarding step)"
            elif step_type == 'boarding':
                return 0, "Passing (no action in boarding step)"
            else:
                return 0, "Passing"
    
    def _get_routing_action(self, observation: Dict, action_space_size: int):
        """Set or update destination."""
        my_position = observation.get('my_position')
        action_mapping = observation.get('action_mapping', {})
        
        # If on edge, can't act
        if isinstance(my_position, tuple):
            return 0, "Passing (on edge)"
        
        current_node = my_position
        
        # Check if we've reached current destination
        if self.current_destination is None or current_node == self.current_destination:
            # Choose new destination
            self.current_destination = self._choose_new_destination(current_node)
        
        # Find action that sets destination to our target
        if self.current_destination is not None:
            details = action_mapping.get('details', {})
            for action_idx in range(1, action_space_size):
                if details.get(action_idx) == self.current_destination:
                    distance = self._euclidean_distance(current_node, self.current_destination)
                    return action_idx, f"Setting destination to node {self.current_destination} (distance-weighted choice, {distance:.1f} units)"
        
        # Default: pass (or set to None)
        return 0, "Passing (no destination or already set)"
    
    def _get_departing_action(self, observation: Dict, action_space_size: int):
        """Choose edge on shortest path to destination."""
        my_position = observation.get('my_position')
        action_mapping = observation.get('action_mapping', {})
        
        # If on edge, must pass
        if isinstance(my_position, tuple):
            return 0, "Passing (already on edge)"
        
        current_node = my_position
        
        # If no destination, pass
        if self.current_destination is None:
            return 0, "Passing (no destination set)"
        
        # Get next node on shortest path
        next_node = self._get_next_node_on_path(current_node, self.current_destination)
        if next_node is None:
            if current_node == self.current_destination:
                return 0, f"Passing (already at destination {self.current_destination})"
            else:
                return 0, f"Passing (no path to destination {self.current_destination})"
        
        # Find action corresponding to edge (current_node, next_node)
        details = action_mapping.get('details', {})
        for action_idx in range(1, action_space_size):
            edge = details.get(action_idx)
            if edge and isinstance(edge, tuple) and edge == (current_node, next_node):
                return action_idx, f"Taking shortest path to destination {self.current_destination} via edge {edge}"
        
        # Edge not found, pass
        return 0, f"Passing (edge to next node {next_node} not available)"
    
    def reset(self):
        """Reset policy state (current destination)."""
        self.current_destination = None
