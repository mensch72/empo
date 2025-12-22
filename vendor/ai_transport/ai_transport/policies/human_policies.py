"""
Human agent policies for AI Transport environment.

Provides abstract base class and concrete implementations for human decision-making.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class HumanPolicy(ABC):
    """Abstract base class for human agent policies."""
    
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        """
        Initialize human policy.
        
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
        pass
    
    @abstractmethod
    def reset(self):
        """Reset policy state (e.g., target destination)."""
        pass


class RandomHumanPolicy(HumanPolicy):
    """
    Completely random policy with configurable passing probabilities by step type.
    
    The agent passes with a certain probability depending on the current step type,
    and otherwise takes a random action from the available options.
    """
    
    def __init__(
        self,
        agent_id: str,
        pass_prob_routing: float = 1.0,
        pass_prob_unboarding: float = 0.8,
        pass_prob_boarding: float = 0.7,
        pass_prob_departing: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize random human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            pass_prob_routing: Probability of passing in routing step (default: 1.0, humans can't act)
            pass_prob_unboarding: Probability of passing in unboarding step (default: 0.8)
            pass_prob_boarding: Probability of passing in boarding step (default: 0.7)
            pass_prob_departing: Probability of passing in departing step (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.pass_probs = {
            'routing': pass_prob_routing,
            'unboarding': pass_prob_unboarding,
            'boarding': pass_prob_boarding,
            'departing': pass_prob_departing
        }
    
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
        pass_prob = self.pass_probs.get(step_type, 0.5)
        
        # Decide whether to pass
        if self.rng.random() < pass_prob:
            return 0, "Passing (random choice)"
        
        # Otherwise, take random action from non-pass options
        if action_space_size <= 1:
            return 0, "Passing (only option)"
        
        action = self.rng.randint(1, action_space_size)
        
        # Get action description from action_mapping if available
        action_mapping = observation.get('action_mapping', {})
        desc = action_mapping.get('description', {}).get(action, 'unknown')
        detail = action_mapping.get('details', {}).get(action, '')
        
        if desc == 'unboard':
            justification = "Unboarding (random choice)"
        elif desc == 'board_vehicle':
            justification = f"Boarding {detail} (random choice)"
        elif desc == 'depart_edge':
            justification = f"Walking to edge {detail} (random choice)"
        else:
            justification = f"Action {action} (random choice)"
            
        return action, justification
    
    def reset(self):
        """Reset policy state (no state to reset for random policy)."""
        pass


class TargetDestinationHumanPolicy(HumanPolicy):
    """
    Policy where human has a target destination and boards vehicles heading that direction.
    
    The human:
    - Has a target destination (node) that changes with some probability rate per real time
    - At boarding steps, boards the vehicle whose destination is closest in Euclidean distance
      to the human's target destination
    - At departing steps, walks toward the target if no suitable vehicle is available
    """
    
    def __init__(
        self,
        agent_id: str,
        network,
        target_change_rate: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize target destination human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            network: NetworkX graph with node coordinates (x, y attributes)
            target_change_rate: Probability rate per second of changing target destination
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.network = network
        self.target_change_rate = target_change_rate
        self.target = None
        self.last_real_time = 0.0
        self.nodes = list(network.nodes())
        
        # Pre-compute node coordinates for distance calculations
        self.node_coords = {}
        for node in self.nodes:
            self.node_coords[node] = (
                network.nodes[node].get('x', 0.0),
                network.nodes[node].get('y', 0.0)
            )

        # Set initial target destination
        self._update_target(0.0)

    
    def _euclidean_distance(self, node1, node2) -> float:
        """Compute Euclidean distance between two nodes."""
        x1, y1 = self.node_coords.get(node1, (0, 0))
        x2, y2 = self.node_coords.get(node2, (0, 0))
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _update_target(self, real_time: float):
        """Update target destination based on time elapsed and change rate."""
        if self.target is None:
            # Initialize target
            self.target = self.rng.choice(self.nodes)
            self.last_real_time = real_time
            return
        
        # Check if target should change based on time elapsed
        time_elapsed = real_time - self.last_real_time
        change_prob = 1.0 - np.exp(-self.target_change_rate * time_elapsed)
        
        if self.rng.random() < change_prob:
            # Change to new random target
            self.target = self.rng.choice(self.nodes)
            self.last_real_time = real_time
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action based on target destination.
        
        For boarding: choose vehicle whose destination is closest to target
        For departing: choose edge leading toward target
        Otherwise: pass
        
        Args:
            observation: Current observation containing step_type, real_time, action_mapping
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        real_time = observation.get('real_time', 0.0)
        action_mapping = observation.get('action_mapping', {})
        
        # Update target destination
        self._update_target(real_time)
        
        if step_type == 'boarding':
            return self._get_boarding_action(observation, action_mapping, action_space_size)
        elif step_type == 'departing':
            return self._get_departing_action(observation, action_mapping, action_space_size)
        else:
            # Routing, unboarding, or on edge - pass
            if step_type == 'routing':
                return 0, "Passing (no action in routing step)"
            elif step_type == 'unboarding':
                return 0, "Passing (not aboard vehicle)"
            else:
                return 0, "Passing"
    
    def _get_boarding_action(self, observation: Dict, action_mapping: Dict, action_space_size: int):
        """Choose vehicle whose destination is closest to target."""
        if action_space_size <= 1 or self.target is None:
            return 0, "Passing (no target or no vehicles available)"
        
        # Get vehicle destinations from observation
        vehicle_destinations = observation.get('vehicle_destinations', {})
        details = action_mapping.get('details', {})
        
        best_action = 0
        best_distance = float('inf')
        best_vehicle = None
        
        for action_idx in range(1, action_space_size):
            vehicle_id = details.get(action_idx)
            if vehicle_id:
                vehicle_dest = vehicle_destinations.get(vehicle_id)
                if vehicle_dest is not None:
                    distance = self._euclidean_distance(vehicle_dest, self.target)
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action_idx
                        best_vehicle = vehicle_id
        
        if best_action > 0:
            return best_action, f"Boarding {best_vehicle} (heading toward target {self.target}, distance {best_distance:.1f})"
        else:
            return 0, f"Passing (no vehicles heading toward target {self.target})"
    
    def _get_departing_action(self, observation: Dict, action_mapping: Dict, action_space_size: int):
        """Choose edge leading toward target destination."""
        if action_space_size <= 1 or self.target is None:
            return 0, "Passing (no target or no edges available)"
        
        my_position = observation.get('my_position')
        if isinstance(my_position, tuple):
            return 0, "Passing (already on edge)"
        
        # Get current node
        current_node = my_position
        if current_node == self.target:
            return 0, f"Passing (already at target {self.target})"
        
        # Find edge that leads toward target
        details = action_mapping.get('details', {})
        best_action = 0
        best_distance = float('inf')
        best_edge = None
        
        for action_idx in range(1, action_space_size):
            edge = details.get(action_idx)
            if edge and isinstance(edge, tuple):
                # Edge is (source, target)
                target_node = edge[1]
                distance = self._euclidean_distance(target_node, self.target)
                if distance < best_distance:
                    best_distance = distance
                    best_action = action_idx
                    best_edge = edge
        
        if best_action > 0:
            return best_action, f"Walking toward target {self.target} via edge {best_edge}"
        else:
            return 0, f"Passing (no edge toward target {self.target})"
    
    def reset(self):
        """Reset policy state (target destination and time)."""
        self.target = None
        self.last_real_time = 0.0
