"""
Base class for Gymnasium environments with state management and transition probability computation capabilities.

This module provides a base class that extends gymnasium.Env with methods
for explicit state management (get_state, set_state) and transition probability
computation.
"""

from abc import abstractmethod
from collections import deque
from typing import List, Dict, Tuple, Any, Optional

import gymnasium as gym
import numpy as np


class WorldModel(gym.Env):
    """
    Base class for Gymnasium environments with explicit state management.
    
    This class extends gymnasium.Env to provide:
    1. get_state(): Get a hashable representation of the complete environment state
    2. set_state(): Restore the environment to a specific state
    3. transition_probabilities(): Compute exact transition probabilities for actions
    
    These methods enable planning algorithms and empowerment computation that require
    explicit enumeration of state spaces and transition dynamics.
    
    Subclasses must implement the abstract methods to provide environment-specific
    state serialization and transition logic.
    """
    
    @abstractmethod
    def get_state(self) -> Any:
        """
        Get the complete state of the environment.
        
        Returns a hashable representation containing everything needed to predict
        the consequences of possible actions. The state must be sufficient to
        restore the environment to an identical configuration via set_state().
        
        Returns:
            A hashable representation of the complete environment state.
            Must be usable as a dictionary key.
        """
        raise NotImplementedError("Subclasses must implement get_state()")
    
    @abstractmethod
    def set_state(self, state: Any) -> None:
        """
        Set the environment to a specific state.
        
        Restores the environment to the exact configuration represented by the
        given state. After calling set_state(s), get_state() should return a
        state equivalent to s.
        
        Args:
            state: A state as returned by get_state()
        """
        raise NotImplementedError("Subclasses must implement set_state()")
    
    @abstractmethod
    def transition_probabilities(
        self, 
        state: Any, 
        actions: List[int]
    ) -> Optional[List[Tuple[float, Any]]]:
        """
        Given a state and vector of actions, return possible transitions with exact probabilities.
        
        This method computes all possible successor states and their probabilities
        given the current state and actions. It is essential for planning algorithms
        that require explicit enumeration of transition dynamics.
        
        Args:
            state: A state tuple as returned by get_state()
            actions: List of action indices, one per agent (for multi-agent envs)
                    or a single action (for single-agent envs)
            
        Returns:
            list: List of (probability, successor_state) tuples describing all
                  possible transitions. Probabilities should sum to 1.0.
                  Returns None if the state is terminal or if actions are invalid.
        """
        raise NotImplementedError("Subclasses must implement transition_probabilities()")
    
    def initial_state(self) -> Any:
        """
        Get the initial state of the environment without permanently resetting it.
        
        This method:
        1. Saves the current state
        2. Calls reset() to get the initial state
        3. Returns the initial state
        4. Restores the environment to its previous state
        
        This is useful for algorithms that need to know the initial state
        without losing the current environment configuration.
        
        Returns:
            The initial state of the environment (as returned by get_state() after reset())
        """
        # Save current state
        saved_state = self.get_state()
        
        # Reset to get initial state
        self.reset()
        init_state = self.get_state()
        
        # Restore previous state
        self.set_state(saved_state)
        
        return init_state
    
    def is_terminal(self, state: Optional[Any] = None) -> bool:
        """
        Check if a state is terminal (no valid transitions exist).
        
        A state is considered terminal if transition_probabilities() returns None
        for all possible actions, indicating that no further transitions are possible.
        
        Args:
            state: The state to check. If None, checks the current state.
        
        Returns:
            True if the state is terminal, False otherwise.
        """
        # Get the state to check
        if state is None:
            state = self.get_state()
        
        # Try a simple action (e.g., all zeros) to check if transitions are possible
        # For multi-agent environments, we need to know the number of agents
        if hasattr(self, 'agents'):
            num_agents = len(self.agents)
        else:
            num_agents = 1
        
        # Use action 0 for all agents as a test action
        test_actions = [0] * num_agents
        
        # Check if transition_probabilities returns None (terminal) or a list (non-terminal)
        transitions = self.transition_probabilities(state, test_actions)
        
        return transitions is None
    
    def step(self, actions: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment using transition probabilities.
        
        This default implementation:
        1. Gets the current state
        2. Calls transition_probabilities() to get possible transitions
        3. Samples from those transitions based on probabilities
        4. Calls set_state() with the sampled successor state
        5. Returns (observation, reward, terminated, truncated, info)
        
        Subclasses may override this for more efficient implementations or
        to add custom logic (e.g., rendering, reward computation).
        
        Args:
            actions: Action(s) to take. For multi-agent environments, this should
                    be a list of actions, one per agent.
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: The new state (as returned by get_state())
                - reward: 0.0 (subclasses should override for actual rewards)
                - terminated: True if the new state is terminal
                - truncated: False (subclasses should override for truncation logic)
                - info: Empty dict (subclasses can add additional info)
        """
        # Ensure actions is a list for multi-agent compatibility
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        
        # Get current state
        current_state = self.get_state()
        
        # Get transition probabilities
        transitions = self.transition_probabilities(current_state, list(actions))
        
        # If terminal state (no transitions), return current state
        if transitions is None:
            return current_state, 0.0, True, False, {}
        
        # Sample from transitions based on probabilities
        probabilities = [prob for prob, _ in transitions]
        successor_states = [state for _, state in transitions]
        
        # Use numpy to sample based on probabilities
        chosen_idx = np.random.choice(len(transitions), p=probabilities)
        new_state = successor_states[chosen_idx]
        
        # Set the environment to the new state
        self.set_state(new_state)
        
        # Check if new state is terminal
        terminated = self.is_terminal(new_state)
        
        # Return observation (the new state), reward, terminated, truncated, info
        # Subclasses should override to provide actual rewards
        return new_state, 0.0, terminated, False, {}
    
    def get_dag(self) -> Tuple[List[Any], Dict[Any, int], List[List[int]]]:
        """
        Efficiently compute the DAG structure of an acyclic finite environment.
        
        This method uses a two-phase algorithm:
        1. BFS to discover all reachable states and edges
        2. Topological sort (Kahn's algorithm) to order states correctly
        
        This ensures that successor states always come after their predecessors,
        even when a state is reachable via multiple paths of different lengths.
        
        Time Complexity: O(|S| + |T|) where |S| is the number of states and |T| is the
        total number of transitions. Phase 1 visits each state once and examines each
        transition once. Phase 2 (topological sort) also runs in O(|S| + |T|).
        
        Space Complexity: O(|S| + |T|) for storing states, edges, and auxiliary structures.
        
        Returns:
            A tuple containing:
            1. states (List): List of all reachable states in topological order
               (successor states always come after predecessor states)
            2. state_to_idx (Dict): Dictionary mapping each state to its index in
               the states list
            3. successors (List[List[int]]): List where successors[i] contains the
               indices of all possible successor states of states[i]
        
        Example:
            >>> states, state_to_idx, successors = env.get_dag()
            >>> # states[0] is the root state
            >>> # state_to_idx[states[i]] == i
            >>> # successors[i] are indices of states reachable from states[i]
            >>> # For any edge i -> j in the DAG: i < j (topological ordering)
        """
        # Reset environment to get root state
        self.reset()
        root_state = self.get_state()
        
        # PHASE 1: Discover all states and edges using BFS
        discovered_states = []  # Temporary list during discovery
        temp_state_to_idx = {}  # Temporary mapping
        edges = {}  # edges[i] = set of successor indices (temporary)
        
        queue = deque([root_state])
        temp_state_to_idx[root_state] = 0
        discovered_states.append(root_state)
        edges[0] = set()
        
        while queue:
            current_state = queue.popleft()
            current_idx = temp_state_to_idx[current_state]
            
            # Restore environment to current state to explore transitions
            self.set_state(current_state)
            num_agents = len(self.agents)
            num_actions = self.action_space.n
            
            # Track unique successor states for this current state
            seen_successors = set()
            
            # Generate all action combinations efficiently
            # For n agents with k actions each: k^n combinations
            total_combinations = num_actions ** num_agents
            
            for combo_idx in range(total_combinations):
                # Convert combo_idx to action tuple
                actions = []
                temp = combo_idx
                for _ in range(num_agents):
                    actions.append(temp % num_actions)
                    temp //= num_actions
                
                # Get transition probabilities for this action combination
                transitions = self.transition_probabilities(current_state, actions)
                
                # If None, state is terminal or actions are invalid
                if transitions is None:
                    continue
                
                # Process all successor states from these transitions
                for prob, successor_state in transitions:
                    # Skip if we've already seen this successor from current state
                    if successor_state in seen_successors:
                        continue
                    
                    seen_successors.add(successor_state)
                    
                    # If this is a new state, add it to discovery list
                    if successor_state not in temp_state_to_idx:
                        temp_state_to_idx[successor_state] = len(discovered_states)
                        discovered_states.append(successor_state)
                        edges[len(discovered_states) - 1] = set()
                        queue.append(successor_state)
                    
                    # Add edge from current to successor
                    successor_idx = temp_state_to_idx[successor_state]
                    edges[current_idx].add(successor_idx)
        
        # PHASE 2: Topological sort using Kahn's algorithm
        num_states = len(discovered_states)
        
        # Compute in-degree for each state
        in_degree = [0] * num_states
        for state_idx in range(num_states):
            for successor_idx in edges[state_idx]:
                in_degree[successor_idx] += 1
        
        # Initialize queue with all states that have in-degree 0
        topo_queue = deque()
        for i in range(num_states):
            if in_degree[i] == 0:
                topo_queue.append(i)
        
        # Topologically sorted order (indices in discovered_states)
        topo_order = []
        
        while topo_queue:
            current_idx = topo_queue.popleft()
            topo_order.append(current_idx)
            
            # Reduce in-degree of successors
            for successor_idx in edges[current_idx]:
                in_degree[successor_idx] -= 1
                if in_degree[successor_idx] == 0:
                    topo_queue.append(successor_idx)
        
        # Verify we got all states (no cycles)
        if len(topo_order) != num_states:
            raise ValueError("Environment contains cycles (not a DAG)")
        
        # PHASE 3: Build final output with new indices
        states = []
        state_to_idx = {}
        old_to_new = {}  # Map old indices to new indices
        
        for new_idx, old_idx in enumerate(topo_order):
            state = discovered_states[old_idx]
            states.append(state)
            state_to_idx[state] = new_idx
            old_to_new[old_idx] = new_idx
        
        # Build successors list with new indices
        successors = [[] for _ in range(num_states)]
        for old_idx in range(num_states):
            new_idx = old_to_new[old_idx]
            for old_succ_idx in edges[old_idx]:
                new_succ_idx = old_to_new[old_succ_idx]
                successors[new_idx].append(new_succ_idx)
        
        return states, state_to_idx, successors
    
    def plot_dag(
        self,
        states: Optional[List[Any]] = None,
        state_to_idx: Optional[Dict[Any, int]] = None,
        successors: Optional[List[List[int]]] = None,
        output_file: Optional[str] = "dag",
        format: str = "png",
        state_labels: Optional[Dict[Any, str]] = None,
        rankdir: str = "TB"
    ) -> str:
        """
        Plot the DAG structure using Graphviz.
        
        If states, state_to_idx, and successors are not provided, they will be
        computed by calling get_dag().
        
        Args:
            states: List of states in topological order (from get_dag)
            state_to_idx: Dictionary mapping states to indices (from get_dag)
            successors: List of successor indices for each state (from get_dag)
            output_file: Output filename without extension (default: "dag")
            format: Output format - 'png', 'pdf', 'svg', etc. (default: "png")
            state_labels: Optional dict mapping states to custom labels for display
            rankdir: Graph direction - 'TB' (top-bottom), 'LR' (left-right), etc.
        
        Returns:
            Path to the generated image file
        
        Raises:
            ImportError: If graphviz is not installed
        
        Example:
            >>> env.plot_dag(output_file="my_dag", format="pdf")
            'my_dag.pdf'
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz is required for plotting. Install with: pip install graphviz"
            )
        
        # Compute DAG if not provided
        if states is None or state_to_idx is None or successors is None:
            states, state_to_idx, successors = self.get_dag()
        
        # Create directed graph
        dot = graphviz.Digraph(comment='State Space DAG')
        dot.attr(rankdir=rankdir)
        dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')
        
        # Add nodes
        for idx, state in enumerate(states):
            # Determine node label
            if state_labels and state in state_labels:
                label = state_labels[state]
            else:
                label = str(state)
            
            # Highlight root node
            if idx == 0:
                dot.node(str(idx), label, fillcolor='lightgreen', style='filled')
            # Highlight terminal nodes (no successors)
            elif len(successors[idx]) == 0:
                dot.node(str(idx), label, fillcolor='lightcoral', style='filled')
            else:
                dot.node(str(idx), label)
        
        # Add edges
        for idx, succ_list in enumerate(successors):
            for succ_idx in succ_list:
                dot.edge(str(idx), str(succ_idx))
        
        # Render to file
        output_path = dot.render(output_file, format=format, cleanup=True)
        
        return output_path
