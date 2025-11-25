"""
Utility functions for working with gym environments.

Note: These functions are provided for backward compatibility.
The preferred approach is to use the corresponding methods directly on
WorldModel instances (env.get_dag() and env.plot_dag()).
"""

from collections import deque
from typing import List, Dict, Tuple, Any, Optional


def get_dag(env) -> Tuple[List[Any], Dict[Any, int], List[List[int]]]:
    """
    Efficiently compute the DAG structure of an acyclic finite gym environment.
    
    This function delegates to the get_dag() method on the environment if available.
    For environments that inherit from WorldModel, this calls env.get_dag() directly.
    For other environments, it uses the standalone implementation.
    
    This function uses a two-phase algorithm:
    1. BFS to discover all reachable states and edges
    2. Topological sort (Kahn's algorithm) to order states correctly
    
    This ensures that successor states always come after their predecessors,
    even when a state is reachable via multiple paths of different lengths.
    
    Time Complexity: O(|S| + |T|) where |S| is the number of states and |T| is the
    total number of transitions. Phase 1 visits each state once and examines each
    transition once. Phase 2 (topological sort) also runs in O(|S| + |T|).
    
    Space Complexity: O(|S| + |T|) for storing states, edges, and auxiliary structures.
    
    Args:
        env: A gym environment with:
            - A method `reset()` that returns the root state
            - A method `transition_probabilities(state, actions)` that returns
              a list of (probability, successor_state) tuples for all possible
              action combinations, or None if the state is terminal
            - An attribute `action_space` with a method `sample()` for generating
              valid actions
    
    Returns:
        A tuple containing:
        1. states (List): List of all reachable states in topological order
           (successor states always come after predecessor states)
        2. state_to_idx (Dict): Dictionary mapping each state to its index in
           the states list
        3. successors (List[List[int]]): List where successors[i] contains the
           indices of all possible successor states of states[i]
    
    Example:
        >>> states, state_to_idx, successors = get_dag(env)
        >>> # states[0] is the root state
        >>> # state_to_idx[states[i]] == i
        >>> # successors[i] are indices of states reachable from states[i]
        >>> # For any edge i -> j in the DAG: i < j (topological ordering)
    """
    # If the environment has a get_dag method (e.g., inherits from WorldModel), use it
    if hasattr(env, 'get_dag') and callable(env.get_dag):
        return env.get_dag()
    
    # Fallback to standalone implementation for environments without the method
    return _get_dag_standalone(env)


def _get_dag_standalone(env) -> Tuple[List[Any], Dict[Any, int], List[List[int]]]:
    """
    Standalone implementation of get_dag for environments without the method.
    
    This is the original implementation that works with any environment
    that provides get_state, set_state, and transition_probabilities.
    """
    # Reset environment to get root state
    env.reset()
    root_state = env.get_state()
    
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
        env.set_state(current_state)
        num_agents = len(env.agents)
        num_actions = env.action_space.n
        
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
            transitions = env.transition_probabilities(current_state, actions)
            
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
    states: List[Any],
    state_to_idx: Dict[Any, int],
    successors: List[List[int]],
    output_file: Optional[str] = "dag",
    format: str = "png",
    state_labels: Optional[Dict[Any, str]] = None,
    rankdir: str = "TB"
) -> str:
    """
    Plot the DAG structure using Graphviz.
    
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
        >>> states, state_to_idx, successors = get_dag(env)
        >>> plot_dag(states, state_to_idx, successors, "my_dag", "pdf")
        'my_dag.pdf'
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz is required for plotting. Install with: pip install graphviz"
        )
    
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
