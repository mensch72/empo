"""
Utility functions for working with gymnasium environments.

This module provides backward-compatible wrapper functions that delegate to
the methods on WorldModel (or compatible environments).

**Preferred Approach:**
For environments inheriting from WorldModel (including MultiGridEnv), use
the instance methods directly:
    - env.get_dag() instead of get_dag(env)
    - env.plot_dag() instead of plot_dag(...)

Functions:
    get_dag: Wrapper that delegates to env.get_dag().
    plot_dag: Wrapper that delegates to env.plot_dag().
"""

from typing import List, Dict, Tuple, Any, Optional


def get_dag(env) -> Tuple[List[Any], Dict[Any, int], List[List[int]]]:
    """
    Compute the DAG structure of an acyclic finite gym environment.
    
    This is a backward-compatible wrapper that delegates to env.get_dag().
    
    Args:
        env: A WorldModel (or compatible environment) with a get_dag() method.
    
    Returns:
        A tuple containing:
        1. states (List): List of all reachable states in topological order
        2. state_to_idx (Dict): Dictionary mapping each state to its index
        3. successors (List[List[int]]): List of successor indices for each state
    
    Example:
        >>> states, state_to_idx, successors = get_dag(env)
    """
    if not hasattr(env, 'get_dag') or not callable(env.get_dag):
        raise TypeError(
            "Environment must have a get_dag() method. "
            "Use an environment that inherits from WorldModel."
        )
    return env.get_dag()


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
    
    This is a backward-compatible wrapper. For environments inheriting from
    WorldModel, you can also use env.plot_dag() directly.
    
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
