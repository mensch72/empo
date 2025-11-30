"""
Environment module for EMPO framework.

This module contains custom MultiGrid environments designed for multi-agent
reinforcement learning experiments with human-robot collaboration scenarios.

Environments:
    OneOrThreeChambersEnv: Large multi-chamber environment (28x16 grid).
        - 15 human agents (red) in upper-center area
        - 2 robot agents (green) in middle
        - Rock and block obstacles for pushing mechanics
        
    OneOrThreeChambersMapEnv: Same as above but using map-based specification.
        - Demonstrates the map string format for environment definition
        
    SmallOneOrTwoChambersMapEnv: Simplified version for tractable computation.
        - Smaller 10x9 grid
        - 2 human agents (yellow) + 1 robot (grey)
        - 8-step timeout for finite state space
        - Suitable for DAG computation and backward induction

Map specification format:
    Environments can be defined using ASCII map strings where each cell
    is represented by a two-character code:
    
    Object codes:
        .. : empty cell
        We : grey wall
        Ay : yellow agent (human)
        Ae : grey agent (robot)
        Ro : rock (pushable by authorized agents)
        Bl : block (pushable by any agent)
        Sw : switch
        Un : unsteady ground
        
    Color codes (second character):
        r : red, g : green, b : blue, p : purple, y : yellow, e : grey

Example usage:
    >>> from src.envs import SmallOneOrTwoChambersMapEnv
    >>> 
    >>> env = SmallOneOrTwoChambersMapEnv()
    >>> obs = env.reset()
    >>> 
    >>> # Environment has 3 agents
    >>> print(f"Agents: {len(env.agents)}")
    >>> 
    >>> # Get state for planning
    >>> state = env.get_state()
    >>> 
    >>> # Compute transitions for action profile
    >>> transitions = env.transition_probabilities(state, [0, 0, 0])
"""

from .one_or_three_chambers import OneOrThreeChambersEnv, OneOrThreeChambersMapEnv, SmallOneOrTwoChambersMapEnv

__all__ = ['OneOrThreeChambersEnv', 'OneOrThreeChambersMapEnv', 'SmallOneOrTwoChambersMapEnv']
