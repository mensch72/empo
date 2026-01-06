"""
Backward Induction for Computing Human Policy Priors and Robot Policies.

This package implements backward induction on the state DAG to compute
goal-conditioned policies for human agents and goal-independent robot policies.

Main functions:
    compute_human_policy_prior: Compute tabular human policy prior via backward induction (Phase 1).
    compute_robot_policy: Compute tabular robot policy via backward induction (Phase 2).

The algorithms work by:
1. Building the DAG of reachable states and transitions
2. Processing states in reverse topological order (from terminal to initial)

Key features:
- Supports parallel computation for large state spaces
- Returns both the policy (prior) and optionally value functions

Parameters:
    beta_h: Human inverse temperature (β). Higher = more deterministic, inf = argmax.
    gamma_h: Human discount factor (γ).

Example usage:
    >>> from empo.backward_induction import compute_human_policy_prior, compute_robot_policy
    >>> from empo.possible_goal import PossibleGoalGenerator
    >>> 
    >>> # Define goal generator (implementation-specific)
    >>> goal_generator = MyGoalGenerator(env)
    >>> 
    >>> # Compute policy prior (Phase 1)
    >>> policy_prior = compute_human_policy_prior(
    ...     world_model=env,
    ...     human_agent_indices=[0, 1],  # agents 0 and 1 are humans
    ...     possible_goal_generator=goal_generator,
    ...     beta_h=10.0,  # high temperature = nearly optimal
    ...     gamma_h=1.0,
    ...     parallel=True
    ... )
    >>> 
    >>> # Compute robot policy (Phase 2)
    >>> robot_policy = compute_robot_policy(
    ...     world_model=env,
    ...     human_agent_indices=[0],
    ...     robot_agent_indices=[1],
    ...     possible_goal_generator=goal_generator,
    ...     human_policy_prior=policy_prior,
    ...     beta_r=5.0
    ... )
    >>> 
    >>> # Use the policies
    >>> action_dist = policy_prior(state, agent_idx=0, goal=my_goal)
    >>> robot_actions = robot_policy.sample(state)

Module structure:
    - helpers: Utility functions for combining action profiles, computing dependency levels
    - phase1: Human policy prior computation (compute_human_policy_prior)
    - phase2: Robot policy computation (compute_robot_policy, TabularRobotPolicy)
"""

# Re-export public API for backward compatibility
from .helpers import (
    default_believed_others_policy,
    combine_action_profiles,
    combine_action_profiles_batch,
    combine_profile_distributions,
    combine_profile_distributions_to_indices,
    compute_dependency_levels_general,
    compute_dependency_levels_fast,
    split_into_batches,
    AttainmentCache,
)

from .phase1 import compute_human_policy_prior

from .phase2 import (
    compute_robot_policy,
    TabularRobotPolicy,
)

__all__ = [
    # Phase 1
    'compute_human_policy_prior',
    # Phase 2
    'compute_robot_policy',
    'TabularRobotPolicy',
    # Types
    'AttainmentCache',
    # Helpers
    'default_believed_others_policy',
    'combine_action_profiles',
    'combine_action_profiles_batch',
    'combine_profile_distributions',
    'combine_profile_distributions_to_indices',
    'compute_dependency_levels_general',
    'compute_dependency_levels_fast',
    'split_into_batches',
]
