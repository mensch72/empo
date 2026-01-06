"""
Helper functions for backward induction computations.

This module provides utility functions used by both Phase 1 (human policy prior)
and Phase 2 (robot policy) backward induction algorithms.
"""

import numpy as np
import numpy.typing as npt
from collections import defaultdict
from itertools import product
from typing import Optional, Callable, List, Tuple, Dict, Any, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:
    from empo.possible_goal import PossibleGoal, PossibleGoalGenerator

# Type aliases for complex types used throughout
State: TypeAlias = Any  # State is typically a hashable tuple from WorldModel.get_state()
ActionProfile = List[int]
TransitionData = Tuple[Tuple[int, ...], List[float], List[State]]  # (action_profile, probs, successor_states)

# Cache for goal attainment values computed during phase 1 and reused in phase 2.
# Structure: state_index -> action_profile_index -> goal -> array of attainment values for successor states.
# Using nested dicts for fast O(1) lookups. The innermost value is an ndarray of 0/1 values
# indicating whether each successor state achieves the goal.
AttainmentCache: TypeAlias = Dict[int, Dict[int, Dict["PossibleGoal", npt.NDArray[np.int8]]]]


def default_believed_others_policy(
    state: State, 
    agent_index: int, 
    action: int, 
    num_agents: int, 
    num_actions: int,
    robot_agent_indices: List[int]
) -> List[Tuple[float, npt.NDArray[np.int64]]]:
    """Default believed others policy - uniform distribution over other humans.
    
    Robot agent positions are set to -1 (placeholder) since they will be
    overwritten by the caller when iterating over robot action profiles.
    """
    # Number of other human agents (exclude self and robots)
    num_other_humans = num_agents - 1 - len(robot_agent_indices)
    uniform_p = 1 / (num_actions ** num_other_humans) if num_other_humans > 0 else 1.0
    # Each action profile for the other human agents gets the same probability.
    # The agent's own action and robot actions are set to -1 since they will be overwritten.
    all_actions = list(range(num_actions))
    robot_set = set(robot_agent_indices)
    return [(uniform_p, np.array(action_profile, dtype=np.int64)) for action_profile in product(*[
        [-1] if (idx == agent_index or idx in robot_set) else all_actions
        for idx in range(num_agents)])]


def collect_goals_by_human(
    goal_generator: "PossibleGoalGenerator",
    human_agent_indices: List[int],
    state: State
) -> Dict[int, List[Tuple["PossibleGoal", float]]]:
    """
    Collect all possible goals for each human agent from a generator.
    
    This function iterates through the goal generator for each human agent
    and collects all (goal, weight) pairs into a dictionary keyed by human
    agent index.
    
    Args:
        goal_generator: A PossibleGoalGenerator that yields (goal, weight) pairs.
        human_agent_indices: List of agent indices that are humans.
        state: Current world state (passed to the generator).
    
    Returns:
        Dict mapping each human agent index to a list of (PossibleGoal, weight)
        tuples as returned by the generator for that human.
    
    Example:
        >>> generator = AllCellsGenerator(env)
        >>> goals_by_human = collect_goals_by_human(generator, [0, 2], state)
        >>> # goals_by_human[0] contains all (goal, weight) pairs for human 0
        >>> # goals_by_human[2] contains all (goal, weight) pairs for human 2
    """
    return {
        human_idx: list(goal_generator.generate(state, human_idx))
        for human_idx in human_agent_indices
    }


def combine_action_profiles(
    human_action_profile: List[int] | npt.NDArray[np.int64],
    robot_action_profile: List[int] | npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None
) -> List[int]:
    """
    Combine human and robot action profiles into a full action profile for all agents.
    
    This function interleaves actions from humans and robots according to their
    respective agent indices to produce a complete action profile where
    full_profile[agent_index] contains the action for that agent.
    
    Args:
        human_action_profile: List/array of actions for human agents, in the order
                             specified by human_agent_indices.
        robot_action_profile: List/array of actions for robot agents, in the order
                             specified by robot_agent_indices.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
    
    Returns:
        List[int]: Full action profile of length num_agents,
                  where each position corresponds to the agent at that index.
    
    Example:
        >>> # 4 agents: agents 0,2 are humans, agents 1,3 are robots
        >>> human_actions = [3, 1]  # human 0 does action 3, human 2 does action 1
        >>> robot_actions = [2, 0]  # robot 1 does action 2, robot 3 does action 0
        >>> combine_action_profiles(human_actions, robot_actions, [0, 2], [1, 3])
        [3, 2, 1, 0]  # full_profile[i] = action of agent i
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    # Use numpy for efficient scatter operation
    full_profile = np.empty(num_agents, dtype=np.int64)
    full_profile[human_agent_indices] = human_action_profile
    full_profile[robot_agent_indices] = robot_action_profile
    
    return full_profile.tolist()


def combine_action_profiles_batch(
    human_action_profiles: npt.NDArray[np.int64],
    robot_action_profile: List[int] | npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None
) -> npt.NDArray[np.int64]:
    """
    Combine multiple human action profiles with a single robot action profile.
    
    Vectorized version for efficiently processing many human action profiles
    at once (e.g., from profile_distribution) with a fixed robot action.
    
    Args:
        human_action_profiles: Array of shape (N, num_humans) containing N human
                              action profiles to combine.
        robot_action_profile: Single robot action profile to use for all combinations.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
    
    Returns:
        np.ndarray: Array of shape (N, num_agents) containing the combined profiles.
    
    Example:
        >>> # 3 agents: agent 0 is human, agents 1,2 are robots
        >>> human_profiles = np.array([[0], [1], [2], [3]])  # 4 human action profiles
        >>> robot_actions = [5, 6]  # robots do actions 5 and 6
        >>> combine_action_profiles_batch(human_profiles, robot_actions, [0], [1, 2])
        array([[0, 5, 6],
               [1, 5, 6],
               [2, 5, 6],
               [3, 5, 6]])
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_profiles = len(human_action_profiles)
    
    # Allocate output array
    full_profiles = np.empty((n_profiles, num_agents), dtype=np.int64)
    
    # Scatter human actions (vectorized across all profiles)
    full_profiles[:, human_agent_indices] = human_action_profiles
    
    # Broadcast robot actions to all profiles
    full_profiles[:, robot_agent_indices] = robot_action_profile
    
    return full_profiles


def combine_profile_distributions(
    human_probs: npt.NDArray[np.floating[Any]],
    human_profiles: npt.NDArray[np.int64],
    robot_probs: npt.NDArray[np.floating[Any]],
    robot_profiles: npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    num_agents: Optional[int] = None,
    min_prob: float = 0.0
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
    """
    Combine human and robot profile distributions into a joint distribution.
    
    Computes the outer product of independent human and robot distributions,
    returning the joint probability distribution over full action profiles.
    
    Args:
        human_probs: Array of shape (N_h,) with probabilities for each human profile.
        human_profiles: Array of shape (N_h, num_humans) with human action profiles.
        robot_probs: Array of shape (N_r,) with probabilities for each robot profile.
        robot_profiles: Array of shape (N_r, num_robots) with robot action profiles.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        num_agents: Total number of agents. If None, computed from indices.
        min_prob: Minimum probability threshold. Profiles with prob <= min_prob
                 are filtered out. Default 0.0 keeps all non-zero entries.
    
    Returns:
        Tuple of:
        - joint_probs: Array of shape (M,) with joint probabilities, where
          M <= N_h * N_r (filtered by min_prob).
        - full_profiles: Array of shape (M, num_agents) with combined profiles.
    
    Example:
        >>> # 3 agents: agent 0 is human, agents 1,2 are robots
        >>> human_probs = np.array([0.3, 0.7])
        >>> human_profiles = np.array([[0], [1]])  # human does action 0 or 1
        >>> robot_probs = np.array([0.4, 0.6])
        >>> robot_profiles = np.array([[2, 3], [4, 5]])  # robots do (2,3) or (4,5)
        >>> probs, profiles = combine_profile_distributions(
        ...     human_probs, human_profiles, robot_probs, robot_profiles,
        ...     [0], [1, 2])
        >>> probs  # [0.3*0.4, 0.3*0.6, 0.7*0.4, 0.7*0.6]
        array([0.12, 0.18, 0.28, 0.42])
        >>> profiles
        array([[0, 2, 3],
               [0, 4, 5],
               [1, 2, 3],
               [1, 4, 5]])
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_human = len(human_probs)
    n_robot = len(robot_probs)
    
    # Compute outer product of probabilities: shape (N_h, N_r)
    joint_probs_2d = np.outer(human_probs, robot_probs)
    
    # Flatten to 1D
    joint_probs_flat = joint_probs_2d.ravel()
    
    # Filter by minimum probability if requested
    if min_prob > 0.0:
        mask = joint_probs_flat > min_prob
        joint_probs_flat = joint_probs_flat[mask]
        
        # Get indices of non-filtered entries
        valid_indices = np.where(mask)[0]
        human_idx = valid_indices // n_robot
        robot_idx = valid_indices % n_robot
    else:
        # All combinations
        human_idx = np.repeat(np.arange(n_human), n_robot)
        robot_idx = np.tile(np.arange(n_robot), n_human)
    
    n_combinations = len(joint_probs_flat)
    
    # Build full profiles array
    full_profiles = np.empty((n_combinations, num_agents), dtype=np.int64)
    full_profiles[:, human_agent_indices] = human_profiles[human_idx]
    full_profiles[:, robot_agent_indices] = robot_profiles[robot_idx]
    
    return joint_probs_flat, full_profiles


def combine_profile_distributions_to_indices(
    human_probs: npt.NDArray[np.floating[Any]],
    human_profiles: npt.NDArray[np.int64],
    robot_probs: npt.NDArray[np.floating[Any]],
    robot_profiles: npt.NDArray[np.int64],
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    action_powers: npt.NDArray[np.int64],
    num_agents: Optional[int] = None,
    min_prob: float = 0.0
) -> Tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.int64]]:
    """
    Combine profile distributions and directly compute action profile indices.
    
    This is an optimized version that skips building the full profiles array
    and directly computes the transition indices needed for value lookups.
    
    Args:
        human_probs: Array of shape (N_h,) with probabilities for each human profile.
        human_profiles: Array of shape (N_h, num_humans) with human action profiles.
        robot_probs: Array of shape (N_r,) with probabilities for each robot profile.
        robot_profiles: Array of shape (N_r, num_robots) with robot action profiles.
        human_agent_indices: List of agent indices that are humans.
        robot_agent_indices: List of agent indices that are robots.
        action_powers: Precomputed array [1, num_actions, num_actions^2, ...] for
                      converting profiles to flat indices.
        num_agents: Total number of agents. If None, computed from indices.
        min_prob: Minimum probability threshold.
    
    Returns:
        Tuple of:
        - joint_probs: Array of shape (M,) with joint probabilities.
        - profile_indices: Array of shape (M,) with flat action profile indices,
          computed as sum(profile[i] * action_powers[i]).
    
    Example:
        >>> # Used in backward induction to directly index transitions
        >>> probs, indices = combine_profile_distributions_to_indices(
        ...     human_probs, human_profiles, robot_probs, robot_profiles,
        ...     human_agent_indices, robot_agent_indices, action_powers)
        >>> for prob, idx in zip(probs, indices):
        ...     _, next_probs, next_states = transitions[state_index][idx]
        ...     # ... compute expected values
    """
    if num_agents is None:
        num_agents = len(human_agent_indices) + len(robot_agent_indices)
    
    n_human = len(human_probs)
    n_robot = len(robot_probs)
    
    # Compute outer product of probabilities
    joint_probs_2d = np.outer(human_probs, robot_probs)
    joint_probs_flat = joint_probs_2d.ravel()
    
    # Precompute contribution of each human profile to the final index
    # human_contrib[i] = sum(human_profiles[i, j] * action_powers[human_agent_indices[j]])
    human_powers = action_powers[human_agent_indices]
    human_contrib = human_profiles @ human_powers  # shape (N_h,)
    
    # Precompute contribution of each robot profile
    robot_powers = action_powers[robot_agent_indices]
    robot_contrib = robot_profiles @ robot_powers  # shape (N_r,)
    
    # Filter by minimum probability if requested
    if min_prob > 0.0:
        mask = joint_probs_flat > min_prob
        joint_probs_flat = joint_probs_flat[mask]
        
        valid_indices = np.where(mask)[0]
        human_idx = valid_indices // n_robot
        robot_idx = valid_indices % n_robot
    else:
        human_idx = np.repeat(np.arange(n_human), n_robot)
        robot_idx = np.tile(np.arange(n_robot), n_human)
    
    # Compute profile indices by adding contributions
    profile_indices = human_contrib[human_idx] + robot_contrib[robot_idx]
    
    return joint_probs_flat, profile_indices


def compute_dependency_levels_general(successors: List[List[int]]) -> List[List[int]]:
    """Compute dependency levels using general topological approach."""
    levels: List[List[int]] = []
    remaining_states = set(range(len(successors)))
    
    while remaining_states:
        # Find states with no dependencies to remaining states (terminal in remaining subgraph)
        current_level: List[int] = []
        for state_idx in remaining_states:
            if not any(succ in remaining_states for succ in successors[state_idx]):
                current_level.append(state_idx)
        
        if not current_level:
            raise ValueError("Circular dependency detected in state graph")
        
        levels.append(current_level)
        remaining_states -= set(current_level)
    
    return levels


def compute_dependency_levels_fast(
    states: List[State], 
    level_fct: Callable[[State], int]
) -> List[List[int]]:
    """Compute dependency levels using level function for faster computation."""
    # Compute level values for all states
    level_values: List[int] = [level_fct(state) for state in states]
    
    # Group states by level value (descending order for backward induction)
    level_groups: Dict[int, List[int]] = defaultdict(list)
    for state_idx, level_val in enumerate(level_values):
        level_groups[level_val].append(state_idx)
    
    # Sort by level value (highest first for backward induction)
    sorted_levels = sorted(level_groups.keys(), reverse=True)
    return [level_groups[level_val] for level_val in sorted_levels]


def split_into_batches(items: List[int], num_batches: Optional[int]) -> List[List[int]]:
    """Split a list into approximately equal batches."""
    if num_batches is None or num_batches <= 1:
        return [items]
    
    batch_size = max(1, len(items) // num_batches)
    batches: List[List[int]] = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if batch:
            batches.append(batch)
    return batches
