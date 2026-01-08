"""
Phase 2: Robot Policy Computation via Backward Induction.

This module implements backward induction on the state DAG to compute
robot policies that maximize human empowerment.

Main function:
    compute_robot_policy: Compute tabular robot policy via backward induction.

The algorithm computes the robot's power-law policy by:
1. Building the DAG of reachable states and transitions
2. Processing states in reverse topological order (from terminal to initial)
3. Computing robot Q-values based on expected future robot values
4. Computing robot policy as power-law distribution over Q-values
5. Computing human expected goal achievement values under robot policy

Attainment Cache:
    Phase 2 automatically reuses the attainment cache computed in Phase 1 if it's
    stored on the world_model (via world_model._attainment_cache). This avoids
    redundant is_achieved() computation between phases.
"""

import numpy as np
import numpy.typing as npt
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Callable, List, Tuple, Dict, Any, Union, overload, Literal

import cloudpickle
from tqdm import tqdm
from scipy.special import logsumexp

from empo.util.memory_monitor import MemoryMonitor
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy
from empo.world_model import WorldModel
from empo.backward_induction.shared_dag import (
    init_shared_dag, get_shared_dag, attach_shared_dag, cleanup_shared_dag
)

from .helpers import (
    State, TransitionData,
    SliceCache, SliceId, SlicedAttainmentCache, make_slice_id,
    compute_dependency_levels_general,
    compute_dependency_levels_fast,
    split_into_batches,
    SlicedList,
    detect_archivable_levels,
    archive_value_slices,
)
from .phase1 import compute_human_policy_prior

# Type aliases
VhValues = List[List[Dict[PossibleGoal, float]]]  # Indexed as Vh_values[state_index][agent_index][goal]
VrValues = npt.NDArray[np.floating[Any]]  # Indexed as Vr_values[state_index]
RobotActionProfile = Tuple[int, ...]
RobotPolicyDict = Dict[State, Dict[RobotActionProfile, float]]  # state -> robot_action_profile -> prob

DEBUG = False  # Set to True for verbose debugging output

# Module-level globals for shared memory in forked processes (Phase 2)
_shared_states: Optional[List[State]] = None
_shared_transitions: Optional[List[List[TransitionData]]] = None
_shared_Vh_values: Optional[VhValues] = None
_shared_Vr_values: Optional[VrValues] = None
_shared_robot_agent_indices: Optional[List[int]] = None
_shared_human_policy_prior_pickle: Optional[bytes] = None
_shared_sliced_cache: Optional[SlicedAttainmentCache] = None
_shared_num_action_profiles: int = 0
_shared_rp_params: Optional[Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float]] = None


def _rp_process_single_state(
    state_index: int,
    state: State,
    states: List[State],
    transitions: List[List[TransitionData]],
    Vh_values: VhValues,
    Vr_values: VrValues,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    robot_action_profiles: List[RobotActionProfile],
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int,
    num_actions: int,
    action_powers: npt.NDArray[np.int64],
    human_policy_prior: TabularHumanPolicyPrior,
    beta_r: float,
    gamma_h: float,
    gamma_r: float,
    zeta: float,
    xi: float,
    eta: float,
    terminal_Vr: float,
    slice_cache: Optional[SliceCache] = None,
) -> Tuple[
    Dict[int, Dict[PossibleGoal, float]],  # vh_results: agent -> goal -> value
    float,  # vr_result
    Optional[Dict[RobotActionProfile, float]]  # robot_policy (None for terminal)
]:
    """Process a single state for Phase 2, returning (vh_results, vr_result, robot_policy).
    
    Unified implementation for sequential, parallel batch, and inline fallback.
    Handles both terminal and non-terminal states correctly.
    
    Args:
        state_index: Index of the state in the states list
        state: The state to process
        states: Full states list (needed for goal achievement checks)
        transitions: Full transitions list (indexed by state_index)
        Vh_values: Human value function (reads from successors)
        Vr_values: Robot value function (reads from successors)
        human_agent_indices: List of human agent indices
        robot_agent_indices: List of robot agent indices
        robot_action_profiles: Precomputed list of robot action profiles
        possible_goal_generator: Generator for possible goals
        num_agents: Total number of agents
        num_actions: Number of actions available
        action_powers: Precomputed powers for action profile indexing
        human_policy_prior: Human policy prior for computing expectations
        beta_r: Robot inverse temperature (power-law parameter)
        gamma_h: Human discount factor
        gamma_r: Robot discount factor
        zeta: Risk-aversion parameter
        xi: Inter-human power-inequality aversion
        eta: Intertemporal power-inequality aversion
        terminal_Vr: Value for terminal states
        slice_cache: Optional SliceCache for this worker's batch (for writing).
            Structure: Dict[state_index, List[Dict[goal, array]]]
    
    Returns:
        Tuple of:
        - vh_results: Dict[agent_index, Dict[goal, float]] - V_h^e values for this state
        - vr_result: float - V_r value for this state
        - robot_policy: Dict[RobotActionProfile, float] or None (None for terminal states)
    """
    if slice_cache is not None:
        this_state_cache = slice_cache[state_index]

    vh_results: Dict[int, Dict[PossibleGoal, float]] = {agent_idx: {} for agent_idx in human_agent_indices}
    action_profile: npt.NDArray[np.int64] = np.zeros(num_agents, dtype=np.int64)
    
    is_terminal = not transitions[state_index]
    
    if is_terminal:
        # Terminal state: V_h^e = 0 for all goals (dict defaults to 0), V_r = terminal_Vr, no robot policy
        if DEBUG:
            print(f"  Terminal state {state_index}")
        return vh_results, terminal_Vr, None
    
    # Non-terminal state: compute Q_r, pi_r, V_h^e, X_h, U_r, V_r
    if DEBUG:
        print(f"  Transient state {state_index}")
    
    # Compute Q_r values for all robot action profiles
    Qr_values = np.zeros(len(robot_action_profiles))
    for robot_action_profile_index, robot_action_profile in enumerate(robot_action_profiles):
        action_profile[robot_agent_indices] = robot_action_profile
        v = 0.0
        for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution(state):
            action_profile[human_agent_indices] = human_action_profile
            action_profile_index = (action_profile @ action_powers).item()
            _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
            v += human_action_profile_prob * np.dot(next_state_probabilities, Vr_values[next_state_indices])
        Qr_values[robot_action_profile_index] = gamma_r * v
    
    # Compute robot policy as power-law policy
    # Use log-space computation for numerical stability:
    # pi_r(a) ∝ (-Q_r(a))^{-beta_r} = exp(-beta_r * log(-Q_r(a)))
    log_neg_Qr = np.log(-Qr_values)  # Q_r values are always negative
    log_powers = -beta_r * log_neg_Qr
    log_normalizer = logsumexp(log_powers)
    ps = np.exp(log_powers - log_normalizer)
    robot_policy = {robot_action_profile: ps[idx] 
                   for idx, robot_action_profile in enumerate(robot_action_profiles)}
    
    # Compute V_h^e, X_h, and U_r values
    powersum = 0.0  # sum over humans of X_h^(-xi)
    for agent_index in human_agent_indices:
        if DEBUG:
            print(f"   Human agent {agent_index}")
            # Check if at least one goal is achieved in this state
            goals_achieved = []
            for pg, _ in possible_goal_generator.generate(state, agent_index):
                achieved = pg.is_achieved(state)
                goals_achieved.append((pg, achieved))
            if not any(a for _, a in goals_achieved):
                print(f"   WARNING: No goal achieved in state {state_index}!")
                for pg, a in goals_achieved:
                    print(f"     {pg}: is_achieved={a}")
        
        xh = 0.0
        some_goal_achieved_with_positive_prob = False
        
        for possible_goal, possible_goal_weight in possible_goal_generator.generate(state, agent_index):
            if DEBUG:
                print(f"    Possible goal: {possible_goal}")
            
            vh = 0.0
            for robot_action_profile_index, robot_action_profile in enumerate(robot_action_profiles):
                action_profile[robot_agent_indices] = robot_action_profile
                v = 0.0
                for human_action_profile_prob, human_action_profile in human_policy_prior.profile_distribution_with_fixed_goal(state, agent_index, possible_goal):
                    action_profile[human_agent_indices] = human_action_profile
                    action_profile_index = (action_profile @ action_powers).item()
                    _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                    
                    # Look up attainment values from Phase 1 cache
                    # The slice_cache is pre-populated with all values for this batch from Phase 1
                    cached = None
                    
                    if slice_cache is not None:
                        cached = this_state_cache[action_profile_index].get(possible_goal)
                    
                    if cached is not None:
                        attainment_values_array = cached
                    else:
                        # Cache miss - compute attainment values
                        # This should rarely happen if Phase 1 populated the cache correctly
                        attainment_values_array = np.fromiter(
                            (possible_goal.is_achieved(states[next_state_index]) 
                             for next_state_index in next_state_indices),
                            dtype=np.int8,
                            count=len(next_state_indices)
                        )
                    
                    if np.dot(next_state_probabilities, attainment_values_array) > 0.0:
                        some_goal_achieved_with_positive_prob = True
                    
                    vhe_values_array = np.fromiter(
                        (Vh_values[next_state_index][agent_index].get(possible_goal, 0)
                         for next_state_index in next_state_indices),
                        dtype=np.float64,
                        count=len(next_state_indices)
                    )
                    # Use np.where to avoid intermediate array allocation
                    v += human_action_profile_prob * np.dot(
                        next_state_probabilities,
                        np.where(attainment_values_array, 1.0, gamma_h * vhe_values_array)
                    )
                vh += ps[robot_action_profile_index] * v
            
            vh_results[agent_index][possible_goal] = vh
            xh += possible_goal_weight * vh**zeta
            
            if DEBUG:
                print(f"      ...Vh = {vh:.4f}")
        
        assert some_goal_achieved_with_positive_prob, \
            f"No goal achievable with positive probability for agent {agent_index} in state {state_index}!"
        
        if xh == 0:
            # xh is zero means no goal has positive expected achievement value
            raise ValueError(
                f"xh=0 for agent {agent_index} in state {state_index}: "
                f"no goal is reachable! State: {state}"
            )
        
        if DEBUG:
            print(f"   ...Xh = {xh:.4f}")
        
        powersum += xh**(-xi)
    
    y = powersum / len(human_agent_indices)  # average over humans
    ur = -(y**eta)
    vr = ur + float(np.dot(ps, Qr_values))
    
    if DEBUG:
        print(f"  ...Ur = {ur:.4f}, Vr = {vr:.4f}")
    
    return vh_results, vr, robot_policy


def _rp_compute_sequential(
    states: List[State], 
    Vh_values: VhValues,  # result is inserted into this!
    Vr_values: VrValues,  # result is inserted into this!
    robot_policy: RobotPolicyDict,  # result is inserted into this! 
    transitions: List[List[Tuple[Tuple[int, ...], List[float], List[int]]]],
    human_agent_indices: List[int], 
    robot_agent_indices: List[int], # the AI coordinates all robots 
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int, 
    num_actions: int, 
    action_powers: npt.NDArray[np.int64],
    human_policy_prior: TabularHumanPolicyPrior, 
    beta_r: float, # softmax parameter for robots' power-law softmax policies
    gamma_h: float, # humans' discount factor
    gamma_r: float, # robots' discount factor
    zeta: float, # robots' risk-aversion
    xi: float, # robots' inter-human power-inequality aversion
    eta: float, # robots' additional intertemporal power-inequality aversion
    terminal_Vr: float = -1e-10,  # must be strictly negative !
    progress_callback: Optional[Callable[[int, int], None]] = None,
    memory_monitor: Optional[MemoryMonitor] = None,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    level_fct: Optional[Callable[[State], int]] = None,
    return_values: bool = False,
    archive_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """Sequential Phase 2 backward induction algorithm.
    
    Processes states in reverse topological order using the unified
    _process_single_state_phase2 helper.
    """
    # Generate all possible robot action profiles (cartesian product of actions for each robot)
    robot_action_profiles: List[RobotActionProfile] = [
        tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
    ]
    
    total_states = len(states)
    
    # In sequential mode, we use a single slice containing all states
    # Retrieve the slice cache populated by Phase 1 (not create a new empty one!)
    slice_cache: Optional[SliceCache] = None
    if sliced_cache is not None:
        all_state_indices = list(range(len(states)))
        slice_id = make_slice_id(all_state_indices)
        slice_cache = sliced_cache.get_slice(slice_id)
    
    # Compute max_successor_levels for archival if level_fct and archive_dir provided
    max_successor_levels: Optional[Dict[int, int]] = None
    archived_levels: Set[int] = set()  # Track already-archived levels
    if level_fct is not None and archive_dir is not None:
        from .helpers import compute_dependency_levels_fast, detect_archivable_levels, archive_value_slices
        from pathlib import Path
        # Build successors list from transitions
        successors = []
        for state_transitions in transitions:
            succ_set = set()
            for action_profile, probs, succ_indices in state_transitions:
                succ_set.update(succ_indices)
            successors.append(list(succ_set))
        _, max_successor_levels, _ = compute_dependency_levels_fast(states, level_fct, successors)
    
    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        state = states[state_index]
        
        # Use unified helper
        vh_results, vr_result, p_result = _rp_process_single_state(
            state_index, state, states, transitions, Vh_values, Vr_values,
            human_agent_indices, robot_agent_indices, robot_action_profiles,
            possible_goal_generator, num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            slice_cache=slice_cache,
        )
        
        # Store results
        for agent_index, agent_vh in vh_results.items():
            Vh_values[state_index][agent_index].update(agent_vh)
        
        Vr_values[state_index] = vr_result
        
        if p_result is not None:
            robot_policy[state] = p_result
        
        # Check memory and update progress AFTER processing each state
        states_processed = total_states - state_index
        if memory_monitor is not None:
            memory_monitor.check(states_processed)
        if progress_callback is not None:
            progress_callback(states_processed, total_states)
        
        # Archive if we just completed a level and archive_dir is set
        if archive_dir is not None and level_fct is not None and max_successor_levels is not None:
            current_state_level = level_fct(state)
            archivable = detect_archivable_levels(current_state_level, max_successor_levels, quiet=quiet)
            # Only archive NEW levels (not already archived)
            new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
            if new_archivable:
                # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                archive_value_slices(
                    Vh_values, states, level_fct, new_archivable,
                    filepath=Path(archive_dir) / "vhe_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                # Archive vr_values (robot values) - convert to list structure for archival
                vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                archive_value_slices(
                    vr_list, states, level_fct, new_archivable,
                    filepath=Path(archive_dir) / "vr_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                archived_levels.update(new_archivable)
    
    # Final archival check: archive any remaining levels after loop completes
    if archive_dir is not None and level_fct is not None:
        # Check if there are any levels we haven't archived yet
        # At this point, all states have been processed, so all levels should be archivable
        all_levels = sorted(set(level_fct(s) for s in states))
        remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
        if remaining_levels:
            # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
            archive_value_slices(
                Vh_values, states, level_fct, remaining_levels,
                filepath=Path(archive_dir) / "vhe_values.pkl",
                return_values=return_values,
                quiet=quiet
            )
            # Archive vr_values (robot values) - convert to list structure for archival
            vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
            archive_value_slices(
                vr_list, states, level_fct, remaining_levels,
                filepath=Path(archive_dir) / "vr_values.pkl",
                return_values=return_values,
                quiet=quiet
            )
            archived_levels.update(remaining_levels)
    
    # Note: slice_cache was retrieved from Phase 1, no need to store it again


def _rp_init_shared_data(
    states: List[State], 
    transitions: List[List[TransitionData]], 
    Vh_values: VhValues, 
    Vr_values: VrValues,
    params: Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float],
    human_policy_prior_pickle: bytes,
    use_shared_memory: bool = False,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    num_action_profiles: int = 0,
) -> None:
    """Initialize shared data for robot policy worker processes.
    
    Args:
        states: List of states (will be stored in shared memory if use_shared_memory=True)
        transitions: List of transitions (will be stored in shared memory if use_shared_memory=True)
        Vh_values: Human value function (always passed via globals)
        Vr_values: Robot value function (always passed via globals)
        params: Parameters tuple
        human_policy_prior_pickle: Pickled human policy prior
        use_shared_memory: If True, states and transitions are already in shared memory
        sliced_cache: Optional SlicedAttainmentCache from Phase 1 for reading
        num_action_profiles: Number of action profiles (needed for slice cache creation)
    """
    global _shared_states, _shared_transitions, _shared_Vh_values, _shared_Vr_values
    global _shared_rp_params, _shared_human_policy_prior_pickle
    global _shared_sliced_cache, _shared_num_action_profiles
    
    if use_shared_memory:
        # DAG is already in shared memory, just store refs as None
        _shared_states = None
        _shared_transitions = None
    else:
        _shared_states = states
        _shared_transitions = transitions
    
    _shared_Vh_values = Vh_values
    _shared_Vr_values = Vr_values
    _shared_rp_params = params
    _shared_human_policy_prior_pickle = human_policy_prior_pickle
    _shared_sliced_cache = sliced_cache
    _shared_num_action_profiles = num_action_profiles


def _rp_process_state_batch(
    state_indices: List[int]
) -> Tuple[Dict[int, Dict[int, Dict[PossibleGoal, float]]], 
           Dict[int, float],
           Dict[State, Dict[RobotActionProfile, float]],
           SliceId,  # slice_id for this batch
           Optional[SliceCache],  # slice cache for this batch (None if not available)
           float]:
    """Process a batch of states for robot policy computation.
    
    Uses module-level shared data (inherited via fork) to avoid copying.
    Returns Vh-values, Vr-values, robot policies, slice_id, slice_cache, plus timing.
    """
    batch_start = time.perf_counter()
    
    # Access shared data - these are guaranteed to be set when called from parallel context
    assert _shared_Vh_values is not None
    assert _shared_Vr_values is not None
    assert _shared_rp_params is not None
    assert _shared_human_policy_prior_pickle is not None
    
    # Try to get states/transitions from shared memory first, fall back to globals
    shared_dag = get_shared_dag()
    if shared_dag is None:
        # Try to attach to shared memory (first call in this worker)
        shared_dag = attach_shared_dag()
    
    if shared_dag is not None:
        states = shared_dag.get_states()
        transitions = shared_dag.get_transitions()
    else:
        # Fall back to module-level globals
        assert _shared_states is not None
        assert _shared_transitions is not None
        states = _shared_states
        transitions = _shared_transitions
    
    assert transitions is not None
    
    Vh_values = _shared_Vh_values
    Vr_values = _shared_Vr_values
    
    (human_agent_indices, robot_agent_indices, possible_goal_generator, 
     num_agents, num_actions, action_powers, beta_r, gamma_h, gamma_r, 
     zeta, xi, eta, terminal_Vr) = _shared_rp_params
    
    # Deserialize human_policy_prior
    human_policy_prior = cloudpickle.loads(_shared_human_policy_prior_pickle)
    # The world_model is excluded from pickling, so we need to set num_actions directly
    # for profile_distribution to work. Use a mock attribute access pattern.
    human_policy_prior._num_actions_override = num_actions
    
    # Generate all possible robot action profiles
    robot_action_profiles: List[RobotActionProfile] = [
        tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
    ]
    
    vh_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    vr_results: Dict[int, float] = {}
    p_results: Dict[State, Dict[RobotActionProfile, float]] = {}
    
    # Retrieve slice cache pre-populated by Phase 1 for this batch
    # Structure: Dict[state_index, List[Dict[goal, array]]]
    slice_id = make_slice_id(state_indices)
    slice_cache: Optional[SliceCache] = None
    if _shared_sliced_cache is not None:
        slice_cache = _shared_sliced_cache.get_slice(slice_id)
    
    for state_index in state_indices:
        state = states[state_index]
        
        # Use unified helper with slice_cache pre-populated by Phase 1
        # The sliced_cache is no longer needed for lookup since slice_cache has all data
        vh_results_state, vr_result, p_result = _rp_process_single_state(
            state_index, state, states, transitions, Vh_values, Vr_values,
            human_agent_indices, robot_agent_indices, robot_action_profiles,
            possible_goal_generator, num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            slice_cache=slice_cache,
        )
        
        vh_results[state_index] = vh_results_state
        vr_results[state_index] = vr_result
        if p_result is not None:
            p_results[state] = p_result
    
    batch_time = time.perf_counter() - batch_start
    return vh_results, vr_results, p_results, slice_id, slice_cache, batch_time


class TabularRobotPolicy(RobotPolicy):
    """
    Tabular (lookup-table) implementation of robot policy.
    
    This implementation stores precomputed robot policy distributions in a dictionary
    structure, indexed by state. The policy maps each state to a distribution over
    robot action profiles (joint actions for all robot agents).
    
    Computed via backward induction on the state DAG in Phase 2.
    
    Attributes:
        world_model: The world model (environment) this policy applies to.
        robot_agent_indices: List of agent indices controlled as robots.
        values: Dict mapping state -> robot_action_profile -> probability.
    """
    
    def __init__(
        self, 
        world_model: WorldModel, 
        robot_agent_indices: List[int], 
        values: RobotPolicyDict
    ):
        """
        Initialize the tabular robot policy.
        
        Args:
            world_model: The world model (environment) this policy applies to.
            robot_agent_indices: List of indices of robot agents.
            values: Precomputed policy lookup table (state -> action_profile -> prob).
        """
        self.world_model = world_model
        self.robot_agent_indices = robot_agent_indices
        self.values = values
        self.num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]
    
    def __call__(self, state) -> Dict[RobotActionProfile, float]:
        """
        Get the robot action profile distribution for a state.
        
        Args:
            state: Current world state.
        
        Returns:
            Dict mapping robot action profiles to probabilities.
        """
        return self.values.get(state, {})
    
    def sample(self, state) -> RobotActionProfile:
        """
        Sample a robot action profile from the policy.
        
        Args:
            state: Current world state.
        
        Returns:
            A tuple of actions, one for each robot agent.
        """
        dist = self(state)
        if not dist:
            # No policy for this state (terminal state?), return random
            return tuple(np.random.randint(0, self.num_actions) for _ in self.robot_agent_indices)
        
        profiles = list(dist.keys())
        probs = np.fromiter((dist[p] for p in profiles), dtype=np.float64, count=len(profiles))
        probs = probs / probs.sum()  # normalize
        idx = np.random.choice(len(profiles), p=probs)
        return profiles[idx]
    
    def reset(self, world_model: WorldModel) -> None:
        """
        Reset the policy at the start of an episode.
        
        Updates the world model reference. For tabular policies, this allows
        the same policy to be used across different instances of the same
        environment type.
        
        Args:
            world_model: The environment/world model for this episode.
        """
        self.world_model = world_model
    
    def get_action(self, state, robot_agent_index: int) -> int:
        """
        Get the action for a specific robot agent.
        
        Samples from the joint policy and returns the action for the specified robot.
        
        Args:
            state: Current world state.
            robot_agent_index: Index of the robot agent.
        
        Returns:
            The action for the specified robot.
        """
        profile = self.sample(state)
        # Find position of robot_agent_index in robot_agent_indices
        pos = self.robot_agent_indices.index(robot_agent_index)
        return profile[pos]


@overload
def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: float = 1.0, 
    gamma_r: float = 1.0,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
) -> TabularRobotPolicy: ...


@overload
def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: float = 1.0, 
    gamma_r: float = 1.0,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: Literal[True],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
) -> Tuple[TabularRobotPolicy, Dict[State, float], Dict[State, Dict[int, Dict[PossibleGoal, float]]]]: ...


def compute_robot_policy(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    robot_agent_indices: List[int],
    possible_goal_generator: Optional[PossibleGoalGenerator] = None,
    human_policy_prior: Optional[TabularHumanPolicyPrior] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: float = 1.0, 
    gamma_r: float = 1.0,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_values: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0,
    sliced_cache: Optional[SlicedAttainmentCache] = None,
    archive_dir: Optional[str] = None,
) -> Union[TabularRobotPolicy, Tuple[TabularRobotPolicy, Dict[State, float], Dict[State, Dict[int, Dict[PossibleGoal, float]]]]]:
    """
    Compute robot policy via backward induction on the state DAG.
    
    This function builds the complete state DAG of the world model and computes
    the robot's power-law policy that aims to maximize human empowerment.
    It simultaneously computes the expected human goal achievement values (V_h^e).
    
    Algorithm overview:
        1. Build the DAG of reachable states using world_model.get_dag()
        2. Compute dependency levels for topological ordering
        3. Process states in reverse topological order:
           - Terminal states: V_h^e(s, g) = 0, V_r(s) = 0
           - Non-terminal states:
             * Q_r(s, a_r) = γ_r * E[V_r(s')] under human_policy_prior
             * π_r(a_r|s) = power-law policy based on Q_r
             * V_h^e(s, g) = E[achievement(s') + (1-achievement(s')) * γ_h * V_h^e(s', g)]
             * X_h(s) = E[V_h^e(s, g)^ζ] (aggregate goal ability)
             * U_r(s) = -(mean(X_h^{-ξ}))^η (intrinsic reward)
             * V_r(s) = U_r(s) + E[Q_r(s, a_r)]
    
    Args:
        world_model: A WorldModel (or MultiGridEnv) with get_state(), set_state(),
                    and transition_probabilities() methods.
        human_agent_indices: List of agent indices representing humans.
        robot_agent_indices: List of agent indices representing robots.
        possible_goal_generator: Generator that yields (goal, weight) pairs for
                                each state and agent. See PossibleGoalGenerator.
                                If None, uses world_model.possible_goal_generator.
        human_policy_prior: Precomputed human policy prior from compute_human_policy_prior().
                           If None, will be computed automatically using the goal generator.
        beta_r: Power-law concentration parameter. Higher = more deterministic.
        gamma_h: Discount factor for human goal achievement values.
        gamma_r: Discount factor for robot values.
        zeta: Risk-aversion parameter for aggregate goal ability.
        xi: Inter-human power-inequality aversion parameter.
        eta: Additional intertemporal power-inequality aversion parameter.
        terminal_Vr: Value for V_r at terminal states. Must be strictly negative
                    to ensure power-law policy is well-defined. Default: -1e-10.
        parallel: If True, use multiprocessing for parallel computation.
                 Requires 'fork' context (works on Linux, may not work on macOS/Windows).
        num_workers: Number of parallel workers. If None, uses mp.cpu_count().
        level_fct: Optional function(state) -> int for fast dependency computation.
        return_values: If True, also return V_r and V_h^e value functions.
        progress_callback: Optional callback(done, total) for progress updates.
        quiet: If True, suppress progress output.
        min_free_memory_fraction: Minimum free memory as fraction of total (0.0-1.0).
            When free memory falls below this threshold, computation pauses for
            memory_pause_duration seconds, then checks again. If still low, raises
            KeyboardInterrupt for graceful shutdown. Set to 0.0 to disable (default).
        memory_check_interval: How often to check memory, in states processed.
        memory_pause_duration: How long to pause (seconds) when memory is low.
        sliced_cache: Optional SlicedAttainmentCache of precomputed goal attainment arrays.
            If not provided, Phase 2 automatically looks for the cache on world_model
            (stored automatically by Phase 1 at world_model._attainment_cache).
            
            **Automatic caching**: Phase 1 now automatically stores its sliced attainment 
            cache on the world_model, so Phase 2 will reuse it without any extra configuration.
            You don't need to pass return_attainment_cache=True to Phase 1 anymore.
            
            The sliced cache structure allows efficient read access without merging overhead.
    
    Returns:
        TabularRobotPolicy: Robot policy that can be called as policy(state).
        
        If return_values=True, returns tuple (robot_policy, Vr_dict, Vh_dict) where:
        - Vr_dict maps state -> float (robot value function)
        - Vh_dict maps state -> agent_idx -> goal -> float (human goal achievement values)
    
    Example:
        >>> # Phase 1 automatically stores attainment cache on world_model
        >>> human_policy = compute_human_policy_prior(env, [0], goal_gen)
        >>>
        >>> # Phase 2 automatically reuses the cache - no extra parameters needed!
        >>> robot_policy = compute_robot_policy(
        ...     env,
        ...     human_agent_indices=[0],
        ...     robot_agent_indices=[1],
        ...     possible_goal_generator=goal_gen,
        ...     human_policy_prior=human_policy,
        ...     beta_r=5.0
        ... )
        >>> 
        >>> state = env.get_state()
        >>> robot_actions = robot_policy.sample(state)  # tuple of actions
    """
    # Use world_model's goal generator if none provided
    if possible_goal_generator is None:
        possible_goal_generator = getattr(world_model, 'possible_goal_generator', None)
        if possible_goal_generator is None:
            raise ValueError(
                "possible_goal_generator must be provided either as an argument "
                "or via world_model.possible_goal_generator (set in config file)"
            )
    
    # Compute human policy prior if not provided
    if human_policy_prior is None:
        human_policy_prior = compute_human_policy_prior(
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            possible_goal_generator=possible_goal_generator,
            parallel=parallel,
            num_workers=num_workers,
            quiet=quiet
        )
    
    robot_policy_values: RobotPolicyDict = {}

    num_agents: int = len(world_model.agents)  # type: ignore[attr-defined]
    num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]

    # Precompute powers for action profile indexing
    action_powers: npt.NDArray[np.int64] = num_actions ** np.arange(num_agents)

    # Serialize human_policy_prior using cloudpickle for parallel mode
    human_policy_prior_pickle = cloudpickle.dumps(human_policy_prior)

    # Get the DAG of the world model
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=quiet)
    
    # Set up default tqdm progress bar if no callback provided
    _pbar: Optional[tqdm[int]] = None
    if progress_callback is None and not quiet:
        _pbar = tqdm(total=len(states), desc="Robot policy backward induction", unit="states")
        def progress_callback(done: int, total: int) -> None:
            if _pbar is not None:
                _pbar.n = done
                _pbar.refresh()
    
    # Initialize value arrays
    Vh_values: VhValues = [[{} for _ in range(num_agents)] for _ in range(len(states))]
    Vr_values: VrValues = np.zeros(len(states))
    
    # Get sliced attainment cache: prioritize explicit parameter, then world_model cache, then create new
    num_action_profiles = num_actions ** num_agents
    if sliced_cache is None:
        # Try to get cache from world_model (automatically stored by Phase 1)
        sliced_cache = getattr(world_model, '_attainment_cache', None)
        if sliced_cache is not None and isinstance(sliced_cache, SlicedAttainmentCache):
            if not quiet:
                print(f"Using sliced attainment cache from world_model ({sliced_cache.num_states()} state entries)")
        else:
            sliced_cache = None  # Wrong type or not set
    if sliced_cache is None:
        # Create empty sliced cache for Phase 2 internal use
        sliced_cache = SlicedAttainmentCache(num_action_profiles)
    
    if parallel and len(states) > 1:
        # Parallel execution using shared memory via fork
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        if not quiet:
            print(f"Using parallel execution with {num_workers} workers")
        
        # Compute dependency levels and max successor levels for archival
        dependency_levels: List[List[int]]
        max_successor_levels: Optional[Dict[int, int]] = None
        level_values_list: Optional[List[int]] = None
        archived_levels: Set[int] = set()  # Track already-archived levels
        if level_fct is not None:
            if not quiet:
                print("Using fast level computation with provided level function")
            # Pass successors for archival max_successor_levels computation
            dependency_levels, max_successor_levels, level_values_list = compute_dependency_levels_fast(
                states, level_fct, successors
            )
        else:
            if not quiet:
                print("Using general level computation")
            dependency_levels = compute_dependency_levels_general(successors)
            max_successor_levels = None
            level_values_list = None
        
        if not quiet:
            print(f"Computed {len(dependency_levels)} dependency levels")
        
        # Initialize shared data for worker processes
        params: Tuple[List[int], List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float, float, float, float, float, float] = (
            human_agent_indices, robot_agent_indices, possible_goal_generator, 
            num_agents, num_actions, action_powers, beta_r, gamma_h, gamma_r, 
            zeta, xi, eta, terminal_Vr
        )
        
        # Use 'fork' context explicitly to ensure shared memory works
        ctx = mp.get_context('fork')
        
        # Initialize shared memory for DAG data to avoid copy-on-write overhead
        if not quiet:
            print("Storing DAG in shared memory...")
        init_shared_dag(states, transitions)
        
        # Create memory monitor if enabled (for parallel mode - check at each level)
        memory_monitor: Optional[MemoryMonitor] = None
        if min_free_memory_fraction > 0.0:
            memory_monitor = MemoryMonitor(
                min_free_fraction=min_free_memory_fraction,
                check_interval=1,  # Check every level in parallel mode
                pause_duration=memory_pause_duration,
                verbose=not quiet,
                enabled=True
            )
        
        # Process each level sequentially, but parallelize within each level
        for level_idx, level in enumerate(dependency_levels):
            # Check memory at the start of each level
            if memory_monitor is not None:
                memory_monitor.check(level_idx)
            
            if DEBUG:
                print(f"Processing level {level_idx} with {len(level)} states")
            
            # Generate all possible robot action profiles (needed for sequential fallback)
            robot_action_profiles: List[RobotActionProfile] = [
                tuple(actions) for actions in product(range(num_actions), repeat=len(robot_agent_indices))
            ]
            
            if len(level) <= num_workers:
                # Few states - process sequentially to avoid overhead
                # Create a slice cache for inline processing
                inline_slice_cache: SliceCache = {
                    state_idx: [{} for _ in range(num_action_profiles)]
                    for state_idx in level
                }
                
                for state_index in level:
                    state = states[state_index]
                    
                    # Use unified helper
                    vh_results, vr_result, p_result = _rp_process_single_state(
                        state_index, state, states, transitions, Vh_values, Vr_values,
                        human_agent_indices, robot_agent_indices, robot_action_profiles,
                        possible_goal_generator, num_agents, num_actions, action_powers,
                        human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
                        slice_cache=inline_slice_cache,
                    )
                    
                    # Store results
                    for agent_index, agent_vh in vh_results.items():
                        Vh_values[state_index][agent_index].update(agent_vh)
                    
                    Vr_values[state_index] = vr_result
                    
                    if p_result is not None:
                        robot_policy_values[state] = p_result
                
                # Store inline slice cache in sliced_cache (states processed sequentially in parallel mode)
                if inline_slice_cache:
                    inline_slice_id = make_slice_id(list(inline_slice_cache.keys()))
                    sliced_cache.store_slice(inline_slice_id, inline_slice_cache)
            else:
                # Many states - parallelize
                # Re-initialize shared data so new workers see updated values from previous levels
                _rp_init_shared_data(states, transitions, Vh_values, Vr_values, params, human_policy_prior_pickle, use_shared_memory=True, sliced_cache=sliced_cache, num_action_profiles=num_action_profiles)
                
                batches = split_into_batches(level, num_workers)
                
                # Create executor per level to ensure workers fork with current values
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                    futures = [executor.submit(_rp_process_state_batch, batch) 
                               for batch in batches if batch]
                    
                    batches_completed = 0
                    for future in as_completed(futures):
                        # Check memory BEFORE collecting result to catch pressure early
                        # Use force=True to bypass interval check since we check per-batch
                        if memory_monitor is not None:
                            memory_monitor.check(batches_completed, force=True)
                        
                        vh_results, vr_results, p_results, slice_id, slice_cache, batch_time = future.result()
                        batches_completed += 1
                        
                        # Merge Vh-values back
                        for state_idx, state_results in vh_results.items():
                            for agent_idx, agent_results in state_results.items():
                                Vh_values[state_idx][agent_idx].update(agent_results)
                        
                        # Merge Vr-values back
                        for state_idx, vr_val in vr_results.items():
                            Vr_values[state_idx] = vr_val
                        
                        # Merge robot policies back
                        robot_policy_values.update(p_results)
                        
                        # Check memory AFTER merging results (this is when memory actually increases)
                        if memory_monitor is not None:
                            memory_monitor.check(batches_completed, force=True)
                        
                        # Note: slice_cache is retrieved from Phase 1, no need to store it again
            
            # Report progress after each level
            if progress_callback:
                states_processed = sum(len(lvl) for lvl in dependency_levels[:level_idx + 1])
                progress_callback(states_processed, len(states))
            
            # Archive completed levels if archive_dir is set
            if archive_dir is not None and max_successor_levels is not None and level_values_list is not None:
                current_level_value = level_values_list[level_idx]
                archivable = detect_archivable_levels(current_level_value, max_successor_levels, quiet=quiet)
                # Only archive NEW levels (not already archived)
                new_archivable = [lvl for lvl in archivable if lvl not in archived_levels]
                if new_archivable:
                    # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                    archive_value_slices(
                        Vh_values, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vhe_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    # Archive vr_values (robot values) - convert to list structure for archival
                    vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                    archive_value_slices(
                        vr_list, states, level_fct, new_archivable,
                        filepath=Path(archive_dir) / "vr_values.pkl",
                        return_values=return_values,
                        quiet=quiet
                    )
                    archived_levels.update(new_archivable)
        
        # Final archival check for parallel mode: archive any remaining levels
        if archive_dir is not None and level_fct is not None:
            all_levels = sorted(set(level_fct(s) for s in states))
            remaining_levels = [lvl for lvl in all_levels if lvl not in archived_levels]
            if remaining_levels:
                # Archive vhe_values (Vh_values in Phase 2 is expected human achievement)
                archive_value_slices(
                    Vh_values, states, level_fct, remaining_levels,
                    filepath=Path(archive_dir) / "vhe_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                # Archive vr_values (robot values) - convert to list structure for archival
                vr_list = [[Vr_values[i]] for i in range(len(Vr_values))]
                archive_value_slices(
                    vr_list, states, level_fct, remaining_levels,
                    filepath=Path(archive_dir) / "vr_values.pkl",
                    return_values=return_values,
                    quiet=quiet
                )
                archived_levels.update(remaining_levels)
        
        # Clean up shared memory after parallel processing
        cleanup_shared_dag()
    
    else:
        # Sequential execution
        # Create memory monitor if enabled
        memory_monitor: Optional[MemoryMonitor] = None
        if min_free_memory_fraction > 0.0:
            memory_monitor = MemoryMonitor(
                min_free_fraction=min_free_memory_fraction,
                check_interval=memory_check_interval,
                pause_duration=memory_pause_duration,
                verbose=not quiet,
                enabled=True
            )
        _rp_compute_sequential(
            states, Vh_values, Vr_values, robot_policy_values, transitions,
            human_agent_indices, robot_agent_indices, possible_goal_generator,
            num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            progress_callback, memory_monitor,
            sliced_cache,
            level_fct, return_values, archive_dir, quiet,
        )

    robot_policy = TabularRobotPolicy(
        world_model=world_model, 
        robot_agent_indices=robot_agent_indices, 
        values=robot_policy_values
    )
    
    if return_values:
        # Convert Vr_values from array to dict
        Vr_dict = {states[idx]: float(Vr_values[idx]) for idx in range(len(states))}
        
        # Convert Vh_values from list-indexed to state-indexed dict
        Vh_dict = {}
        for state_idx, state in enumerate(states):
            if any(Vh_values[state_idx][agent_idx] for agent_idx in human_agent_indices):
                Vh_dict[state] = {agent_idx: Vh_values[state_idx][agent_idx] 
                                 for agent_idx in human_agent_indices
                                 if Vh_values[state_idx][agent_idx]}
        
        if _pbar is not None:
            _pbar.close()
        return robot_policy, Vr_dict, Vh_dict
    
    if _pbar is not None:
        _pbar.close()
    return robot_policy
