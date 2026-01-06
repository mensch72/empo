"""
Phase 1: Human Policy Prior Computation via Backward Induction.

This module implements backward induction on the state DAG to compute
goal-conditioned Boltzmann policies for human agents.

Main function:
    compute_human_policy_prior: Compute tabular human policy prior via backward induction.

The algorithm works by:
1. Building the DAG of reachable states and transitions
2. Processing states in reverse topological order (from terminal to initial)

Key features:
- Supports parallel computation for large state spaces
- Returns both the policy (prior) and optionally value functions

Parameters:
    beta_h: Human inverse temperature (β). Higher = more deterministic, inf = argmax.
    gamma_h: Human discount factor (γ).
"""

import numpy as np
import numpy.typing as npt
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional, Callable, List, Tuple, Dict, Any, Union, overload, Literal, TypeAlias

import cloudpickle
from tqdm import tqdm

from empo.memory_monitor import MemoryMonitor
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_model import WorldModel
from empo.shared_dag import (
    init_shared_dag, get_shared_dag, attach_shared_dag, cleanup_shared_dag
)

from .helpers import (
    State, TransitionData,
    default_believed_others_policy,
    compute_dependency_levels_general,
    compute_dependency_levels_fast,
    split_into_batches,
)

# Type aliases
VhValues = List[List[Dict[PossibleGoal, float]]]  # Indexed as Vh_values[state_index][agent_index][goal]
HumanPolicyDict = Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]]  # state -> agent -> goal -> probs

DEBUG = False  # Set to True for verbose debugging output
PROFILE_PARALLEL = os.environ.get('PROFILE_PARALLEL', '').lower() in ('1', 'true', 'yes')

# Module-level globals for shared memory in forked processes
# These are set before spawning workers and inherited copy-on-write
_shared_states: Optional[List[State]] = None
_shared_transitions: Optional[List[List[TransitionData]]] = None
_shared_Vh_values: Optional[VhValues] = None
_shared_params: Optional[Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], List[int], List[List[int]], float, float]] = None
_shared_believed_others_policy_pickle: Optional[bytes] = None  # cloudpickle'd believed_others_policy function


def _hpp_process_single_state(
    state_index: int,
    state: State,
    states: List[State],
    transitions: List[List[TransitionData]],
    Vh_values: VhValues,
    human_agent_indices: List[int],
    possible_goal_generator: PossibleGoalGenerator,
    num_actions: int,
    action_powers: npt.NDArray[np.int64],
    believed_others_policy: Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]],
    robot_agent_indices: List[int],
    robot_action_profiles: List[List[int]],
    beta_h: float,
    gamma_h: float,
) -> Tuple[Dict[int, Dict[PossibleGoal, float]], Optional[Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]]]:
    """Process a single state, returning (v_results, p_results).
    
    Unified implementation for sequential, parallel batch, and inline fallback.
    Handles both terminal and non-terminal states correctly.
    
    Args:
        state_index: Index of the state in the states list
        state: The state to process
        transitions: Full transitions list (indexed by state_index)
        Vh_values: Value function (reads from successors, may write to state_index)
        human_agent_indices: List of agent indices to compute for
        possible_goal_generator: Generator for possible goals
        num_actions: Number of actions available
        action_powers: Precomputed powers for action profile indexing
        believed_others_policy: Function for beliefs about other agents
        beta_h: Inverse temperature (can be float('inf') for argmax)
        gamma_h: Discount factor
    
    Returns:
        Tuple of:
        - v_results: Dict[agent_index, Dict[goal, float]] - V-values for this state
        - p_results: Dict[agent_index, Dict[goal, ndarray]] - policies (None for terminal states)
    """
    actions = range(num_actions)
    v_results: Dict[int, Dict[PossibleGoal, float]] = {}
    p_results: Optional[Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]] = None
    
    is_terminal = not transitions[state_index]
    
    if is_terminal:
        # Terminal state: V_h^m = 0 for all goals
        if DEBUG:
            print(f"  Terminal state {state_index}")
        return v_results, None

    # Non-terminal state: compute both V-values and policies
    p_results = {}
    for agent_index in human_agent_indices:
        v_results[agent_index] = {}
        p_results[agent_index] = {}
        
        for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
            if possible_goal.is_achieved(state):
                # Goal achieved already in this state: remain still, V=0
                v_results[agent_index][possible_goal] = 0
                ps = np.zeros(num_actions)
                ps[0] = 1.0  # Assume action 0 is 'stay still'
                p_results[agent_index][possible_goal] = ps
                if DEBUG:
                    print(f"    Goal achieved in state, agent {agent_index}: V = 0, remain still policy")
            else:
                # Compute Q values as expected future V values
                expected_Vs: npt.NDArray[np.floating[Any]] = np.zeros(num_actions)
                for action in actions:
                    v_accum: float = 0.0
                    for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                        worst_expectation = float('inf')
                        action_profile[agent_index] = action
                        for robot_action_profile in robot_action_profiles:
                            action_profile[robot_agent_indices] = robot_action_profile
                            action_profile_index = int(np.dot(action_profile, action_powers))
                            _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]

                            attainment_values_array: npt.NDArray[np.floating[Any]] = np.array([
                                possible_goal.is_achieved(states[next_state_index]) 
                                for next_state_index in next_state_indices
                            ])

                            # Get V values from successors (use .get for parallel safety)
                            v_values_array: npt.NDArray[np.floating[Any]] = np.array([
                                Vh_values[next_state_indices[i]][agent_index].get(possible_goal, 0)
                                for i in range(len(next_state_indices))
                            ])
                            continuation_values_array = attainment_values_array + (1-attainment_values_array) * v_values_array
                            expectation = float(np.dot(next_state_probabilities, continuation_values_array))
                            if expectation < worst_expectation:
                                worst_expectation = expectation
                        v_accum += action_profile_prob * worst_expectation
                    expected_Vs[action] = v_accum
                
                q = gamma_h * expected_Vs
                
                # Boltzmann policy (numerically stable softmax)
                if beta_h == float('inf'):
                    # Infinite beta: deterministic argmax policy
                    max_q = np.max(q)
                    p = np.zeros_like(q)
                    max_indices = np.where(q == max_q)[0]
                    p[max_indices] = 1.0 / len(max_indices)  # Uniform over max actions
                else:
                    scaled_q = beta_h * (q - np.max(q))  # Subtract max for numerical stability
                    p = np.exp(scaled_q)
                    p /= np.sum(p)
                
                v_results[agent_index][possible_goal] = float(np.sum(p * q))
                p_results[agent_index][possible_goal] = p
                
                if DEBUG:
                    print(f"    Agent {agent_index}, goal {possible_goal}: V = {v_results[agent_index][possible_goal]:.4f}")
    
    return v_results, p_results


def _hpp_compute_sequential(
    states: List[State], 
    Vh_values: VhValues,  # result is inserted into this!
    system2_policies: HumanPolicyDict,  # result is inserted into this! 
    transitions: List[List[Tuple[Tuple[int, ...], List[float], List[int]]]],
    human_agent_indices: List[int], 
    possible_goal_generator: PossibleGoalGenerator,
    num_agents: int, 
    num_actions: int, 
    action_powers: npt.NDArray[np.int64],
    believed_others_policy: Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]], 
    robot_agent_indices: List[int],
    robot_action_profiles: List[List[int]],
    beta_h: float, 
    gamma_h: float,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    memory_monitor: Optional[MemoryMonitor] = None
) -> None:
    """Sequential backward induction algorithm.
    
    Processes states in reverse topological order using the unified
    _process_single_state helper.
    """
    total_states = len(states)
    
    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        # Check memory periodically (using step = states processed)
        states_processed = total_states - state_index
        if memory_monitor is not None:
            memory_monitor.check(states_processed)
        # Update progress bar
        if progress_callback is not None:
            progress_callback(states_processed, total_states)
        if DEBUG:
            print(f"Processing state {state_index}")
        
        state = states[state_index]
        
        # Use unified helper
        v_results, p_results = _hpp_process_single_state(
            state_index, state, states, transitions, Vh_values,
            human_agent_indices, possible_goal_generator,
            num_actions, action_powers, believed_others_policy,
            robot_agent_indices, robot_action_profiles,
            beta_h, gamma_h
        )
        
        # Store results back into Vh_values and system2_policies
        for agent_index, agent_v in v_results.items():
            Vh_values[state_index][agent_index].update(agent_v)
        
        if p_results is not None:
            system2_policies[state] = p_results


def _hpp_init_shared_data(
    states: List[State], 
    transitions: List[List[TransitionData]], 
    Vh_values: VhValues, 
    params: Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float],
    believed_others_policy_pickle: Optional[bytes] = None,
    use_shared_memory: bool = False
) -> None:
    """Initialize shared data for worker processes.
    
    Args:
        states: List of states (will be stored in shared memory if use_shared_memory=True)
        transitions: List of transitions (will be stored in shared memory if use_shared_memory=True)
        Vh_values: Value function (always passed via globals, updated by workers)
        params: Parameters tuple
        believed_others_policy_pickle: Pickled policy function
        use_shared_memory: If True, store states and transitions in shared memory
    """
    global _shared_states, _shared_transitions, _shared_Vh_values, _shared_params, _shared_believed_others_policy_pickle
    
    if use_shared_memory:
        # Store DAG in shared memory to avoid copy-on-write
        init_shared_dag(states, transitions)
        _shared_states = None  # Will be loaded from shared memory in workers
        _shared_transitions = None
    else:
        _shared_states = states
        _shared_transitions = transitions
    
    _shared_Vh_values = Vh_values
    _shared_params = params
    _shared_believed_others_policy_pickle = believed_others_policy_pickle


def _hpp_process_state_batch(
    state_indices: List[int]
) -> Tuple[Dict[int, Dict[int, Dict[PossibleGoal, float]]], 
           Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]], 
           float]:
    """Process a batch of states that can be computed in parallel.
    
    Uses module-level shared data (inherited via fork) to avoid copying.
    Returns both V-values and policies for non-terminal states, plus timing.
    """
    batch_start = time.perf_counter()
    
    # Access shared data - these are guaranteed to be set when called from parallel context
    assert _shared_Vh_values is not None
    assert _shared_params is not None
    
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
    (human_agent_indices, possible_goal_generator, num_agents, num_actions, 
     action_powers, robot_agent_indices, robot_action_profiles, beta_h, gamma_h) = _shared_params
    
    v_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    p_results: Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]] = {}
    
    # Deserialize believed_others_policy if custom one was provided via cloudpickle
    if _shared_believed_others_policy_pickle is not None:
        believed_others_policy = cloudpickle.loads(_shared_believed_others_policy_pickle)
    else:
        # Create default believed others policy function
        believed_others_policy = lambda state, agent_index, action: default_believed_others_policy(
            state, agent_index, action, num_agents, num_actions, robot_agent_indices)
    
    for state_index in state_indices:
        state = states[state_index]
        
        # Use unified helper
        state_v_results, state_p_results = _hpp_process_single_state(
            state_index, state, states, transitions, Vh_values,
            human_agent_indices, possible_goal_generator,
            num_actions, action_powers, believed_others_policy,
            robot_agent_indices, robot_action_profiles,
            beta_h, gamma_h
        )
        
        v_results[state_index] = state_v_results
        if state_p_results is not None:
            p_results[state] = state_p_results
    
    batch_time = time.perf_counter() - batch_start
    return v_results, p_results, batch_time


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0
) -> TabularHumanPolicyPrior: ...


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *, 
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[True],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0
) -> Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]]]: ...


def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: Optional[PossibleGoalGenerator] = None, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, npt.NDArray[np.int64]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False,
    min_free_memory_fraction: float = 0.1,
    memory_check_interval: int = 100,
    memory_pause_duration: float = 60.0
) -> Union[TabularHumanPolicyPrior, Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]]]]:
    """
    Compute human policy prior via backward induction on the state DAG.
    
    This function builds the complete state DAG of the world model and computes
    goal-conditioned Boltzmann policies for all human agents in all non-terminal
    states. The result is a TabularHumanPolicyPrior that can be used to query
    action distributions.
    
    Algorithm overview:
        1. Build the DAG of reachable states using world_model.get_dag()
        2. Compute dependency levels for topological ordering
        3. Process states in reverse topological order:
           - Terminal states: V(s, g) = is_achieved(g, s)
           - Non-terminal states: 
             * Q(s, a, g) = γ * E[V(s', g)] under believed_others_policy
             * π(a|s,g) = softmax(β * Q(s, *, g))
             * V(s, g) = Σ_a π(a|s,g) * Q(s, a, g)
    
    Args:
        world_model: A WorldModel (or MultiGridEnv) with get_state(), set_state(),
                    and transition_probabilities() methods.
        human_agent_indices: List of agent indices to compute policies for.
                            Other agents are modeled by believed_others_policy.
        possible_goal_generator: Generator that yields (goal, weight) pairs for
                                each state and agent. See PossibleGoalGenerator.
                                If None, uses world_model.possible_goal_generator
                                (which can be set via config file 'possible_goals' key).
        believed_others_policy: Function(state, agent_index, action) -> List[(prob, action_profile)]
                               specifying beliefs about other agents' actions.
                               action_profile must be a numpy array of int64.
                               If None, uses uniform distribution over all action profiles.
        beta_h: Inverse temperature for Boltzmann policy. Higher = more deterministic.
                Use float('inf') for pure argmax (greedy) policy.
        gamma_h: Discount factor for future rewards.
        parallel: If True, use multiprocessing for parallel computation.
                 Requires 'fork' context (works on Linux, may not work on macOS/Windows).
        num_workers: Number of parallel workers. If None, uses mp.cpu_count().
        level_fct: Optional function(state) -> int that returns the "level" of a state
                  for fast dependency computation. States at higher levels are processed
                  first. If None, uses general topological sort (slower for large DAGs).
        return_Vh: If True, also return the computed value function.
        progress_callback: Optional callback(done, total) for progress updates.
        quiet: If True, suppress progress output.
        min_free_memory_fraction: Minimum free memory as fraction of total (0.0-1.0).
            When free memory falls below this threshold, computation pauses for
            memory_pause_duration seconds, then checks again. If still low, raises
            KeyboardInterrupt for graceful shutdown. Set to 0.0 to disable (default).
        memory_check_interval: How often to check memory, in states processed.
        memory_pause_duration: How long to pause (seconds) when memory is low.
    
    Returns:
        TabularHumanPolicyPrior: Policy prior that can be called as prior(state, agent, goal).
        
        If return_Vh=True, returns tuple (policy_prior, Vh_values_dict) where
        Vh_values_dict maps state -> agent_idx -> goal -> float.
    
    Performance notes:
        - State space must be finite and acyclic (DAG structure)
        - Time complexity: O(|S| * |A|^n * |G|) where |S|=states, |A|=actions,
          n=agents, |G|=goals per state
        - Memory: O(|S| * n * |G|) for storing V-values and policies
        - Parallel mode provides ~linear speedup for large state spaces
    
    Example:
        >>> env = SmallOneOrTwoChambersMapEnv()
        >>> 
        >>> class ReachGoal(PossibleGoalGenerator):
        ...     def generate(self, state, agent_idx):
        ...         yield (MyGoal(env, target=(3, 7)), 1.0)
        >>> 
        >>> policy = compute_human_policy_prior(
        ...     env, 
        ...     human_agent_indices=[0], 
        ...     possible_goal_generator=ReachGoal(env),
        ...     beta_h=5.0
        ... )
        >>> 
        >>> state = env.get_state()
        >>> action_dist = policy(state, 0, my_goal)  # numpy array of probabilities
    """
    # Use world_model's goal generator if none provided
    if possible_goal_generator is None:
        possible_goal_generator = getattr(world_model, 'possible_goal_generator', None)
        if possible_goal_generator is None:
            raise ValueError(
                "possible_goal_generator must be provided either as an argument "
                "or via world_model.possible_goal_generator (set in config file)"
            )
    
    human_policy_priors: HumanPolicyDict = {}  # these will be a mixture of system-1 and system-2 policies

    # Q_vectors = {}
    system2_policies: HumanPolicyDict = {}  # these will be Boltzmann policies with fixed inverse temperature beta for now
    # V_values will be indexed as V_values[state_index][agent_index][possible_goal]
    # Using nested lists for faster access on first two levels

    num_agents: int = len(world_model.agents)  # type: ignore[attr-defined]
    num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]

    # Compute robot agent indices and all possible robot action profiles
    robot_agent_indices: List[int] = [i for i in range(num_agents) if i not in human_agent_indices]
    robot_action_profiles: List[List[int]] = [list(profile) for profile in product(range(num_actions), repeat=len(robot_agent_indices))] if robot_agent_indices else [[]]

    if believed_others_policy is None:
        # Create wrapper for sequential execution (parallel uses default directly)
        believed_others_policy = lambda state, agent_index, action: default_believed_others_policy(
            state, agent_index, action, num_agents, num_actions, robot_agent_indices)
        believed_others_policy_pickle: Optional[bytes] = None  # No need to pickle the default
    else:
        # Serialize custom believed_others_policy using cloudpickle for parallel mode
        # cloudpickle can serialize lambdas, closures, and other functions that
        # standard pickle cannot handle
        believed_others_policy_pickle = cloudpickle.dumps(believed_others_policy)

    # Precompute powers for action profile indexing
    action_powers: npt.NDArray[np.int64] = num_actions ** np.arange(num_agents)

    # first get the dag of the world model:
    states, state_to_idx, successors, transitions = world_model.get_dag(return_probabilities=True, quiet=quiet)
    
    # Set up default tqdm progress bar if no callback provided
    _pbar: Optional[tqdm[int]] = None
    if progress_callback is None and not quiet:
        _pbar = tqdm(total=len(states), desc="Backward induction", unit="states")
        def progress_callback(done: int, total: int) -> None:
            if _pbar is not None:
                _pbar.n = done
                _pbar.refresh()
    
    # Initialize V_values as nested lists for faster access
    Vh_values: VhValues = [[{} for _ in range(num_agents)] for _ in range(len(states))]
    
    if parallel and len(states) > 1:
        # Parallel execution using shared memory via fork
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        if not quiet:
            print(f"Using parallel execution with {num_workers} workers")
        
        # Compute dependency levels
        dependency_levels: List[List[int]]
        if level_fct is not None:
            if not quiet:
                print("Using fast level computation with provided level function")
            dependency_levels = compute_dependency_levels_fast(states, level_fct)
        else:
            if not quiet:
                print("Using general level computation")
            dependency_levels = compute_dependency_levels_general(successors)
        
        if not quiet:
            print(f"Computed {len(dependency_levels)} dependency levels")
        
        # Initialize shared data for worker processes
        # On Linux (fork), workers inherit these as copy-on-write
        params: Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], List[int], List[List[int]], float, float] = (
            human_agent_indices, possible_goal_generator, num_agents, num_actions,
            action_powers, robot_agent_indices, robot_action_profiles, beta_h, gamma_h
        )
        
        # Use 'fork' context explicitly to ensure shared memory works
        ctx = mp.get_context('fork')
        
        # Initialize DAG in shared memory to avoid copy-on-write overhead
        # This is done once before processing levels
        if not quiet:
            print("Storing DAG in shared memory...")
        init_shared_dag(states, transitions)
        
        # Profiling counters
        if PROFILE_PARALLEL:
            prof_states_parallel: int = 0
            prof_states_sequential: int = 0
            prof_batches: int = 0
            prof_fork_time = 0.0
            prof_submit_time = 0.0
            prof_wait_time = 0.0
            prof_merge_v_time = 0.0
            prof_merge_p_time = 0.0
            prof_seq_in_par_time = 0.0
            prof_total_parallel_time = 0.0
            prof_batch_times: List[float] = []  # Worker timing for each batch
        
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
            
            if len(level) <= num_workers:
                # Few states - process sequentially to avoid overhead
                if PROFILE_PARALLEL:
                    _t0 = time.perf_counter()
                    prof_states_sequential += len(level)
                for state_index in level:
                    state = states[state_index]
                    
                    # Use unified helper
                    v_results, p_results = _hpp_process_single_state(
                        state_index, state, states, transitions, Vh_values,
                        human_agent_indices, possible_goal_generator,
                        num_actions, action_powers, believed_others_policy,
                        robot_agent_indices, robot_action_profiles,
                        beta_h, gamma_h
                    )
                    
                    # Store results
                    for agent_index, agent_v in v_results.items():
                        Vh_values[state_index][agent_index].update(agent_v)
                    
                    if p_results is not None:
                        system2_policies[state] = p_results
                
                if PROFILE_PARALLEL:
                    prof_seq_in_par_time += time.perf_counter() - _t0
            else:
                # Many states - parallelize
                if PROFILE_PARALLEL:
                    _level_t0 = time.perf_counter()
                    prof_states_parallel += len(level)
                
                # Re-initialize shared data so new workers see updated Vh_values from previous levels
                # Also pass the cloudpickle'd believed_others_policy for custom policy support
                # DAG (states, transitions) is already in shared memory, only update Vh_values
                _hpp_init_shared_data(states, transitions, Vh_values, params, believed_others_policy_pickle, use_shared_memory=True)
                
                # Only pass state indices - workers access shared data via globals
                batches = split_into_batches(level, num_workers)
                if PROFILE_PARALLEL:
                    prof_batches += len(batches)
                
                # Create executor per level to ensure workers fork with current Vh_values
                if PROFILE_PARALLEL:
                    _fork_t0 = time.perf_counter()
                with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
                    if PROFILE_PARALLEL:
                        prof_fork_time += time.perf_counter() - _fork_t0
                    
                    # Submit all batches (just the indices, not the data!)
                    if PROFILE_PARALLEL:
                        _submit_t0 = time.perf_counter()
                    futures = [executor.submit(_hpp_process_state_batch, batch) 
                               for batch in batches if batch]
                    if PROFILE_PARALLEL:
                        prof_submit_time += time.perf_counter() - _submit_t0
                    
                    # Collect results and merge back
                    for future in as_completed(futures):
                        if PROFILE_PARALLEL:
                            _wait_t0 = time.perf_counter()
                        v_results, p_results, batch_time = future.result()
                        if PROFILE_PARALLEL:
                            prof_wait_time += time.perf_counter() - _wait_t0
                            prof_batch_times.append(batch_time)
                        
                        # Merge Vh-values back into shared Vh_values
                        if PROFILE_PARALLEL:
                            _merge_v_t0 = time.perf_counter()
                        for state_idx, state_results in v_results.items():
                            for agent_idx, agent_results in state_results.items():
                                Vh_values[state_idx][agent_idx].update(agent_results)
                        if PROFILE_PARALLEL:
                            prof_merge_v_time += time.perf_counter() - _merge_v_t0
                        
                        # Merge policies back
                        if PROFILE_PARALLEL:
                            _merge_p_t0 = time.perf_counter()
                        for state, state_policies in p_results.items():
                            if state not in system2_policies:
                                system2_policies[state] = {}
                            for agent_idx, agent_policies in state_policies.items():
                                if agent_idx not in system2_policies[state]:
                                    system2_policies[state][agent_idx] = {}
                                system2_policies[state][agent_idx].update(agent_policies)
                        if PROFILE_PARALLEL:
                            prof_merge_p_time += time.perf_counter() - _merge_p_t0
                
                if PROFILE_PARALLEL:
                    prof_total_parallel_time += time.perf_counter() - _level_t0
            
            # Report progress after each level
            if progress_callback:
                states_processed = sum(len(lvl) for lvl in dependency_levels[:level_idx + 1])
                progress_callback(states_processed, len(states))
        
        # Print profiling results
        if PROFILE_PARALLEL:
            overhead = prof_fork_time + prof_submit_time + prof_wait_time + prof_merge_v_time + prof_merge_p_time
            print("\n=== Parallelization Overhead Profile ===")
            print(f"States processed in parallel: {prof_states_parallel}")
            print(f"States processed sequentially: {prof_states_sequential}")
            print(f"Batches submitted: {prof_batches}")
            print(f"\nTime breakdown (parallel levels only):")
            print(f"  Total parallel level time:  {prof_total_parallel_time:.4f}s")
            print(f"  Fork overhead:              {prof_fork_time:.4f}s")
            print(f"  Submit overhead:            {prof_submit_time:.4f}s")
            print(f"  Wait + unpickle (result):   {prof_wait_time:.4f}s")
            print(f"  Merge Vh-values:            {prof_merge_v_time:.4f}s")
            print(f"  Merge policies:             {prof_merge_p_time:.4f}s")
            print(f"  Sequential in parallel mode:{prof_seq_in_par_time:.4f}s")
            print(f"\n  Overhead total:             {overhead:.4f}s")
            if prof_total_parallel_time > 0:
                print(f"  Overhead percentage:        {100*overhead/prof_total_parallel_time:.1f}%")
            
            # Worker load balance analysis
            if prof_batch_times:
                sum_batch = sum(prof_batch_times)
                min_batch = min(prof_batch_times)
                max_batch = max(prof_batch_times)
                mean_batch = sum_batch / len(prof_batch_times)
                print(f"\nWorker batch times ({len(prof_batch_times)} batches):")
                print(f"  Min batch time:             {min_batch:.4f}s")
                print(f"  Max batch time:             {max_batch:.4f}s")
                print(f"  Mean batch time:            {mean_batch:.4f}s")
                print(f"  Sum of all batch times:     {sum_batch:.4f}s")
                if prof_total_parallel_time > 0:
                    theoretical_speedup = sum_batch / prof_total_parallel_time
                    print(f"  Theoretical max speedup:    {theoretical_speedup:.2f}x")
                    print(f"  Load imbalance (max/mean):  {max_batch/mean_batch:.2f}x")
            print("=========================================")
        
        # Clean up shared memory after parallel processing
        cleanup_shared_dag()
    
    else:
        # Sequential execution (original algorithm)
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
        _hpp_compute_sequential(states, Vh_values, system2_policies, transitions,
                         human_agent_indices, possible_goal_generator,
                         num_agents, num_actions, action_powers,
                         believed_others_policy, robot_agent_indices, robot_action_profiles,
                         beta_h, gamma_h,
                         progress_callback, memory_monitor)
    
    human_policy_priors = system2_policies # TODO: mix with system-1 policies!

    policy_prior = TabularHumanPolicyPrior(
        world_model=world_model, human_agent_indices=human_agent_indices, 
        possible_goal_generator=possible_goal_generator, values=human_policy_priors
    )
    
    if return_Vh:
        # Convert V_values from list-indexed to state-indexed dict
        Vh_values_dict = {}
        for state_idx, state in enumerate(states):
            if any(Vh_values[state_idx][agent_idx] for agent_idx in range(num_agents)):
                Vh_values_dict[state] = {agent_idx: Vh_values[state_idx][agent_idx] 
                                        for agent_idx in human_agent_indices
                                        if Vh_values[state_idx][agent_idx]}
        if _pbar is not None:
            _pbar.close()
        return policy_prior, Vh_values_dict
    
    if _pbar is not None:
        _pbar.close()
    return policy_prior
