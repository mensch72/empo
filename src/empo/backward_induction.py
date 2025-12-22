"""
Backward Induction for Computing Human Policy Priors and Robot Policies.

This module implements backward induction on the state DAG to compute
goal-conditioned policies for human agents and goal-independent robot policies. 

Main functions:
    compute_human_policy_prior: Compute tabular human policy prior via backward induction.
    compute_robot_policy: Compute tabular robot policy via backward induction.

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
    >>> from empo.backward_induction import compute_human_policy_prior
    >>> from empo.possible_goal import PossibleGoalGenerator
    >>> 
    >>> # Define goal generator (implementation-specific)
    >>> goal_generator = MyGoalGenerator(env)
    >>> 
    >>> # Compute policy prior
    >>> policy_prior = compute_human_policy_prior(
    ...     world_model=env,
    ...     human_agent_indices=[0, 1],  # agents 0 and 1 are humans
    ...     possible_goal_generator=goal_generator,
    ...     beta_h=10.0,  # high temperature = nearly optimal
    ...     gamma_h=1.0,
    ...     parallel=True
    ... )
    >>> 
    >>> # Use the policy prior
    >>> action_dist = policy_prior(state, agent_idx=0, goal=my_goal)
"""

import numpy as np
import numpy.typing as npt
import time
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict
from typing import Optional, Callable, List, Tuple, Dict, Any, Union, overload, Literal, TypeAlias

import cloudpickle
from tqdm import tqdm

from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.human_policy_prior import TabularHumanPolicyPrior
from empo.world_model import WorldModel

# Type aliases for complex types used throughout
State: TypeAlias = Any  # State is typically a hashable tuple from WorldModel.get_state()
ActionProfile = List[int]
TransitionData = Tuple[Tuple[int, ...], List[float], List[State]]  # (action_profile, probs, successor_states)
VhValues = List[List[Dict[PossibleGoal, float]]]  # Indexed as Vh_values[state_index][agent_index][goal]
HumanPolicyDict = Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]]  # state -> agent -> goal -> probs

VrValues = List[float]  # Indexed as Vr_values[state_index]
RobotPolicyDict = Dict[State, npt.NDArray[np.floating[Any]]]  # state -> probs

DEBUG = False  # Set to True for verbose debugging output
PROFILE_PARALLEL = os.environ.get('PROFILE_PARALLEL', '').lower() in ('1', 'true', 'yes')

# Module-level globals for shared memory in forked processes
# These are set before spawning workers and inherited copy-on-write
_shared_states: Optional[List[State]] = None
_shared_transitions: Optional[List[List[TransitionData]]] = None
_shared_Vh_values: Optional[VhValues] = None
_shared_believed_others_policy_pickle: Optional[bytes] = None  # cloudpickle'd believed_others_policy function

_shared_Vr_values: Optional[VrValues] = None


######################
### Helper methods ###
######################

def default_believed_others_policy(
    state: State, 
    agent_index: int, 
    action: int, 
    num_agents: int, 
    num_actions: int
) -> List[Tuple[float, List[int]]]:
    """Default believed others policy - uniform distribution."""
    uniform_p = 1 / num_actions**(num_agents - 1)
    # each action profile for the other (!) agents gets the same probability, and the agent's own action is always put to -1 since it will be overwritten in the loop below:
    all_actions = list(range(num_actions))
    return [(uniform_p, list(action_profile)) for action_profile in product(*[
        [-1] if idx == agent_index else all_actions
        for idx in range(num_agents)])]

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


###############################################
### PHASE 1: HUMAN POLICY PRIOR COMPUTATION ###
###############################################


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
    believed_others_policy: Callable[[State, int, int], List[Tuple[float, List[int]]]], 
    beta_h: float, 
    gamma_h: float
) -> None:
    """Original sequential algorithm.
    
    Processes a batch of independent states:
    - Terminal states: V(s, g) = is_achieved(g, s)
    - Non-terminal states: 
        * Q(s, a, g) = γ * E[V(s', g)] under believed_others_policy
        * π(a|s,g) = softmax(β * Q(s, *, g))
        * V(s, g) = Σ_a π(a|s,g) * Q(s, a, g)
    """
    actions = range(num_actions)
    
    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        if DEBUG:
            print(f"Processing state {state_index}")
        state = states[state_index]
        is_terminal = not transitions[state_index]
        
        if is_terminal:
            if DEBUG:
                print(f"  Terminal state")
            # in terminal states, policy and Q values are undefined, only Vh values need computation:
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"  Human agent {agent_index}")
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    v = Vh_values[state_index][agent_index][possible_goal] = possible_goal.is_achieved(state)
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}, Vh = {v:.4f}")
        else:
            ps = system2_policies[state] = {}
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"  Human agent {agent_index}")
                psi = ps[agent_index] = {}
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}")
                    # if the goal is achieved in that state, the human will not care about future rewards and use a uniform policy:
                    if possible_goal.is_achieved(state):
                        if DEBUG:
                            print(f"      Goal achieved in this state; using uniform policy")
                        Vh_values[state_index][agent_index][possible_goal] = 1
                        psi[possible_goal] = np.ones(num_actions) / num_actions
                    else:
                        # otherwise, compute the Q values as expected future V values, and the policy as a Boltzmann policy based on those Q values:
                        expected_Vs: npt.NDArray[np.floating[Any]] = np.zeros(num_actions)
                        for action in actions:
                            v_accum: float = 0.0
                            for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                                action_profile[agent_index] = action
                                # convert profile [a,b,c] into index a + b*num_actions + c*num_actions*num_actions ...
                                # Optimized base conversion using precomputed powers
                                action_profile_index = int(np.dot(action_profile, action_powers))
                                _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                                # Vectorized computation using numpy
                                v_values_array: npt.NDArray[np.floating[Any]] = np.array([
                                    Vh_values[next_state_indices[i]][agent_index][possible_goal] 
                                    for i in range(len(next_state_indices))
                                ])
                                v_accum += action_profile_prob * float(np.dot(next_state_probabilities, v_values_array))
                            expected_Vs[action] = v_accum
                        q = gamma_h * expected_Vs
                        # Boltzmann policy (numerically stable softmax):
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
                        psi[possible_goal] = p
                        v_result = Vh_values[state_index][agent_index][possible_goal] = float(np.sum(p * q))
                        if DEBUG:
                            print(f"      Goal not achieved; Vh = {v_result:.4f}")


def _hpp_init_shared_data(
    states: List[State], 
    transitions: List[List[TransitionData]], 
    Vh_values: VhValues, 
    params: Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float],
    believed_others_policy_pickle: Optional[bytes] = None
) -> None:
    """Initialize shared data for worker processes."""
    global _shared_states, _shared_transitions, _shared_Vh_values, _shared_params, _shared_believed_others_policy_pickle
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
    assert _shared_states is not None
    assert _shared_transitions is not None
    assert _shared_Vh_values is not None
    assert _shared_params is not None
    
    states = _shared_states
    transitions = _shared_transitions
    Vh_values = _shared_Vh_values
    (human_agent_indices, possible_goal_generator, num_agents, num_actions, 
     action_powers, beta_h, gamma_h) = _shared_params
    
    actions = range(num_actions)
    v_results: Dict[int, Dict[int, Dict[PossibleGoal, float]]] = {}
    p_results: Dict[State, Dict[int, Dict[PossibleGoal, npt.NDArray[np.floating[Any]]]]] = {}
    
    # Deserialize believed_others_policy if custom one was provided via cloudpickle
    if _shared_believed_others_policy_pickle is not None:
        believed_others_policy = cloudpickle.loads(_shared_believed_others_policy_pickle)
    else:
        # Create default believed others policy function
        believed_others_policy = lambda state, agent_index, action: default_believed_others_policy(
            state, agent_index, action, num_agents, num_actions)
    
    for state_index in state_indices:
        state = states[state_index]
        v_results[state_index] = {}
        
        # Check if terminal - transitions[state_index] is empty list for terminal states
        is_terminal = not transitions[state_index]
        
        if is_terminal:
            # Terminal state: only Vh-values, no policy
            for agent_index in human_agent_indices:
                v_results[state_index][agent_index] = {}
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    v_results[state_index][agent_index][possible_goal] = possible_goal.is_achieved(state)
        else:
            # Non-terminal state: compute both Vh-values and policies
            p_results[state] = {}
            for agent_index in human_agent_indices:
                v_results[state_index][agent_index] = {}
                p_results[state][agent_index] = {}
                
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    if possible_goal.is_achieved(state):
                        # Goal achieved, treated as episode end: uniform policy, Vh=1
                        v_results[state_index][agent_index][possible_goal] = 1
                        p_results[state][agent_index][possible_goal] = np.ones(num_actions) / num_actions
                    else:
                        # Compute Q values
                        expected_Vs: npt.NDArray[np.floating[Any]] = np.zeros(num_actions)
                        for action in actions:
                            v_accum: float = 0.0
                            for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                                action_profile[agent_index] = action
                                action_profile_index = int(np.dot(action_profile, action_powers))
                                _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                                
                                # Get Vh values from shared data
                                v_values_array: npt.NDArray[np.floating[Any]] = np.array([
                                    Vh_values[next_state_indices[i]][agent_index].get(possible_goal, 0)
                                    for i in range(len(next_state_indices))
                                ])
                                v_accum += action_profile_prob * float(np.dot(next_state_probabilities, v_values_array))
                            
                            expected_Vs[action] = v_accum
                        
                        q = gamma_h * expected_Vs
                        p = np.exp(beta_h * q)
                        p /= np.sum(p)
                        
                        # Store both Vh-value and policy
                        v_results[state_index][agent_index][possible_goal] = float(np.sum(p * q))
                        p_results[state][agent_index][possible_goal] = p
    
    batch_time = time.perf_counter() - batch_start
    return v_results, p_results, batch_time


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: PossibleGoalGenerator, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, List[int]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[False] = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False
) -> TabularHumanPolicyPrior: ...


@overload
def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: PossibleGoalGenerator, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, List[int]]]]] = None, 
    *, 
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: Literal[True],
    progress_callback: Optional[Callable[[int, int], None]] = None,
    quiet: bool = False
) -> Tuple[TabularHumanPolicyPrior, Dict[State, Dict[int, Dict[PossibleGoal, float]]]]: ...


def compute_human_policy_prior(
    world_model: WorldModel, 
    human_agent_indices: List[int], 
    possible_goal_generator: PossibleGoalGenerator, 
    believed_others_policy: Optional[Callable[[State, int, int], List[Tuple[float, List[int]]]]] = None, 
    *,
    beta_h: float = 10.0, 
    gamma_h: float = 1.0, 
    parallel: bool = False, 
    num_workers: Optional[int] = None, 
    level_fct: Optional[Callable[[State], int]] = None, 
    return_Vh: bool = False,
    quiet: bool = False
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
        believed_others_policy: Function(state, agent_index, action) -> List[(prob, action_profile)]
                               specifying beliefs about other agents' actions.
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
    human_policy_priors: HumanPolicyDict = {}  # these will be a mixture of system-1 and system-2 policies

    # Q_vectors = {}
    system2_policies: HumanPolicyDict = {}  # these will be Boltzmann policies with fixed inverse temperature beta for now
    # V_values will be indexed as V_values[state_index][agent_index][possible_goal]
    # Using nested lists for faster access on first two levels

    num_agents: int = len(world_model.agents)  # type: ignore[attr-defined]
    num_actions: int = world_model.action_space.n  # type: ignore[attr-defined]
    actions = range(num_actions)

    if believed_others_policy is None:
        # Create wrapper for sequential execution (parallel uses default directly)
        believed_others_policy = lambda state, agent_index, action: default_believed_others_policy(
            state, agent_index, action, num_agents, num_actions)
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
        params: Tuple[List[int], PossibleGoalGenerator, int, int, npt.NDArray[np.int64], float, float] = (
            human_agent_indices, possible_goal_generator, num_agents, num_actions,
            action_powers, beta_h, gamma_h
        )
        
        # Use 'fork' context explicitly to ensure shared memory works
        ctx = mp.get_context('fork')
        
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
        
        # Process each level sequentially, but parallelize within each level
        for level_idx, level in enumerate(dependency_levels):
            if DEBUG:
                print(f"Processing level {level_idx} with {len(level)} states")
            
            if len(level) <= num_workers:
                # Few states - process sequentially to avoid overhead
                if PROFILE_PARALLEL:
                    _t0 = time.perf_counter()
                    prof_states_sequential += len(level)
                for state_index in level:
                    state = states[state_index]
                    is_terminal = not transitions[state_index]
                    
                    if is_terminal:
                        for agent_index in human_agent_indices:
                            for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                                Vh_values[state_index][agent_index][possible_goal] = possible_goal.is_achieved(state)
                    else:
                        ps = system2_policies[state] = {}
                        for agent_index in human_agent_indices:
                            psi = ps[agent_index] = {}
                            for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                                if possible_goal.is_achieved(state):
                                    Vh_values[state_index][agent_index][possible_goal] = 1
                                    psi[possible_goal] = np.ones(num_actions) / num_actions
                                else:
                                    expected_Vs = np.zeros(num_actions)
                                    for action in actions:
                                        v = 0
                                        for action_profile_prob, action_profile in believed_others_policy(state, agent_index, action):
                                            action_profile[agent_index] = action
                                            action_profile_index = np.dot(action_profile, action_powers)
                                            _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                                            v_values_array = np.array([Vh_values[next_state_indices[i]][agent_index][possible_goal] 
                                                                      for i in range(len(next_state_indices))])
                                            v += action_profile_prob * np.dot(next_state_probabilities, v_values_array)
                                        expected_Vs[action] = v
                                    q = gamma_h * expected_Vs
                                    p = np.exp(beta_h * q)
                                    p /= np.sum(p)
                                    psi[possible_goal] = p
                                    Vh_values[state_index][agent_index][possible_goal] = np.sum(p * q)
                if PROFILE_PARALLEL:
                    prof_seq_in_par_time += time.perf_counter() - _t0
            else:
                # Many states - parallelize
                if PROFILE_PARALLEL:
                    _level_t0 = time.perf_counter()
                    prof_states_parallel += len(level)
                
                # Re-initialize shared data so new workers see updated Vh_values from previous levels
                # Also pass the cloudpickle'd believed_others_policy for custom policy support
                _hpp_init_shared_data(states, transitions, Vh_values, params, believed_others_policy_pickle)
                
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
    
    else:
        # Sequential execution (original algorithm)
        _hpp_compute_sequential(states, Vh_values, system2_policies, transitions,
                         human_agent_indices, possible_goal_generator,
                         num_agents, num_actions, action_powers,
                         believed_others_policy, beta_h, gamma_h,
                         progress_callback)
    
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


#########################################
### PHASE 2: ROBOT POLICY COMPUTATION ###  --> under construction!
#########################################

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
    human_policy_prior: Callable[[State, int, int], List[Tuple[float, List[int]]]], 
    beta_r: float, # softmax parameter for robots' power-law softmax policies
    gamma_h: float, # humans' discount factor
    gamma_r: float, # robots' discount factor
    zeta: float, # robots' risk-aversion
    xi: float, # robots' inter-human power-inequality aversion
    eta: float # robots' additional intertemporal power-inequality aversion
) -> None:
    """(under construction)
    """
    actions = range(num_actions)
    
    # TO CLARIFY:

    # loop over the nodes in reverse topological order:
    for state_index in range(len(states)-1, -1, -1):
        if DEBUG:
            print(f"Processing state {state_index}")
        state = states[state_index]
        is_terminal = not transitions[state_index]
        
        if is_terminal:
            # in terminal states, policy and Q values are undefined, only Vh, Xh, Ur, Vr values need computation:
            if DEBUG:
                print(f"  Terminal state")
            ur_inner = 0
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"   Human agent {agent_index}")
                xh = 0
                for possible_goal, weight in possible_goal_generator.generate(state, agent_index):
                    vh = Vh_values[state_index][agent_index][possible_goal] = possible_goal.is_achieved(state)
                    xh += weight * vh**zeta
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}, Vh = {vh:.4f}")
                if DEBUG:
                    print(f"   ...Xh = {xh:.4f}")
                ur_inner += xh**(-xi)
            vr = ur = -ur_inner**eta
            if DEBUG:
                print(f"  ...Vr = Ur = {vr:.4f}")
        else:
            if DEBUG:
                print(f"  Transient state")
            ur_inner = 0
            ps = robot_policy[state] = []
            for agent_index in human_agent_indices:
                if DEBUG:
                    print(f"   Human agent {agent_index}")
                xh = 0
                for possible_goal, _ in possible_goal_generator.generate(state, agent_index):
                    if DEBUG:
                        print(f"    Possible goal: {possible_goal}")
                    # if the goal is achieved in that state, the human does not care about future, hence Vh=1:
                    if possible_goal.is_achieved(state):
                        if DEBUG:
                            print(f"      Goal achieved in this state")
                        vh = Vh_values[state_index][agent_index][possible_goal] = 1
                    else:
                        # otherwise, first compute the human Vh value as a discounted expection over future Vh values given policy prior marginalized over others possible goals:
                        expected_Vh: float = 0.0
                        for action_profile_prob, action_profile in human_policy_prior.profile_distribution(state, agent_index, possible_goal):
                            # convert profile [a,b,c] into index a + b*num_actions + c*num_actions*num_actions ...
                            # Optimized base conversion using precomputed powers
                            action_profile_index = int(np.dot(action_profile, action_powers))
                            _, next_state_probabilities, next_state_indices = transitions[state_index][action_profile_index]
                            # Vectorized computation using numpy
                            vh_values_array: npt.NDArray[np.floating[Any]] = np.array([
                                Vh_values[next_state_indices[i]][agent_index][possible_goal] 
                                for i in range(len(next_state_indices))
                            ])
                            expected_Vh += action_profile_prob * float(np.dot(next_state_probabilities, vh_values_array))
                        vh_result = Vh_values[state_index][agent_index][possible_goal] = gamma_h * expected_Vh
                        if DEBUG:
                            print(f"      Goal not achieved; V = {vh_result:.4f}")
                    xh += weight * vh**zeta
                if DEBUG:
                    print(f"   ...Xh = {xh:.4f}")
                ur_inner += xh**(-xi)
            ur = -ur_inner**eta
            vr = # TODO!
            if DEBUG:
                print(f"  ...Vr = Ur = {vr:.4f}")
        Vr_values[state_index] = vr
