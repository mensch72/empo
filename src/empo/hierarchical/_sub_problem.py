"""Sub-problem DAG construction for hierarchical backward induction.

Builds a restricted state DAG rooted at a micro-level state, expanding
only feasible action profiles and marking states as terminal when
``return_control()`` fires.
"""

from collections import deque
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


def build_sub_problem_dag(
    micro_env: Any,
    mapper: Any,
    coarse_action_profile: Tuple[int, ...],
    root_state: Any,
    *,
    quiet: bool = True,
) -> Tuple[
    List[Any],                  # states  (topological order)
    Dict[Any, int],             # state_to_idx
    List[List[Tuple[Tuple[int, ...], List[float], List[int]]]],  # transitions
    List[bool],                 # terminal_mask
]:
    """Build a feasibility-filtered sub-problem DAG for one macro action.

    Starting from *root_state*, performs a BFS over the micro-level state
    space, but:

    - Only expands action profiles where ``mapper.is_feasible()`` is True.
    - Marks successor states where ``mapper.return_control()`` is True as
      **terminal** (they will not be expanded further).
    - The root state itself is never terminal.

    After BFS discovery, a Kahn topological sort is applied to ensure that
    every successor state has a strictly higher index than its predecessor.
    This guarantees correct reverse-order processing for backward induction,
    even when cross-edges exist in the BFS tree.

    Args:
        micro_env: The micro-level WorldModel (must support ``get_state``,
            ``set_state``, ``transition_probabilities``).
        mapper: The LevelMapper connecting coarse to fine level.
        coarse_action_profile: The active macro action profile.
        root_state: The micro-state to start from.
        quiet: Suppress progress output.

    Returns:
        A 4-tuple ``(states, state_to_idx, transitions, terminal_mask)``
        where:

        - *states*: list of micro-states in topological order.
        - *state_to_idx*: dict mapping micro-state → index.
        - *transitions*: per-state list of ``(action_profile, probs,
          succ_indices)`` tuples (only feasible actions).
        - *terminal_mask*: boolean list — ``True`` for terminal states.
    """
    num_agents: int = len(micro_env.agents)
    num_actions: int = micro_env.action_space.n

    # Iterate action profiles lazily to avoid large memory spike
    from itertools import product as itertools_product

    # ── Phase 1: BFS discovery ──────────────────────────────────
    bfs_states: List[Any] = []
    bfs_state_to_idx: Dict[Any, int] = {}
    bfs_terminal_mask: List[bool] = []
    bfs_transitions: List[List[Tuple[Tuple[int, ...], List[float], List[int]]]] = []

    # Enqueue root
    bfs_state_to_idx[root_state] = 0
    bfs_states.append(root_state)
    bfs_terminal_mask.append(False)  # root is never terminal
    bfs_transitions.append([])

    queue: deque[int] = deque([0])

    pbar = None
    if not quiet:
        pbar = tqdm(desc="Sub-problem DAG", unit=" states", leave=False)

    old_state = micro_env.get_state()
    try:
        while queue:
            src_idx = queue.popleft()
            src_state = bfs_states[src_idx]

            if bfs_terminal_mask[src_idx]:
                continue  # Don't expand terminal states

            micro_env.set_state(src_state)

            state_trans: List[Tuple[Tuple[int, ...], List[float], List[int]]] = []

            for ap in itertools_product(range(num_actions), repeat=num_agents):
                # Feasibility filter
                if not mapper.is_feasible(coarse_action_profile, src_state, ap):
                    continue

                # Get transition outcomes
                outcomes = micro_env.transition_probabilities(src_state, list(ap))

                if outcomes is None:
                    continue  # Terminal state — skip

                probs: List[float] = []
                succ_indices: List[int] = []

                for prob, succ_state in outcomes:
                    if prob <= 0.0:
                        continue

                    if succ_state not in bfs_state_to_idx:
                        succ_idx = len(bfs_states)
                        bfs_state_to_idx[succ_state] = succ_idx
                        bfs_states.append(succ_state)

                        # Determine if successor is terminal
                        is_term = mapper.return_control(
                            coarse_action_profile, src_state, ap, succ_state
                        )
                        bfs_terminal_mask.append(is_term)
                        bfs_transitions.append([])

                        if not is_term:
                            queue.append(succ_idx)
                    else:
                        succ_idx = bfs_state_to_idx[succ_state]

                    probs.append(prob)
                    succ_indices.append(succ_idx)

                if probs:
                    state_trans.append((ap, probs, succ_indices))

            bfs_transitions[src_idx] = state_trans

            if pbar is not None:
                pbar.update(1)
    finally:
        micro_env.set_state(old_state)
        if pbar is not None:
            pbar.close()

    # ── Phase 2: Kahn topological sort ──────────────────────────
    n = len(bfs_states)
    in_degree = [0] * n
    for src_idx in range(n):
        for _, _, succ_idxs in bfs_transitions[src_idx]:
            for si in succ_idxs:
                if si != src_idx:  # defensive: skip if same state (not a DAG edge)
                    in_degree[si] += 1

    topo_queue: deque[int] = deque()
    for i in range(n):
        if in_degree[i] == 0:
            topo_queue.append(i)

    topo_order: List[int] = []  # bfs_idx → position in topo_order
    while topo_queue:
        node = topo_queue.popleft()
        topo_order.append(node)
        for _, _, succ_idxs in bfs_transitions[node]:
            for si in succ_idxs:
                if si != node:  # defensive: skip self-transitions
                    in_degree[si] -= 1
                    if in_degree[si] == 0:
                        topo_queue.append(si)

    # Build old→new index mapping
    old_to_new = [0] * n
    for new_idx, old_idx in enumerate(topo_order):
        old_to_new[old_idx] = new_idx

    # Remap everything to topological order
    states: List[Any] = [bfs_states[old] for old in topo_order]
    terminal_mask: List[bool] = [bfs_terminal_mask[old] for old in topo_order]
    state_to_idx: Dict[Any, int] = {s: i for i, s in enumerate(states)}
    transitions: List[List[Tuple[Tuple[int, ...], List[float], List[int]]]] = []
    for old in topo_order:
        remapped: List[Tuple[Tuple[int, ...], List[float], List[int]]] = []
        for ap, probs, succ_idxs in bfs_transitions[old]:
            remapped.append((ap, probs, [old_to_new[si] for si in succ_idxs]))
        transitions.append(remapped)

    return states, state_to_idx, transitions, terminal_mask
