"""
Shared helper for computing simplified (goal-agnostic) X_h TD targets.

Used by both the DQN-based and PPO-based Phase 2 trainers when
``use_simplified_x_h=True``.

The simplified recursion is::

    X_h(s) = 1 + gamma_h^zeta * sum_{s'} q_h(s,s')^zeta * X_h(s')

where::

    q_h(s,s') = max_{a_h} P(s'|s, a_h, pi_{-h})
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def _joint_distribution_numpy(
    marginals: List[np.ndarray],
) -> List[Tuple[float, List[int]]]:
    """Compute joint distribution from independent marginals.

    Fast paths for 0, 1, and 2 agents; general N-agent fallback.
    Returns a list of ``(probability, action_list)`` pairs with non-zero
    probability only.
    """
    if not marginals:
        return [(1.0, [])]
    if len(marginals) == 1:
        m = marginals[0]
        return [(float(m[a]), [int(a)]) for a in np.nonzero(m > 0)[0]]
    if len(marginals) == 2:
        m0, m1 = marginals[0], marginals[1]
        joint = np.outer(m0, m1)
        ii, jj = np.nonzero(joint > 0)
        probs = joint[ii, jj]
        return [
            (float(probs[k]), [int(ii[k]), int(jj[k])])
            for k in range(len(probs))
        ]
    # General case: iterative outer product
    from itertools import product as itertools_product

    indices = [np.nonzero(m > 0)[0] for m in marginals]
    result = []
    for combo in itertools_product(*indices):
        p = 1.0
        for m, a in zip(marginals, combo):
            p *= float(m[a])
        if p > 0:
            result.append((p, [int(a) for a in combo]))
    return result


def _to_prob_array(dist) -> np.ndarray:
    """Convert an action distribution (dict, list, or ndarray) to a
    normalised numpy probability array."""
    if isinstance(dist, dict):
        arr = np.array([dist[i] for i in range(len(dist))], dtype=np.float64)
    else:
        arr = np.asarray(dist, dtype=np.float64)
    s = arr.sum()
    if s > 0:
        arr /= s
    else:
        arr[:] = 1.0 / len(arr)
    return arr


def compute_simplified_x_h_td_targets(
    states: List[Any],

    human_indices: List[int],
    *,
    gamma_h: float,
    zeta: float,
    epsilon_h: float,
    num_actions: int,
    num_agents: int,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    evaluate_x_h_target_fn: Callable[[List[Any], List[int]], List[float]],
    robot_policy_per_state: Dict[Any, torch.Tensor],
    action_index_to_tuple: Callable[[int], Tuple[int, ...]],
    other_human_probs_fn: Callable[[Any, int], np.ndarray],
    world_model: Any,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute simplified X_h TD targets via the goal-agnostic recursion.

    This is a per-sample (stochastic) TD estimator for the fixed-point
    equation::

        X_h(s) = 1 + gamma_h^zeta * sum_{s'} q_h(s,s')^zeta * X_h(s')

    For each observed transition ``(s, s')`` the per-sample target is::

        target(s, s') = 1 + gamma_h^zeta * q_h(s, s')^zeta * X_h_target(s')

    where::

        q_h(s, s') = max_{a_h} P(s'|s, a_h, pi_{-h}(·|s))

    With bounded rationality ``(epsilon_h > 0)``::

        q_h = (1 - eps) * max_{a_h} P(...) + eps * mean_{a_h} P(...)

    Parameters
    ----------
    states : list
        Source states, one per sample.
    human_indices : list[int]
        Focal human agent index for each sample.
    gamma_h, zeta, epsilon_h : float
        Theory parameters.
    num_actions : int
        Number of actions per agent.
    num_agents : int
        Total number of agents (humans + robots).
    human_agent_indices, robot_agent_indices : list[int]
        Agent index lists.
    evaluate_x_h_target_fn : callable
        Function ``(next_states, h_indices) -> List[float]`` returning target values for (ns, h_idx).
    robot_policy_per_state : dict[state → Tensor]
        Maps each unique state to a 1-D tensor of robot joint-action
        probabilities.
    action_index_to_tuple : callable
        Flat robot joint-action index → per-robot action tuple.
    other_human_probs_fn : callable ``(state, agent_index) → np.ndarray``
        Returns a normalised probability array over actions for the given
        human agent (goal-agnostic / marginal).  Framework-agnostic:
        callers wrap their own ``HumanPolicyPrior`` or PPO callable.
    world_model : WorldModel
        Provides ``transition_probabilities(state, actions)``.
    device : str
        Torch device for the returned tensor.

    Returns
    -------
    Tensor of shape ``(batch,)`` with TD targets.
    """
    gamma_h_zeta = gamma_h ** zeta

    # Cache other-human joint distributions per (state, h_idx)
    other_joint_cache: Dict[Any, Tuple[List[int], list]] = {}
    # Cache P(s' | s, a_h, pi_-h) mass maps per (state, h_idx).
    # Each entry is a list of length num_actions where element a_h is a dict
    # mapping successor states to total probability mass under that action.
    transition_mass_cache: Dict[Any, List[Dict[Any, float]]] = {}

    # Step 1: Collect transition mass caches and required target evaluations
    required_evals = set()
    states_data = []

    for i, (state, h_idx) in enumerate(zip(states, human_indices)):
        pi_r = robot_policy_per_state[state]  # (num_robot_joint_actions,)

        # Compute (or look up) the joint distribution of other humans.
        cache_key = (state, h_idx)
        if cache_key not in other_joint_cache:
            other_humans = [h for h in human_agent_indices if h != h_idx]
            if other_humans:
                marginals = [
                    _to_prob_array(other_human_probs_fn(state, oh))
                    for oh in other_humans
                ]
                dist = _joint_distribution_numpy(marginals)
            else:
                other_humans = []
                dist = [(1.0, [])]
            other_joint_cache[cache_key] = (other_humans, dist)
        other_humans, other_joint_dist = other_joint_cache[cache_key]

        mass_key = (state, h_idx)
        if mass_key not in transition_mass_cache:
            per_action_next_mass: List[Dict[Any, float]] = []
            for a_h_val in range(num_actions):
                next_mass: Dict[Any, float] = {}
                for a_r_idx in range(pi_r.shape[0]):
                    robot_prob = float(pi_r[a_r_idx])
                    if robot_prob == 0.0:
                        continue
                    robot_action_tuple = action_index_to_tuple(a_r_idx)
                    for oh_prob, oh_actions in other_joint_dist:
                        w = robot_prob * oh_prob
                        if w == 0.0:
                            continue
                        actions = [0] * num_agents
                        actions[h_idx] = a_h_val
                        for r, r_a in zip(robot_agent_indices, robot_action_tuple):
                            actions[r] = r_a
                        for oh, oa in zip(other_humans, oh_actions):
                            actions[oh] = oa
                        trans = world_model.transition_probabilities(state, actions)
                        if trans is None:
                            continue
                        for prob, ns in trans:
                            next_mass[ns] = next_mass.get(ns, 0.0) + w * float(prob)
                per_action_next_mass.append(next_mass)
                for ns in next_mass:
                    required_evals.add((ns, h_idx))
            transition_mass_cache[mass_key] = per_action_next_mass

        states_data.append((state, h_idx, transition_mass_cache[mass_key]))

    # Step 2: Batch compute required targets
    ns_list = []
    h_idx_list = []
    x_h_cache = {}
    
    for ns, h_idx in required_evals:
        if world_model.is_terminal(ns):
            x_h_cache[(ns, h_idx)] = 1.0
        else:
            ns_list.append(ns)
            h_idx_list.append(h_idx)
            
    if ns_list:
        target_vals = evaluate_x_h_target_fn(ns_list, h_idx_list)
        for ns, h_idx, val in zip(ns_list, h_idx_list, target_vals):
            x_h_cache[(ns, h_idx)] = val

    # Step 3: Compute exact sum over all reachable next_states
    targets = []
    for state, h_idx, per_action_next_mass in states_data:
        # Find all unique next_states reachable from this state
        all_ns = set()
        for next_mass in per_action_next_mass:
            all_ns.update(next_mass.keys())
            
        sum_s_prime = 0.0
        for ns in all_ns:
            p_per_ah = [
                m.get(ns, 0.0) for m in per_action_next_mass
            ]
            max_p = max(p_per_ah)
            if epsilon_h > 0.0:
                mean_p = sum(p_per_ah) / num_actions
                q_h = (1.0 - epsilon_h) * max_p + epsilon_h * mean_p
            else:
                q_h = max_p
                
            if q_h > 0.0:
                sum_s_prime += (q_h ** zeta) * x_h_cache[(ns, h_idx)]
                
        targets.append(1.0 + gamma_h_zeta * sum_s_prime)

    return torch.tensor(targets, device=device, dtype=torch.float32)