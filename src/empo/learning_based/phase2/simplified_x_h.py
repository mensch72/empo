"""
Shared helper for computing simplified (goal-agnostic) X_h TD targets.

Used by both the DQN-based and PPO-based Phase 2 trainers when
``use_simplified_x_h=True``.

The simplified recursion is::

    X_h(s) = 1 + gamma_h^zeta * sum_{s'} inverse_dynamics(s,s')^zeta * X_h(s')

where::

    inverse_dynamics(s,s') = max_{a_h} P(s'|s, a_h, pi_{-h})
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

def _joint_distribution_numpy(marginals: List[np.ndarray]) -> List[Tuple[float, List[int]]]:
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
        return [(float(probs[k]), [int(ii[k]), int(jj[k])]) for k in range(len(probs))]
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
    next_states: List[Any],
    human_indices: List[int],
    *,
    gamma_h: float,
    zeta: float,
    epsilon_h: float,
    num_actions: int,
    num_agents: int,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    x_h_next_values: torch.Tensor,
    robot_policy_per_state: Optional[Dict[Any, torch.Tensor]] = None,
    action_index_to_tuple: Optional[Callable[[int], Tuple[int, ...]]] = None,
    other_human_probs_fn: Optional[Callable[[Any, int], np.ndarray]] = None,
    world_model: Optional[Any] = None,
    inverse_dynamics_network: Optional[Any] = None,
    device: str = "cpu",
) -> torch.Tensor:
    gamma_h_zeta = gamma_h ** zeta

    # =========================================================================
    # OPTION 1: Use sample-based inverse_dynamics neural network output to approximate X_h
    # =========================================================================
    if inverse_dynamics_network is not None:
        targets = []
        for i, (state, ns, h_idx) in enumerate(zip(states, next_states, human_indices)):
            x_h_sp = float(x_h_next_values[i]) if x_h_next_values.dim() > 0 else float(x_h_next_values)
            
            with torch.no_grad():
                logits = inverse_dynamics_network(state, ns, world_model, h_idx, device)
                p_theta = torch.softmax(logits, dim=-1).cpu().numpy()  # Extract max probability distribution

            if other_human_probs_fn is not None:
                pi_h = _to_prob_array(other_human_probs_fn(state, h_idx))
            else:
                pi_h = np.ones(num_actions) / num_actions
            
            pi_h = np.maximum(pi_h, 1e-8)
            ratio = p_theta / pi_h
            
            max_ratio = np.max(ratio)
            if epsilon_h > 0.0:
                mean_ratio = np.mean(ratio)
                inv_dyn_val = (1.0 - epsilon_h) * max_ratio + epsilon_h * mean_ratio
            else:
                inv_dyn_val = max_ratio

            # Approximation: target(s,s') = 1 + gamma_h^zeta * (max(p_theta/pi))^zeta * X_h(s') 
            targets.append(1.0 + gamma_h_zeta * (inv_dyn_val ** zeta) * x_h_sp)

        return torch.tensor(targets, dtype=torch.float32, device=device)

    # =========================================================================
    # OPTION 2: Use exact world model P(s'|s,a_h,pi_-h) computation 
    # =========================================================================
    other_joint_cache: Dict[Any, Tuple[List[int], list]] = {}
    transition_mass_cache: Dict[Any, List[Dict[Any, float]]] = {}
    
    # Pre-collect states to compute transition mass caching over reachable next states
    states_data = []

    for i, (state, h_idx) in enumerate(zip(states, human_indices)):
        pi_r = robot_policy_per_state[state] 

        cache_key = (state, h_idx)
        if cache_key not in other_joint_cache:
            other_humans = [h for h in human_agent_indices if h != h_idx]
            if other_humans:
                marginals = [_to_prob_array(other_human_probs_fn(state, oh)) for oh in other_humans]
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
                        for prob, possible_ns in trans:
                            next_mass[possible_ns] = next_mass.get(possible_ns, 0.0) + w * float(prob)
                per_action_next_mass.append(next_mass)
            transition_mass_cache[mass_key] = per_action_next_mass
            
        states_data.append((state, h_idx, transition_mass_cache[mass_key]))

    targets = []
    
    # Original specific bugfix: calculate sum over *all* reachable next_states rather than just the sampled one
    # But wait, since we now have `x_h_next_values` matching `next_states` batch exactly, doing full exact evaluation
    # of *all* reachable next states requires evaluating X_h(s') for all possible outcomes.
    # To fix our DQN backwards compatibility and keep exact math running, we just return the sample estimator
    # over the observed next_state using the exact probabilities of `next_state` if that's what was evaluated.
    
    # Wait, the exact exact target means:
    # 1 + gamma_h_zeta * sum_{s''} (P(s''|...)) * X_h(s'')
    # If we are strictly estimating from rollouts, we should compute: 
    # target = 1 + gamma_h_zeta * inv_dyn_val ** zeta * X_h(s')
    # Because we don't have X_h(s'') for all s'', only X_h(s') passed as x_h_next_values
    
    for i, (state, ns, h_idx) in enumerate(zip(states, next_states, human_indices)):
        mass_key = (state, h_idx)
        per_action_next_mass = transition_mass_cache[mass_key]
        
        # Compute probability for THIS specific observed next_state
        p_per_ah = [
            m.get(ns, 0.0) 
            for m in per_action_next_mass
        ]
        max_p = max(p_per_ah)
        if epsilon_h > 0.0:
            mean_p = sum(p_per_ah) / num_actions
            inverse_dynamics = (1.0 - epsilon_h) * max_p + epsilon_h * mean_p
        else:
            inverse_dynamics = max_p

        x_h_sp = float(x_h_next_values[i]) if x_h_next_values.dim() > 0 else float(x_h_next_values)
        targets.append(1.0 + gamma_h_zeta * (inverse_dynamics ** zeta) * x_h_sp)

    return torch.tensor(targets, device=device, dtype=torch.float32)
