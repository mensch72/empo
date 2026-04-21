"""
Shared helper for computing simplified (goal-agnostic) X_h TD targets.

Used by both the DQN-based and PPO-based Phase 2 trainers when
``use_simplified_x_h=True``.

The simplified recursion is::

    X_h(s) = 1 + gamma_h^zeta * sum_{s'} q_h(s,s')^zeta * X_h(s')

where::

    q_h(s,s') = max_{a_h} P(s'|s, a_h, pi_{-h})

When an inverse-dynamics model ``P_theta(a_h | s, s')`` is available, this
helper can also form the sampled model-free Bayes-ratio approximation agreed
for the learning-based implementation.
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
    robot_policy_per_state: Dict[Any, torch.Tensor],
    action_index_to_tuple: Callable[[int], Tuple[int, ...]],
    other_human_probs_fn: Callable[[Any, int], np.ndarray],
    world_model: Any,
    inverse_dynamics_network: Optional[Any] = None,
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

    If ``inverse_dynamics_network`` is provided, the helper instead uses the
    sampled Bayes-ratio approximation::

        q_h(s, s') \\propto max_{a_h} P_theta(a_h | s, s') / pi_h(a_h | s)

    with the same bounded-rationality mixing applied to the ratio term.

    Parameters
    ----------
    states, next_states : list
        Source / successor states, one per sample.
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
    x_h_next_values : Tensor, shape ``(batch,)``
        Pre-computed ``X_h_target(s')`` for every next_state.
        Terminal states should already have value 1.0.
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
        Used by the inverse-dynamics network for tensorisation and, when no
        inverse-dynamics model is provided, for exact
        ``transition_probabilities(state, actions)`` fallback.
    inverse_dynamics_network : optional
        If provided, uses the model-free Bayes-ratio target instead of the
        exact transition-mass computation.
    device : str
        Torch device for the returned tensor.

    Returns
    -------
    Tensor of shape ``(batch,)`` with TD targets.
    """
    gamma_h_zeta = gamma_h ** zeta

    # Cache other-human joint distributions per (state, h_idx)
    other_joint_cache: Dict[Any, Tuple[List[int], list]] = {}
    # Cache P(s' | s, a_h, pi_-h) mass maps per (state, h_idx) for the exact
    # fallback path. Each entry is a list of length num_actions where element
    # a_h is a dict mapping successor states to total probability mass under
    # that action.
    transition_mass_cache: Dict[Any, List[Dict[Any, float]]] = {}

    targets = []
    for i, (state, next_state, h_idx) in enumerate(
        zip(states, next_states, human_indices)
    ):
        pi_r = robot_policy_per_state[state]  # (num_robot_joint_actions,)

        if inverse_dynamics_network is not None:
            pi_h = _to_prob_array(other_human_probs_fn(state, h_idx))
            with torch.no_grad():
                inverse_logits = inverse_dynamics_network.forward(
                    state,
                    next_state,
                    world_model,
                    h_idx,
                    device=device,
                )
                inverse_probs = torch.softmax(inverse_logits, dim=-1)
            ratio_per_ah = (
                inverse_probs.squeeze(0)
                .detach()
                .to(dtype=torch.float64)
                .cpu()
                .numpy()
                / np.clip(pi_h, 1e-12, None)
            )

            max_ratio = float(np.max(ratio_per_ah))
            if epsilon_h > 0.0:
                mean_ratio = float(np.mean(ratio_per_ah))
                q_h = (1.0 - epsilon_h) * max_ratio + epsilon_h * mean_ratio
            else:
                q_h = max_ratio
        else:
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
                transition_mass_cache[mass_key] = per_action_next_mass

            p_per_ah = [
                per_action_next_mass.get(next_state, 0.0)
                for per_action_next_mass in transition_mass_cache[mass_key]
            ]

            max_p = max(p_per_ah)
            if epsilon_h > 0.0:
                mean_p = sum(p_per_ah) / num_actions
                q_h = (1.0 - epsilon_h) * max_p + epsilon_h * mean_p
            else:
                q_h = max_p

        x_h_sp = (
            float(x_h_next_values[i])
            if x_h_next_values.dim() > 0
            else float(x_h_next_values)
        )
        targets.append(1.0 + gamma_h_zeta * (q_h ** zeta) * x_h_sp)

    return torch.tensor(targets, device=device, dtype=torch.float32)
