"""Hierarchical backward induction: compute a hierarchical robot policy.

Implements ``compute_hierarchical_robot_policy()`` which:

1. Computes the macro-level (M^0) robot policy fully via
   ``compute_robot_policy()``.
2. Returns a ``HierarchicalRobotPolicy`` that computes micro-level
   sub-problem policies on demand during rollouts (no caching).
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from empo.possible_goal import PossibleGoalGenerator
from empo.human_policy_prior import HumanPolicyPrior
from empo.backward_induction.phase2 import compute_robot_policy
from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel
from empo.hierarchical.hierarchical_robot_policy import HierarchicalRobotPolicy


def compute_hierarchical_robot_policy(
    hierarchical_model: HierarchicalWorldModel,
    human_agent_indices: List[int],
    robot_agent_indices: List[int],
    possible_goal_generators: List[PossibleGoalGenerator],
    human_policy_priors: Optional[List[Optional[HumanPolicyPrior]]] = None,
    *,
    beta_r: float = 10.0,
    gamma_h: Optional[float] = None,
    gamma_r: Optional[float] = None,
    rho_h: Optional[float] = None,
    rho_r: Optional[float] = None,
    zeta: float = 1.0,
    xi: float = 1.0,
    eta: float = 1.0,
    terminal_Vr: float = -1e-10,
    quiet: bool = False,
) -> HierarchicalRobotPolicy:
    """Compute a hierarchical robot policy via top-down backward induction.

    1. Computes the macro-level (M^0) robot policy fully via
       ``compute_robot_policy()``.
    2. Returns a ``HierarchicalRobotPolicy`` that computes micro-level
       sub-problem policies on demand during rollouts (no caching).

    Args:
        hierarchical_model: A two-level ``HierarchicalWorldModel``.
        human_agent_indices: Indices of human agents (same across levels).
        robot_agent_indices: Indices of robot agents.
        possible_goal_generators: One ``PossibleGoalGenerator`` per level.
            ``possible_goal_generators[0]`` is for M^0 (macro),
            ``possible_goal_generators[1]`` is for M^1 (micro).
        human_policy_priors: Optional list of ``HumanPolicyPrior`` per level.
            ``human_policy_priors[0]`` is for M^0, ``[1]`` for M^1.
            If ``None``, both are auto-computed.
        beta_r: Power-law concentration parameter.
        gamma_h: Human discount factor (0 < gamma_h ≤ 1).
        gamma_r: Robot discount factor (0 < gamma_r ≤ 1).
        rho_h: Continuous-time human discount rate.
        rho_r: Continuous-time robot discount rate.
        zeta: Risk-aversion parameter.
        xi: Inter-human power-inequality aversion.
        eta: Additional intertemporal power-inequality aversion.
        terminal_Vr: V_r value for terminal states (must be strictly negative).
        quiet: Suppress progress output.

    Returns:
        A ``HierarchicalRobotPolicy``.

    Raises:
        ValueError: If fewer than 2 goal generators are provided, or if
            both gamma and rho are specified simultaneously.
    """
    if hierarchical_model.num_levels != 2:
        raise ValueError(
            f"compute_hierarchical_robot_policy currently requires exactly "
            f"2 levels, got {hierarchical_model.num_levels}"
        )

    if len(possible_goal_generators) < 1:
        raise ValueError(
            f"Need at least 1 possible_goal_generator (for M^0), "
            f"got {len(possible_goal_generators)}"
        )

    macro_env = hierarchical_model.coarsest()
    micro_env = hierarchical_model.finest()

    # ── Resolve discount parameters ────────────────────────────
    if gamma_h is not None and rho_h is not None:
        raise ValueError("Specify at most one of gamma_h or rho_h, not both.")
    if gamma_r is not None and rho_r is not None:
        raise ValueError("Specify at most one of gamma_r or rho_r, not both.")

    if gamma_h is not None:
        _rho_h = 0.0 if gamma_h == 1.0 else -math.log(gamma_h)
        _gamma_h = gamma_h
    elif rho_h is not None:
        _rho_h = rho_h
        _gamma_h = math.exp(-rho_h)
    else:
        _gamma_h = 1.0
        _rho_h = 0.0

    if gamma_r is not None:
        _rho_r = 0.0 if gamma_r == 1.0 else -math.log(gamma_r)
        _gamma_r = gamma_r
    elif rho_r is not None:
        _rho_r = rho_r
        _gamma_r = math.exp(-rho_r)
    else:
        _gamma_r = 1.0
        _rho_r = 0.0

    # ── Extract per-level priors ───────────────────────────────
    macro_prior = None
    micro_prior = None
    if human_policy_priors is not None:
        if len(human_policy_priors) >= 1:
            macro_prior = human_policy_priors[0]
        if len(human_policy_priors) >= 2:
            micro_prior = human_policy_priors[1]

    # ── Step 1: full macro-level solve ─────────────────────────
    if not quiet:
        print("=== Hierarchical Backward Induction: Macro-Level (M^0) ===")

    macro_policy, macro_Vr, macro_Xh = compute_robot_policy(
        macro_env,
        human_agent_indices=human_agent_indices,
        robot_agent_indices=robot_agent_indices,
        possible_goal_generator=possible_goal_generators[0],
        human_policy_prior=macro_prior,
        beta_r=beta_r,
        gamma_h=_gamma_h,
        gamma_r=_gamma_r,
        zeta=zeta,
        xi=xi,
        eta=eta,
        terminal_Vr=terminal_Vr,
        return_values=True,
        quiet=quiet,
    )

    if not quiet:
        print(
            f"  Macro-level solve complete: {len(macro_Vr)} states, "
            f"{len(macro_policy.values)} policy entries"
        )

    # ── Step 2: construct HierarchicalRobotPolicy ──────────────
    micro_goal_gen = (
        possible_goal_generators[1] if len(possible_goal_generators) >= 2 else None
    )

    h_policy = HierarchicalRobotPolicy(
        hierarchical_model=hierarchical_model,
        macro_policy=macro_policy,
        macro_Vr=macro_Vr,
        macro_Xh=macro_Xh,
        robot_agent_indices=robot_agent_indices,
        human_agent_indices=human_agent_indices,
        micro_goal_generator=micro_goal_gen,
        micro_human_policy_prior=micro_prior,
        beta_r=beta_r,
        gamma_h=_gamma_h,
        gamma_r=_gamma_r,
        rho_h=_rho_h,
        rho_r=_rho_r,
        zeta=zeta,
        xi=xi,
        eta=eta,
        terminal_Vr=terminal_Vr,
        quiet=quiet,
    )

    if not quiet:
        print("=== Hierarchical policy ready (micro sub-problems on demand) ===")

    return h_policy
