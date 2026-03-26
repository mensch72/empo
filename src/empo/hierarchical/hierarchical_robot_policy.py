"""Hierarchical robot policy that operates across multiple levels.

Maintains the current control level and delegates to the appropriate
level's policy.  Computes micro-level sub-problem policies on demand
(no caching across macro decisions — each macro-to-micro transition
triggers a fresh sub-problem solve, but the resulting micro policy is
reused for the duration of that macro action).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from empo.robot_policy import RobotPolicy
from empo.backward_induction.phase2 import TabularRobotPolicy
from empo.hierarchical.hierarchical_world_model import HierarchicalWorldModel


class HierarchicalRobotPolicy(RobotPolicy):
    """A robot policy that operates across two levels of a hierarchical model.

    At the start of each macro decision point, the macro-level policy selects a
    coarse action profile.  Control then transfers to the micro level, where a
    sub-problem policy is computed on demand and used to select fine actions
    until the coarse action completes, fails, or is aborted.

    Usage::

        policy = compute_hierarchical_robot_policy(...)
        policy.reset(hierarchical_model)
        while not done:
            action = policy.sample(micro_state)

    Attributes:
        hierarchical_model: The two-level HierarchicalWorldModel.
        macro_policy: Pre-computed TabularRobotPolicy for M^0.
        macro_Vr: Robot value function V_r(s^0) from the macro-level solve.
        macro_Xh: Human goal-achievement values from the macro-level solve.
        robot_agent_indices: Indices of robot agents.
        human_agent_indices: Indices of human agents.
    """

    def __init__(
        self,
        hierarchical_model: HierarchicalWorldModel,
        macro_policy: TabularRobotPolicy,
        macro_Vr: Dict[Any, float],
        macro_Xh: Dict,
        *,
        robot_agent_indices: List[int],
        human_agent_indices: List[int],
        micro_goal_generator: Any = None,
        micro_human_policy_prior: Any = None,
        beta_r: float = 10.0,
        gamma_h: float = 1.0,
        gamma_r: float = 1.0,
        rho_h: float = 0.0,
        rho_r: float = 0.0,
        zeta: float = 1.0,
        xi: float = 1.0,
        eta: float = 1.0,
        terminal_Vr: float = -1e-10,
        quiet: bool = True,
    ):
        self.hierarchical_model = hierarchical_model
        self.macro_policy = macro_policy
        self.macro_Vr = macro_Vr
        self.macro_Xh = macro_Xh
        self.robot_agent_indices = robot_agent_indices
        self.human_agent_indices = human_agent_indices

        # Parameters for on-demand micro sub-problem computation
        self.micro_goal_generator = micro_goal_generator
        self.micro_human_policy_prior = micro_human_policy_prior
        self.beta_r = beta_r
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.rho_h = rho_h
        self.rho_r = rho_r
        self.zeta = zeta
        self.xi = xi
        self.eta = eta
        self.terminal_Vr = terminal_Vr
        self.quiet = quiet

        # Control state
        self._current_coarse_action_profile: Optional[Tuple[int, ...]] = None
        self._current_coarse_state: Optional[Any] = None
        self._current_sub_policy: Optional[RobotPolicy] = None

    # ------------------------------------------------------------------
    # RobotPolicy interface
    # ------------------------------------------------------------------

    def sample(self, state: Any) -> Any:
        """Sample a micro-level action for the given micro-state.

        Logic:

        1. If control is at the macro level (no active coarse action profile):
           a. Compute macro-state from micro-state via ``super_state()``.
           b. Sample macro-action from ``macro_policy``.
           c. Expand the robot-only macro profile to a full per-agent profile
              (``MACRO_PASS`` for non-robot agents).
           d. If the full coarse profile is all ``MACRO_PASS``, stay at the
              macro level (return a default micro action without transferring
              control).
           e. Otherwise, store as current coarse action profile and transfer
              control to the micro level.
        2. Compute the sub-problem policy for the current context and sample
           a micro-level action.
        3. Rely on ``observe_transition()`` (called after ``env.step()``) for
           return-control / abort detection.

        Args:
            state: A micro-level state (fine-level).

        Returns:
            A robot-only micro-level action profile (tuple of ints, one per
            robot agent).
        """
        from empo.hierarchical.macro_grid_env import MACRO_PASS

        mapper = self.hierarchical_model.mappers[0]
        micro_env = self.hierarchical_model.finest()
        num_agents = len(micro_env.agents)

        # ── Step 1: decide macro action if needed ──────────────
        if self._current_coarse_action_profile is None:
            coarse_state = mapper.super_state(state)
            self._current_coarse_state = coarse_state

            # macro_policy.sample() returns a *robot-only* profile
            robot_coarse = self.macro_policy.sample(coarse_state)

            # Expand to a full per-agent coarse profile:
            # non-robot agents default to MACRO_PASS.
            full_coarse = [MACRO_PASS] * num_agents
            for i, r_idx in enumerate(self.robot_agent_indices):
                if r_idx >= num_agents:
                    raise IndexError(
                        f"robot_agent_index {r_idx} out of range for "
                        f"{num_agents} agents"
                    )
                full_coarse[r_idx] = robot_coarse[i]
            full_coarse = tuple(full_coarse)

            # If every agent's coarse action is PASS, there is no walk
            # target for return_control() to trigger on.  Stay at the
            # macro level and emit a default micro action.
            if all(a == MACRO_PASS for a in full_coarse):
                self._current_coarse_action_profile = None
                self._current_coarse_state = None
                self._current_sub_policy = None
                # Return a non-moving micro action: prefer "still", then "left"
                # (a turn in place), and only action 0 as a last resort.
                still_action = getattr(micro_env.actions, 'still', None)
                if still_action is None:
                    # No still — use "left" (turn in place) if available
                    left_action = getattr(micro_env.actions, 'left', None)
                    still_action = left_action if left_action is not None else 0
                return tuple(still_action for _ in self.robot_agent_indices)

            self._current_coarse_action_profile = full_coarse
            # Invalidate previous sub-policy
            self._current_sub_policy = None

        # ── Step 2: compute/retrieve micro sub-problem policy ──
        if self._current_sub_policy is None:
            self._current_sub_policy = self._compute_sub_policy(
                self._current_coarse_action_profile,
                state,
            )

        # Sample from the sub-policy (returns robot-only profile)
        micro_action_profile = self._current_sub_policy.sample(state)

        return micro_action_profile

    def observe_transition(
        self, state: Any, action_profile: Tuple[int, ...], successor_state: Any
    ) -> None:
        """Notify the policy of a completed micro-level transition.

        Must be called after each ``env.step()`` so that control can be
        returned to the macro level at the appropriate time.

        Args:
            state: Micro-state before the transition.
            action_profile: The micro action profile that was taken.
            successor_state: Micro-state after the transition.
        """
        if self._current_coarse_action_profile is None:
            return  # Already returned control (e.g. via abort in sample())

        mapper = self.hierarchical_model.mappers[0]
        if mapper.return_control(
            self._current_coarse_action_profile,
            state,
            action_profile,
            successor_state,
        ):
            self._current_coarse_action_profile = None
            self._current_coarse_state = None
            self._current_sub_policy = None

    def reset(self, world_model: Any = None) -> None:
        """Reset control state at the start of an episode.

        Args:
            world_model: Optionally update the hierarchical model reference.
        """
        self._current_coarse_action_profile = None
        self._current_coarse_state = None
        self._current_sub_policy = None
        if world_model is not None and isinstance(world_model, HierarchicalWorldModel):
            self.hierarchical_model = world_model

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_coarse_action_profile(self) -> Optional[Tuple[int, ...]]:
        """The currently active macro-level action profile, or None."""
        return self._current_coarse_action_profile

    @property
    def at_macro_level(self) -> bool:
        """True if control is at the macro level (no active coarse action)."""
        return self._current_coarse_action_profile is None

    # ------------------------------------------------------------------
    # Sub-problem computation
    # ------------------------------------------------------------------

    def _compute_sub_policy(
        self,
        coarse_action_profile: Tuple[int, ...],
        micro_state: Any,
    ) -> RobotPolicy:
        """Compute a micro-level sub-problem policy on demand.

        No caching is performed across macro decisions — each sub-problem is
        solved fresh since it is unlikely to encounter the same sub-problem
        twice in practice.

        The sub-problem is a restricted DAG rooted at *micro_state* where:

        - **Feasible actions only**: only action profiles accepted by
          ``mapper.is_feasible()`` are expanded.
        - **Return-control edges**: transition edges where
          ``mapper.return_control()`` fires use the pre-computed macro-level
          ``V_r(σ^0(s^1))`` as the successor value.
        - **Discounting**: uses per-transition durations from M^1.

        Args:
            coarse_action_profile: The sampled macro action profile.
            micro_state: The current micro-state to start from.

        Returns:
            A ``RobotPolicy`` for the sub-problem.
        """
        from empo.hierarchical._sub_problem import build_sub_problem_dag

        mapper = self.hierarchical_model.mappers[0]
        micro_env = self.hierarchical_model.finest()

        # Build restricted sub-problem DAG
        sub_states, _, sub_transitions, sub_terminal_mask = (
            build_sub_problem_dag(
                micro_env=micro_env,
                mapper=mapper,
                coarse_action_profile=coarse_action_profile,
                root_state=micro_state,
                quiet=self.quiet,
            )
        )

        if not sub_transitions[0]:
            # Edge case: root state has no feasible outgoing transitions
            # (e.g., return_control fires on all successors immediately).
            # Return control to macro immediately so the policy doesn't
            # get stuck at the micro level.
            self._current_coarse_action_profile = None
            self._current_coarse_state = None
            self._current_sub_policy = None
            return self._trivial_policy(micro_env)

        # Collect state indices that need macro V_r:
        # 1. States in terminal_mask (reached only via return-control)
        # 2. Any state that is a successor on a return-control edge
        rc_state_indices: set = set()
        for state_trans in sub_transitions:
            for _, _, succ_indices, rc_flags in state_trans:
                for si, rc in zip(succ_indices, rc_flags):
                    if rc:
                        rc_state_indices.add(si)

        terminal_Vr_map: Dict[int, float] = {}
        for idx, is_term in enumerate(sub_terminal_mask):
            if is_term or idx in rc_state_indices:
                s = sub_states[idx]
                coarse_s = mapper.super_state(s)
                terminal_Vr_map[idx] = self.macro_Vr.get(
                    coarse_s, self.terminal_Vr
                )

        # Run Phase 2 backward induction on the sub-DAG
        sub_policy = self._backward_induction_on_sub_dag(
            micro_env,
            sub_states,
            sub_transitions,
            sub_terminal_mask,
            terminal_Vr_map,
        )
        return sub_policy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _backward_induction_on_sub_dag(
        self,
        micro_env: Any,
        states: List[Any],
        transitions: List[
            List[Tuple[Tuple[int, ...], List[float], List[int], List[bool]]]
        ],
        terminal_mask: List[bool],
        terminal_Vr_map: Dict[int, float],
    ) -> TabularRobotPolicy:
        """Run Phase 2 backward induction on a pre-built sub-problem DAG.

        This is a simplified, self-contained backward induction that operates
        on the sub-DAG data structures directly (no ``get_dag()`` call).

        Per-edge ``rc_flags`` (return-control flags) are used to select the
        correct successor value: macro-level ``terminal_Vr_map`` for
        return-control edges, and the sub-DAG–computed ``Vr_values`` for
        normal edges.

        Args:
            micro_env: The micro-level WorldModel.
            states: Sub-problem states in topological order.
            transitions: Per-state transition data with per-edge rc_flags.
            terminal_mask: Boolean per state — True for terminal sub-states.
            terminal_Vr_map: Terminal V_r overrides (from macro solve) for
                states reachable via return-control edges.

        Returns:
            TabularRobotPolicy for the sub-problem.
        """
        from itertools import product as itertools_product
        from scipy.special import logsumexp

        num_agents = len(micro_env.agents)
        num_actions = micro_env.action_space.n
        action_powers = num_actions ** np.arange(num_agents)

        robot_action_profiles = [
            tuple(a)
            for a in itertools_product(
                range(num_actions), repeat=len(self.robot_agent_indices)
            )
        ]

        # Initialise value arrays
        n = len(states)
        Vr_values = np.zeros(n)
        Vh_values: List[List[Dict]] = [[{} for _ in range(num_agents)] for _ in range(n)]
        robot_policy_dict: Dict[Any, Dict[Tuple[int, ...], float]] = {}

        # Resolve discount rates
        rho_h = self.rho_h
        rho_r = self.rho_r
        gamma_h = self.gamma_h
        beta_r = self.beta_r
        zeta = self.zeta
        xi = self.xi
        eta = self.eta

        # Prepare human policy prior — use uniform if none provided
        hpp = self.micro_human_policy_prior
        if hpp is None:
            hpp = _UniformHumanPrior(
                self.human_agent_indices, num_actions
            )

        # Iterate in reverse topological order
        action_profile_arr = np.zeros(num_agents, dtype=np.int64)

        for idx in range(n - 1, -1, -1):
            state = states[idx]
            state_trans = transitions[idx]

            if terminal_mask[idx] or not state_trans:
                # Terminal: use macro Vr or default
                Vr_values[idx] = terminal_Vr_map.get(idx, self.terminal_Vr)
                continue

            # ── Build per-state transition lookup table ────────
            # Maps encoded action_profile_index →
            #   (probs, succ_indices, rc_flags)
            # for O(1) access in inner loops below.
            ap_index_to_trans: Dict[
                int, Tuple[List[float], List[int], List[bool]]
            ] = {}
            for t_ap, t_probs, t_succs, t_rc in state_trans:
                t_ap_index = int(
                    sum(a * (num_actions ** i) for i, a in enumerate(t_ap))
                )
                ap_index_to_trans[t_ap_index] = (t_probs, t_succs, t_rc)

            # ── Q_r computation ────────────────────────────────
            Qr_values = np.zeros(len(robot_action_profiles))
            rap_feasible = np.zeros(len(robot_action_profiles), dtype=bool)
            duration_weights_per_rap = (
                np.zeros(len(robot_action_profiles)) if rho_r > 0.0 else None
            )

            h_dist = list(hpp.profile_distribution(state))
            for rap_idx, rap in enumerate(robot_action_profiles):
                action_profile_arr[self.robot_agent_indices] = rap
                v = 0.0
                dw = 0.0
                for h_prob, h_profile in h_dist:
                    action_profile_arr[self.human_agent_indices] = h_profile
                    ap_index = int(
                        (action_profile_arr * action_powers).sum()
                    )
                    entry = ap_index_to_trans.get(ap_index)
                    if entry is not None:
                        t_probs, t_succs, t_rc = entry
                        rap_feasible[rap_idx] = True
                        probs_arr = np.array(t_probs)
                        # Per-edge V_r: use terminal_Vr for RC edges
                        succ_vr = np.array([
                            terminal_Vr_map.get(si, self.terminal_Vr)
                            if rc else Vr_values[si]
                            for si, rc in zip(t_succs, t_rc)
                        ])
                        if rho_r > 0.0:
                            trans_list = [
                                (float(p), states[si])
                                for p, si in zip(t_probs, t_succs)
                            ]
                            durations = np.array(
                                micro_env.transition_durations(
                                    state,
                                    action_profile_arr.tolist(),
                                    trans_list,
                                )
                            )
                            disc_r = np.exp(-rho_r * durations)
                            v += h_prob * float(
                                np.dot(probs_arr, disc_r * succ_vr)
                            )
                            dw_factors = -np.expm1(-rho_r * durations) / rho_r
                            dw += h_prob * float(
                                np.dot(probs_arr, dw_factors)
                            )
                        else:
                            v += h_prob * float(
                                np.dot(probs_arr, succ_vr)
                            )
                Qr_values[rap_idx] = v
                if rho_r > 0.0:
                    duration_weights_per_rap[rap_idx] = dw

            # ── Robot policy: power-law distribution ───────────
            # Only feasible robot action profiles participate in the
            # distribution; infeasible ones get zero probability.
            # Clamp Q_r to ensure all values are ≤ terminal_Vr < 0;
            # the power-law formula π_r(a) ∝ (-Q_r(a))^{-β_r} requires
            # strictly negative Q_r to avoid log(0) = -inf.
            if not rap_feasible.any():
                # Degenerate: no RAP had any matching transition;
                # assign uniform over all profiles.
                ps = np.ones(len(robot_action_profiles)) / len(robot_action_profiles)
            else:
                Qr_values = np.minimum(Qr_values, self.terminal_Vr)
                log_neg_Qr = np.log(-Qr_values)
                log_powers = -beta_r * log_neg_Qr
                # Mask infeasible RAPs so they get zero probability
                log_powers[~rap_feasible] = -np.inf
                log_norm = logsumexp(log_powers)
                ps = np.exp(log_powers - log_norm)
            robot_policy_dict[state] = {
                rap: ps[i] for i, rap in enumerate(robot_action_profiles)
            }

            # ── V_h^e, X_h, U_r, V_r ─────────────────────────
            powersum = 0.0
            for agent_index in self.human_agent_indices:
                vh_agent: Dict = {}
                Vh_values[idx][agent_index] = vh_agent
                xh = 0.0

                if self.micro_goal_generator is None:
                    continue  # Skip goal computation if no generator

                for goal, gw in self.micro_goal_generator.generate(
                    state, agent_index
                ):
                    h_goal_dist = list(
                        hpp.profile_distribution_with_fixed_goal(
                            state, agent_index, goal
                        )
                    )
                    vh = 0.0
                    for rap_idx, rap in enumerate(robot_action_profiles):
                        action_profile_arr[self.robot_agent_indices] = rap
                        v = 0.0
                        for h_prob, h_profile in h_goal_dist:
                            action_profile_arr[self.human_agent_indices] = (
                                h_profile
                            )
                            ap_index = int(
                                (action_profile_arr * action_powers).sum()
                            )
                            entry = ap_index_to_trans.get(ap_index)
                            if entry is not None:
                                t_probs, t_succs, t_rc = entry
                                probs_arr = np.array(t_probs)
                                att = np.array(
                                    [
                                        goal.is_achieved(states[si])
                                        for si in t_succs
                                    ]
                                )
                                # For RC edges, V_h^e defaults to 0
                                # (sub-problem ends, no further tracking)
                                vhe_succ = np.array(
                                    [
                                        0.0 if rc
                                        else Vh_values[si][
                                            agent_index
                                        ].get(goal, 0.0)
                                        for si, rc in zip(
                                            t_succs, t_rc
                                        )
                                    ]
                                )
                                if rho_h > 0.0:
                                    trans_list = [
                                        (float(p), states[si])
                                        for p, si in zip(
                                            t_probs, t_succs
                                        )
                                    ]
                                    durations = np.array(
                                        micro_env.transition_durations(
                                            state,
                                            action_profile_arr.tolist(),
                                            trans_list,
                                        )
                                    )
                                    disc_h = np.exp(-rho_h * durations)
                                    vals = np.where(
                                        att, 1.0, disc_h * vhe_succ
                                    )
                                else:
                                    vals = np.where(
                                        att, 1.0, gamma_h * vhe_succ
                                    )
                                v += h_prob * float(
                                    np.dot(probs_arr, vals)
                                )
                        vh += ps[rap_idx] * v
                    if vh != 0.0:
                        vh_agent[goal] = float(vh)
                    xh += gw * vh ** zeta

                if xh > 0:
                    powersum += xh ** (-xi)

            if powersum > 0:
                y = powersum / len(self.human_agent_indices)
                ur = -(y ** eta)
            else:
                ur = 0.0

            if rho_r > 0.0:
                edw = float(np.dot(ps, duration_weights_per_rap))
                Vr_values[idx] = edw * ur + float(np.dot(ps, Qr_values))
            else:
                Vr_values[idx] = ur + float(np.dot(ps, Qr_values))

        return TabularRobotPolicy(
            micro_env, list(self.robot_agent_indices), robot_policy_dict
        )

    def _trivial_policy(self, micro_env: Any) -> RobotPolicy:
        """Return a policy for degenerate sub-problems.

        Produces a policy whose ``sample()`` always returns a safe non-moving
        action for each robot agent (``still`` if available, else ``left``).
        """
        still = getattr(micro_env.actions, 'still', None)
        if still is None:
            left = getattr(micro_env.actions, 'left', None)
            still = left if left is not None else 0
        default_profile = tuple(still for _ in self.robot_agent_indices)

        class _ConstantPolicy(RobotPolicy):
            """Policy that always returns a fixed action profile."""

            def sample(self, state: Any) -> Tuple[int, ...]:
                return default_profile

        return _ConstantPolicy()


class _UniformHumanPrior:
    """Minimal uniform human policy prior for sub-problem backward induction.

    Assigns equal probability to every joint action profile of the human
    agents.  Used as a fallback when no ``TabularHumanPolicyPrior`` is
    provided for the micro level.
    """

    def __init__(self, human_agent_indices: List[int], num_actions: int):
        from itertools import product as itertools_product

        self._human_agent_indices = human_agent_indices
        self._num_actions = num_actions

        self._profiles = [
            tuple(a)
            for a in itertools_product(
                range(num_actions), repeat=len(human_agent_indices)
            )
        ]
        self._prob = 1.0 / len(self._profiles) if self._profiles else 1.0

    def profile_distribution(self, state: Any):
        """Yield (probability, human_action_profile) for all profiles."""
        for p in self._profiles:
            yield self._prob, p

    def profile_distribution_with_fixed_goal(self, state: Any, agent_index: int, goal: Any):
        """Same as profile_distribution (uniform ignores goals)."""
        return self.profile_distribution(state)
