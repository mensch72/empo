"""
Tools WorldModel — Exchange and use of tools in a shared workshop.

This module implements a multi-agent WorldModel where agents exchange tools
through a workshop with spatial constraints on communication and reachability.

State:
    Constant (stored on the environment object):
        - can_hear[i,j]: directed graph of audibility (Waxman random graph)
        - can_reach[i,j]: directed graph of physical reachability (Waxman, faster decay)
        - can_grab[i,j]: directed subgraph of can_reach for grabbing from hands

    Mutable (hashable state tuple):
        - remaining_steps: int
        - has_on_workbench[i,k]: bool matrix — which tools are on whose workbench
        - holds[i,k]: bool matrix — which agent holds which tool
        - has_requested[i,k]: bool matrix — which agent has requested which tool

Actions per agent i:
    Agent i has ``n_tools + n_give_targets(i)`` actions where
    ``n_give_targets(i) = sum(can_reach[i, :])``.

    0..m-1                 : acquire tool k — if already holding k do nothing;
                             else if reachable/grabbable take it; otherwise request it.
    m..m+|give_targets|-1  : give held tool to agent give_targets[j]
                             (only agents j with can_reach[i, j])

Transition dynamics:
    With probability (1 - p_failure) all actions are attempted.
    With probability p_failure, one agent (drawn from p_fail[i]) has their
    action fail.  Actions are resolved in order of agent index (lowest first);
    later agents whose actions conflict with earlier ones simply fail.
    This keeps the number of successor states ≤ n + 1.

Goals:
    - HoldGoal(i, k): agent i holds tool k
    - WorkbenchGoal(i, k): tool k is on agent i's workbench
    - IdleGoal(i): agent i is not holding any tool (hands free)

The goal set is designed so that in every reachable state, at least one
goal per agent is already attained (required by Phase 2 backward induction).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple
import random as _stdlib_random

import gymnasium as gym
import numpy as np

from empo.human_policy_prior import HumanPolicyPrior
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler
from empo.world_model import WorldModel

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------
ACTION_PASS = 0  # kept for backward-compat references; acquire(0) is the new "idle"


def _action_acquire(k: int) -> int:
    """Action index for 'acquire tool k'."""
    return k


def _action_give(agent_idx: int, give_targets: List[int]) -> int:
    """Action index for 'give to agent j' (relative to *give_targets* list).

    ``give_targets`` is the precomputed list of agent indices this agent can
    reach (``env.give_targets[i]``).  Returns
    ``n_tools + give_targets.index(agent_idx)``.

    Note: ``n_tools`` is not passed here; the caller must know that the
    give_targets list starts right after the acquire actions.  Use
    ``_action_give_idx`` for the raw positional variant.
    """
    # We don't know n_tools here; this helper is mainly used in tests.
    # The give actions start at index n_tools, and give_targets.index gives
    # the offset.  However, since n_tools is not available, we store the
    # offset only and the caller adds n_tools.  For backward-compat with
    # existing test calls like _action_give(1, env.give_targets[0]) where
    # the caller expects the *absolute* action index, we cannot compute it
    # without n_tools.  So raise if called directly — tests should use
    # _action_give_idx instead or pass n_tools explicitly.
    raise NotImplementedError(
        "_action_give cannot compute absolute action index without n_tools. "
        "Use _action_give_idx(give_targets.index(agent_idx), n_tools) instead."
    )


def _action_give_idx(j_pos: int, n_tools: int) -> int:
    """Action index for give at position *j_pos* (0-based) in give_targets."""
    return n_tools + j_pos


def decode_action(action: int, n_tools: int, give_targets: List[int]):
    """Decode an action index into (type_str, param).

    Returns one of:
        ('acquire', k)     — tool index k
        ('give', j)        — target agent index j (absolute)
    """
    if 0 <= action < n_tools:
        return "acquire", action
    pos = action - n_tools
    if 0 <= pos < len(give_targets):
        return "give", give_targets[pos]
    raise ValueError(
        f"Invalid action index {action} for n_tools={n_tools}, "
        f"give_targets={give_targets}"
    )


def action_name(action: int, n_tools: int, give_targets: List[int]) -> str:
    """Human-readable name for an action index."""
    atype, param = decode_action(action, n_tools, give_targets)
    if atype == "acquire":
        return f"acq T{param}"
    if atype == "give":
        return f"give→A{param}"
    return "?"


# ---------------------------------------------------------------------------
# ToolsWorldModel
# ---------------------------------------------------------------------------
class ToolsWorldModel(WorldModel):
    """WorldModel for tool exchange in a shared workshop.

    Overrides ``step()`` and ``reset()`` to follow the ``Discrete(1)``
    observation-space contract (returns dummy observation ``0``; use
    ``get_state()`` for the full state tuple).  Transition sampling in
    ``step()`` uses the environment RNG (``self._rng``) so that
    ``reset(seed=...)`` makes rollouts reproducible.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # ----- construction -----
    def __init__(
        self,
        n_agents: int = 4,
        n_tools: int = 6,
        max_steps: int = 30,
        p_failure: float = 0.1,
        p_fail: Optional[np.ndarray] = None,
        robot_agent_indices: Optional[List[int]] = None,
        human_agent_indices_list: Optional[List[int]] = None,
        seed: Optional[int] = None,
        # Waxman parameters (defaults for human agents)
        waxman_hear_alpha: float = 0.8,
        waxman_hear_beta: float = 0.5,
        waxman_reach_alpha: float = 0.6,
        waxman_reach_beta: float = 0.3,
        grab_prob: float = 0.5,
        # Robot Waxman overrides (None = use human defaults)
        robot_waxman_hear_alpha: Optional[float] = None,
        robot_waxman_hear_beta: Optional[float] = None,
        robot_waxman_reach_alpha: Optional[float] = None,
        robot_waxman_reach_beta: Optional[float] = None,
        robot_grab_prob: Optional[float] = None,
        # Optional pre-specified constant state
        agent_positions: Optional[np.ndarray] = None,
        can_hear: Optional[np.ndarray] = None,
        can_reach: Optional[np.ndarray] = None,
        can_grab: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.n_tools = n_tools
        self.max_steps = max_steps
        self.p_failure = p_failure
        self.render_mode = render_mode

        # Agent roles
        self._robot_indices: List[int] = (
            robot_agent_indices if robot_agent_indices is not None else [0]
        )
        self._human_indices: List[int] = (
            human_agent_indices_list
            if human_agent_indices_list is not None
            else [i for i in range(n_agents) if i not in self._robot_indices]
        )

        # Validate agent index assignments
        provided_robot = robot_agent_indices is not None
        provided_human = human_agent_indices_list is not None

        def _validate_role_indices(name: str, indices: List[int]) -> None:
            if not indices:
                raise ValueError(f"{name} must contain at least one agent index")
            for idx in indices:
                if not isinstance(idx, int):
                    raise ValueError(
                        f"{name} entries must be integers in [0, {n_agents - 1}], got {idx!r}"
                    )
                if idx < 0 or idx >= n_agents:
                    raise ValueError(
                        f"{name} entries must be in [0, {n_agents - 1}], got {idx}"
                    )
            if len(set(indices)) != len(indices):
                raise ValueError(f"{name} must not contain duplicate agent indices")

        _validate_role_indices("robot_agent_indices", self._robot_indices)
        _validate_role_indices("human_agent_indices_list", self._human_indices)

        # If both roles were explicitly provided, require disjointness and full coverage.
        if provided_robot and provided_human:
            robot_set = set(self._robot_indices)
            human_set = set(self._human_indices)
            overlap = robot_set & human_set
            if overlap:
                raise ValueError(
                    "robot_agent_indices and human_agent_indices_list must be disjoint; "
                    f"overlapping indices: {sorted(overlap)}"
                )
            all_agents = set(range(n_agents))
            if robot_set | human_set != all_agents:
                raise ValueError(
                    "When both robot_agent_indices and human_agent_indices_list are provided, "
                    "their union must cover all agents "
                    f"(expected {sorted(all_agents)}, got "
                    f"robots={sorted(robot_set)}, humans={sorted(human_set)})"
                )

        # Validate and normalise failure probability
        if not 0.0 <= p_failure <= 1.0:
            raise ValueError(f"p_failure must be in [0, 1], got {p_failure}")

        if p_fail is not None:
            p_fail = np.asarray(p_fail, dtype=np.float64)
            if len(p_fail) != n_agents:
                raise ValueError(f"p_fail length {len(p_fail)} != n_agents {n_agents}")
            if (p_fail < 0).any():
                raise ValueError("p_fail entries must be non-negative")
            total = p_fail.sum()
            if total <= 0:
                raise ValueError("p_fail must have positive sum")
            self.p_fail = p_fail / total
        else:
            self.p_fail = np.ones(n_agents, dtype=np.float64) / n_agents

        # Action space (per-agent, computed after graphs are built)
        # Placeholder — will be set after can_reach is known.
        self.observation_space = gym.spaces.Discrete(1)

        # Required by WorldModel / DAG builder
        self.agents = list(range(n_agents))

        # Store construction parameters for reproducibility / pickling
        self._init_seed = seed
        self._waxman_params = dict(
            hear_alpha=waxman_hear_alpha,
            hear_beta=waxman_hear_beta,
            reach_alpha=waxman_reach_alpha,
            reach_beta=waxman_reach_beta,
            grab_prob=grab_prob,
        )
        self._robot_waxman_params = dict(
            hear_alpha=robot_waxman_hear_alpha,
            hear_beta=robot_waxman_hear_beta,
            reach_alpha=robot_waxman_reach_alpha,
            reach_beta=robot_waxman_reach_beta,
            grab_prob=robot_grab_prob,
        )

        # RNG for graph / tool initialisation
        self._rng = np.random.RandomState(seed)

        # --- constant state (graphs) ---
        if agent_positions is not None:
            self.agent_positions = np.array(agent_positions, dtype=np.float64)
            if self.agent_positions.shape != (n_agents, 2):
                raise ValueError(
                    f"agent_positions shape {self.agent_positions.shape} "
                    f"!= expected ({n_agents}, 2)"
                )
        else:
            self.agent_positions = self._rng.rand(n_agents, 2)
            # Place robot(s) at centre
            for ri in self._robot_indices:
                self.agent_positions[ri] = [0.5, 0.5]

        expected_shape = (n_agents, n_agents)
        robot_set = set(self._robot_indices)

        # Effective Waxman params per source agent (robot overrides if given)
        def _ha(i: int) -> float:
            if i in robot_set and robot_waxman_hear_alpha is not None:
                return robot_waxman_hear_alpha
            return waxman_hear_alpha

        def _hb(i: int) -> float:
            if i in robot_set and robot_waxman_hear_beta is not None:
                return robot_waxman_hear_beta
            return waxman_hear_beta

        def _ra(i: int) -> float:
            if i in robot_set and robot_waxman_reach_alpha is not None:
                return robot_waxman_reach_alpha
            return waxman_reach_alpha

        def _rb(i: int) -> float:
            if i in robot_set and robot_waxman_reach_beta is not None:
                return robot_waxman_reach_beta
            return waxman_reach_beta

        def _gp(i: int) -> float:
            if i in robot_set and robot_grab_prob is not None:
                return robot_grab_prob
            return grab_prob

        self.can_hear = (
            np.array(can_hear, dtype=bool)
            if can_hear is not None
            else self._waxman_graph_per_agent(_ha, _hb)
        )
        if self.can_hear.shape != expected_shape:
            raise ValueError(
                f"can_hear shape {self.can_hear.shape} != expected {expected_shape}"
            )
        np.fill_diagonal(self.can_hear, True)

        self.can_reach = (
            np.array(can_reach, dtype=bool)
            if can_reach is not None
            else self._waxman_graph_per_agent(_ra, _rb)
        )
        if self.can_reach.shape != expected_shape:
            raise ValueError(
                f"can_reach shape {self.can_reach.shape} != expected {expected_shape}"
            )
        np.fill_diagonal(self.can_reach, True)

        if can_grab is not None:
            self.can_grab = np.array(can_grab, dtype=bool)
        else:
            self.can_grab = self.can_reach.copy()
            for i in range(n_agents):
                for j in range(n_agents):
                    if i != j and self.can_grab[i, j]:
                        if self._rng.rand() >= _gp(i):
                            self.can_grab[i, j] = False
        if self.can_grab.shape != expected_shape:
            raise ValueError(
                f"can_grab shape {self.can_grab.shape} != expected {expected_shape}"
            )
        np.fill_diagonal(self.can_grab, True)

        # Per-agent give targets (agents reachable by agent i)
        self.give_targets: List[List[int]] = [
            [j for j in range(n_agents) if self.can_reach[i, j]]
            for i in range(n_agents)
        ]

        # Per-agent action counts: acquire(m) + give(|give_targets[i]|)
        self._n_actions_per_agent: List[int] = [
            n_tools + len(self.give_targets[i]) for i in range(n_agents)
        ]

        # action_space is set to the max for Gymnasium compatibility
        self.n_actions = max(self._n_actions_per_agent)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # --- mutable state ---
        self._remaining: int = max_steps
        self._workbench = np.zeros((n_agents, n_tools), dtype=bool)
        self._holds = np.zeros((n_agents, n_tools), dtype=bool)
        self._requested = np.zeros((n_agents, n_tools), dtype=bool)
        self._init_tools()

    # ----- helpers for construction -----
    def _waxman_graph_per_agent(
        self,
        alpha_fn,
        beta_fn,
    ) -> np.ndarray:
        """Generate a directed Waxman-style connectivity graph.

        ``alpha_fn(i)`` and ``beta_fn(i)`` return per-source-agent parameters.
        """
        n = self.n_agents
        pos = self.agent_positions
        L = np.sqrt(2.0)
        g = np.eye(n, dtype=bool)
        for i in range(n):
            a = float(np.clip(alpha_fn(i), 0.0, 1.0))
            b = beta_fn(i)
            if b <= 0.0:
                raise ValueError(f"waxman beta must be positive; got {b!r}")
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    p = a * np.exp(-d / (b * L))
                    p = float(np.clip(p, 0.0, 1.0))
                    if self._rng.rand() < p:
                        g[i, j] = True
        return g

    def _init_tools(self):
        """Clear all mutable state and place each tool on a random workbench."""
        self._workbench[:] = False
        self._holds[:] = False
        self._requested[:] = False
        for k in range(self.n_tools):
            owner = self._rng.randint(self.n_agents)
            self._workbench[owner, k] = True

    # ----- WorldModel interface -----
    @property
    def n_actions_per_agent(self) -> List[int]:
        return self._n_actions_per_agent

    @property
    def human_agent_indices(self) -> List[int]:
        return self._human_indices

    @property
    def robot_agent_indices(self) -> List[int]:
        return self._robot_indices

    def reset(self, *, seed=None, options=None):
        """Reset the environment.

        Uses Gymnasium keyword-only signature for compatibility with
        wrappers.  ``seed`` re-seeds the internal RNG; ``options`` is
        accepted but unused.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._remaining = self.max_steps
        self._init_tools()
        self._dag_cache = None
        return 0, {}

    def step(self, actions):
        """Apply actions and return ``(obs, reward, done, trunc, info)``.

        The observation is always ``0`` (matching ``observation_space =
        Discrete(1)``).  Use :meth:`get_state` for the full state tuple.
        """
        if not isinstance(actions, (list, tuple)):
            actions = [actions]
        current_state = self.get_state()
        transitions = self.transition_probabilities(current_state, list(actions))
        if transitions is None:
            return 0, 0.0, True, False, {}
        probabilities = [prob for prob, _ in transitions]
        successor_states = [state for _, state in transitions]
        chosen_idx = self._rng.choice(len(transitions), p=probabilities)
        self.set_state(successor_states[chosen_idx])
        terminated = self.is_terminal(successor_states[chosen_idx])
        return 0, 0.0, terminated, False, {}

    def get_state(self) -> Hashable:
        return (
            self._remaining,
            tuple(tuple(int(x) for x in row) for row in self._workbench),
            tuple(tuple(int(x) for x in row) for row in self._holds),
            tuple(tuple(int(x) for x in row) for row in self._requested),
        )

    def set_state(self, state) -> None:
        self._remaining = state[0]
        self._workbench = np.array(state[1], dtype=bool)
        self._holds = np.array(state[2], dtype=bool)
        self._requested = np.array(state[3], dtype=bool)

    def perceived_state(self, state, agent_index: int):
        """Perceived state masks ``has_requested`` entries the agent cannot hear."""
        if not 0 <= agent_index < self.n_agents:
            raise ValueError(
                f"agent_index {agent_index} must be between 0 and "
                f"{self.n_agents - 1} (inclusive)"
            )
        remaining, workbench, holds, requested = state
        m = self.n_tools
        new_req: list = []
        for j in range(self.n_agents):
            if self.can_hear[agent_index, j]:
                new_req.append(requested[j])
            else:
                new_req.append(tuple(0 for _ in range(m)))
        return (remaining, workbench, holds, tuple(new_req))

    # ----- action feasibility (operates on tuple state components) -----
    def is_feasible(self, action: int, agent: int, workbench, holds) -> bool:
        """Check whether *action* is feasible for *agent* given state arrays.

        ``workbench`` and ``holds`` can be tuples-of-tuples **or** lists-of-lists.
        ``requested`` is not needed for feasibility checks.
        """
        gt = self.give_targets[agent]
        atype, param = decode_action(action, self.n_tools, gt)
        if atype == "acquire":
            return True  # acquire is always a valid choice (may be no-op)
        if atype == "give":
            return any(holds[agent])
        return False

    # ----- apply a single agent's action (mutates lists in-place) -----
    @staticmethod
    def _apply(
        agent: int,
        action_type: str,
        param: int,
        n_tools: int,
        n_agents: int,
        can_reach: np.ndarray,
        can_grab: np.ndarray,
        wb: List[List[int]],
        hd: List[List[int]],
        rq: List[List[int]],
    ) -> bool:
        """Apply one agent's action in-place; return True if it succeeded."""

        if action_type == "acquire":
            k = param
            # Already holding → no-op (success)
            if hd[agent][k]:
                return True
            # Try to take from reachable workbench or grabbable hand
            source, stype = None, None
            for j in range(n_agents):
                if can_reach[agent, j] and wb[j][k]:
                    source, stype = j, "wb"
                    break
            if source is None:
                for j in range(n_agents):
                    if can_grab[agent, j] and hd[j][k]:
                        source, stype = j, "hd"
                        break
            if source is not None:
                # take path: put down current tool, grab k
                for k2 in range(n_tools):
                    if hd[agent][k2]:
                        hd[agent][k2] = 0
                        wb[agent][k2] = 1
                        break
                if stype == "wb":
                    wb[source][k] = 0
                else:
                    hd[source][k] = 0
                hd[agent][k] = 1
                rq[agent][k] = 0
                return True
            # Cannot take → request path
            if hd[agent][k] or wb[agent][k]:
                return False  # already have it on own workbench → no request
            # cancel previous request and set new one
            for k2 in range(n_tools):
                rq[agent][k2] = 0
            rq[agent][k] = 1
            return True

        if action_type == "give":
            j = param
            if not can_reach[agent, j]:
                return False
            held = None
            for k in range(n_tools):
                if hd[agent][k]:
                    held = k
                    break
            if held is None:
                return False
            hd[agent][held] = 0
            wb[j][held] = 1
            rq[j][held] = 0
            return True

        return False

    # ----- transition probabilities -----
    def transition_probabilities(self, state, actions):
        remaining = state[0]
        if remaining <= 0:
            return None

        n = self.n_agents
        m = self.n_tools

        if len(actions) != n:
            raise ValueError(
                f"Expected {n} actions (one per agent), got {len(actions)}"
            )

        results: Dict[Any, float] = {}

        for fail_idx in range(n + 1):
            if fail_idx == 0:
                prob = 1.0 - self.p_failure
                failed = -1
            else:
                prob = self.p_failure * self.p_fail[fail_idx - 1]
                failed = fail_idx - 1
            if prob <= 0.0:
                continue

            # mutable copies
            wb = [list(row) for row in state[1]]
            hd = [list(row) for row in state[2]]
            rq = [list(row) for row in state[3]]

            claimed_tools: set = set()
            for i in range(n):
                if i == failed:
                    continue
                gt = self.give_targets[i]
                atype, param = decode_action(actions[i], m, gt)
                if atype == "acquire":
                    if param in claimed_tools:
                        pass  # tool already claimed this timestep
                    else:
                        ok = self._apply(
                            i, atype, param, m, n,
                            self.can_reach, self.can_grab, wb, hd, rq,
                        )
                        if ok and hd[i][param]:
                            claimed_tools.add(param)
                elif atype == "give":
                    self._apply(
                        i, atype, param, m, n,
                        self.can_reach, self.can_grab, wb, hd, rq,
                    )

            new_state = (
                remaining - 1,
                tuple(tuple(r) for r in wb),
                tuple(tuple(r) for r in hd),
                tuple(tuple(r) for r in rq),
            )
            results[new_state] = results.get(new_state, 0.0) + prob

        if not results:
            return None
        return [(p, s) for s, p in results.items()]

    # ----- reconstruction helpers (for parallel DAG) -----
    def _get_construction_kwargs(self) -> dict:
        rwp = self._robot_waxman_params
        return dict(
            n_agents=self.n_agents,
            n_tools=self.n_tools,
            max_steps=self.max_steps,
            p_failure=self.p_failure,
            p_fail=self.p_fail,
            robot_agent_indices=self._robot_indices,
            human_agent_indices_list=self._human_indices,
            seed=self._init_seed,
            agent_positions=self.agent_positions,
            can_hear=self.can_hear,
            can_reach=self.can_reach,
            can_grab=self.can_grab,
            render_mode=self.render_mode,
            robot_waxman_hear_alpha=rwp["hear_alpha"],
            robot_waxman_hear_beta=rwp["hear_beta"],
            robot_waxman_reach_alpha=rwp["reach_alpha"],
            robot_waxman_reach_beta=rwp["reach_beta"],
            robot_grab_prob=rwp["grab_prob"],
        )

    # ----- rendering -----
    def render(self, goals=None):
        """Render current state using matplotlib.

        Args:
            goals: Optional list of PossibleGoal objects to visualise.

        Returns:
            numpy RGB array (H, W, 3) when ``render_mode == "rgb_array"``,
            else ``None``.
        """
        if self.render_mode == "rgb_array":
            return render_tools_state(self, goals=goals)
        return None


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
class HoldGoal(PossibleGoal):
    """Goal: agent *agent_idx* holds tool *tool_idx*."""

    def __init__(self, env, agent_idx: int, tool_idx: int, index: Optional[int] = None):
        super().__init__(env, index=index)
        self.agent_idx = agent_idx
        self.tool_idx = tool_idx
        self._hash = hash((0, agent_idx, tool_idx))
        super()._freeze()

    def is_achieved(self, state) -> int:
        _remaining, _wb, holds, _rq = state
        return int(holds[self.agent_idx][self.tool_idx])

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, HoldGoal)
            and self.agent_idx == other.agent_idx
            and self.tool_idx == other.tool_idx
        )

    def __repr__(self):
        return f"HoldGoal(A{self.agent_idx}, T{self.tool_idx})"


class WorkbenchGoal(PossibleGoal):
    """Goal: tool *tool_idx* is on agent *agent_idx*'s workbench."""

    def __init__(self, env, agent_idx: int, tool_idx: int, index: Optional[int] = None):
        super().__init__(env, index=index)
        self.agent_idx = agent_idx
        self.tool_idx = tool_idx
        self._hash = hash((1, agent_idx, tool_idx))
        super()._freeze()

    def is_achieved(self, state) -> int:
        _remaining, wb, _holds, _rq = state
        return int(wb[self.agent_idx][self.tool_idx])

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, WorkbenchGoal)
            and self.agent_idx == other.agent_idx
            and self.tool_idx == other.tool_idx
        )

    def __repr__(self):
        return f"WorkbenchGoal(A{self.agent_idx}, T{self.tool_idx})"


class IdleGoal(PossibleGoal):
    """Goal: agent *agent_idx* is not holding any tool (hands free).

    This goal ensures that the set of possible goals covers the full state
    space: in every reachable state at least one goal per agent is already
    attained.  Without it, a state where the agent holds nothing and has an
    empty workbench would leave no goal achieved, violating the Phase 2
    backward-induction precondition.
    """

    def __init__(self, env, agent_idx: int, index: Optional[int] = None):
        super().__init__(env, index=index)
        self.agent_idx = agent_idx
        # Use int tag 2 to distinguish from HoldGoal(0,...) and WorkbenchGoal(1,...)
        self._hash = hash((2, agent_idx))
        super()._freeze()

    def is_achieved(self, state) -> int:
        _remaining, _wb, holds, _rq = state
        return int(not any(holds[self.agent_idx]))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, IdleGoal) and self.agent_idx == other.agent_idx

    def __repr__(self):
        return f"IdleGoal(A{self.agent_idx})"


# ---------------------------------------------------------------------------
# Goal generator / sampler
# ---------------------------------------------------------------------------
class ToolsGoalGenerator(PossibleGoalGenerator):
    """Enumerate all Hold, Workbench, and Idle goals for every human agent.

    The goal set is designed so that in every reachable state, at least one
    goal per agent is already attained:

    * Agent holds tool *k* → ``HoldGoal(i, k)`` achieved
    * Agent holds nothing, tool *k* on workbench → ``WorkbenchGoal(i, k)`` achieved
    * Agent holds nothing, empty workbench → ``IdleGoal(i)`` achieved
    """

    def __init__(self, env: ToolsWorldModel, indexed: bool = True):
        super().__init__(env, indexed=indexed)
        # Pre-build per-agent goal lists for O(1) generate() calls.
        self._agent_goals: Dict[int, List[PossibleGoal]] = {}
        idx = 0
        for i in env.human_agent_indices:
            gs: List[PossibleGoal] = []
            for k in range(env.n_tools):
                gs.append(HoldGoal(env, i, k, index=idx))
                idx += 1
                gs.append(WorkbenchGoal(env, i, k, index=idx))
                idx += 1
            gs.append(IdleGoal(env, i, index=idx))
            idx += 1
            self._agent_goals[i] = gs
        # Per-agent weight: 1/n_goals_per_agent so weights sum to 1 for each agent.
        # This is consistent with the sampler (uniform sampling, weight=1.0):
        #   generator_weight = sampler_weight * p(goal)  =>  1/n = 1.0 * (1/n)
        self._n_goals_per_agent = 2 * env.n_tools + 1

    def generate(
        self, state, human_agent_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        weight = 1.0 / self._n_goals_per_agent if self._n_goals_per_agent > 0 else 1.0
        for g in self._agent_goals.get(human_agent_index, []):
            yield g, weight


class ToolsGoalSampler(PossibleGoalSampler):
    """Uniformly sample a Hold, Workbench, or Idle goal for a given human agent."""

    def __init__(self, env: ToolsWorldModel, indexed: bool = True):
        super().__init__(env, indexed=indexed)
        # pre-build per-agent goal lists
        self._agent_goals: Dict[int, List[PossibleGoal]] = {}
        idx = 0
        for i in env.human_agent_indices:
            gs: List[PossibleGoal] = []
            for k in range(env.n_tools):
                gs.append(HoldGoal(env, i, k, index=idx))
                idx += 1
                gs.append(WorkbenchGoal(env, i, k, index=idx))
                idx += 1
            gs.append(IdleGoal(env, i, index=idx))
            idx += 1
            self._agent_goals[i] = gs

    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        """Sample a goal uniformly at random for *human_agent_index*.

        Returns ``(goal, 1.0)`` — the weight is 1.0 because the generator
        already accounts for the uniform distribution (weight = 1/n_goals).
        """
        gs = self._agent_goals[human_agent_index]
        g = _stdlib_random.choice(gs)
        return g, 1.0


# ---------------------------------------------------------------------------
# Heuristic human policy prior
# ---------------------------------------------------------------------------
class ToolsHeuristicPolicy(HumanPolicyPrior):
    """Goal-conditioned heuristic policy for the tools environment.

    Implements the strategy described in the issue for HoldGoal and WorkbenchGoal.
    Uses the acquire/give action encoding (no pass action).
    """

    def __init__(
        self,
        env: ToolsWorldModel,
        possible_goal_generator: ToolsGoalGenerator,
        beta: float = 5.0,
    ):
        super().__init__(env, env.human_agent_indices)
        self._goal_gen = possible_goal_generator
        if beta < 0 or np.isnan(beta):
            raise ValueError(f"beta must be non-negative (inf allowed), got {beta}")
        self._beta = beta

    # ------ shortest path on can_reach ------
    @staticmethod
    def _bfs_shortest(graph: np.ndarray, src: int, targets) -> Optional[List[int]]:
        """BFS shortest path from *src* to any node in *targets* over *graph*.

        Returns the path as a list of node indices (including src and target),
        or None if unreachable.
        """
        n = graph.shape[0]
        visited = [False] * n
        parent = [-1] * n
        visited[src] = True
        queue: deque = deque([src])
        target_set = set(targets)
        while queue:
            u = queue.popleft()
            if u in target_set and u != src:
                path = []
                c = u
                while c != -1:
                    path.append(c)
                    c = parent[c]
                return path[::-1]
            for v in range(n):
                if graph[u, v] and not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
        return None

    # ------ core policy logic ------
    def _action_distribution(
        self, state, agent_idx: int, goal: PossibleGoal
    ) -> np.ndarray:
        """Return softmax action distribution for *agent_idx* pursuing *goal*."""
        env = self.world_model
        n, m = env.n_agents, env.n_tools
        gt = env.give_targets[agent_idx]
        na = env._n_actions_per_agent[agent_idx]
        _remaining, wb, holds, _requested = state

        # decode perceived state for this agent
        p_state = env.perceived_state(state, agent_idx)
        _, _, _, p_req = p_state

        logits = np.full(na, -1e9)

        # Find a "no-op" action: acquire a tool we already hold
        noop_action = None
        for k in range(m):
            if holds[agent_idx][k]:
                noop_action = _action_acquire(k)
                break
        if noop_action is not None:
            logits[noop_action] = 0.0  # baseline

        if isinstance(goal, HoldGoal) and goal.agent_idx == agent_idx:
            k = goal.tool_idx
            # acquire(k) handles: no-op if held, take if reachable, request otherwise
            logits[_action_acquire(k)] = 5.0

            # If already requested k and holding something, help by giving
            if p_req[agent_idx][k] and any(holds[agent_idx]):
                self._give_along_path(agent_idx, holds, p_req, logits, n, m, gt)

            return _softmax(logits, self._beta)

        if isinstance(goal, WorkbenchGoal) and goal.agent_idx == agent_idx:
            k = goal.tool_idx
            # 1) already on own workbench → prefer no-op
            if wb[agent_idx][k]:
                if noop_action is not None:
                    logits[noop_action] = 10.0
                else:
                    logits[:] = 0.0  # no tool held, uniform
                return _softmax(logits, self._beta)

            # 2) holding k → give to self (puts on own workbench)
            if holds[agent_idx][k]:
                try:
                    j_pos = gt.index(agent_idx)
                    logits[_action_give_idx(j_pos, m)] = 5.0
                except ValueError:
                    pass
                return _softmax(logits, self._beta)

            # 3) don't have it → acquire it
            logits[_action_acquire(k)] = 5.0
            return _softmax(logits, self._beta)

        if isinstance(goal, IdleGoal) and goal.agent_idx == agent_idx:
            if not any(holds[agent_idx]):
                # Already idle → uniform
                logits[:] = 0.0
                return _softmax(logits, self._beta)
            # Holding something → give to self
            try:
                j_pos = gt.index(agent_idx)
                logits[_action_give_idx(j_pos, m)] = 5.0
            except ValueError:
                pass
            return _softmax(logits, self._beta)

        # goal does not apply to this agent → uniform over feasible actions
        for a in range(na):
            if env.is_feasible(a, agent_idx, wb, holds):
                logits[a] = 0.0
        return _softmax(logits, self._beta)

    def _give_along_path(
        self,
        agent_idx: int,
        holds,
        perceived_requested,
        logits: np.ndarray,
        n: int,
        m: int,
        give_targets: List[int],
    ):
        """Set logit for giving the currently-held tool toward a requester."""
        env = self.world_model
        held = None
        for k2 in range(m):
            if holds[agent_idx][k2]:
                held = k2
                break
        if held is None:
            return

        requesters = [j for j in range(n) if perceived_requested[j][held]]
        if requesters:
            target = min(requesters)
            path = self._bfs_shortest(env.can_reach, agent_idx, [target])
            if path and len(path) >= 2:
                next_hop = path[1]
                try:
                    j_pos = give_targets.index(next_hop)
                    logits[_action_give_idx(j_pos, m)] = 4.0
                except ValueError:
                    pass

    # ------ HumanPolicyPrior interface ------
    def __call__(self, state, human_agent_index: int, possible_goal=None):
        if possible_goal is not None:
            return self._action_distribution(state, human_agent_index, possible_goal)

        # marginalise over goals
        na = self.world_model._n_actions_per_agent[human_agent_index]
        dist = np.zeros(na, dtype=np.float64)
        count = 0
        for goal, weight in self._goal_gen.generate(state, human_agent_index):
            dist += weight * self._action_distribution(state, human_agent_index, goal)
            count += 1
        if count > 0:
            dist /= dist.sum() + 1e-30
        else:
            dist[:] = 1.0 / len(dist)
        return dist


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_TOOL_MARKERS = ["o", "s", "^", "D", "p", "h", "*", "P", "X", "v"]
_AGENT_COLORS = [
    "#999999",  # grey for robots
    "#e6194B",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
]


def _agent_color(idx: int, robot_indices: List[int]) -> str:
    if idx in robot_indices:
        return _AGENT_COLORS[0]
    return _AGENT_COLORS[(idx % (len(_AGENT_COLORS) - 1)) + 1]


def render_tools_state(env: ToolsWorldModel, goals=None, ax=None, figsize=(8, 8)):
    """Render the current state of a :class:`ToolsWorldModel`.

    Returns an RGB numpy array (H, W, 3) suitable for video frames.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Tools Workshop  (t_remaining={env._remaining})")
    ax.axis("off")

    pos = env.agent_positions
    n, m = env.n_agents, env.n_tools

    # ---- graph edges ----
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if env.can_grab[i, j]:
                ax.annotate(
                    "",
                    xy=pos[j],
                    xytext=pos[i],
                    arrowprops=dict(arrowstyle="->", color="grey", lw=2.0, alpha=0.35),
                )
            elif env.can_reach[i, j]:
                ax.annotate(
                    "",
                    xy=pos[j],
                    xytext=pos[i],
                    arrowprops=dict(arrowstyle="->", color="grey", lw=1.0, alpha=0.3),
                )
            elif env.can_hear[i, j]:
                ax.annotate(
                    "",
                    xy=pos[j],
                    xytext=pos[i],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="grey",
                        lw=0.5,
                        alpha=0.2,
                        linestyle="dotted",
                    ),
                )

    # ---- agents ----
    state = env.get_state()
    _rem, wb, holds, req = state

    for i in range(n):
        c = _agent_color(i, env._robot_indices)
        label = f"R{i}" if i in env._robot_indices else f"H{i}"
        ax.plot(pos[i, 0], pos[i, 1], "o", color=c, markersize=18, zorder=5)
        ax.text(
            pos[i, 0],
            pos[i, 1],
            label,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
            zorder=6,
        )

        # held tool overlay
        for k in range(m):
            if holds[i][k]:
                angle = 2 * np.pi * k / m - np.pi / 2
                dx, dy = 0.03 * np.cos(angle), 0.03 * np.sin(angle)
                marker = _TOOL_MARKERS[k % len(_TOOL_MARKERS)]
                ax.plot(
                    pos[i, 0] + dx,
                    pos[i, 1] + dy,
                    marker,
                    color="black",
                    markersize=10,
                    zorder=7,
                )
                ax.text(
                    pos[i, 0] + dx + 0.015,
                    pos[i, 1] + dy,
                    f"T{k}",
                    fontsize=6,
                    color="black",
                    zorder=7,
                )

        # workbench tools (small shapes around agent)
        for k in range(m):
            if wb[i][k]:
                angle = 2 * np.pi * k / m - np.pi / 2
                dx, dy = 0.06 * np.cos(angle), 0.06 * np.sin(angle)
                marker = _TOOL_MARKERS[k % len(_TOOL_MARKERS)]
                ax.plot(
                    pos[i, 0] + dx,
                    pos[i, 1] + dy,
                    marker,
                    color="dimgrey",
                    markersize=7,
                    zorder=4,
                )
                ax.text(
                    pos[i, 0] + dx + 0.015,
                    pos[i, 1] + dy,
                    f"T{k}",
                    fontsize=5,
                    color="dimgrey",
                    zorder=4,
                )

    # ---- has_requested arrows (solid blue, parallel to goal arcs) ----
    _REQ_OFFSET = 0.015  # perpendicular offset for parallel look
    for i in range(n):
        for k in range(m):
            if req[i][k]:
                tool_pos = _tool_render_pos(k, wb, holds, pos, n)
                if tool_pos is not None:
                    diff = tool_pos - pos[i]
                    length = np.linalg.norm(diff)
                    if length > 1e-9:
                        perp = np.array([-diff[1], diff[0]]) / length
                    else:
                        perp = np.array([0.0, 0.0])
                    off = _REQ_OFFSET * perp
                    ax.annotate(
                        "",
                        xy=tool_pos + off,
                        xytext=pos[i] + off,
                        arrowprops=dict(
                            arrowstyle="->",
                            color="royalblue",
                            lw=2.5,
                            connectionstyle="arc3,rad=-0.3",
                        ),
                    )

    # ---- goal arrows (dashed blue) ----
    if goals:
        for g in goals:
            if isinstance(g, (HoldGoal, WorkbenchGoal)):
                tool_pos = _tool_render_pos(g.tool_idx, wb, holds, pos, n)
                if tool_pos is not None:
                    ax.annotate(
                        "",
                        xy=tool_pos,
                        xytext=pos[g.agent_idx],
                        arrowprops=dict(
                            arrowstyle="->",
                            color="blue",
                            lw=3.0,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=-0.3",
                        ),
                    )
            elif isinstance(g, IdleGoal):
                # Draw a dashed blue circle around the agent to show "idle" goal
                circle = plt.Circle(
                    pos[g.agent_idx],
                    0.06,
                    fill=False,
                    edgecolor="blue",
                    linestyle="dashed",
                    linewidth=3.0,
                )
                ax.add_patch(circle)

    if own_fig:
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    return None


def _tool_render_pos(k, wb, holds, positions, n):
    """Find approximate render position of tool k.

    Must match the placement logic in :func:`render_tools_state`.
    Angle is constant per tool index; distance is short (held) or long (workbench).
    """
    m = len(wb[0])  # number of tools
    angle = 2 * np.pi * k / m - np.pi / 2
    for i in range(n):
        if holds[i][k]:
            dx, dy = 0.03 * np.cos(angle), 0.03 * np.sin(angle)
            return positions[i] + np.array([dx, dy])
        if wb[i][k]:
            dx, dy = 0.06 * np.cos(angle), 0.06 * np.sin(angle)
            return positions[i] + np.array([dx, dy])
    return None


def _tool_owner(k, wb, holds, n):
    """Return ``(agent_index, 'held'|'wb')`` for tool *k*, or ``(None, None)``."""
    for i in range(n):
        if holds[i][k]:
            return i, "held"
        if wb[i][k]:
            return i, "wb"
    return None, None


def _draw_partial_arc(ax, start, end, frac, rad=-0.2, color="royalblue",
                      lw=2.5, alpha=0.7):
    """Draw fraction *frac* of a quadratic Bezier arc from *start* to *end*.

    The control point is offset perpendicular to the start→end midpoint by
    ``rad * ||end - start||``, matching matplotlib's ``arc3,rad=`` semantics.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    diff = end - start
    length = np.linalg.norm(diff)
    if length < 1e-9 or frac <= 0:
        return
    mid = (start + end) / 2
    perp = np.array([diff[1], -diff[0]])  # 90° CW (matches matplotlib arc3)
    ctrl = mid + rad * perp  # control point (same convention as mpl)
    n_pts = max(int(50 * frac), 2)
    ts = np.linspace(0, frac, n_pts)
    pts = np.column_stack([
        (1 - ts)**2 * start[0] + 2 * (1 - ts) * ts * ctrl[0] + ts**2 * end[0],
        (1 - ts)**2 * start[1] + 2 * (1 - ts) * ts * ctrl[1] + ts**2 * end[1],
    ])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round")


def render_tools_transition(
    env: ToolsWorldModel,
    old_state,
    new_state,
    actions: List[int],
    goals=None,
    n_interp: int = 10,
    figsize=(8, 8),
) -> List[np.ndarray]:
    """Render interpolated frames showing tool motion between two states.

    For each moved tool a T-shaped indicator is drawn:

    * *give*: red line from the giver extending toward the tool with a
      T-bar just **before** the tool (a "pusher").
    * *take*: green line from the taker extending toward the tool with a
      T-bar just **behind** the tool.

    Returns a list of *n_interp* RGB numpy frames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = env.agent_positions
    n, m = env.n_agents, env.n_tools

    _rem_old, wb_old, holds_old, req_old = old_state
    _rem_new, wb_new, holds_new, req_new = new_state

    # Compute old / new render positions for each tool
    old_tool_pos = {}
    new_tool_pos = {}
    for k in range(m):
        op = _tool_render_pos(k, wb_old, holds_old, pos, n)
        np_ = _tool_render_pos(k, wb_new, holds_new, pos, n)
        if op is not None:
            old_tool_pos[k] = op
        if np_ is not None:
            new_tool_pos[k] = np_

    # Determine which tools moved and the cause
    moves: Dict[int, dict] = {}
    for k in range(m):
        if k not in old_tool_pos or k not in new_tool_pos:
            continue
        if not np.allclose(old_tool_pos[k], new_tool_pos[k], atol=1e-6):
            old_owner, _ = _tool_owner(k, wb_old, holds_old, n)
            move_type = None
            actor = None
            for i in range(n):
                atype, param = decode_action(actions[i], m, env.give_targets[i])
                if atype == "give" and old_owner == i:
                    move_type = "give"
                    actor = i
                    break
                if atype == "acquire" and param == k and holds_new[i][k]:
                    move_type = "take"
                    actor = i
                    break
            if move_type is None:
                new_owner, _ = _tool_owner(k, wb_new, holds_new, n)
                move_type = "take"
                actor = new_owner
            moves[k] = {
                "old": old_tool_pos[k],
                "new": new_tool_pos[k],
                "type": move_type,
                "actor": actor,
            }

    static_tools = set(range(m)) - set(moves.keys())

    frames: List[np.ndarray] = []
    for fi in range(n_interp):
        t = (fi + 1) / (n_interp + 1)  # 0 < t < 1

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"Tools Workshop  (t_remaining={_rem_old})")
        ax.axis("off")

        # ---- graph edges ----
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if env.can_grab[i, j]:
                    ax.annotate("", xy=pos[j], xytext=pos[i],
                                arrowprops=dict(arrowstyle="->", color="grey",
                                                lw=2.0, alpha=0.35))
                elif env.can_reach[i, j]:
                    ax.annotate("", xy=pos[j], xytext=pos[i],
                                arrowprops=dict(arrowstyle="->", color="grey",
                                                lw=1.0, alpha=0.3))
                elif env.can_hear[i, j]:
                    ax.annotate("", xy=pos[j], xytext=pos[i],
                                arrowprops=dict(arrowstyle="->", color="grey",
                                                lw=0.5, alpha=0.2,
                                                linestyle="dotted"))

        # ---- agents ----
        for i in range(n):
            c = _agent_color(i, env._robot_indices)
            label = f"R{i}" if i in env._robot_indices else f"H{i}"
            ax.plot(pos[i, 0], pos[i, 1], "o", color=c, markersize=18, zorder=5)
            ax.text(pos[i, 0], pos[i, 1], label, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=6)

        # ---- static tools (old state positions) ----
        for k in static_tools:
            if k in old_tool_pos:
                tp = old_tool_pos[k]
                _, kind = _tool_owner(k, wb_old, holds_old, n)
                marker = _TOOL_MARKERS[k % len(_TOOL_MARKERS)]
                color = "black" if kind == "held" else "dimgrey"
                msize = 10 if kind == "held" else 7
                zord = 7 if kind == "held" else 4
                ax.plot(tp[0], tp[1], marker, color=color,
                        markersize=msize, zorder=zord)
                ax.text(tp[0] + 0.015, tp[1], f"T{k}",
                        fontsize=6 if kind == "held" else 5,
                        color=color, zorder=zord)

        # ---- moving tools (interpolated) + T-shaped indicators ----
        for k, mv in moves.items():
            tool_pos = (1 - t) * mv["old"] + t * mv["new"]
            marker = _TOOL_MARKERS[k % len(_TOOL_MARKERS)]
            ax.plot(tool_pos[0], tool_pos[1], marker, color="black",
                    markersize=10, zorder=8)
            ax.text(tool_pos[0] + 0.015, tool_pos[1], f"T{k}",
                    fontsize=6, color="black", zorder=8)

            # T-shaped indicator
            actor = mv["actor"]
            if actor is not None:
                actor_pos = pos[actor]
                direction = tool_pos - actor_pos
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    unit = direction / dist
                    perp = np.array([-unit[1], unit[0]])

                    if mv["type"] == "give":
                        # Pusher: T-bar just before the tool (between actor and tool)
                        tbar_center = tool_pos - unit * 0.015
                    else:
                        # Take: T-bar just behind the tool (far side from actor)
                        tbar_center = tool_pos + unit * 0.015

                    tbar_half = 0.02
                    tbar_left = tbar_center + perp * tbar_half
                    tbar_right = tbar_center - perp * tbar_half

                    clr = "red" if mv["type"] == "give" else "green"
                    ax.plot([actor_pos[0], tool_pos[0]],
                            [actor_pos[1], tool_pos[1]],
                            color=clr, lw=2.5, zorder=9, alpha=0.7)
                    ax.plot([tbar_left[0], tbar_right[0]],
                            [tbar_left[1], tbar_right[1]],
                            color=clr, lw=3.0, zorder=9, alpha=0.7)

        # ---- goal arrows ----
        if goals:
            for g in goals:
                if isinstance(g, (HoldGoal, WorkbenchGoal)):
                    tp = _tool_render_pos(g.tool_idx, wb_old, holds_old, pos, n)
                    if g.tool_idx in moves:
                        mv = moves[g.tool_idx]
                        tp = (1 - t) * mv["old"] + t * mv["new"]
                    if tp is not None:
                        ax.annotate("", xy=tp, xytext=pos[g.agent_idx],
                                    arrowprops=dict(arrowstyle="->", color="blue",
                                                    lw=3.0, linestyle="dashed",
                                                    connectionstyle="arc3,rad=-0.3"))
                elif isinstance(g, IdleGoal):
                    circle = plt.Circle(pos[g.agent_idx], 0.06, fill=False,
                                        edgecolor="blue", linestyle="dashed",
                                        linewidth=3.0)
                    ax.add_patch(circle)

        # ---- persistent request arcs (already in old_state matrix) ----
        _REQ_OFFSET = 0.015
        for i in range(n):
            for k in range(m):
                if req_old[i][k]:
                    tp = _tool_render_pos(k, wb_old, holds_old, pos, n)
                    if k in moves:
                        mv = moves[k]
                        tp = (1 - t) * mv["old"] + t * mv["new"]
                    if tp is not None:
                        diff = tp - pos[i]
                        length = np.linalg.norm(diff)
                        if length > 1e-9:
                            perp = np.array([-diff[1], diff[0]]) / length
                        else:
                            perp = np.array([0.0, 0.0])
                        off = _REQ_OFFSET * perp
                        ax.annotate("", xy=tp + off, xytext=pos[i] + off,
                                    arrowprops=dict(arrowstyle="->",
                                                    color="royalblue", lw=2.5,
                                                    connectionstyle="arc3,rad=-0.3"))

        # ---- animated request arcs (acquire that resulted in a new request) ----
        for i in range(n):
            atype, aparam = decode_action(actions[i], m, env.give_targets[i])
            if atype == "acquire":
                k = aparam
                # Only animate if this is a newly-placed request
                if req_old[i][k] or not req_new[i][k]:
                    continue
                tp = _tool_render_pos(k, wb_old, holds_old, pos, n)
                if k in moves:
                    mv = moves[k]
                    tp = (1 - t) * mv["old"] + t * mv["new"]
                if tp is not None:
                    diff = tp - pos[i]
                    length = np.linalg.norm(diff)
                    if length > 1e-9:
                        perp = np.array([-diff[1], diff[0]]) / length
                    else:
                        perp = np.array([0.0, 0.0])
                    off = _REQ_OFFSET * perp
                    _draw_partial_arc(ax, pos[i] + off, tp + off, t,
                                      rad=-0.3, color="royalblue",
                                      lw=2.5, alpha=0.7)

        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        frames.append(buf)

    return frames


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_tools_env(
    n_agents: int = 4,
    n_tools: int = 6,
    max_steps: int = 30,
    p_failure: float = 0.1,
    seed: Optional[int] = None,
    robot_agent_indices: Optional[List[int]] = None,
    render_mode: Optional[str] = None,
    **kwargs,
) -> ToolsWorldModel:
    """Create a :class:`ToolsWorldModel` with sensible defaults."""
    return ToolsWorldModel(
        n_agents=n_agents,
        n_tools=n_tools,
        max_steps=max_steps,
        p_failure=p_failure,
        seed=seed,
        robot_agent_indices=robot_agent_indices,
        render_mode=render_mode,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Video helper
# ---------------------------------------------------------------------------
def save_tools_video(frames: List[np.ndarray], path: str, fps: int = 2):
    """Save a list of RGB frames as an mp4 video."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    if not frames:
        return
    h, w, _ = frames[0].shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(idx):
        im.set_data(frames[idx])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    try:
        writer = FFMpegWriter(fps=fps)
        anim.save(path, writer=writer)
    except Exception:
        # fallback: save as gif
        gif_path = path.rsplit(".", 1)[0] + ".gif"
        anim.save(gif_path, writer="pillow", fps=fps)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------
def _softmax(logits: np.ndarray, beta: float) -> np.ndarray:
    """Numerically stable softmax with temperature *beta*.

    When *beta* is ``inf``, returns a uniform distribution over the
    maximum-logit actions (deterministic argmax with tie-breaking).
    """
    if np.isinf(beta):
        max_val = logits.max()
        mask = logits == max_val
        out = np.zeros_like(logits, dtype=np.float64)
        out[mask] = 1.0 / mask.sum()
        return out
    x = beta * logits
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    if s == 0:
        return np.ones_like(logits) / len(logits)
    return e / s
