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

Actions per agent i (total = 1 + 2*n_tools + n_agents):
    0               : pass
    1..m            : take tool k  (k = action - 1)
    m+1..m+n        : give to agent j  (j = action - m - 1)
    m+n+1..2m+n     : request tool k  (k = action - m - n - 1)

Transition dynamics:
    With probability (1 - p_failure) all actions are attempted.
    With probability p_failure, one agent (drawn from p_fail[i]) has their
    action fail.  Actions are resolved in order of agent index (lowest first);
    later agents whose actions conflict with earlier ones simply fail.
    This keeps the number of successor states ≤ n + 1.

Goals:
    - HoldGoal(i, k): agent i holds tool k
    - WorkbenchGoal(i, k): tool k is on agent i's workbench
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
ACTION_PASS = 0


def _action_take(k: int) -> int:
    """Action index for 'take tool k'."""
    return 1 + k


def _action_give(j: int, n_tools: int) -> int:
    """Action index for 'give to agent j'."""
    return 1 + n_tools + j


def _action_request(k: int, n_tools: int, n_agents: int) -> int:
    """Action index for 'request tool k'."""
    return 1 + n_tools + n_agents + k


def decode_action(action: int, n_tools: int, n_agents: int):
    """Decode an action index into (type_str, param).

    Returns one of:
        ('pass', None)
        ('take', k)        — tool index k
        ('give', j)        — target agent index j
        ('request', k)     — tool index k
    """
    if action == ACTION_PASS:
        return "pass", None
    if 1 <= action <= n_tools:
        return "take", action - 1
    if n_tools + 1 <= action <= n_tools + n_agents:
        return "give", action - n_tools - 1
    if n_tools + n_agents + 1 <= action <= 2 * n_tools + n_agents:
        return "request", action - n_tools - n_agents - 1
    raise ValueError(
        f"Invalid action index {action} for n_tools={n_tools}, n_agents={n_agents}"
    )


def action_name(action: int, n_tools: int, n_agents: int) -> str:
    """Human-readable name for an action index."""
    atype, param = decode_action(action, n_tools, n_agents)
    if atype == "pass":
        return "pass"
    if atype == "take":
        return f"take T{param}"
    if atype == "give":
        return f"give→A{param}"
    if atype == "request":
        return f"req T{param}"
    return "?"


# ---------------------------------------------------------------------------
# ToolsWorldModel
# ---------------------------------------------------------------------------
class ToolsWorldModel(WorldModel):
    """WorldModel for tool exchange in a shared workshop.

    Inherits the default ``step()`` from :class:`WorldModel`, which calls
    ``transition_probabilities()`` and samples from it — no override needed.
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
        # Waxman parameters
        waxman_hear_alpha: float = 0.8,
        waxman_hear_beta: float = 0.5,
        waxman_reach_alpha: float = 0.6,
        waxman_reach_beta: float = 0.3,
        grab_prob: float = 0.5,
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

        # Action space: pass + take(m) + give(n) + request(m)
        self.n_actions = 1 + 2 * n_tools + n_agents
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # Planning code uses get_state() for the full state tuple.
        # observation_space is a dummy placeholder (same convention as MacroGridEnv).
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

        # RNG for graph / tool initialisation
        self._rng = np.random.RandomState(seed)

        # --- constant state (graphs) ---
        if agent_positions is not None:
            self.agent_positions = np.array(agent_positions, dtype=np.float64)
        else:
            self.agent_positions = self._rng.rand(n_agents, 2)
            # Place robot(s) at centre
            for ri in self._robot_indices:
                self.agent_positions[ri] = [0.5, 0.5]

        self.can_hear = (
            np.array(can_hear, dtype=bool)
            if can_hear is not None
            else self._waxman_graph(waxman_hear_alpha, waxman_hear_beta)
        )
        np.fill_diagonal(self.can_hear, True)

        self.can_reach = (
            np.array(can_reach, dtype=bool)
            if can_reach is not None
            else self._waxman_graph(waxman_reach_alpha, waxman_reach_beta)
        )
        np.fill_diagonal(self.can_reach, True)

        if can_grab is not None:
            self.can_grab = np.array(can_grab, dtype=bool)
        else:
            self.can_grab = self.can_reach.copy()
            mask = self._rng.rand(n_agents, n_agents) >= grab_prob
            self.can_grab[mask] = False
        np.fill_diagonal(self.can_grab, True)

        # --- mutable state ---
        self._remaining: int = max_steps
        self._workbench = np.zeros((n_agents, n_tools), dtype=bool)
        self._holds = np.zeros((n_agents, n_tools), dtype=bool)
        self._requested = np.zeros((n_agents, n_tools), dtype=bool)
        self._init_tools()

    # ----- helpers for construction -----
    def _waxman_graph(self, alpha: float, beta: float) -> np.ndarray:
        n = self.n_agents
        pos = self.agent_positions
        L = np.sqrt(2.0)
        g = np.eye(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    p = alpha * np.exp(-d / (beta * L))
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
        chosen_idx = np.random.choice(len(transitions), p=probabilities)
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
        atype, param = decode_action(action, self.n_tools, self.n_agents)
        if atype == "pass":
            return True
        if atype == "take":
            k = param
            for j in range(self.n_agents):
                if self.can_reach[agent, j] and workbench[j][k]:
                    return True
                if self.can_grab[agent, j] and holds[j][k]:
                    return True
            return False
        if atype == "give":
            j = param
            if not self.can_reach[agent, j]:
                return False
            return any(holds[agent])
        if atype == "request":
            k = param
            return not holds[agent][k] and not workbench[agent][k]
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
        if action_type == "pass":
            return True

        if action_type == "take":
            k = param
            # find source
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
            if source is None:
                return False
            # put down currently held tool (if any)
            for k2 in range(n_tools):
                if hd[agent][k2]:
                    hd[agent][k2] = 0
                    wb[agent][k2] = 1
                    break
            # remove from source
            if stype == "wb":
                wb[source][k] = 0
            else:
                hd[source][k] = 0
            # agent now holds tool k
            hd[agent][k] = 1
            # cancel own request for k
            rq[agent][k] = 0
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
            # cancel j's request for this tool
            rq[j][held] = 0
            return True

        if action_type == "request":
            k = param
            if hd[agent][k] or wb[agent][k]:
                return False
            # cancel previous request
            for k2 in range(n_tools):
                rq[agent][k2] = 0
            rq[agent][k] = 1
            return True

        return False

    # ----- transition probabilities -----
    def transition_probabilities(self, state, actions):
        remaining = state[0]
        if remaining <= 0:
            return None

        n = self.n_agents
        m = self.n_tools
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

            # Apply actions sequentially in agent-index order.  Each
            # action is applied against the *evolving* state, so a
            # higher-index agent whose action conflicts with an earlier
            # agent's successful action will naturally fail (e.g., tool
            # already moved).  This is the intended priority mechanism:
            # lower-index agents have implicit priority.
            for i in range(n):
                if i == failed:
                    continue
                atype, param = decode_action(actions[i], m, n)
                # feasibility check against current evolving state
                if atype == "pass":
                    pass  # always feasible, no effect
                elif atype == "take":
                    self._apply(
                        i, atype, param, m, n, self.can_reach, self.can_grab, wb, hd, rq
                    )
                elif atype == "give":
                    self._apply(
                        i, atype, param, m, n, self.can_reach, self.can_grab, wb, hd, rq
                    )
                elif atype == "request":
                    self._apply(
                        i, atype, param, m, n, self.can_reach, self.can_grab, wb, hd, rq
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

    # step() is inherited from WorldModel — it calls transition_probabilities
    # and samples from it, which is the desired behaviour.

    # ----- reconstruction helpers (for parallel DAG) -----
    def _get_construction_kwargs(self) -> dict:
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


# ---------------------------------------------------------------------------
# Goal generator / sampler
# ---------------------------------------------------------------------------
class ToolsGoalGenerator(PossibleGoalGenerator):
    """Enumerate all Hold and Workbench goals for every human agent."""

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
            self._agent_goals[i] = gs
        # Per-agent weight: 1/n_goals_per_agent so weights sum to 1 for each agent.
        # This is consistent with the sampler (uniform sampling, weight=1.0):
        #   generator_weight = sampler_weight * p(goal)  =>  1/n = 1.0 * (1/n)
        self._n_goals_per_agent = 2 * env.n_tools

    def generate(
        self, state, human_agent_index: int
    ) -> Iterator[Tuple[PossibleGoal, float]]:
        weight = 1.0 / self._n_goals_per_agent if self._n_goals_per_agent > 0 else 1.0
        for g in self._agent_goals.get(human_agent_index, []):
            yield g, weight


class ToolsGoalSampler(PossibleGoalSampler):
    """Uniformly sample a Hold or Workbench goal for a given human agent."""

    def __init__(self, env: ToolsWorldModel, indexed: bool = True):
        super().__init__(env, indexed=indexed)
        self._env = env
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
            self._agent_goals[i] = gs

    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        gs = self._agent_goals[human_agent_index]
        g = _stdlib_random.choice(gs)
        return g, 1.0


# ---------------------------------------------------------------------------
# Heuristic human policy prior
# ---------------------------------------------------------------------------
class ToolsHeuristicPolicy(HumanPolicyPrior):
    """Goal-conditioned heuristic policy for the tools environment.

    Implements the strategy described in the issue for HoldGoal and WorkbenchGoal.
    """

    def __init__(
        self,
        env: ToolsWorldModel,
        possible_goal_generator: ToolsGoalGenerator,
        beta: float = 5.0,
    ):
        super().__init__(env, env.human_agent_indices)
        self._env = env
        self._goal_gen = possible_goal_generator
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
        env = self._env
        n, m = env.n_agents, env.n_tools
        num_actions = env.n_actions
        _remaining, wb, holds, _requested = state

        # decode perceived state for this agent
        p_state = env.perceived_state(state, agent_idx)
        _, _, _, p_req = p_state

        logits = np.full(num_actions, -1e9)
        logits[ACTION_PASS] = 0.0  # pass always feasible, baseline logit

        if isinstance(goal, HoldGoal) and goal.agent_idx == agent_idx:
            k = goal.tool_idx
            # 1) already holding k → pass
            if holds[agent_idx][k]:
                logits[ACTION_PASS] = 10.0
                return _softmax(logits, self._beta)

            # 2) k on reachable workbench or grabbable hand → take
            for j in range(n):
                if env.can_reach[agent_idx, j] and wb[j][k]:
                    logits[_action_take(k)] = 5.0
                if env.can_grab[agent_idx, j] and holds[j][k]:
                    logits[_action_take(k)] = 5.0

            if logits[_action_take(k)] > 0:
                return _softmax(logits, self._beta)

            # 3) not reachable: request k if not already requested
            if not p_req[agent_idx][k]:
                logits[_action_request(k, m, n)] = 4.0
                return _softmax(logits, self._beta)

            # 4) already requested k → give held tool along shortest path
            self._give_along_path(agent_idx, holds, p_req, logits, n, m)
            return _softmax(logits, self._beta)

        if isinstance(goal, WorkbenchGoal) and goal.agent_idx == agent_idx:
            k = goal.tool_idx
            # 1) already on own workbench → pass
            if wb[agent_idx][k]:
                logits[ACTION_PASS] = 10.0
                return _softmax(logits, self._beta)

            # 2) holding k → give to self
            if holds[agent_idx][k]:
                logits[_action_give(agent_idx, m)] = 5.0
                return _softmax(logits, self._beta)

            # 3) k on reachable workbench or grabbable hand → take
            for j in range(n):
                if env.can_reach[agent_idx, j] and wb[j][k]:
                    logits[_action_take(k)] = 5.0
                if env.can_grab[agent_idx, j] and holds[j][k]:
                    logits[_action_take(k)] = 5.0
            if logits[_action_take(k)] > 0:
                return _softmax(logits, self._beta)

            # 4) not reachable: request k if not already requested
            if not p_req[agent_idx][k]:
                logits[_action_request(k, m, n)] = 4.0
                return _softmax(logits, self._beta)

            # 5) already requested k → give held tool along shortest path
            self._give_along_path(agent_idx, holds, p_req, logits, n, m)
            return _softmax(logits, self._beta)

        # goal does not apply to this agent → uniform over feasible actions
        for a in range(num_actions):
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
    ):
        """Set logit for giving the currently-held tool toward a requester."""
        env = self._env
        # find held tool
        held = None
        for k2 in range(m):
            if holds[agent_idx][k2]:
                held = k2
                break
        if held is None:
            return  # nothing to give

        # find agents who requested held tool (smallest-index first)
        requesters = [j for j in range(n) if perceived_requested[j][held]]
        if requesters:
            path = self._bfs_shortest(env.can_reach, agent_idx, requesters)
            if path and len(path) >= 2:
                next_hop = path[1]
                logits[_action_give(next_hop, m)] = 4.0
                return

        # no requester or unreachable → keep the tool (pass)

    # ------ HumanPolicyPrior interface ------
    def __call__(self, state, human_agent_index: int, possible_goal=None):
        if possible_goal is not None:
            return self._action_distribution(state, human_agent_index, possible_goal)

        # marginalise over goals
        dist = np.zeros(self._env.n_actions, dtype=np.float64)
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
                marker = _TOOL_MARKERS[k % len(_TOOL_MARKERS)]
                ax.plot(
                    pos[i, 0],
                    pos[i, 1] + 0.04,
                    marker,
                    color="black",
                    markersize=10,
                    zorder=7,
                )
                ax.text(
                    pos[i, 0] + 0.03,
                    pos[i, 1] + 0.04,
                    f"T{k}",
                    fontsize=6,
                    color="black",
                    zorder=7,
                )

        # workbench tools (small shapes around agent)
        wb_tools = [k for k in range(m) if wb[i][k]]
        for ti, k in enumerate(wb_tools):
            angle = 2 * np.pi * ti / max(len(wb_tools), 1) - np.pi / 2
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

    # ---- has_requested arrows (solid blue) ----
    for i in range(n):
        for k in range(m):
            if req[i][k]:
                # find where tool k currently is
                tool_pos = _tool_render_pos(k, wb, holds, pos, n)
                if tool_pos is not None:
                    ax.annotate(
                        "",
                        xy=tool_pos,
                        xytext=pos[i],
                        arrowprops=dict(
                            arrowstyle="->",
                            color="royalblue",
                            lw=1.2,
                            connectionstyle="arc3,rad=0.2",
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
                            lw=1.5,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=-0.3",
                        ),
                    )

    if own_fig:
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf

    return None


def _tool_render_pos(k, wb, holds, positions, n):
    """Find approximate render position of tool k."""
    for i in range(n):
        if holds[i][k]:
            return positions[i] + np.array([0.0, 0.04])
        if wb[i][k]:
            return positions[i] + np.array([0.06, 0.0])
    return None


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
    """Numerically stable softmax with temperature *beta*."""
    x = beta * logits
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    if s == 0:
        return np.ones_like(logits) / len(logits)
    return e / s
